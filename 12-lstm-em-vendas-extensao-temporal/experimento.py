"""Monta os exemplos, treina a LSTM e prevê as origens do teste. Aula «LSTM em vendas: extensão temporal».

Este arquivo não imprime nada. O comparar_lstm.py e o minha_comparacao.py importam daqui.

Uma configuração é um dicionário com estas chaves:
    horizonte       dias entre a noite da origem e o dia previsto (7 no ajuste da semana)
    coluna          "unidades" ou "pedidos"
    janela          quantos dias, terminando na origem, a LSTM lê
    estado          tamanho do h e do c
    calendario      True: a LSTM vê os eventos dos dias da janela e o calendário do dia previsto
    fim_do_treino   último dia previsto que entra no treino (data)
    fim_da_escolha  último dia previsto que a validação usa (data); o teste começa depois
    validacao       "depois_do_treino": a validação são os dias previstos entre os dois fins
                    "sorteio": 20% dos exemplos sorteados entre todos, como o Caio fez
    normalizar_com  "treino": média e desvio do log da venda só até o fim do treino
                    "tudo": média e desvio de 2024 e 2025 inteiros
    primeira_origem e ultima_origem   as noites do teste
    retreino        "nenhum": treina uma vez, no fim da escolha; "mensal": treina na primeira noite do
                    teste e de novo na última noite de cada mês, com a venda até ali
    sorteios        lista com os números do sorteio dos pesos iniciais
    epocas, taxa    do Adam
"""
import numpy as np
import pandas as pd

import lstm as L
import serie as S

AULA = {"horizonte": 7, "coluna": "unidades", "janela": 28, "estado": 16, "calendario": True,
        "fim_do_treino": "2025-05-31", "fim_da_escolha": "2025-08-31", "validacao": "depois_do_treino",
        "normalizar_com": "treino", "primeira_origem": "2025-08-31", "ultima_origem": "2025-12-24",
        "retreino": "nenhum", "sorteios": [0, 1, 2, 3, 4], "epocas": 300, "taxa": 0.01}

CHAVES = set(AULA)


def validar(cfg):
    """Recusa a configuração que não dá para rodar. Não julga se ela respeita o corte: isso é dos testes."""
    faltam = sorted(CHAVES - set(cfg))
    sobram = sorted(set(cfg) - CHAVES)
    if faltam or sobram:
        raise SystemExit(f"configuração recusada: faltam {faltam}, sobram {sobram}")
    if cfg["coluna"] not in ("unidades", "pedidos") or cfg["validacao"] not in ("depois_do_treino", "sorteio") \
            or cfg["normalizar_com"] not in ("treino", "tudo") or cfg["retreino"] not in ("nenhum", "mensal"):
        raise SystemExit("configuração recusada: coluna, validacao, normalizar_com ou retreino com valor que não existe")
    if not (1 <= int(cfg["horizonte"]) <= 60 and 7 <= int(cfg["janela"]) <= 120 and 2 <= int(cfg["estado"]) <= 64):
        raise SystemExit("configuração recusada: horizonte de 1 a 60, janela de 7 a 120, estado de 2 a 64")
    datas = [pd.Timestamp(cfg[k]) for k in ("fim_do_treino", "fim_da_escolha", "primeira_origem", "ultima_origem")]
    if not (S.INICIO < datas[0] <= datas[1] and datas[2] <= datas[3] <= S.FIM):
        raise SystemExit("configuração recusada: as datas precisam estar em ordem e dentro de 2024 e 2025")
    if not cfg["sorteios"]:
        raise SystemExit("configuração recusada: pelo menos um sorteio")


def exemplos(d, origens, cfg, media, desvio):
    """Uma linha por origem com dia previsto usável: a janela até a origem, o calendário do dia, o alvo."""
    h, T = int(cfg["horizonte"]), int(cfg["janela"])
    X, C, y, dias = [], [], [], []
    for o in origens:
        dia = o + pd.Timedelta(days=h)
        if dia > S.FIM or np.isnan(d.at[dia, "usavel"]) or o - pd.Timedelta(days=T - 1) < S.INICIO:
            continue
        X.append(S.janela(d, o, T, media, desvio, cfg["calendario"]))
        C.append(S.calendario_do_dia(d, dia) if cfg["calendario"] else np.zeros(0))
        y.append((np.log(d.at[dia, "usavel"]) - media) / desvio)
        dias.append(dia)
    return np.array(X), np.array(C).reshape(len(X), -1), np.array(y), dias


def fins(cfg, corte):
    """O fim do treino e o fim da escolha para um treino feito na noite `corte`."""
    corte = pd.Timestamp(corte)
    if cfg["retreino"] == "mensal" and corte != pd.Timestamp(cfg["fim_da_escolha"]):
        return corte - pd.DateOffset(months=3), corte
    return pd.Timestamp(cfg["fim_do_treino"]), corte


def normalizacao(d, cfg, fim_treino):
    """Média e desvio do log da venda usável, até o fim do treino ou na série inteira."""
    ate = fim_treino if cfg["normalizar_com"] == "treino" else S.FIM
    serie = np.log(d.loc[:ate, "usavel"].dropna())
    return float(serie.mean()), float(serie.std())


def ajustar(d, cfg, corte, sorteio):
    """Treina uma rede com os dias previstos até `corte`. Retorna a rede e um resumo do treino."""
    h = int(cfg["horizonte"])
    fim_treino, fim_escolha = fins(cfg, corte)
    media, desvio = normalizacao(d, cfg, fim_treino)
    if cfg["validacao"] == "sorteio":
        origens = pd.date_range(S.INICIO, S.FIM - pd.Timedelta(days=h))
        X, C, y, dias = exemplos(d, origens, cfg, media, desvio)
        val = np.random.default_rng(100).random(len(y)) < 0.2
        tr = ~val
    else:
        origens = pd.date_range(S.INICIO, fim_escolha - pd.Timedelta(days=h))
        X, C, y, dias = exemplos(d, origens, cfg, media, desvio)
        tr = np.array([x <= fim_treino for x in dias])
        val = ~tr
    rede = L.nova_rede(X.shape[2], int(cfg["estado"]), C.shape[1], sorteio)
    rede, epoca, curva = L.treinar(rede, (X[tr], C[tr], y[tr]), (X[val], C[val], y[val]),
                                   epocas=int(cfg["epocas"]), taxa=float(cfg["taxa"]))
    info = {"media": media, "desvio": desvio, "epoca": epoca, "curva": curva, "n_treino": int(tr.sum()),
            "n_validacao": int(val.sum()), "dias_validacao": [x for x, v in zip(dias, val) if v],
            "fim_treino": fim_treino, "fim_escolha": fim_escolha, "parametros": L.parametros(rede),
            "validacao_X": (X[val], C[val], y[val])}
    return rede, info


def prever(d, cfg, rede, info, origem):
    """A previsão do dia origem + horizonte, feita só com a venda até a noite da origem."""
    dia = origem + pd.Timedelta(days=int(cfg["horizonte"]))
    if d.at[dia, "fechada"]:                 # o calendário já sabe: loja fechada, previsão zero
        return 0.0
    X = S.janela(d, origem, int(cfg["janela"]), info["media"], info["desvio"], cfg["calendario"])[None]
    C = (S.calendario_do_dia(d, dia) if cfg["calendario"] else np.zeros(0))[None]
    z, _ = L.frente(rede, X, C)
    return float(np.exp(info["media"] + info["desvio"] * z[0]))


def na_validacao(info, rede):
    """A previsão de cada exemplo da validação, de volta em unidades (ou pedidos), e a venda real."""
    X, C, y = info["validacao_X"]
    z, _ = L.frente(rede, X, C)
    return np.exp(info["media"] + info["desvio"] * y), np.exp(info["media"] + info["desvio"] * z)


def rodar(cfg, d=None):
    """Treina cada sorteio, prevê todas as origens do teste e junta tudo numa tabela, um dia previsto por linha."""
    validar(cfg)
    d = S.carregar(cfg["coluna"]) if d is None else d
    h = int(cfg["horizonte"])
    origens = pd.date_range(cfg["primeira_origem"], cfg["ultima_origem"])
    linhas = [{"origem": o, "dia": o + pd.Timedelta(days=h)} for o in origens]
    tab = pd.DataFrame(linhas)
    tab["real"] = [d.at[x, "real"] for x in tab["dia"]]
    tab["avaliado"] = [bool(not np.isnan(d.at[x, "real"]) and not d.at[x, "incompleto"]) for x in tab["dia"]]
    tab["evento"] = [bool(d.at[x, "evento"]) for x in tab["dia"]]
    redes = {}
    for s in cfg["sorteios"]:
        if cfg["retreino"] == "mensal":       # a primeira noite do teste e a última noite de cada mês
            cortes = sorted({origens[0]} | {o for o in origens if (o + pd.Timedelta(days=1)).day == 1})
        else:
            cortes = [pd.Timestamp(cfg["fim_da_escolha"])]
        ajustes = [(c, *ajustar(d, cfg, c, s)) for c in cortes]
        prev = []
        for o in origens:
            c, rede, info = [a for a in ajustes if a[0] <= o][-1] if cfg["retreino"] == "mensal" else ajustes[0]
            prev.append(prever(d, cfg, rede, info, o))
        tab[f"lstm {s}"] = prev
        redes[s] = ajustes
    return tab, redes, d


def com_baselines(tab, d, h):
    """Acrescenta a semana passada, a régua e o calendário + AR(1), nas mesmas origens."""
    tab = tab.copy()
    tab["semana passada"] = [0.0 if d.at[x, "fechada"] else S.semana_passada(d, x, h) for x in tab["dia"]]
    tab["mesmo dia, 4 semanas"] = [0.0 if d.at[x, "fechada"] else S.mesmo_dia_4_semanas(d, x, h) for x in tab["dia"]]
    tab["calendário + AR(1)"] = [0.0 if d.at[x, "fechada"] else S.calendario_ar1(d, o, x)
                                 for o, x in zip(tab["origem"], tab["dia"])]
    return tab


def teste_do_futuro(cfg, d, origens_teste, vezes=3.0):
    """Multiplica por `vezes` toda a venda depois de uma noite e vê o que muda no que a LSTM recebe.

    Para o treino, a noite é o fim da escolha: a normalização e os exemplos de treino e de validação
    não podem mudar. Para cada origem do teste, a janela e o calendário do dia previsto também não.
    Retorna a lista do que mudou; vazia quer dizer que nada do futuro entra.
    """
    mudou = []
    corte = pd.Timestamp(cfg["fim_da_escolha"])
    fim_treino, _ = fins(cfg, corte)
    f = d.copy()
    f.loc[f.index > corte, "usavel"] *= vezes
    m1, m2 = normalizacao(d, cfg, fim_treino), normalizacao(f, cfg, fim_treino)
    if not np.allclose(m1, m2):
        mudou.append(f"normalização: média {m1[0]:.4f} vira {m2[0]:.4f}")
    h = int(cfg["horizonte"])
    if cfg["validacao"] == "sorteio":
        origens = pd.date_range(S.INICIO, S.FIM - pd.Timedelta(days=h))
    else:
        origens = pd.date_range(S.INICIO, corte - pd.Timedelta(days=h))
    a, b = exemplos(d, origens, cfg, *m1), exemplos(f, origens, cfg, *m1)
    if len(a[2]) != len(b[2]) or not (np.allclose(a[0], b[0]) and np.allclose(a[2], b[2])):
        mudou.append(f"exemplos de treino e validação: {int(np.sum(~np.isclose(a[2], b[2])))} alvos mudam")
    for o in origens_teste:
        g = d.copy()
        g.loc[g.index > o, "usavel"] *= vezes
        dia = o + pd.Timedelta(days=h)
        j1 = S.janela(d, o, int(cfg["janela"]), *m1, cfg["calendario"])
        j2 = S.janela(g, o, int(cfg["janela"]), *m1, cfg["calendario"])
        if not np.allclose(j1, j2) or not np.allclose(S.calendario_do_dia(d, dia), S.calendario_do_dia(g, dia)):
            mudou.append(f"janela da origem {o:%d/%m/%Y}")
    return mudou
