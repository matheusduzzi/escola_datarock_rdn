"""Coloca a LSTM na mesma comparação do curso Séries temporais. Aula «LSTM em vendas: extensão temporal».

Rode de dentro da pasta da aula:
    python3 comparar_lstm.py                      # tudo o que a aula mostra, partes 1 a 8
    python3 comparar_lstm.py --janela 56          # prática guiada: só a LSTM com calendário, com outra janela
    python3 comparar_lstm.py --retreino mensal    # prática guiada: treina de novo no fim de cada mês
    python3 comparar_lstm.py --estado 4           # prática guiada: uma LSTM menor

O ajuste da semana: na noite da origem, a Bia prevê o dia que fica 7 dias depois. As origens do
teste vão de 31/08/2025 a 24/12/2025. Todos os modelos preveem os mesmos dias, feitos nas mesmas
noites, e o erro é medido nos dias que nenhum deles viu.
"""
import argparse
import time

import numpy as np
import pandas as pd

import experimento as E
import lstm as L
import serie as S

REGUA = "mesmo dia, 4 semanas"


def lstm_media(tab, cfg):
    """A previsão da LSTM: a média das previsões das redes dos sorteios, regra decidida antes do teste."""
    return tab[[f"lstm {s}" for s in cfg["sorteios"]]].mean(axis=1)


def linha(nome, real, prev, largura=26):
    vies = (np.sum(prev) - np.sum(real)) / np.sum(real) * 100
    print(f"  {nome:{largura}}{S.br(np.mean(np.abs(real - prev))):>8}{S.br(S.wmape(real, prev), 1) + '%':>9}"
          f"{S.sinal(vies) + '%':>8}{S.br(S.custo(real, prev), 0):>13}")


def cabecalho(largura=26):
    print(f"  {'':{largura}}{'MAE':>8}{'WMAPE':>9}{'viés':>8}{'custo (R$)':>13}")


def semanas(av, nomes, contra):
    av = av.assign(semana=(av["dia"] - av["dia"].iloc[0]).dt.days // 7)
    erro_ref = (av["real"] - av[contra]).abs()
    print(f"  {'':26}{'semanas em que erra menos':>27}{'erro a menos por dia':>23}")
    for nome in nomes:
        ganho = erro_ref - (av["real"] - av[nome]).abs()          # positivo: o modelo errou menos
        por = ganho.groupby(av["semana"])
        medias = por.mean()
        faixa = 2 * medias.std(ddof=1) / np.sqrt(len(medias))
        print(f"  {nome:26}{f'{int((por.sum() > 0).sum())} de {len(medias)}':>27}"
              f"{S.sinal(ganho.mean()) + ' ± ' + S.br(faixa, 1):>23}")


def uma_mudanca_por_vez(d):
    """A conta do Caio e o que acontece quando eu mudo uma coisa por vez até chegar no plano da aula."""
    caio = dict(E.AULA, horizonte=1, validacao="sorteio", normalizar_com="tudo")
    passos = [
        ("a conta do Caio", caio, "validação sorteada"),
        ("a mesma rede, nos dias do teste", caio, "teste"),
        ("7 dias antes, validação sorteada", dict(caio, horizonte=7), "validação sorteada"),
        ("7 dias antes, corte no tempo", dict(caio, horizonte=7, validacao="depois_do_treino"), "teste"),
        ("e normalizar só com o treino", dict(E.AULA), "teste"),
    ]
    print(f"  {'':34}{'dias medidos':>27}{'MAE':>8}{'WMAPE':>9}")
    guardado = {}
    for nome, cfg, onde in passos:
        chave = repr(sorted(cfg.items()))
        if chave not in guardado:
            guardado[chave] = E.rodar(cfg, d)
        tab, redes, _ = guardado[chave]
        if onde == "validação sorteada":
            reais, prevs = zip(*[E.na_validacao(redes[s][0][2], redes[s][0][1]) for s in cfg["sorteios"]])
            real, prev = reais[0], np.mean(prevs, axis=0)
            dias = redes[cfg["sorteios"][0]][0][2]["dias_validacao"]
            onde_txt = f"{len(real)} sorteados de 2024 e 2025"
            depois = sum(1 for x in dias if x > pd.Timestamp(cfg["fim_da_escolha"]))
        else:
            av = tab[tab["avaliado"]]
            real, prev = av["real"].to_numpy(), lstm_media(av, cfg).to_numpy()
            onde_txt = f"{len(real)} de 07/09 a 31/12/2025" if cfg["horizonte"] == 7 else \
                f"{len(real)} de 01/09 a 25/12/2025"
            depois = None
        print(f"  {nome:34}{onde_txt:>27}{S.br(np.mean(np.abs(real - prev))):>8}"
              f"{S.br(S.wmape(real, prev), 1) + '%':>9}")
        if depois is not None and nome == "a conta do Caio":
            print(f"  {'':34}{f'{depois} deles depois de 31/08/2025':>27}")
        if nome == "a mesma rede, nos dias do teste":
            val = set(redes[cfg["sorteios"][0]][0][2]["dias_validacao"])
            no_treino = sum(1 for x in tab.loc[tab["avaliado"], "dia"] if x not in val)
            print(f"  {'':34}{f'{no_treino} deles estavam no treino':>27}")
    return caio, guardado


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--janela", type=int, help="dias que a LSTM lê, terminando na origem (a aula usa 28)")
    ap.add_argument("--retreino", choices=["nenhum", "mensal"], help="treinar de novo no fim de cada mês")
    ap.add_argument("--estado", type=int, help="tamanho do h e do c (a aula usa 16)")
    a = ap.parse_args()
    inicio = time.time()
    d = S.carregar("unidades")
    cfg = dict(E.AULA)
    so_pratica = any(x is not None for x in (a.janela, a.retreino, a.estado))
    if a.janela is not None:
        cfg["janela"] = a.janela
    if a.retreino is not None:
        cfg["retreino"] = a.retreino
    if a.estado is not None:
        cfg["estado"] = a.estado

    if so_pratica:
        tab, redes, _ = E.rodar(cfg, d)
        tab = E.com_baselines(tab, d, 7)
        av = tab[tab["avaliado"]].assign(lstm=lambda t: lstm_media(t, cfg))
        info = redes[0][0][2]
        print(f"LSTM com calendário: janela {cfg['janela']} dias, estado {cfg['estado']}, retreino {cfg['retreino']}, "
              f"{info['parametros']} parâmetros, {len(cfg['sorteios'])} sorteios")
        print(f"  {len(av)} dias previstos 7 dias antes, de 07/09 a 31/12/2025")
        cabecalho()
        for nome in (REGUA, "calendário + AR(1)"):
            linha(nome, av["real"], av[nome])
        linha("LSTM com calendário", av["real"], av["lstm"])
        val = np.mean([redes[s][0][2]["curva"][redes[s][0][2]["epoca"]][2] for s in cfg["sorteios"]])
        print(f"  perda de validação, média dos sorteios (dias de 01/06 a 31/08/2025): {S.br(val, 4)}")
        print("  erro de cada sorteio (WMAPE): " + "; ".join(
            f"{S.br(S.wmape(av['real'], av[f'lstm {s}']), 1)}%" for s in cfg["sorteios"]))
        print(f"  treinos: {sum(len(v) for v in redes.values())}; tempo: {time.time() - inicio:.0f} s")
        return

    print("1. A conta do Caio e o que muda, uma coisa por vez (LSTM com calendário, média de 5 sorteios)")
    caio, guardado = uma_mudanca_por_vez(d)
    print()

    print("2. O plano: o ajuste da semana, com o corte do curso Séries temporais")
    tab, redes, _ = guardado[repr(sorted(cfg.items()))]
    info = redes[0][0][2]
    print(f"  treino: dias previstos de {pd.Timestamp('2024-01-01') + pd.Timedelta(days=cfg['janela'] - 1 + 7):%d/%m/%Y} "
          f"a {info['fim_treino']:%d/%m/%Y} ({info['n_treino']} exemplos)")
    print(f"  validação, para escolher a época: dias previstos de 01/06/2025 a {info['fim_escolha']:%d/%m/%Y} "
          f"({info['n_validacao']} exemplos)")
    print(f"  teste: origens de {pd.Timestamp(cfg['primeira_origem']):%d/%m/%Y} a {pd.Timestamp(cfg['ultima_origem']):%d/%m/%Y} "
          f"({len(tab)} noites), {int(tab['avaliado'].sum())} dias com venda real completa")
    print(f"  cada exemplo: {cfg['janela']} dias até a noite da origem; o dia previsto fica {cfg['horizonte']} dias depois")
    print(f"  normalização: média {S.br(info['media'], 4)} e desvio {S.br(info['desvio'], 4)} do log da venda, "
          f"só até {info['fim_treino']:%d/%m/%Y}")
    origens = list(pd.date_range(cfg["primeira_origem"], cfg["ultima_origem"], freq="14D"))
    mudou = E.teste_do_futuro(cfg, d, origens)
    print(f"  teste do futuro (venda depois de cada noite multiplicada por 3, {len(origens)} origens): "
          + ("nada muda, o plano não lê o futuro" if not mudou else f"muda {mudou}"))
    mudou_caio = E.teste_do_futuro(caio, d, origens)
    print("  o mesmo teste na conta do Caio: " + "; ".join(mudou_caio))
    print()

    tab = E.com_baselines(tab, d, 7)
    so_venda = dict(cfg, calendario=False)
    tab_sv, redes_sv, _ = E.rodar(so_venda, d)
    tab["LSTM só com a venda"] = lstm_media(tab_sv, so_venda)
    tab["LSTM com calendário"] = lstm_media(tab, cfg)
    av = tab[tab["avaliado"]]
    nomes = ["semana passada", REGUA, "LSTM só com a venda", "LSTM com calendário", "calendário + AR(1)"]
    print(f"3. Fora do tempo, nos mesmos {len(av)} dias, previstos 7 dias antes")
    cabecalho()
    for nome in nomes:
        linha(nome, av["real"], av[nome])
    print(f"  viés: soma de (previsão − real) sobre a venda; custo: faltar R$ {S.br(S.FALTA, 0)}, "
          f"sobrar R$ {S.br(S.SOBRA, 0)} por unidade")
    print()

    print("4. Cinco sorteios dos pesos iniciais, a mesma rede e os mesmos dados")
    print(f"  {'sorteio':>9}{'época escolhida':>17}{'perda de validação':>20}{'WMAPE no teste':>16}")
    for s in cfg["sorteios"]:
        i = redes[s][0][2]
        print(f"  {s:>9}{i['epoca']:>17}{S.br(i['curva'][i['epoca']][2], 4):>20}"
              f"{S.br(S.wmape(av['real'], av[f'lstm {s}']), 1) + '%':>16}")
    val = np.mean([redes[s][0][2]["curva"][redes[s][0][2]["epoca"]][2] for s in cfg["sorteios"]])
    print(f"  {'média dos 5':>9}{'':>17}{S.br(val, 4):>20}")
    print(f"  {'média das 5 previsões':>46}{S.br(S.wmape(av['real'], av['LSTM com calendário']), 1) + '%':>16}")
    print()

    print("5. Dias comuns e dias de evento (promoção ou feriado com a loja aberta)")
    for rotulo, parte in (("comuns", av[~av["evento"]]), ("de evento", av[av["evento"]])):
        print(f"  {len(parte)} dias {rotulo}: " + ", ".join(
            f"{n} {S.br(S.wmape(parte['real'], parte[n]), 1)}%" for n in nomes[1:]))
    x = av.set_index("dia").loc[pd.Timestamp("2025-11-28")]
    print(f"  Black Friday 28/11, venda {S.br(x['real'], 0)}: " + ", ".join(f"{n} {S.br(x[n], 0)}" for n in nomes[1:]))
    print()

    print(f"6. Semana a semana, contra a régua ({REGUA})")
    semanas(av, ["LSTM só com a venda", "LSTM com calendário", "calendário + AR(1)"], REGUA)
    print("  e contra o calendário + AR(1)")
    semanas(av, ["LSTM com calendário"], "calendário + AR(1)")
    print("  erro a menos por dia: média e ± 2 erros-padrão das médias semanais, em unidades")
    print()

    print("7. O tamanho de cada um")
    print(f"  LSTM com calendário: {info['parametros']} parâmetros, {info['n_treino']} exemplos de treino, "
          f"{len(cfg['sorteios'])} treinos de até {cfg['epocas']} épocas")
    print(f"  LSTM só com a venda: {redes_sv[0][0][2]['parametros']} parâmetros")
    cols = S.colunas_calendario(d, d.loc[:"2025-08-31"].index)
    print(f"  calendário + AR(1): {cols.shape[1]} pesos da regressão e 1 do AR(1), reajustados em cada uma das "
          f"{len(tab)} noites")
    X, C, y, _ = E.exemplos(d, pd.date_range("2024-02-01", "2024-03-01"), cfg, info["media"], info["desvio"])
    rede = L.nova_rede(X.shape[2], cfg["estado"], C.shape[1], 0)
    pior = L.checar_gradiente(rede, X, C, y)
    print(f"  checagem da volta pelo tempo: em 72 pesos sorteados, a maior diferença relativa entre o gradiente "
          f"e a diferença finita é {pior:.1e}")
    print()

    print("8. O que a LSTM fez com a Black Friday")
    o = pd.Timestamp("2025-11-21")
    for nome, t in (("com calendário", tab), ("só com a venda", tab_sv)):
        print(f"  LSTM {nome}, origem 21/11, previsão para 28/11: "
              + "; ".join(S.br(t.set_index("origem").at[o, f"lstm {s}"], 0) for s in cfg["sorteios"]))
    pos = cfg["estado"] + 10                     # na densa: o h, depois seno e cosseno (4), dia da semana (6), eventos
    efeitos = [np.exp(info["desvio"] * redes[s][0][1]["w"][pos]) - 1 for s in cfg["sorteios"]]
    print("  LSTM com calendário, quanto a coluna da Black Friday muda a previsão do dia: "
          + "; ".join(S.sinal(100 * e, 0) + "%" for e in efeitos))
    tr = pd.Timestamp("2024-11-29")
    print(f"  no treino, uma Black Friday: 29/11/2024, venda {S.br(d.at[tr, 'real'], 0)}")
    print(f"tempo total: {time.time() - inicio:.0f} s")


if __name__ == "__main__":
    main()
