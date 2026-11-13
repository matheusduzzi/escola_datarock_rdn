"""O exercício da aula «LSTM em vendas: extensão temporal»: a previsão de amanhã da tarefa 5.

Rode de dentro da pasta da aula:
    python3 minha_comparacao.py

A CONFIG abaixo é a do Caio. Você muda os valores e roda de novo. As chaves são as do
experimento.py; o começo daquele arquivo diz o que cada uma faz.

A comparação é a da tarefa 5 do diagnóstico: a previsão de cada dia de novembro e dezembro de
2025 sai na noite anterior, depois que a loja fecha, e só usa o que já se sabia nessa hora. Os
dias medidos e as três previsões simples são as do tarefa5.py, para os números baterem.
"""
import numpy as np
import pandas as pd

import experimento as E
import serie as S

CONFIG = {"horizonte": 7, "coluna": "pedidos", "janela": 28, "estado": 16, "calendario": True,
          "fim_do_treino": "2025-08-31", "fim_da_escolha": "2025-10-31", "validacao": "sorteio",
          "normalizar_com": "tudo", "primeira_origem": "2025-10-25", "ultima_origem": "2025-12-24",
          "retreino": "nenhum", "sorteios": [0, 1, 2], "epocas": 300, "taxa": 0.01}

CORTE = pd.Timestamp("2025-11-01")


def dias_da_tarefa5(d):
    """Os dias medidos na tarefa 5 e as três previsões dela, com a mesma conta do tarefa5.py."""
    serie = d["real"]
    t = pd.DataFrame({"real": serie, "ontem": serie.shift(1), "semana passada": serie.shift(7)})
    t["média antes do corte"] = serie[serie.index < CORTE].mean()
    t = t[(t.index >= CORTE) & ~d["fechada"]].dropna()
    return t


def comparar(cfg, d=None):
    """Roda a LSTM com a CONFIG e junta com a tarefa 5 nos dias dela. Retorna a tabela e o resumo do treino."""
    E.validar(cfg)
    d = S.carregar(cfg["coluna"]) if d is None else d
    t5 = dias_da_tarefa5(d)
    tab, redes, _ = E.rodar(cfg, d)
    lstm = tab.set_index("dia")[[f"lstm {s}" for s in cfg["sorteios"]]].mean(axis=1)
    t5["LSTM"] = lstm.reindex(t5.index)
    h = int(cfg["horizonte"])
    t5["calendário + AR(1)"] = [S.calendario_ar1(d, x - pd.Timedelta(days=h), x) for x in t5.index]
    return t5, redes, d


def main():
    cfg = CONFIG
    E.validar(cfg)
    print("configuração: " + ", ".join(f"{k} {v}" for k, v in cfg.items()))
    d = S.carregar(cfg["coluna"])
    origens = list(pd.date_range(cfg["primeira_origem"], cfg["ultima_origem"], freq="10D"))
    mudou = E.teste_do_futuro(cfg, d, origens)
    print("teste do futuro: " + ("nada muda, a configuração não lê o futuro" if not mudou else "; ".join(mudou)))
    t5, redes, d = comparar(cfg, d)
    info = redes[cfg["sorteios"][0]][0][2]
    print(f"treino: {info['n_treino']} exemplos; validação: {info['n_validacao']} exemplos; "
          f"épocas escolhidas: {', '.join(str(redes[s][0][2]['epoca']) for s in cfg['sorteios'])}")
    h = int(cfg["horizonte"])
    feitas = [x - pd.Timedelta(days=h) for x in t5.index[t5["LSTM"].notna()]]
    if feitas:
        print(f"previsões da LSTM feitas {h} dia(s) antes, origens de {min(feitas):%d/%m/%Y} a {max(feitas):%d/%m/%Y}")
    falta = int(t5["LSTM"].isna().sum())
    print(f"dias da tarefa 5: {len(t5)}, de {t5.index[0]:%d/%m/%Y} a {t5.index[-1]:%d/%m/%Y}; "
          f"sem previsão da LSTM: {falta}")
    print(f"\nerro médio absoluto nos dias da tarefa 5 ({cfg['coluna']} por dia; média {S.br(t5['real'].mean(), 1)})")
    av = t5.dropna()
    for nome in ["ontem", "semana passada", "média antes do corte", "LSTM", "calendário + AR(1)"]:
        print(f"  {nome:24}{S.br(np.mean(np.abs(av['real'] - av[nome])), 1):>8}")
    if len(av) != len(t5):
        print(f"  (só {len(av)} dos {len(t5)} dias têm previsão da LSTM; a comparação não é nos mesmos dias)")
    semana = (av.index - av.index[0]).days // 7
    ganho = (av["real"] - av["semana passada"]).abs() - (av["real"] - av["LSTM"]).abs()
    medias = ganho.groupby(semana).mean()
    faixa = 2 * medias.std(ddof=1) / np.sqrt(len(medias))
    print(f"\nsemana a semana, LSTM contra a semana passada: erra menos em {int((ganho.groupby(semana).sum() > 0).sum())} "
          f"de {len(medias)} semanas; erro a menos por dia {S.sinal(ganho.mean())} ± {S.br(faixa, 1)}")


if __name__ == "__main__":
    main()
