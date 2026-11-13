"""Os testes do exercício da aula «LSTM em vendas: extensão temporal».

Rode de dentro da pasta da aula:
    python3 testar_minha_comparacao.py

Eles leem a CONFIG do minha_comparacao.py e dizem o que ainda não respeita a tarefa 5 ou o corte.
Com a configuração do Caio, vários falham e o programa sai com código 1. É o esperado.
Os testes não treinam a rede; quem treina é o minha_comparacao.py.
"""
import importlib
import sys

import numpy as np
import pandas as pd

import experimento as E
import serie as S

falhas, passaram = [], []


def teste(nome, cond, detalhe=""):
    (passaram if cond else falhas).append(nome)
    if not cond:
        print(f"FALHOU {nome}: {detalhe}")


def main():
    try:
        mc = importlib.import_module("minha_comparacao")
        cfg = dict(mc.CONFIG)
    except Exception as erro:                       # arquivo com erro de sintaxe, CONFIG apagada...
        print(f"FALHOU importar_minha_comparacao: {type(erro).__name__}: {erro}")
        sys.exit(1)
    try:
        E.validar(cfg)
        ok = True
    except SystemExit as erro:
        ok = False
        teste("configuracao_valida", False, str(erro))
    if not ok:
        sys.exit(1)
    teste("configuracao_valida", True)

    teste("contrato_da_tarefa5", int(cfg["horizonte"]) == 1 and cfg["coluna"] == "pedidos",
          f"a tarefa 5 prevê os pedidos de cada dia na noite anterior; a CONFIG tem horizonte {cfg['horizonte']} "
          f"e coluna {cfg['coluna']!r}")
    h = int(cfg["horizonte"])
    primeiro = pd.Timestamp(cfg["primeira_origem"]) + pd.Timedelta(days=h)
    ultimo = pd.Timestamp(cfg["ultima_origem"]) + pd.Timedelta(days=h)
    teste("dias_previstos_da_tarefa5", primeiro == pd.Timestamp("2025-11-01") and ultimo == pd.Timestamp("2025-12-31"),
          f"as origens preveem de {primeiro:%d/%m/%Y} a {ultimo:%d/%m/%Y}; a tarefa 5 mede de 01/11 a 31/12/2025")
    teste("escolha_antes_do_teste", pd.Timestamp(cfg["fim_da_escolha"]) <= pd.Timestamp(cfg["primeira_origem"]),
          f"fim_da_escolha {cfg['fim_da_escolha']} passa da primeira noite do teste, {cfg['primeira_origem']}")
    dias_val = (pd.Timestamp(cfg["fim_da_escolha"]) - pd.Timestamp(cfg["fim_do_treino"])).days
    teste("validacao_no_tempo", cfg["validacao"] == "depois_do_treino" and dias_val >= 28,
          f"validacao {cfg['validacao']!r} com {dias_val} dias entre o fim do treino e o fim da escolha; "
          "a validação precisa ser um pedaço do tempo, depois do treino, com pelo menos 28 dias")
    d = S.carregar(cfg["coluna"])
    origens = list(pd.date_range(cfg["primeira_origem"], cfg["ultima_origem"], freq="10D"))
    mudou = E.teste_do_futuro(cfg, d, origens)
    teste("teste_do_futuro_vazio", not mudou, "; ".join(mudou))
    mesma = {k: cfg[k] for k in ("janela", "estado", "calendario", "retreino", "sorteios", "epocas", "taxa")}
    aula = {"janela": 28, "estado": 16, "calendario": True, "retreino": "nenhum", "sorteios": [0, 1, 2],
            "epocas": 300, "taxa": 0.01}
    teste("mesma_rede_da_aula", mesma == aula, f"a rede tem que ser a da aula, com 3 sorteios: {mesma}")
    t5 = mc.dias_da_tarefa5(S.carregar("pedidos"))
    teste("os_60_dias_da_tarefa5", len(t5) == 60 and np.isclose(round(np.mean(np.abs(t5["real"] - t5["semana passada"])), 1), 31.2),
          f"{len(t5)} dias; a função dias_da_tarefa5 do minha_comparacao.py precisa ficar como veio")

    if falhas:
        print(f"{len(falhas)} de {len(falhas) + len(passaram)} testes falharam")
        sys.exit(1)
    print(f"ok: {len(passaram)} testes passaram")


if __name__ == "__main__":
    main()
