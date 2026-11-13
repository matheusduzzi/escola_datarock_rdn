"""Exercício da aula «RNN, LSTM e GRU»: a GRU que o Caio quer treinar na aula 12.

Escreva a função passo_gru. O resto do arquivo já está pronto: ele roda a sua função na janela de
60 dias, conta quantas vezes ela rodou, conta os parâmetros e mede quanto da memória do dia 1
chega no último dia. Rode de dentro da pasta da aula:

    python3 testar_minha_gru.py
    python3 minha_gru.py

Antes de você escrever a função, os dois comandos param com NotImplementedError.
"""
import csv
from pathlib import Path

import numpy as np

DADOS = Path(__file__).resolve().parent.parent / "dados" / "vendas"
CORTE = "2025-11-01"
ENTRADAS = 4      # pedidos, promoção, feriado, loja fechada
ESTADO = 16


def passo_gru(x, h, p):
    """Um passo da GRU, no formato do PyTorch.

    x: as entradas do dia, array de tamanho X.
    h: o estado anterior, array de tamanho H.
    p: dicionário com os pesos, os blocos sempre na ordem r, z, n:
       p["W_ih"] (3H, X), p["W_hh"] (3H, H), p["b_ih"] (3H,), p["b_hh"] (3H,)

    Retorna o estado novo, array de tamanho H. Não muda h nem p.
    Para com ValueError se o formato de h, x ou de algum peso não bater com H e X.
    """
    raise NotImplementedError("escreva passo_gru")


# ------------------------------------------------------------------ daqui para baixo já está pronto

def ler_janela(dias):
    with open(DADOS / "calendario.csv", encoding="utf-8") as f:
        cal = {r["data"]: r for r in csv.DictReader(f)}
    with open(DADOS / "venda_diaria.csv", encoding="utf-8") as f:
        venda = [r for r in csv.DictReader(f) if r["data"] < CORTE][-dias:]
    xs = []
    for r in venda:
        c = cal[r["data"]]
        xs.append(np.array([(int(r["pedidos"]) - 100) / 50, float(bool(c["promocao"])),
                            float(bool(c["feriado"])), float(c["loja_fechada"])]))
    return [r["data"] for r in venda], xs


def pesos_da_proposta():
    """Pesos de partida da proposta (ainda não treinados). O viés do update começa em 2,2."""
    g = np.random.default_rng(12)
    p = {"W_ih": g.normal(0, 0.3, (3 * ESTADO, ENTRADAS)),
         "W_hh": g.normal(0, 0.3, (3 * ESTADO, ESTADO)),
         "b_ih": np.zeros(3 * ESTADO),
         "b_hh": np.zeros(3 * ESTADO)}
    p["b_ih"][ESTADO:2 * ESTADO] = 2.2
    return p


class Contador:
    def __init__(self, passo):
        self.passo, self.vezes = passo, 0

    def __call__(self, *args):
        self.vezes += 1
        return self.passo(*args)


def ultimo_h(passo, xs, p, empurrao=None):
    h = np.zeros(ESTADO)
    for t, x in enumerate(xs):
        h = passo(x, h, p)
        if t == 0 and empurrao is not None:
            h = h + empurrao
    return h


def main():
    p = pesos_da_proposta()
    print(f"A GRU da proposta: {ENTRADAS} entradas por dia, estado de {ESTADO}, densa {ESTADO} -> 1")
    for nome in ("W_ih", "W_hh", "b_ih", "b_hh"):
        print(f"  {nome}: formato {p[nome].shape}, {p[nome].size} números")
    da_gru = sum(v.size for v in p.values())
    densa = ESTADO + 1
    print(f"  parâmetros da GRU: {da_gru}; da densa: {densa}; total: {da_gru + densa}")
    print()
    sem_n = {k: v.copy() for k, v in p.items()}
    sem_n["W_hh"][2 * ESTADO:] = 0.0          # o h antigo não entra no valor novo n
    print("  janela   de      até     vezes que passo_gru rodou   sobra do dia 1   sobra, h antigo só pelo update")
    direcao = np.ones(ESTADO) / np.sqrt(ESTADO)
    eps = 1e-4
    for dias in (28, 60):
        datas, xs = ler_janela(dias)
        cont = Contador(passo_gru)
        ultimo_h(cont, xs, p)
        sobras = []
        for q in (p, sem_n):
            mais = ultimo_h(passo_gru, xs, q, eps * direcao)
            menos = ultimo_h(passo_gru, xs, q, -eps * direcao)
            sobras.append(np.linalg.norm(mais - menos) / (2 * eps))
        d0, d1 = datas[0], datas[-1]
        print(f"  {dias:6d}   {d0[8:]}/{d0[5:7]}   {d1[8:]}/{d1[5:7]}   {cont.vezes:25d}   {sobras[0]:14.1e}"
              f"   {sobras[1]:30.1e}")
    print()
    print("  sobra = quanto o último h muda quando o h do dia 1 muda 1 (empurro 0.0001 para os dois lados)")
    print("  na última coluna, as linhas do n em W_hh valem zero: o h antigo só passa pelo update")


if __name__ == "__main__":
    main()
