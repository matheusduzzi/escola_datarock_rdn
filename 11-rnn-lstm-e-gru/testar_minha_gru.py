"""Testes do exercício da aula «RNN, LSTM e GRU».

Rode de dentro da pasta da aula:  python3 testar_minha_gru.py
Os testes usam a GRU de um número da parte 6 da aula e casos em que um portão está todo aberto
ou todo fechado. Eles não usam os pesos da proposta do Caio.
"""
import sys

import numpy as np

from minha_gru import passo_gru

FECHA, ABRE = -50.0, 50.0   # viés que deixa a sigmoide em 0 ou em 1


def pesos(H, X, W_ih=None, W_hh=None, b_ih=None, b_hh=None):
    return {"W_ih": np.zeros((3 * H, X)) if W_ih is None else np.array(W_ih, dtype=float),
            "W_hh": np.zeros((3 * H, H)) if W_hh is None else np.array(W_hh, dtype=float),
            "b_ih": np.zeros(3 * H) if b_ih is None else np.array(b_ih, dtype=float),
            "b_hh": np.zeros(3 * H) if b_hh is None else np.array(b_hh, dtype=float)}


def perto(obtido, esperado, conta, tol=1e-6):
    obtido = np.asarray(obtido, dtype=float)
    if obtido.shape != np.shape(esperado):
        raise AssertionError(f"{conta} retornou formato {obtido.shape}; esperado {np.shape(esperado)}")
    if not np.allclose(obtido, esperado, atol=tol):
        raise AssertionError(f"{conta} retornou {np.round(obtido, 4).tolist()}; "
                             f"esperado {np.round(esperado, 4).tolist()}")


def recusa(*args):
    try:
        resultado = passo_gru(*args)
    except ValueError:
        return
    raise AssertionError(f"passo_gru retornou {resultado!r}; precisava parar com ValueError")


AULA = pesos(1, 1, W_ih=[[1.0], [0.5], [1.0]], W_hh=[[0.0], [0.0], [0.5]], b_ih=[0.0, 2.0, 0.0])


def passo_da_aula():
    # parte 6: x = 0,22, h anterior = 0,016, update perto de 0,89, h novo perto de 0,038
    perto(passo_gru(np.array([0.22]), np.array([0.016]), AULA), [0.038], "passo da parte 6", tol=1e-3)


def update_fechado_mantem():
    p = pesos(1, 1, W_ih=[[1.0], [0.0], [1.0]], b_ih=[0.0, ABRE, 0.0])
    perto(passo_gru(np.array([0.5]), np.array([0.7]), p), [0.7], "update em 1 (h anterior 0,7)")


def update_aberto_troca():
    p = pesos(1, 1, W_ih=[[0.0], [0.0], [1.0]], b_ih=[0.0, FECHA, 0.0])
    perto(passo_gru(np.array([0.5]), np.array([0.7]), p), [np.tanh(0.5)], "update em 0 (h vira n = tanh(0,5))")


def vies_do_estado_conta():
    # o viés do update está em b_hh, não em b_ih: o update fica em 0 e h vira n
    p = pesos(1, 1, W_ih=[[0.0], [0.0], [1.0]], b_hh=[0.0, FECHA, 0.0])
    perto(passo_gru(np.array([0.5]), np.array([0.7]), p), [np.tanh(0.5)], "viés do update em b_hh")


def reset_fechado_ignora_h():
    p = pesos(1, 1, W_ih=[[0.0], [0.0], [1.0]], W_hh=[[0.0], [0.0], [2.0]], b_ih=[FECHA, FECHA, 0.0])
    for h in (-0.9, 0.9):
        perto(passo_gru(np.array([0.3]), np.array([h]), p), [np.tanh(0.3)], f"reset em 0 com h anterior {h}")


def vies_hn_dentro_do_reset():
    # PyTorch: n = tanh(W_in x + b_in + r * (W_hn h + b_hn)); com r em 0, b_hn some
    p = pesos(1, 1, b_ih=[FECHA, FECHA, 0.0], b_hh=[0.0, 0.0, 3.0])
    perto(passo_gru(np.array([0.0]), np.array([0.4]), p), [0.0], "reset em 0 com b_hn = 3")


def reset_aberto_usa_h():
    p = pesos(1, 1, W_hh=[[0.0], [0.0], [2.0]], b_ih=[ABRE, FECHA, 0.0], b_hh=[0.0, 0.0, 0.5])
    perto(passo_gru(np.array([0.0]), np.array([0.4]), p), [np.tanh(2.0 * 0.4 + 0.5)], "reset em 1")


def blocos_com_estado_dois():
    # estado de 2: linhas r0 r1 z0 z1 n0 n1. A coordenada 0 é a GRU da aula; a 1 tem o update em 1.
    p = pesos(2, 1, W_ih=[[1.0], [0.0], [0.5], [0.0], [1.0], [0.0]],
              W_hh=[[0, 0], [0, 0], [0, 0], [0, 0], [0.5, 0], [0, 0]],
              b_ih=[0.0, 0.0, 2.0, ABRE, 0.0, 0.0])
    h = np.array([0.016, -0.3])
    copia = h.copy()
    perto(passo_gru(np.array([0.22]), h, p), [0.038, -0.3], "estado de 2", tol=1e-3)
    if not np.array_equal(h, copia):
        raise AssertionError("passo_gru mudou o h que recebeu")


def formato_errado_para():
    p = pesos(2, 3)
    recusa(np.zeros(3), np.zeros(3), p)                     # h com 3, estado é 2
    recusa(np.zeros(2), np.zeros(2), p)                     # x com 2, entradas são 3
    ruim = dict(p, W_hh=np.zeros((6, 3)))
    recusa(np.zeros(3), np.zeros(2), ruim)
    ruim = dict(p, b_hh=np.zeros(5))
    recusa(np.zeros(3), np.zeros(2), ruim)
    ruim = dict(p, b_ih=np.zeros(1))                        # o NumPy somaria esse viés sem reclamar
    recusa(np.zeros(3), np.zeros(2), ruim)


TESTES = [passo_da_aula, update_fechado_mantem, update_aberto_troca, vies_do_estado_conta,
          reset_fechado_ignora_h, vies_hn_dentro_do_reset, reset_aberto_usa_h, blocos_com_estado_dois,
          formato_errado_para]


def main():
    falhas = 0
    for teste in TESTES:
        try:
            teste()
        except NotImplementedError:
            raise
        except Exception as erro:  # noqa: BLE001
            falhas += 1
            print(f"FALHOU {teste.__name__}: {type(erro).__name__}: {erro}")
    if falhas:
        print(f"{falhas} de {len(TESTES)} testes falharam")
        sys.exit(1)
    print(f"ok: {len(TESTES)} testes passaram")


if __name__ == "__main__":
    main()
