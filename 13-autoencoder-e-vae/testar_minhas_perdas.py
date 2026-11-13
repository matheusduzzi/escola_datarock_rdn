"""Testes do exercício da aula «Autoencoder e VAE».

Rode de dentro da pasta da aula:  python3 testar_minhas_perdas.py
Os testes usam os números das partes 1, 2 e 5 da aula, não os dois casos do Caio.
"""
import math
import sys

import numpy as np

from minhas_perdas import kl_normal, perda_vae, reconstrucao_bernoulli

PEDACO = [0.0, 1.0, 1.0, 0.0]
REC_A = [0.01, 0.99, 0.99, 0.45]
REC_B = [0.2, 0.8, 0.8, 0.2]
LOG_025 = math.log(0.25)        # sigma 0,5
LOG_001 = math.log(0.01)        # sigma 0,1


def perto(obtido, esperado, conta, tol=1e-4):
    if isinstance(obtido, (str, bytes)) or obtido is None:
        raise AssertionError(f"{conta} retornou {obtido!r}; esperado um número")
    obtido = np.asarray(obtido, dtype=float)
    if obtido.shape != np.shape(esperado):
        raise AssertionError(f"{conta} retornou formato {obtido.shape}; esperado {np.shape(esperado)}")
    if not np.all(np.isfinite(obtido)) or not np.allclose(obtido, esperado, atol=tol):
        raise AssertionError(f"{conta} retornou {np.round(obtido, 4).tolist()}; "
                             f"esperado {np.round(esperado, 4).tolist()}")


def recusa(funcao, *args, **kw):
    try:
        resultado = funcao(*args, **kw)
    except ValueError:
        return
    raise AssertionError(f"{funcao.__name__} retornou {resultado!r}; precisava parar com ValueError")


def kl_da_parte5():
    # parte 5: mu 1, sigma 0,5 deu 0,8181 pela integral; mu 0, sigma 0,1 deu 1,8076
    perto(kl_normal([1.0], [LOG_025]), 0.8181, "kl_normal([1], [log 0,25])")
    perto(kl_normal([0.0], [LOG_001]), 1.8076, "kl_normal([0], [log 0,01])")
    perto(kl_normal([0.0], [math.log(9.0)]), 2.9014, "kl_normal([0], [log 9])")


def kl_zero_no_centro():
    perto(kl_normal([0.0, 0.0], [0.0, 0.0]), 0.0, "kl_normal([0, 0], [0, 0])")


def kl_soma_dimensoes():
    # mu 1, sigma 0,5 numa dimensão e mu 2, sigma 1 na outra: 0,8181 + 2,0
    perto(kl_normal([1.0, 2.0], [LOG_025, 0.0]), 2.8181, "kl_normal([1, 2], [log 0,25, 0])")


def kl_por_foto():
    mu = np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
    lv = np.array([[LOG_025, 0.0], [0.0, 0.0], [LOG_001, 0.0]])
    perto(kl_normal(mu, lv), [2.8181, 0.0, 1.8076], "kl_normal com 3 fotos")


def bernoulli_da_parte1():
    perto(reconstrucao_bernoulli(PEDACO, REC_A), 0.6280, "reconstrucao_bernoulli(pedaço, A)", tol=5e-4)
    perto(reconstrucao_bernoulli(PEDACO, REC_B), 0.8926, "reconstrucao_bernoulli(pedaço, B)", tol=5e-4)


def bernoulli_pixel_cinza():
    # parte 2: pixel 0,5 com reconstrução 0,3 dá 0,7803; com 0,5, log de 2
    perto(reconstrucao_bernoulli([0.5], [0.3]), 0.7803, "reconstrucao_bernoulli([0,5], [0,3])")
    perto(reconstrucao_bernoulli([0.5], [0.5]), math.log(2), "reconstrucao_bernoulli([0,5], [0,5])")


def bernoulli_extremos():
    # reconstrução 0 num pixel 1: com o limite, -log(0,000001) = 13,8155, não infinito
    perto(reconstrucao_bernoulli([1.0, 0.0], [0.0, 1.0]), 2 * 13.8155, "reconstrucao_bernoulli([1, 0], [0, 1])", tol=1e-3)
    perto(reconstrucao_bernoulli([1.0, 0.0], [1.0, 0.0]), 0.0, "reconstrucao_bernoulli([1, 0], [1, 0])", tol=1e-4)


def bernoulli_por_foto():
    perto(reconstrucao_bernoulli(np.array([PEDACO, PEDACO]), np.array([REC_A, REC_B])), [0.6280, 0.8926],
          "reconstrucao_bernoulli com 2 fotos", tol=5e-4)


def perda_com_peso():
    perto(perda_vae(PEDACO, REC_A, [1.0], [LOG_025]), 0.6280 + 0.8181, "perda_vae(pedaço, A, peso 1)", tol=5e-4)
    perto(perda_vae(PEDACO, REC_A, [1.0], [LOG_025], peso_kl=0.5), 0.6280 + 0.40905, "perda_vae(peso 0,5)", tol=5e-4)
    perto(perda_vae(PEDACO, REC_A, [1.0], [LOG_025], peso_kl=0.0), 0.6280, "perda_vae(peso 0)", tol=5e-4)


def recusas():
    recusa(kl_normal, [0.0, 1.0], [0.0])
    recusa(reconstrucao_bernoulli, [0.0, 1.0], [0.5])
    recusa(reconstrucao_bernoulli, [1.2, 0.0], [0.5, 0.5])
    recusa(reconstrucao_bernoulli, [-0.1, 0.0], [0.5, 0.5])
    recusa(perda_vae, PEDACO, REC_A, [1.0], [LOG_025], peso_kl=-1.0)


TESTES = [kl_da_parte5, kl_zero_no_centro, kl_soma_dimensoes, kl_por_foto, bernoulli_da_parte1,
          bernoulli_pixel_cinza, bernoulli_extremos, bernoulli_por_foto, perda_com_peso, recusas]


def main():
    falhas = 0
    for teste in TESTES:
        try:
            teste()
        except NotImplementedError:
            raise
        except Exception as erro:  # noqa: BLE001
            falhas += 1
            print(f"FALHOU {teste.__name__}: {erro}")
    if falhas:
        print(f"{falhas} de {len(TESTES)} testes falharam")
        return 1
    print(f"ok: {len(TESTES)} testes passaram")
    return 0


if __name__ == "__main__":
    sys.exit(main())
