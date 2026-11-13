"""Testes da função resumir() do meu_resumo.py. De dentro da pasta da aula:

    python3 testar_meu_resumo.py

Não precisa dos modelos: os testes usam arrays pequenos montados à mão.
"""
import sys

import numpy as np

from meu_resumo import resumir

falhas = []


def caso(nome):
    def registrar(f):
        try:
            f()
        except AssertionError as e:
            falhas.append(f"{nome}: {e}")
        except NotImplementedError:
            raise
        except Exception as e:  # erro inesperado também é falha do teste
            falhas.append(f"{nome}: {type(e).__name__}: {e}")
        return f
    return registrar


def lote():
    """3 mensagens com 4, 2 e 1 tokens de verdade; o preenchimento vale 999 para aparecer se for lido."""
    est = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
    mascara = np.array([[True, True, True, True],
                        [True, True, False, False],
                        [True, False, False, False]])
    est[~mascara] = 999.0
    return est, mascara


def para_com_valueerror(nome_do_erro, *args):
    try:
        resumir(*args)
    except ValueError:
        return
    except NotImplementedError:
        raise
    except Exception as e:
        raise AssertionError(f"{nome_do_erro}: esperava ValueError, veio {type(e).__name__}")
    raise AssertionError(f"{nome_do_erro}: esperava ValueError, a função retornou")


@caso("primeiro token")
def _():
    est, mascara = lote()
    r = resumir(est, mascara, "primeiro")
    assert r.shape == (3, 2), f"formato {r.shape}, esperava (3, 2)"
    assert np.array_equal(r, est[:, 0]), f"saiu {r.tolist()}, esperava {est[:, 0].tolist()}"


@caso("último token de verdade")
def _():
    est, mascara = lote()
    r = resumir(est, mascara, "ultimo")
    certo = np.stack([est[0, 3], est[1, 1], est[2, 0]])
    assert r.shape == (3, 2), f"formato {r.shape}, esperava (3, 2)"
    assert not (r == 999.0).any(), "leu uma posição de preenchimento (999)"
    assert np.array_equal(r, certo), f"saiu {r.tolist()}, esperava {certo.tolist()}"


@caso("sem preenchimento, o último é a última coluna")
def _():
    est = np.random.default_rng(0).normal(size=(2, 5, 3))
    mascara = np.ones((2, 5), dtype=bool)
    assert np.array_equal(resumir(est, mascara, "ultimo"), est[:, -1])


@caso("uma mensagem só, de um token")
def _():
    est = np.array([[[1.0, 2.0, 3.0]]])
    mascara = np.array([[True]])
    for jeito in ("primeiro", "ultimo"):
        r = resumir(est, mascara, jeito)
        assert r.shape == (1, 3) and np.array_equal(r[0], [1.0, 2.0, 3.0]), f"{jeito}: saiu {r.tolist()}"


@caso("não muda as entradas")
def _():
    est, mascara = lote()
    copia_est, copia_mascara = est.copy(), mascara.copy()
    for jeito in ("primeiro", "ultimo"):
        r = resumir(est, mascara, jeito)
        r[...] = -1.0
    assert np.array_equal(est, copia_est), "estados mudou"
    assert np.array_equal(mascara, copia_mascara), "mascara mudou"


@caso("jeito desconhecido")
def _():
    est, mascara = lote()
    for jeito in ("media", "último", "Primeiro", ""):
        para_com_valueerror(f"jeito {jeito!r}", est, mascara, jeito)


@caso("formatos que não batem")
def _():
    est, mascara = lote()
    para_com_valueerror("estados com 2 dimensões", est[:, :, 0], mascara, "primeiro")
    para_com_valueerror("mascara com uma coluna a menos", est, mascara[:, :3], "ultimo")
    para_com_valueerror("mascara com uma linha a menos", est, mascara[:2], "ultimo")


@caso("mascara que não é True/False")
def _():
    est, mascara = lote()
    para_com_valueerror("mascara de 0 e 1", est, mascara.astype(int), "ultimo")


@caso("mensagem sem nenhum token")
def _():
    est, mascara = lote()
    mascara = mascara.copy()
    mascara[2] = False
    para_com_valueerror("linha toda False, último", est, mascara, "ultimo")
    para_com_valueerror("linha toda False, primeiro", est, mascara, "primeiro")


@caso("buraco ou preenchimento à esquerda")
def _():
    est, mascara = lote()
    buraco = mascara.copy()
    buraco[0] = [True, False, True, True]
    para_com_valueerror("True depois de False no meio", est, buraco, "ultimo")
    esquerda = mascara.copy()
    esquerda[1] = [False, False, True, True]
    para_com_valueerror("preenchimento à esquerda", est, esquerda, "primeiro")


if __name__ == "__main__":
    total = 10
    if falhas:
        print(f"{len(falhas)} de {total} testes falharam:")
        for f in falhas:
            print("  -", f)
        sys.exit(1)
    print(f"ok: {total} testes passaram")
