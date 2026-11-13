"""Testes das funções retropropagar e gradiente_numerico do meus_gradientes.py.

Rode de dentro da pasta da aula:  python3 testar_meus_gradientes.py
Os testes usam a rede da aula (4 pixels, 3 neurônios, 2 notas), não a rede de camisa e casaco:
eles mostram que o código faz a conta da aula, e a conta do exercício continua com você.
"""
import math

import numpy as np

from meus_gradientes import gradiente_numerico, retropropagar

REDE = {
    "W1": np.array([[1, 2, 0, -1], [-1, 0, 2, 1], [0, -2, 1, -1]], dtype=float),
    "b1": np.array([0, 0.5, 1], dtype=float),
    "W2": np.array([[2, -1, 1], [-1, 2, 1]], dtype=float),
    "b2": np.array([0, 0.5], dtype=float),
}
FOTO = [1, 0.5, 0, 1]
OUTRA = [0, 0, 1, 1]
A = 1 / (1 + math.e)  # 0,2689: a probabilidade que a rede dá a calça na foto da aula

# Os gradientes da seção 4 da aula, para a foto da aula com a classe camiseta.
ESPERADO = {
    "W2": np.array([[-A, -A / 2, 0], [A, A / 2, 0]]),
    "b2": np.array([-A, A]),
    "W1": np.array([[-3 * A, -1.5 * A, 0, -3 * A], [3 * A, 1.5 * A, 0, 3 * A], [0, 0, 0, 0]]),
    "b1": np.array([-3 * A, 3 * A, 0]),
}

# A outra foto, (0, 0, 1, 1), também camiseta: z1 = (-1; 3,5; 1), a ReLU passa (0; 1; 1),
# notas (-2,5; 8,5) e p - y = (-B; B), com B a probabilidade da calça.
B = 1 / (1 + math.exp(-11))
DA_OUTRA = {
    "W2": np.array([[0, -3.5 * B, -B], [0, 3.5 * B, B]]),
    "b2": np.array([-B, B]),
    "W1": np.array([[0, 0, 0, 0], [0, 0, 3 * B, 3 * B], [0, 0, 0, 0]]),
    "b1": np.array([0, 3 * B, 0]),
}
# O lote da aula com as duas fotos: a média dos gradientes de cada uma (seção 7 da aula).
DO_LOTE = {k: (ESPERADO[k] + DA_OUTRA[k]) / 2 for k in ESPERADO}


def copia():
    return {k: v.copy() for k, v in REDE.items()}


def perto(a, b, tol=1e-7):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and np.allclose(a, b, atol=tol, rtol=0)


def testar_gradientes_da_foto_da_aula():
    g = retropropagar(FOTO, 0, copia())
    for k in ("W2", "b2", "W1", "b1"):
        assert perto(g[k], ESPERADO[k]), f"d{k} deu\n{g[k]}\na seção 4 da aula tem\n{np.round(ESPERADO[k], 4)}"


def testar_formatos_iguais_aos_parametros():
    g = retropropagar(np.array([FOTO, OUTRA]), [0, 0], copia())
    for k, v in REDE.items():
        assert np.shape(g.get(k)) == v.shape, f"d{k} tem formato {np.shape(g.get(k))}, o parâmetro tem {v.shape}"


def testar_neuronio_apagado_sem_gradiente():
    g = retropropagar(FOTO, 0, copia())
    assert perto(g["W1"][2], [0, 0, 0, 0]) and abs(g["b1"][2]) < 1e-12, \
        "o neurônio 3 deu -1 antes da ReLU nessa foto: os pesos dele tinham que ter gradiente 0"


def testar_lote_e_a_media_das_fotos():
    lote = retropropagar(np.array([FOTO, OUTRA]), [0, 0], copia())
    um = retropropagar(FOTO, 0, copia())
    outro = retropropagar(OUTRA, 0, copia())
    for k in REDE:
        media = (um[k] + outro[k]) / 2
        assert perto(lote[k], media), f"d{k} do lote não é a média das duas fotos (a soma dá o dobro)"


def testar_gradientes_do_lote_da_aula():
    g = retropropagar(np.array([FOTO, OUTRA]), [0, 0], copia())
    for k in ("W2", "b2", "W1", "b1"):
        assert perto(g[k], DO_LOTE[k], tol=1e-6), \
            f"no lote da aula, d{k} deu\n{g[k]}\na conta das duas fotos dá\n{np.round(DO_LOTE[k], 4)}"


def testar_gradiente_numerico_da_foto_da_aula():
    rede = copia()
    for k in REDE:
        n = gradiente_numerico(FOTO, 0, rede, k)
        assert perto(n, ESPERADO[k], tol=1e-6), \
            f"gradiente_numerico de {k} deu\n{n}\na aula tem\n{np.round(ESPERADO[k], 4)}"


def testar_gradiente_numerico_volta_os_pesos():
    rede = copia()
    for k in REDE:
        gradiente_numerico(FOTO, 0, rede, k)
        assert np.array_equal(rede[k], REDE[k]), f"gradiente_numerico deixou {k} diferente do que era"


def testar_retropropagar_nao_mexe_na_rede():
    rede = copia()
    retropropagar(np.array([FOTO, OUTRA]), [0, 0], rede)
    for k in REDE:
        assert np.array_equal(rede[k], REDE[k]), f"retropropagar mudou {k}"


def testar_classe_que_nao_existe():
    for classe in (2, -1):
        try:
            retropropagar(FOTO, classe, copia())
        except ValueError:
            continue
        raise AssertionError(f"retropropagar aceitou a classe {classe} numa rede de 2 classes; "
                             "tinha que parar com ValueError")


def testar_classes_a_menos_no_lote():
    try:
        retropropagar(np.array([FOTO, OUTRA]), [0], copia())
    except ValueError:
        return
    raise AssertionError("retropropagar aceitou 2 fotos com 1 classe; tinha que parar com ValueError")


if __name__ == "__main__":
    import sys

    testes = [(nome, f) for nome, f in sorted(globals().items()) if nome.startswith("testar_")]
    falhas = 0
    for nome, teste in testes:
        try:
            teste()
        except AssertionError as erro:
            falhas += 1
            print(f"FALHOU {nome}: {erro}")
        except NotImplementedError:
            raise
        except Exception as erro:
            falhas += 1
            print(f"FALHOU {nome}: {type(erro).__name__}: {erro}")
    if falhas:
        print(f"{falhas} de {len(testes)} testes falharam")
        sys.exit(1)
    print(f"ok: {len(testes)} testes passaram")
