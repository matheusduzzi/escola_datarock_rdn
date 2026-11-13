"""Testes do exercício da aula «Atenção Q/K/V». Rode de dentro da pasta da aula:

    python3 testar_minha_atencao.py

Os testes usam o exemplo de três palavras da aula e casos em que a resposta sai de uma conta curta.
Eles não usam a camada do Caio: essa conta é sua.
"""
import math
import sys
import traceback

import numpy as np

from minha_atencao import atencao

# O exemplo da aula, seções 2 a 5: uma cabeça, d_k = d_v = 2. W_O = identidade deixa a saída igual.
X_AULA = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]], dtype=float)
WQ_AULA = np.array([[1, 0], [0, 1], [0, 0], [0, 1]], dtype=float)
WK_AULA = np.array([[0, 1], [0, 0], [1, 0], [0, 1]], dtype=float)
WV_AULA = np.array([[1, 0], [0, -2], [1, 0], [0, 1]], dtype=float)
I2 = np.eye(2)
PESOS_AULA = np.array([[0.503, 0.248, 0.248], [0.164, 0.164, 0.673], [0.401, 0.198, 0.401]])
SAIDA_AULA = np.array([[1.255, 0.000], [1.000, 0.509], [1.203, 0.203]])


def perto(a, b, tol=1e-3):
    return np.shape(a) == np.shape(b) and np.allclose(a, b, atol=tol)


def teste_exemplo_da_aula():
    pesos, saida = atencao(X_AULA, WQ_AULA, WK_AULA, WV_AULA, I2, 1)
    assert np.shape(pesos) == (1, 3, 3), f"pesos {np.shape(pesos)}; esperado (1, 3, 3)"
    assert perto(pesos[0], PESOS_AULA), f"pesos {np.round(pesos[0], 3).tolist()}; esperado {PESOS_AULA.tolist()}"
    assert perto(saida, SAIDA_AULA), f"saída {np.round(saida, 3).tolist()}; esperado {SAIDA_AULA.tolist()}"


def teste_linhas_somam_um():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 6))
    args = (X, rng.normal(size=(6, 4)), rng.normal(size=(6, 4)), rng.normal(size=(6, 6)), rng.normal(size=(6, 3)), 2)
    for causal in (False, True):
        pesos, _ = atencao(*args, causal=causal)
        assert np.shape(pesos) == (2, 5, 5), f"pesos {np.shape(pesos)}; esperado (2, 5, 5)"
        assert np.all(pesos >= 0), "peso negativo"
        assert np.allclose(pesos.sum(axis=2), 1), \
            f"causal={causal}: somas das linhas {np.round(pesos.sum(axis=2), 3).tolist()}"


def teste_notas_grandes():
    pesos, saida = atencao(1000 * X_AULA, WQ_AULA, WK_AULA, WV_AULA, I2, 1)
    assert np.all(np.isfinite(pesos)) and np.all(np.isfinite(saida)), "nan ou inf com notas grandes"
    assert np.allclose(pesos.sum(axis=2), 1), "linhas não somam 1 com notas grandes"


def teste_divide_pela_raiz_de_dk_da_cabeca():
    # Duas cabeças de d_k = 4. Na cabeça 1, a palavra 0 dá nota 4 a ela mesma e 0 à palavra 1.
    # Dividindo por raiz(4) = 2, os pesos são 1 / (1 + e^-2) e o resto. Raiz de 8 daria outro número.
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    W_Q = np.zeros((2, 8))
    W_Q[0, :4] = 1
    W_K = np.zeros((2, 8))
    W_K[0, :4] = 1
    W_V = np.zeros((2, 2))
    W_V[0, 0] = 1
    pesos, _ = atencao(X, W_Q, W_K, W_V, np.eye(2), 2)
    p = 1 / (1 + math.exp(-2))
    assert perto(pesos[0, 0], [p, 1 - p]), \
        f"cabeça 1, linha 0: {np.round(pesos[0, 0], 3).tolist()}; esperado {[round(p, 3), round(1 - p, 3)]}"
    assert perto(pesos[1], np.full((2, 2), 0.5)), \
        f"cabeça 2 (notas 0): {np.round(pesos[1], 3).tolist()}; esperado 0,5 em tudo"


def teste_causal_olha_so_para_tras():
    pesos, _ = atencao(X_AULA, WQ_AULA, WK_AULA, WV_AULA, I2, 1, causal=True)
    p = pesos[0]
    assert perto(p[0], [1, 0, 0]), f"primeira linha {np.round(p[0], 3).tolist()}; esperado [1, 0, 0]"
    assert perto(p[1], [0.5, 0.5, 0]), f"segunda linha {np.round(p[1], 3).tolist()}; esperado [0,5, 0,5, 0]"
    assert np.allclose(np.triu(p, 1), 0), "peso acima da diagonal com causal=True"


def teste_causal_ultima_linha_igual():
    sem, s_sem = atencao(X_AULA, WQ_AULA, WK_AULA, WV_AULA, I2, 1)
    com, s_com = atencao(X_AULA, WQ_AULA, WK_AULA, WV_AULA, I2, 1, causal=True)
    assert perto(sem[0, -1], com[0, -1]), "a última palavra vê todas, mas a linha dela mudou com a máscara"
    assert perto(s_sem[-1], s_com[-1]), "a saída da última palavra mudou com a máscara"


def teste_cabecas_em_blocos_seguidos():
    # Cabeça 1 (colunas 0 a 2 de W_Q e W_K) só com zeros: notas 0 e pesos iguais.
    # Cabeça 2 (colunas 3 a 5) puxa todas as palavras para a palavra 2.
    X = np.eye(3)
    W_Q = np.zeros((3, 6))
    W_Q[:, 3] = 2.0
    W_K = np.zeros((3, 6))
    W_K[2, 3] = 3.0
    W_V = np.zeros((3, 4))
    pesos, _ = atencao(X, W_Q, W_K, W_V, np.zeros((4, 1)), 2)
    assert perto(pesos[0], np.full((3, 3), 1 / 3)), \
        f"cabeça 1: {np.round(pesos[0], 3).tolist()}; esperado 1/3 em tudo"
    e = math.exp(6 / math.sqrt(3))
    assert perto(pesos[1][:, 2], np.full(3, e / (e + 2))), \
        f"cabeça 2, coluna da palavra 2: {np.round(pesos[1][:, 2], 3).tolist()}; esperado {round(e / (e + 2), 3)}"


def teste_saida_junta_cabecas_e_passa_por_W_O():
    # Notas 0 nas duas cabeças: a saída de cada cabeça é a média das linhas de V daquela cabeça.
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    W_Q = np.zeros((2, 2))
    W_K = np.zeros((2, 2))
    W_V = np.array([[2.0, 0.0, 10.0, 0.0],      # cabeça 1: colunas 0 e 1; cabeça 2: colunas 2 e 3
                    [0.0, 4.0, 0.0, 20.0]])
    W_O = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    _, saida = atencao(X, W_Q, W_K, W_V, W_O, 2)
    # médias: cabeça 1 [1, 2]; cabeça 2 [5, 10]; lado a lado [1, 2, 5, 10]; vezes W_O: [1, 2, 5 - 10]
    assert perto(saida, np.array([[1.0, 2.0, -5.0], [1.0, 2.0, -5.0]])), \
        f"saída {np.round(saida, 3).tolist()}; esperado [1, 2, -5] nas duas linhas"


def teste_formato_errado_para():
    casos = {
        "X com uma dimensão só": (X_AULA[0], WQ_AULA, WK_AULA, WV_AULA, I2, 1),
        "W_Q com 3 linhas, d_model é 4": (X_AULA, WQ_AULA[:3], WK_AULA, WV_AULA, I2, 1),
        "W_K com 3 colunas, W_Q com 2": (X_AULA, WQ_AULA, np.ones((4, 3)), WV_AULA, I2, 1),
        "W_V com 3 linhas": (X_AULA, WQ_AULA, WK_AULA, WV_AULA[:3], I2, 1),
        "2 cabeças e 3 colunas em W_V": (X_AULA, np.ones((4, 4)), np.ones((4, 4)), np.ones((4, 3)), np.eye(2), 2),
        "2 cabeças, W_Q com 4 colunas e W_K com 6": (X_AULA, np.ones((4, 4)), np.ones((4, 6)), np.ones((4, 2)), I2, 2),
        "W_O com 3 linhas, W_V tem 2 colunas": (X_AULA, WQ_AULA, WK_AULA, WV_AULA, np.ones((3, 2)), 1),
        "W_O com 1 linha, W_V tem 2 colunas": (X_AULA, WQ_AULA, WK_AULA, WV_AULA, np.ones((1, 2)), 1),
    }
    for nome, args in casos.items():
        try:
            atencao(*args)
        except ValueError:
            continue
        except Exception as erro:  # noqa: BLE001
            raise AssertionError(f"{nome}: parou com {type(erro).__name__}, não com ValueError") from None
        raise AssertionError(f"{nome}: rodou sem erro")


TESTES = [teste_exemplo_da_aula, teste_linhas_somam_um, teste_notas_grandes, teste_divide_pela_raiz_de_dk_da_cabeca,
          teste_causal_olha_so_para_tras, teste_causal_ultima_linha_igual, teste_cabecas_em_blocos_seguidos,
          teste_saida_junta_cabecas_e_passa_por_W_O, teste_formato_errado_para]


if __name__ == "__main__":
    falhas = 0
    for teste in TESTES:
        nome = teste.__name__[len("teste_"):]
        try:
            with np.errstate(all="ignore"):
                teste()
        except NotImplementedError:
            traceback.print_exc()
            sys.exit(1)
        except Exception as erro:  # noqa: BLE001
            falhas += 1
            print(f"FALHOU {nome}: {type(erro).__name__}: {erro}")
    if falhas:
        sys.exit(1)
    print(f"ok: {len(TESTES)} testes passaram")
