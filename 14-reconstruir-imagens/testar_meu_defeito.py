"""Testa a função etiqueta do meu_defeito.py. Rode de dentro da pasta da aula:  python3 testar_meu_defeito.py"""
import sys

import numpy as np

from meu_defeito import etiqueta

testes = []


def teste(f):
    testes.append(f)
    return f


def foto_vazia(n=1):
    return np.zeros((n, 784))


@teste
def quadrado_no_lugar():
    saida = etiqueta(foto_vazia(), 2, 5, lado=3).reshape(28, 28)
    esperado = np.zeros((28, 28))
    esperado[2:5, 5:8] = 1.0
    assert np.array_equal(saida, esperado), "o quadrado de lado 3 em (2, 5) deveria acender as linhas 2 a 4 e as colunas 5 a 7"


@teste
def linha_e_coluna_na_ordem_certa():
    saida = etiqueta(foto_vazia(), 0, 20, lado=2).reshape(28, 28)
    assert saida[0, 20] == 1.0 and saida[20, 0] == 0.0, "linha é a posição de cima para baixo; coluna, da esquerda para a direita"


@teste
def branco_e_1():
    saida = etiqueta(foto_vazia(), 0, 0)
    assert saida.max() == 1.0 and set(np.unique(saida)) == {0.0, 1.0}, "o branco é 1,0, na escala das fotos (0 a 1)"


@teste
def lado_padrao_6():
    saida = etiqueta(foto_vazia(), 0, 0)
    assert saida.sum() == 36, f"com o lado padrão, 36 pixels acendem: vieram {saida.sum():g}"


@teste
def resto_da_foto_igual():
    x = np.random.default_rng(0).random((5, 784))
    saida = etiqueta(x, 10, 12)
    fora = np.ones((28, 28), dtype=bool)
    fora[10:16, 12:18] = False
    assert np.array_equal(saida.reshape(5, 28, 28)[:, fora], x.reshape(5, 28, 28)[:, fora]), "fora do quadrado, nada muda"


@teste
def todas_as_fotos_ganham():
    saida = etiqueta(foto_vazia(4), 3, 3).reshape(4, 28, 28)
    assert all(saida[i, 3:9, 3:9].min() == 1.0 for i in range(4)), "cada foto do lote ganha a etiqueta"


@teste
def entrada_nao_muda():
    x = np.random.default_rng(1).random((3, 784))
    copia = x.copy()
    saida = etiqueta(x, 0, 0)
    assert np.array_equal(x, copia), "a função mudou as fotos que entraram: faça uma cópia"
    assert saida.shape == (3, 784), f"a saída tem que ter o formato da entrada: veio {saida.shape}"


@teste
def cabe_encostado_na_borda():
    saida = etiqueta(foto_vazia(), 22, 22).reshape(28, 28)
    assert saida[22:28, 22:28].min() == 1.0, "o quadrado de lado 6 em (22, 22) cabe: vai até a linha e a coluna 27"


@teste
def recusa_o_que_nao_cabe():
    for lin, col in ((23, 0), (0, 23), (30, 30)):
        try:
            etiqueta(foto_vazia(), lin, col)
        except ValueError:
            continue
        raise AssertionError(f"o quadrado de lado 6 em ({lin}, {col}) passa da borda: tinha que dar ValueError")


@teste
def recusa_posicao_negativa():
    for lin, col in ((-1, 0), (0, -3)):
        try:
            etiqueta(foto_vazia(), lin, col)
        except ValueError:
            continue
        raise AssertionError(f"({lin}, {col}) está fora da foto: tinha que dar ValueError")


@teste
def recusa_lado_zero():
    try:
        etiqueta(foto_vazia(), 0, 0, lado=0)
    except ValueError:
        return
    raise AssertionError("lado 0 não é etiqueta: tinha que dar ValueError")


@teste
def recusa_formato_errado():
    try:
        etiqueta(np.zeros((2, 28, 28)), 0, 0)
    except ValueError:
        return
    raise AssertionError("fotos em (fotos, 28, 28) não são (fotos, 784): tinha que dar ValueError")


falhou = []
for t in testes:
    try:
        t()
    except NotImplementedError:
        print("A função etiqueta ainda não foi escrita (NotImplementedError).")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        falhou.append(t.__name__)
        print(f"falhou: {t.__name__}: {e}")
if falhou:
    print(f"{len(falhou)} de {len(testes)} testes falharam")
    sys.exit(1)
print(f"ok: {len(testes)} testes passaram")
