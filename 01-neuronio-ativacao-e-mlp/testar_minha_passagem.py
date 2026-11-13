"""Testes das funções passagem e contar_parametros do minha_passagem.py.

Rode de dentro da pasta da aula:  python3 testar_minha_passagem.py
Os testes usam a rede da aula (4 pixels, 3 neurônios, 2 notas), não a rede de calçados:
eles mostram que o código faz a conta da aula, e a conta do exercício continua com você.
"""
import numpy as np

from minha_passagem import contar_parametros, passagem

W1 = np.array([[1, 2, 0, -1], [-1, 0, 2, 1], [0, -2, 1, -1]], dtype=float)
b1 = np.array([0, 0.5, 1], dtype=float)
W2 = np.array([[2, -1, 1], [-1, 2, 1]], dtype=float)
b2 = np.array([0, 0.5], dtype=float)
REDE = [(W1, b1, True), (W2, b2, False)]
FOTO = [1, 0.5, 0, 1]
OUTRA = [0, 0, 1, 1]


def perto(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and np.allclose(a, b, atol=1e-9)


def testar_notas_da_aula():
    notas = passagem(FOTO, REDE)
    assert perto(notas, [1.5, 0.5]), f"as notas deram {notas}, a aula tem (1,5; 0,5)"


def testar_camada_escondida_com_relu():
    h = passagem(FOTO, REDE[:1])
    assert perto(h, [1, 0.5, 0]), f"a camada escondida deu {h}, a aula tem (1; 0,5; 0): a ReLU zera o -1"


def testar_ultima_camada_sem_relu():
    notas = passagem(OUTRA, REDE)
    assert perto(notas, [-2.5, 8.5]), f"as notas da foto (0, 0, 1, 1) deram {notas}, a aula tem (-2,5; 8,5)"


def testar_lote_de_duas_fotos():
    notas = passagem(np.array([FOTO, OUTRA]), REDE)
    assert np.shape(notas) == (2, 2), f"o lote de 2 fotos saiu com formato {np.shape(notas)}, a aula tem (2, 2)"
    assert perto(notas, [[1.5, 0.5], [-2.5, 8.5]]), f"as notas do lote deram {notas}, uma linha por foto"


def testar_sem_relu_vira_uma_camada():
    sem = [(W1, b1, False), (W2, b2, False)]
    unica = [(W2 @ W1, W2 @ b1 + b2, False)]
    assert perto(passagem(FOTO, sem), passagem(FOTO, unica)), "sem ReLU, as duas camadas tinham que dar o mesmo que W2 W1"
    assert perto(passagem(FOTO, sem), [0.5, -0.5]), "sem ReLU, a aula tem (0,5; -0,5)"


def testar_entrada_de_tamanho_errado():
    try:
        passagem([1, 0.5, 0], REDE)
    except ValueError:
        return
    raise AssertionError("passagem aceitou 3 pixels numa rede de 4 entradas; tinha que parar com ValueError")


def testar_vies_de_tamanho_errado():
    try:
        passagem(FOTO, [(W1, np.array([0.5]), True), (W2, b2, False)])
    except ValueError:
        return
    raise AssertionError("passagem aceitou um viés só para 3 neurônios; tinha que parar com ValueError")


def testar_parametros_da_aula():
    for tamanhos, esperado in (([4, 3, 2], 23), ([784, 10], 7850), ([784, 64, 10], 50890),
                               ([784, 32, 32, 32, 10], 27562)):
        total = contar_parametros(tamanhos)
        assert total == esperado, f"contar_parametros({tamanhos}) deu {total}, a aula tem {esperado}"


def testar_parametros_invalidos():
    for tamanhos in ([784], [], [784, 0, 10]):
        try:
            contar_parametros(tamanhos)
        except ValueError:
            continue
        raise AssertionError(f"contar_parametros aceitou {tamanhos}; tinha que parar com ValueError")


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
