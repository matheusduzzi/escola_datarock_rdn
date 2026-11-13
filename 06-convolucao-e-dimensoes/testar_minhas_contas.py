"""Testes do exercício da aula «Convolução e dimensões».

Rode de dentro da pasta da aula:  python3 testar_minhas_contas.py
Os testes usam os números da aula (o pedaço 6 x 6 e a rede do Caio), não a rede B do exercício.
"""
import sys

from minhas_contas import dimensao_saida, parametros_conv, parametros_densa, percorrer

REDE_DA_AULA = {
    0: [("conv", 32, 3, 1, 0), ("pool", 2, 2), ("conv", 64, 3, 1, 0), ("pool", 2, 2),
        ("densa", 128), ("densa", 10)],
    1: [("conv", 32, 3, 1, 1), ("pool", 2, 2), ("conv", 64, 3, 1, 1), ("pool", 2, 2),
        ("densa", 128), ("densa", 10)],
}


def igual(obtido, esperado, conta):
    if type(obtido) is not int:
        raise AssertionError(f"{conta} retornou {obtido!r} ({type(obtido).__name__}); precisa ser um inteiro")
    if obtido != esperado:
        raise AssertionError(f"{conta} retornou {obtido}; a aula dá {esperado}")


def recusa(funcao, *args):
    try:
        resultado = funcao(*args)
    except ValueError:
        return
    raise AssertionError(f"{funcao.__name__}{args} retornou {resultado!r}; precisava parar com ValueError")


def conv_sem_borda():
    igual(dimensao_saida(28, 3), 26, "dimensao_saida(28, 3)")
    igual(dimensao_saida(6, 3), 4, "dimensao_saida(6, 3)")


def pooling():
    igual(dimensao_saida(26, 2, 2), 13, "dimensao_saida(26, 2, 2)")
    igual(dimensao_saida(13, 2, 2), 6, "dimensao_saida(13, 2, 2)")
    igual(dimensao_saida(11, 2, 2), 5, "dimensao_saida(11, 2, 2)")


def passo_dois():
    igual(dimensao_saida(6, 3, 2), 2, "dimensao_saida(6, 3, 2)")
    igual(dimensao_saida(6, 3, 2, 1), 3, "dimensao_saida(6, 3, 2, 1)")


def borda_que_mantem():
    igual(dimensao_saida(6, 3, 1, 1), 6, "dimensao_saida(6, 3, 1, 1)")
    igual(dimensao_saida(28, 3, 1, 1), 28, "dimensao_saida(28, 3, 1, 1)")
    igual(dimensao_saida(2, 3, 1, 1), 2, "dimensao_saida(2, 3, 1, 1)")


def dimensao_recusa():
    recusa(dimensao_saida, 2, 3)
    recusa(dimensao_saida, 6, 3, 0)
    recusa(dimensao_saida, 6, 3, 1, -1)


def conv_parametros():
    igual(parametros_conv(1, 32, 3), 320, "parametros_conv(1, 32, 3)")
    igual(parametros_conv(32, 64, 3), 18496, "parametros_conv(32, 64, 3)")
    igual(parametros_conv(3, 4, 3), 112, "parametros_conv(3, 4, 3)")


def conv_grande():
    igual(parametros_conv(3, 96, 11), 34944, "parametros_conv(3, 96, 11)")


def densa_parametros():
    igual(parametros_densa(1600, 128), 204928, "parametros_densa(1600, 128)")
    igual(parametros_densa(784, 128), 100480, "parametros_densa(784, 128)")
    igual(parametros_densa(128, 10), 1290, "parametros_densa(128, 10)")


def parametros_recusa():
    recusa(parametros_conv, 0, 32, 3)
    recusa(parametros_conv, 1, 32, 0)
    recusa(parametros_densa, 10, 0)


def rede_do_caio():
    for borda, achatar, total in [(0, "1600", 225034), (1, "3136", 421642)]:
        linhas = percorrer(REDE_DA_AULA[borda], 28, 1)
        obtido = dict((nome, saida) for nome, saida, _ in linhas).get("achatar")
        if obtido != achatar:
            raise AssertionError(f"rede do Caio com borda {borda}: achatar deu {obtido}; a aula dá {achatar}")
        igual(sum(p for _, _, p in linhas), total, f"total da rede do Caio com borda {borda}")


TESTES = [conv_sem_borda, pooling, passo_dois, borda_que_mantem, dimensao_recusa,
          conv_parametros, conv_grande, densa_parametros, parametros_recusa, rede_do_caio]

if __name__ == "__main__":
    falhas = 0
    for teste in TESTES:
        try:
            teste()
        except NotImplementedError:
            raise
        except AssertionError as erro:
            falhas += 1
            print(f"FALHOU {teste.__name__}: {erro}")
        except Exception as erro:
            falhas += 1
            print(f"FALHOU {teste.__name__}: parou com {type(erro).__name__}: {erro}")
    if falhas:
        print(f"{falhas} de {len(TESTES)} testes falharam")
        sys.exit(1)
    print(f"ok: {len(TESTES)} testes passaram")
