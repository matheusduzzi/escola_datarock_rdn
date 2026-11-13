"""Testes da função treinar_com_paciencia do meu_ciclo.py.

Rode de dentro da pasta da aula:  python3 testar_meu_ciclo.py
Os testes não usam as fotos. Uns usam uma rede de mentira, cuja perda de validação segue um roteiro
conhecido: assim dá para saber em que época o treino tem que parar. Outros treinam uma rede pequena
de verdade e comparam com o mlp.ciclo da aula.
"""
import sys

import numpy as np

import meu_ciclo
import mlp

ORIGINAIS = {nome: getattr(mlp, nome) for nome in ("uma_epoca", "avaliar")}
N_TREINO, N_VALIDACAO = 6, 4
X = np.zeros((N_TREINO + N_VALIDACAO, 1))
Y = np.zeros(N_TREINO + N_VALIDACAO, dtype=int)
I_TREINO, I_VALIDACAO = np.arange(N_TREINO), np.arange(N_TREINO, N_TREINO + N_VALIDACAO)


class Roteiro:
    """Rede de mentira: cada época soma 1 no viés, e a perda de validação da época e é roteiro[e]."""

    def __init__(self, perdas):
        self.perdas = list(perdas) + [9.0] * 50
        self.epocas = 0
        self.ordens = []
        self.rngs = []

    def uma_epoca(self, rede, Xt, yt, taxa, lote, rng):
        rede[0]["b"][0] += 1
        self.epocas += 1
        self.rngs.append(rng)
        self.ordens.append(tuple(rng.permutation(10)))
        return 1

    def avaliar(self, Xa, ya, rede, lote=10000):
        e = int(round(rede[0]["b"][0]))
        perda = self.perdas[e] if len(ya) == N_VALIDACAO else 3.0 - 0.1 * e
        return perda, 0.5, np.zeros(len(ya), dtype=int)


def rede_de_mentira():
    return [{"W": np.zeros((1, 1)), "b": np.zeros(1)}]


def com_roteiro(perdas, max_epocas, paciencia, i_treino=I_TREINO, i_validacao=I_VALIDACAO, roteiro=None):
    roteiro = roteiro or Roteiro(perdas)
    mlp.uma_epoca, mlp.avaliar = roteiro.uma_epoca, roteiro.avaliar
    for nome in ("uma_epoca", "avaliar"):  # caso o meu_ciclo.py tenha importado a função direto
        if nome in vars(meu_ciclo):
            setattr(meu_ciclo, nome, getattr(roteiro, nome))
    rede = rede_de_mentira()
    try:
        saida = meu_ciclo.treinar_com_paciencia(rede, X, Y, i_treino, i_validacao, max_epocas, paciencia,
                                               taxa=0.1, lote=2, sorteio=0)
    finally:
        for nome, f in ORIGINAIS.items():
            setattr(mlp, nome, f)
            if nome in vars(meu_ciclo):
                setattr(meu_ciclo, nome, f)
    return saida, rede, roteiro


def conferir_parada(perdas, max_epocas, paciencia, melhor_esperada, ultima_esperada):
    (melhor_rede, melhor_epoca, curva), rede, roteiro = com_roteiro(perdas, max_epocas, paciencia)
    assert melhor_epoca == melhor_esperada, f"melhor época {melhor_epoca}; o esperado é {melhor_esperada}"
    assert len(curva) == ultima_esperada + 1, (f"a curva tem {len(curva)} linhas; o treino tinha que parar na "
                                               f"época {ultima_esperada}, com {ultima_esperada + 1} linhas")
    assert roteiro.epocas == ultima_esperada, f"{roteiro.epocas} épocas treinadas; o esperado é {ultima_esperada}"
    assert [c["epoca"] for c in curva] == list(range(ultima_esperada + 1)), "a curva tem que ir de 0 até a última época"
    assert [c["perda_validacao"] for c in curva] == roteiro.perdas[:ultima_esperada + 1], "perda de validação da curva diferente da rede"
    assert melhor_rede[0]["b"][0] == melhor_esperada, (
        f"a melhor rede está na época {melhor_rede[0]['b'][0]:.0f}, não na {melhor_esperada}: ela é uma cópia "
        "feita na hora certa, ou a mesma rede que continuou treinando?")
    return rede, roteiro


def testar_para_depois_da_paciencia():
    # melhora até a época 4 (0,55), piora nas épocas 5 e 6: com paciência 2, para na 6
    conferir_parada([1.0, 0.8, 0.6, 0.65, 0.55, 0.7, 0.72], max_epocas=20, paciencia=2,
                    melhor_esperada=4, ultima_esperada=6)


def testar_volta_a_melhorar_dentro_da_paciencia():
    # piora na 3, melhora de novo na 4: com paciência 2, o treino não para na 3 e segue até a 6
    conferir_parada([1.0, 0.8, 0.6, 0.7, 0.5, 0.9, 0.9], max_epocas=20, paciencia=2,
                    melhor_esperada=4, ultima_esperada=6)


def testar_melhora_sem_recorde_nao_zera_a_paciencia():
    # recorde na época 1 (0,5); a época 3 (0,6) melhora sobre a 2 (0,7), mas não bate o recorde:
    # com paciência 2, as épocas 2 e 3 contam como sem melhora e o treino para na 3
    conferir_parada([1.0, 0.5, 0.7, 0.6, 0.65, 0.62], max_epocas=20, paciencia=2,
                    melhor_esperada=1, ultima_esperada=3)


def testar_empate_nao_e_melhora():
    # a época 2 empata com a 1; com paciência 3, as épocas 2, 3 e 4 contam como sem melhora
    conferir_parada([1.0, 0.5, 0.5, 0.6, 0.7], max_epocas=20, paciencia=3, melhor_esperada=1, ultima_esperada=4)


def testar_nunca_melhora():
    conferir_parada([1.0, 1.1, 1.2, 1.3], max_epocas=20, paciencia=2, melhor_esperada=0, ultima_esperada=2)


def testar_para_em_max_epocas():
    conferir_parada([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], max_epocas=5, paciencia=2,
                    melhor_esperada=5, ultima_esperada=5)


def testar_sorteio_criado_uma_vez():
    _, roteiro = conferir_parada([1.0, 0.9, 0.8, 0.7, 0.6], max_epocas=4, paciencia=2,
                                 melhor_esperada=4, ultima_esperada=4)
    assert len(set(map(id, roteiro.rngs))) == 1, "cada época recebeu um sorteio novo; crie o sorteio uma vez, antes do loop"
    assert len(set(roteiro.ordens)) > 1, "a ordem das fotos saiu igual em todas as épocas"


def testar_recusa_sem_treinar():
    casos = [
        ("max_epocas 0", dict(max_epocas=0, paciencia=2)),
        ("paciencia 0", dict(max_epocas=5, paciencia=0)),
        ("validação vazia", dict(max_epocas=5, paciencia=2, i_validacao=np.array([], dtype=int))),
        ("foto 5 no treino e na validação", dict(max_epocas=5, paciencia=2, i_validacao=np.array([5, 6, 7, 8]))),
    ]
    for nome, args in casos:
        roteiro = Roteiro([1.0, 0.9])
        try:
            com_roteiro([1.0, 0.9], roteiro=roteiro, **args)
        except ValueError:
            assert roteiro.epocas == 0, f"{nome}: levantou ValueError, mas depois de treinar {roteiro.epocas} épocas"
            continue
        raise AssertionError(f"{nome}: tinha que levantar ValueError, e o treino rodou {roteiro.epocas} épocas")


def tarefa_pequena():
    rng = np.random.default_rng(7)
    Xp = rng.normal(size=(120, 4))
    yp = (Xp @ rng.normal(size=(4, 3))).argmax(axis=1)
    return Xp, yp, np.arange(90), np.arange(90, 120)


def testar_igual_ao_ciclo_da_aula():
    # com paciência do tamanho de max_epocas, o treino não para antes: tem que dar o mesmo que o mlp.ciclo
    Xp, yp, it, iv = tarefa_pequena()
    a = mlp.nova_rede([4, 8, 3], sorteio=1)
    b = mlp.nova_rede([4, 8, 3], sorteio=1)
    ma, ea, ca = meu_ciclo.treinar_com_paciencia(a, Xp, yp, it, iv, 12, 12, taxa=0.5, lote=16, sorteio=3)
    mb, eb, cb = mlp.ciclo(b, Xp, yp, it, iv, 12, taxa=0.5, lote=16, sorteio=3)
    assert ea == eb, f"melhor época {ea}; o mlp.ciclo da aula dá {eb}"
    assert len(ca) == len(cb), f"a curva tem {len(ca)} linhas; a do mlp.ciclo tem {len(cb)}"
    for la, lb in zip(ca, cb):
        assert set(la) == set(lb), f"campos da linha diferentes: {sorted(la)}"
        assert all(abs(la[k] - lb[k]) < 1e-12 for k in lb), f"época {lb['epoca']}: {la} e o mlp.ciclo dá {lb}"


def testar_melhor_rede_e_a_da_curva():
    Xp, yp, it, iv = tarefa_pequena()
    rede = mlp.nova_rede([4, 8, 3], sorteio=1)
    inicial = mlp.copiar(rede)
    melhor, e, curva = meu_ciclo.treinar_com_paciencia(rede, Xp, yp, it, iv, 30, 3, taxa=0.5, lote=16, sorteio=3)
    zero = mlp.linha_da_curva(0, inicial, Xp, yp, it, iv)
    assert all(abs(curva[0][k] - zero[k]) < 1e-12 for k in zero), "a linha 0 da curva tem que ser a rede antes do treino"
    perda_v = mlp.avaliar(Xp[iv], yp[iv], melhor)[0]
    assert abs(perda_v - curva[e]["perda_validacao"]) < 1e-12, (
        f"a melhor rede dá perda {perda_v:.6f} na validação, e a curva diz {curva[e]['perda_validacao']:.6f} na época {e}")
    assert melhor is not rede and melhor[0]["W"] is not rede[0]["W"], "a melhor rede tem que ser uma cópia"


TESTES = [testar_para_depois_da_paciencia, testar_volta_a_melhorar_dentro_da_paciencia,
          testar_melhora_sem_recorde_nao_zera_a_paciencia, testar_empate_nao_e_melhora, testar_nunca_melhora, testar_para_em_max_epocas, testar_sorteio_criado_uma_vez, testar_recusa_sem_treinar,
          testar_igual_ao_ciclo_da_aula, testar_melhor_rede_e_a_da_curva]

if __name__ == "__main__":
    falhas = 0
    for t in TESTES:
        try:
            t()
        except NotImplementedError as erro:
            raise SystemExit(f"NotImplementedError: {erro}")
        except Exception as erro:  # noqa: BLE001 - o teste mostra qualquer erro do aluno
            falhas += 1
            print(f"FALHOU {t.__name__}: {type(erro).__name__}: {erro}")
    if falhas:
        print(f"{falhas} de {len(TESTES)} testes falharam")
        sys.exit(1)
    print(f"ok: {len(TESTES)} testes passaram")
