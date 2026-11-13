"""Exercício da aula «Neurônio, ativação e MLP»: a rede de calçados da Bia.

Escreva as duas funções marcadas com NotImplementedError. Depois rode, de dentro da pasta da aula:

    python3 testar_minha_passagem.py
    python3 minha_passagem.py

Enquanto as funções não existem, os dois comandos param com NotImplementedError.
"""
import numpy as np

# A rede que separa tênis de sandália: 3 números de entrada, duas camadas escondidas com ReLU
# (3 e 2 neurônios) e 2 notas na saída. Cada camada é (W, b, usa_relu).
CALCADOS = [
    (np.array([[1, -2, 0.5], [0, 1, -1], [-1, 0, 1]]), np.array([0.5, 1, -0.5]), True),
    (np.array([[1, 1, -2], [0.5, -1, 2]]), np.array([0, -1]), True),
    (np.array([[2, -1], [-1, 2]]), np.array([0, 0.5]), False),
]
CLASSES = ["tênis", "sandália"]
FOTO = [1, 0.5, 2]

# As duas propostas do fornecedor, em número de neurônios por camada, da entrada à saída.
PROPOSTA_A = [784, 512, 10]
PROPOSTA_B = [784, 64, 64, 64, 64, 10]


def relu(z):
    return np.maximum(z, 0)


def passagem(x, camadas):
    """Retorna a saída da última camada para x.

    x pode ser uma foto (vetor) ou um lote (uma foto por linha), como lista ou array.
    Cada camada é (W, b, usa_relu), com W no formato (neurônios, entradas), como no PyTorch.
    Pare com ValueError se o número de colunas de W não for o tamanho da entrada
    ou se b não tiver um valor por neurônio.
    """
    raise NotImplementedError("escreva passagem(x, camadas)")


def contar_parametros(tamanhos):
    """Retorna quantos pesos e vieses a rede aprende.

    tamanhos é a lista de neurônios por camada, da entrada à saída, por exemplo [4, 3, 2].
    Pare com ValueError se a lista tiver menos de dois tamanhos ou algum tamanho menor que 1.
    """
    raise NotImplementedError("escreva contar_parametros(tamanhos)")


if __name__ == "__main__":
    print(f"Foto: {FOTO}")
    for k in range(1, len(CALCADOS) + 1):
        print(f"  saída da camada {k}: {passagem(FOTO, CALCADOS[:k])}")
    notas = passagem(FOTO, CALCADOS)
    print(f"  maior nota: {CLASSES[int(np.argmax(notas))]}")
    print()
    print(f"Parâmetros da rede de calçados: {contar_parametros([3, 3, 2, 2])}")
    print(f"Proposta A {PROPOSTA_A}: {contar_parametros(PROPOSTA_A)}")
    print(f"Proposta B {PROPOSTA_B}: {contar_parametros(PROPOSTA_B)}")
