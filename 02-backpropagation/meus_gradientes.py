"""Exercício da aula «Backpropagation»: os gradientes da rede que separa camisa de casaco.

Escreva as duas funções marcadas com NotImplementedError. Depois rode, de dentro da pasta da aula:

    python3 testar_meus_gradientes.py
    python3 meus_gradientes.py

Enquanto as funções não existem, os dois comandos param com NotImplementedError.
"""
import numpy as np

from gradientes_do_caio import gradientes_do_caio

# A rede da Bia para camisa e casaco: 3 números de entrada, uma camada escondida de
# 3 neurônios com ReLU e 2 notas na saída. W no formato (neurônios, entradas).
CAMISA_CASACO = {
    "W1": np.array([[1, 0, 1], [0, 1, 1], [-1, 1, 0]], dtype=float),
    "b1": np.array([0, 0.5, 0], dtype=float),
    "W2": np.array([[1, -1, 2], [0, 1, 1]], dtype=float),
    "b2": np.array([0, 0], dtype=float),
}
CLASSES = ["camisa", "casaco"]
FOTO = [1, -1, 2]  # é uma camisa: classe 0

# A rede e as fotos da aula, para a tarefa 3. As duas fotos são camisetas: classe 0.
REDE_DA_AULA = {
    "W1": np.array([[1, 2, 0, -1], [-1, 0, 2, 1], [0, -2, 1, -1]], dtype=float),
    "b1": np.array([0, 0.5, 1], dtype=float),
    "W2": np.array([[2, -1, 1], [-1, 2, 1]], dtype=float),
    "b2": np.array([0, 0.5], dtype=float),
}
FOTO_DA_AULA = [1, 0.5, 0, 1]
LOTE_DA_AULA = [[1, 0.5, 0, 1], [0, 0, 1, 1]]


def notas(x, rede):
    """Passagem direta da aula «Neurônio, ativação e MLP»: uma linha de notas por foto."""
    X = np.atleast_2d(np.asarray(x, dtype=float))
    h = np.maximum(X @ rede["W1"].T + rede["b1"], 0)
    return h @ rede["W2"].T + rede["b2"]


def probabilidades(x, rede):
    z = notas(x, rede)
    z = z - z.max(axis=1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)


def perda(x, classe, rede):
    """Log-loss média: a média de -log da probabilidade da classe certa de cada foto."""
    p = probabilidades(x, rede)
    classes = np.atleast_1d(classe)
    return float(-np.log(p[np.arange(len(p)), classes]).mean())


def retropropagar(x, classe, rede):
    """Retorna um dicionário com o gradiente da perda média para W1, b1, W2 e b2.

    x é uma foto (vetor) ou um lote (uma foto por linha); classe é o número da classe
    certa de cada foto (um número para uma foto, uma lista para o lote).
    Cada gradiente tem o mesmo formato do parâmetro. Não mexa nos arrays de rede.
    Pare com ValueError se alguma classe não existir na rede ou se o número de
    classes não for o número de fotos.
    """
    raise NotImplementedError("escreva retropropagar(x, classe, rede)")


def gradiente_numerico(x, classe, rede, nome, eps=1e-5):
    """Retorna o gradiente de rede[nome] por diferença finita central.

    Para cada número do parâmetro: some eps, calcule a perda, subtraia eps, calcule de novo,
    divida a diferença por 2 * eps e volte o número ao valor original.
    """
    raise NotImplementedError("escreva gradiente_numerico(x, classe, rede, nome)")


def diferenca_relativa(a, b):
    """A maior diferença entre a e b, dividida pelo maior dos dois. 0 quando os dois são 0."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    topo = np.maximum(np.abs(a), np.abs(b))
    d = np.where(topo < 1e-12, 0.0, np.abs(a - b) / np.where(topo < 1e-12, 1.0, topo))
    return float(d.max())


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, floatmode="maxprec")
    rede = {k: v.copy() for k, v in CAMISA_CASACO.items()}
    g = retropropagar(FOTO, 0, rede)
    num = {k: gradiente_numerico(FOTO, 0, rede, k) for k in rede}
    nova = {k: rede[k] - 0.1 * g[k] for k in rede}
    caio_foto = gradientes_do_caio(FOTO_DA_AULA, 0, REDE_DA_AULA)
    caio_lote = gradientes_do_caio(LOTE_DA_AULA, [0, 0], REDE_DA_AULA)
    num_foto = {k: gradiente_numerico(FOTO_DA_AULA, 0, REDE_DA_AULA, k) for k in REDE_DA_AULA}
    num_lote = {k: gradiente_numerico(LOTE_DA_AULA, [0, 0], REDE_DA_AULA, k) for k in REDE_DA_AULA}

    print(f"Foto: {FOTO}, classe certa: {CLASSES[0]}")
    print(f"  notas: {notas(FOTO, rede)[0]}")
    print(f"  probabilidades: {probabilidades(FOTO, rede)[0]}")
    print(f"  perda: {perda(FOTO, 0, rede):.4f}")
    for k in ("W2", "b2", "W1", "b1"):
        print(f"  d{k} =\n{g[k]}" if g[k].ndim == 2 else f"  d{k} = {g[k]}")
    print("  diferença relativa para a diferença finita: "
          + ", ".join(f"{k} {diferenca_relativa(g[k], num[k]):.1e}" for k in rede))
    print(f"  um passo de taxa 0.1: perda {perda(FOTO, 0, rede):.4f} -> {perda(FOTO, 0, nova):.4f}")
    print()
    print("Gradientes do Caio, comparados com a diferença finita, na rede da aula")
    print("  foto da aula: " + ", ".join(f"{k} {diferenca_relativa(caio_foto[k], num_foto[k]):.1e}" for k in REDE_DA_AULA))
    print("  lote da aula: " + ", ".join(f"{k} {diferenca_relativa(caio_lote[k], num_lote[k]):.1e}" for k in REDE_DA_AULA))
