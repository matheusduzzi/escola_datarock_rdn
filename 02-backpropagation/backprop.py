"""As contas da aula «Backpropagation», na rede pequena da aula «Neurônio, ativação e MLP».

Rode de dentro da pasta da aula:  python3 backprop.py
A saída sai igual toda vez: os pesos são fixos e não há sorteio.

O script segue a aula em ordem, conta por conta, para as fotos da aula. Ele não traz funções
prontas de retropropagação nem de diferença finita: essas você escreve no exercício.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True, floatmode="maxprec")

# A rede da aula 01: 4 pixels, uma camada escondida de 3 neurônios com ReLU e 2 notas.
REDE = {
    "W1": np.array([[1, 2, 0, -1], [-1, 0, 2, 1], [0, -2, 1, -1]], dtype=float),
    "b1": np.array([0, 0.5, 1], dtype=float),
    "W2": np.array([[2, -1, 1], [-1, 2, 1]], dtype=float),
    "b2": np.array([0, 0.5], dtype=float),
}
CLASSES = ["camiseta", "calça"]
FOTO = np.array([1, 0.5, 0, 1], dtype=float)  # é uma camiseta
OUTRA = np.array([0, 0, 1, 1], dtype=float)   # também é uma camiseta, e a rede diz calça
DOBRA = np.array([1, 0, 0, 0.5], dtype=float)  # o neurônio 2 dá exatamente 0 antes da ReLU
CAMISETA = np.array([1.0, 0.0])               # o rótulo y de uma camiseta


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def perda(X, classes, rede):
    """Log-loss média do lote: -log da probabilidade da classe certa."""
    X = np.atleast_2d(X)
    classes = np.atleast_1d(classes)
    notas = np.maximum(X @ rede["W1"].T + rede["b1"], 0) @ rede["W2"].T + rede["b2"]
    z = notas - notas.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(X)), classes].mean())


def diferenca_relativa(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    topo = np.maximum(np.abs(a), np.abs(b))
    d = np.where(topo < 1e-12, 0.0, np.abs(a - b) / np.where(topo < 1e-12, 1.0, topo))
    return float(d.max())


def mostrar(nome, valor):
    print(f"  {nome} =\n{valor}" if np.ndim(valor) == 2 else f"  {nome} = {valor}")


if __name__ == "__main__":
    rede = {k: v.copy() for k, v in REDE.items()}

    # A foto da aula: a passagem direta e as sete linhas da seção 4, uma por uma.
    z1 = rede["W1"] @ FOTO + rede["b1"]
    h = np.maximum(z1, 0)
    notas = rede["W2"] @ h + rede["b2"]
    p = softmax(notas)
    d_notas = p - CAMISETA                  # perda -> notas
    dW2 = d_notas[:, None] @ h[None, :]   # coluna (p - y) vezes linha h
    db2 = d_notas
    d_h = d_notas @ rede["W2"]              # notas -> h: o "W2 de lado"
    d_z1 = d_h * (z1 > 0)                   # h -> z1: a ReLU deixa passar ou zera
    dW1 = d_z1[:, None] @ FOTO[None, :]   # coluna perda -> z1 vezes linha x
    db1 = d_z1
    g = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # O lote das duas camisetas: as mesmas sete linhas, com uma foto por linha.
    L = np.array([FOTO, OUTRA])
    Y = np.array([CAMISETA, CAMISETA])
    z1_l = L @ rede["W1"].T + rede["b1"]
    h_l = np.maximum(z1_l, 0)
    notas_l = h_l @ rede["W2"].T + rede["b2"]
    d_notas_l = softmax(notas_l) - Y        # sem dividir por 2: o gradiente da soma das perdas
    d_z1_l = (d_notas_l @ rede["W2"]) * (z1_l > 0)
    g_soma = {"W1": d_z1_l.T @ L, "b1": d_z1_l.sum(axis=0), "W2": d_notas_l.T @ h_l, "b2": d_notas_l.sum(axis=0)}
    g_media = {k: v / 2 for k, v in g_soma.items()}   # dividido por 2: o gradiente da média

    # A foto da dobra: as sete linhas de novo.
    z1_d = rede["W1"] @ DOBRA + rede["b1"]
    h_d = np.maximum(z1_d, 0)
    d_notas_d = softmax(rede["W2"] @ h_d + rede["b2"]) - CAMISETA
    d_z1_d = (d_notas_d @ rede["W2"]) * (z1_d > 0)
    g_dobra = {"W1": d_z1_d[:, None] @ DOBRA[None, :], "b1": d_z1_d, "W2": d_notas_d[:, None] @ h_d[None, :],
               "b2": d_notas_d}

    # Diferença finita nas três situações: cada número de cada parâmetro sobe e desce 0,00001,
    # a perda é recalculada nos dois casos e o número volta ao que era.
    num = {}
    for caso, X, classes in (("foto", FOTO, 0), ("lote", L, [0, 0]), ("dobra", DOBRA, 0)):
        num[caso] = {}
        for nome, w in rede.items():
            inclinacao = np.zeros_like(w)
            for i in np.ndindex(w.shape):
                guarda = w[i]
                w[i] = guarda + 1e-5
                mais = perda(X, classes, rede)
                w[i] = guarda - 1e-5
                menos = perda(X, classes, rede)
                w[i] = guarda
                inclinacao[i] = (mais - menos) / 2e-5
            num[caso][nome] = inclinacao

    print("1. A perda de uma foto")
    mostrar("notas", notas)
    mostrar("probabilidades", p)
    print(f"  classe certa: {CLASSES[0]}")
    print(f"  perda = -log({p[0]:.4f}) = {perda(FOTO, 0, rede):.4f}")

    print()
    print("2. Um peso de cada vez, pela regra da cadeia")
    print(f"  W2[camiseta, 1]: (p - y) x h1 = {d_notas[0]:.4f} x {h[0]:.0f} = {dW2[0, 0]:.4f}")
    print(f"  W1[1, 1]: soma dos dois caminhos x ReLU' x x1 = "
          f"({d_notas[0]:.4f} x 2 + {d_notas[1]:.4f} x -1) x 1 x 1 = {dW1[0, 0]:.4f}")
    for nome in ("W2", "W1"):
        guarda = rede[nome][0, 0]
        rede[nome][0, 0] = guarda + 1e-5
        mais = perda(FOTO, 0, rede)
        rede[nome][0, 0] = guarda - 1e-5
        menos = perda(FOTO, 0, rede)
        rede[nome][0, 0] = guarda
        print(f"  {nome}[1, 1] mexido 0,00001 para cada lado: ({mais:.8f} - {menos:.8f}) / 0,00002 = "
              f"{(mais - menos) / 2e-5:.4f}")

    print()
    print("3. Todos os gradientes, do fim para o começo")
    mostrar("perda -> notas (p - y)", d_notas)
    mostrar("dW2", dW2)
    mostrar("db2", db2)
    mostrar("perda -> h (W2 de lado vezes p - y)", d_h)
    mostrar("ReLU deixa passar (z1 > 0)", (z1 > 0).astype(int))
    mostrar("perda -> z1", d_z1)
    mostrar("dW1", dW1)
    mostrar("db1", db1)

    print()
    print("4. Conferência por diferença finita")
    for nome in rede:
        print(f"  {nome:3} maior diferença relativa: {diferenca_relativa(g[nome], num['foto'][nome]):.1e}")
    n_par = sum(v.size for v in rede.values())
    print(f"  parâmetros: {n_par}; passagens da diferença finita: {2 * n_par}; "
          f"caminho de trás: 1 passagem para a frente e 1 para trás")
    grande = 784 * 256 + 256 + 256 * 10 + 10
    print(f"  rede 784 -> 256 -> 10: {grande} parâmetros, {2 * grande} passagens por passo na diferença finita")

    print()
    print("5. Um passo de gradiente")
    soma_g2 = sum(float((v ** 2).sum()) for v in g.values())
    print(f"  soma dos gradientes ao quadrado: {soma_g2:.4f}")
    for taxa in (0.01, 0.5):
        nova = {k: rede[k] - taxa * g[k] for k in rede}
        antes, depois = perda(FOTO, 0, rede), perda(FOTO, 0, nova)
        print(f"  taxa {taxa}: perda {antes:.4f} -> {depois:.4f}; mudou {depois - antes:.4f}, "
              f"o gradiente previa {-taxa * soma_g2:.4f}")
    nova = {k: rede[k] - 0.5 * g[k] for k in rede}
    notas_nova = nova["W2"] @ np.maximum(nova["W1"] @ FOTO + nova["b1"], 0) + nova["b2"]
    print(f"  depois do passo de taxa 0.5: notas {notas_nova}")
    print(f"  a outra camiseta, (0, 0, 1, 1): perda {perda(OUTRA, 0, rede):.4f} -> {perda(OUTRA, 0, nova):.4f}")

    print()
    print("6. Um lote de duas fotos, as duas camisetas")
    mostrar("notas", notas_l)
    print(f"  perda de cada foto: {perda(FOTO, 0, rede):.4f} e {perda(OUTRA, 0, rede):.4f}; média {perda(L, [0, 0], rede):.4f}")
    mostrar("ReLU deixa passar, uma linha por foto", (z1_l > 0).astype(int))
    mostrar("dW1 (média das duas fotos)", g_media["W1"])
    print("  diferença relativa da média: "
          + ", ".join(f"{k} {diferenca_relativa(g_media[k], num['lote'][k]):.1e}" for k in rede))
    print("  diferença relativa da soma:  "
          + ", ".join(f"{k} {diferenca_relativa(g_soma[k], num['lote'][k]):.1e}" for k in rede))

    print()
    print("7. Na dobra da ReLU")
    mostrar("z1", z1_d)
    mostrar("dW1 linha 2, caminho de trás", g_dobra["W1"][1])
    mostrar("dW1 linha 2, diferença finita", num["dobra"]["W1"][1])
    print("  maior diferença relativa: "
          + ", ".join(f"{k} {diferenca_relativa(g_dobra[k], num['dobra'][k]):.1e}" for k in rede))
