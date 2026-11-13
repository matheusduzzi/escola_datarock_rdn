"""Aula «Atenção Q/K/V» do curso Redes neurais: tudo o que a aula mostra, partes 1 a 9.

    python3 atencao.py

Só usa o NumPy. Os números das palavras e as três matrizes de pesos foram escolhidos à mão,
para a conta caber no papel. Num modelo treinado, eles saem do treino.
"""
import numpy as np

PALAVRAS = ["pedido", "não", "chegou"]

# Cada palavra vira 4 números (linhas = palavras, colunas = d_model = 4).
X = np.array([[1, 0, 1, 0],    # pedido
              [0, 1, 0, 1],    # não
              [1, 0, 0, 1]],   # chegou
             dtype=float)

# Pesos das projeções: 4 linhas (d_model) e 2 colunas (d_k = d_v = 2).
W_Q = np.array([[1, 0], [0, 1], [0, 0], [0, 1]], dtype=float)
W_K = np.array([[0, 1], [0, 0], [1, 0], [0, 1]], dtype=float)
W_V = np.array([[1, 0], [0, -2], [1, 0], [0, 1]], dtype=float)


def softmax(z):
    """Softmax por linha, estável: tira o maior de cada linha antes do exp (tarefa 3 do diagnóstico)."""
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def atencao(X, W_Q, W_K, W_V, causal=False):
    """Atenção de uma cabeça: retorna as notas divididas, os pesos e a saída."""
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    notas = Q @ K.T / np.sqrt(K.shape[1])
    if causal:
        notas = np.where(np.tril(np.ones_like(notas)) == 1, notas, -np.inf)
    pesos = softmax(notas)
    return notas, pesos, pesos @ V


def forma(m):
    return f"({m.shape[0]} x {m.shape[1]})"


def matriz(nome, m, rotulos=None, casas=3):
    print(f"  {nome} {forma(m)}")
    for i, linha in enumerate(m):
        rot = f"{rotulos[i]:<7}" if rotulos else ""
        print("    " + rot + "".join(f"{v:9.{casas}f}" if np.isfinite(v) else "     -inf" for v in linha))


def tabela_pesos(pesos):
    print("    " + " " * 7 + "".join(f"{p:>9}" for p in PALAVRAS) + "     soma")
    for i, linha in enumerate(pesos):
        print(f"    {PALAVRAS[i]:<7}" + "".join(f"{v:9.3f}" for v in linha) + f"    {linha.sum():.3f}")


def parte1():
    print("1. As três palavras viram números")
    matriz("X", X, PALAVRAS, casas=0)
    print("  cada linha é uma palavra; cada uma tem d_model = 4 números")


def parte2():
    print("\n2. As projeções: Q = X W_Q, K = X W_K, V = X W_V")
    for nome, W in (("W_Q", W_Q), ("W_K", W_K), ("W_V", W_V)):
        matriz(nome, W, casas=0)
    for nome, W in (("Q", W_Q), ("K", W_K), ("V", W_V)):
        print(f"  {nome} = X {forma(X)} x W_{nome} {forma(W)} = {forma(X @ W)}")
    for nome, W in (("Q", W_Q), ("K", W_K), ("V", W_V)):
        matriz(nome, X @ W, PALAVRAS, casas=0)


def parte3():
    print("\n3. As notas: Q K^T, dividido pela raiz de d_k")
    Q, K = X @ W_Q, X @ W_K
    print(f"  Q {forma(Q)} x K^T {forma(K.T)} = {forma(Q @ K.T)}")
    matriz("Q K^T (linha = quem procura, coluna = quem é achado)", Q @ K.T, PALAVRAS, casas=0)
    print(f"  d_k = {K.shape[1]}, raiz de d_k = {np.sqrt(K.shape[1]):.3f}")
    matriz("Q K^T / raiz(2)", Q @ K.T / np.sqrt(2), PALAVRAS)


def parte4():
    print("\n4. Os pesos: softmax de cada linha")
    notas, pesos, _ = atencao(X, W_Q, W_K, W_V)
    tabela_pesos(pesos)
    print(f"  pesos {forma(pesos)}: uma linha por palavra que procura, uma coluna por palavra achada")
    e = np.exp(notas[1] - notas[1].max())
    print("  a linha de 'não', passo a passo:")
    print("    notas divididas: " + "  ".join(f"{v:.3f}" for v in notas[1]))
    print("    menos a maior:   " + "  ".join(f"{v:.3f}" for v in notas[1] - notas[1].max()))
    print("    exp:             " + "  ".join(f"{v:.3f}" for v in e) + f"   soma {e.sum():.3f}")
    print("    dividido pela soma: " + "  ".join(f"{v:.3f}" for v in e / e.sum()))


def parte5():
    print("\n5. A saída: pesos x V")
    _, pesos, saida = atencao(X, W_Q, W_K, W_V)
    V = X @ W_V
    print(f"  pesos {forma(pesos)} x V {forma(V)} = {forma(saida)}")
    matriz("saída", saida, PALAVRAS)
    p = pesos[1]
    print("  a linha de 'não': " + " + ".join(f"{p[j]:.3f} x {V[j].astype(int).tolist()}" for j in range(3))
          + " = [" + ", ".join(f"{v:.3f}" for v in saida[1]) + "]")


def parte6():
    print("\n6. Por que dividir pela raiz de d_k")
    rng = np.random.default_rng(15)
    print("  q e k sorteados com média 0 e desvio 1; 10.000 pares por linha; 10 palavras por frase")
    print("  d_k    desvio de q.k    maior peso sem dividir    maior peso dividindo")
    for d in (2, 16, 64, 512):
        q = rng.standard_normal((10_000, d))
        k = rng.standard_normal((10_000, d))
        desvio = (q * k).sum(axis=1).std()
        qs = rng.standard_normal((2_000, 1, d))
        ks = rng.standard_normal((2_000, 10, d))
        n = (qs * ks).sum(axis=2)
        sem = softmax(n).max(axis=1).mean()
        com = softmax(n / np.sqrt(d)).max(axis=1).mean()
        print(f"  {d:>4}    {desvio:10.2f}         {sem:10.3f}               {com:10.3f}")
    print("  raiz de 2, 16, 64 e 512: " + ", ".join(f"{np.sqrt(d):.2f}" for d in (2, 16, 64, 512)))


def parte7():
    print("\n7. A máscara causal: cada palavra só olha para ela e para as anteriores")
    notas, pesos, saida = atencao(X, W_Q, W_K, W_V, causal=True)
    matriz("notas divididas, com -inf acima da diagonal", notas, PALAVRAS)
    tabela_pesos(pesos)
    _, sem_mascara, _ = atencao(X, W_Q, W_K, W_V)
    print("  a última linha é igual sem máscara: " + str(bool(np.allclose(pesos[-1], sem_mascara[-1]))))


def parte8():
    print("\n8. A ordem das palavras")
    ordem = [2, 1, 0]
    _, pesos, saida = atencao(X[ordem], W_Q, W_K, W_V)
    _, p0, s0 = atencao(X, W_Q, W_K, W_V)
    trocada = [PALAVRAS[i] for i in ordem]
    print("  frase trocada: " + " ".join(trocada))
    matriz("saída", saida, trocada)
    print("  cada palavra tem a mesma saída que antes: " + str(bool(np.allclose(saida, s0[ordem]))))


def parte9():
    print("\n9. Do exemplo ao tamanho do artigo de 2017 (modelo base)")
    d_model, h = 512, 8
    d_k = d_v = d_model // h
    n = 60
    print(f"  d_model = {d_model}, cabeças = {h}, d_k = d_v = {d_model} / {h} = {d_k}; frase de n = {n} tokens")
    print("  uma cabeça:")
    for nome, (a, b) in [("X", (n, d_model)), ("W_Q, W_K", (d_model, d_k)), ("W_V", (d_model, d_v)),
                         ("Q, K", (n, d_k)), ("V", (n, d_v)), ("Q K^T e pesos", (n, n)), ("saída", (n, d_v))]:
        print(f"    {nome:<15} ({a} x {b})")
    print(f"  {h} saídas lado a lado: ({n} x {h * d_v}); W_O ({h * d_v} x {d_model}); resultado ({n} x {d_model})")
    proj = h * (2 * d_model * d_k + d_model * d_v) + h * d_v * d_model
    print(f"  pesos das projeções na camada: {h} x (2 x {d_model} x {d_k} + {d_model} x {d_v}) + {h * d_v} x {d_model}"
          f" = {proj:,}".replace(",", "."))
    print(f"  notas por cabeça: {n} x {n} = {n * n:,}".replace(",", ".")
          + f"; nas {h} cabeças: {h * n * n:,}".replace(",", "."))
    print(f"  passos em fila entre o token 1 e o último: atenção 1; rede recorrente {n - 1}")
    for m in (60, 120, 600):
        print(f"  n = {m:>3}: notas por cabeça {m * m:>7,}".replace(",", "."))


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    parte1(); parte2(); parte3(); parte4(); parte5(); parte6(); parte7(); parte8(); parte9()
