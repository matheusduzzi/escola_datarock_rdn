"""Tudo o que a aula «Convolução e dimensões» mostra, partes 1 a 6.

Rode de dentro da pasta da aula:  python3 convolucao.py
Só precisa do NumPy. Não usa as fotos do Fashion MNIST: o pedaço de foto da parte 1 está escrito aqui,
e a rede da parte 5 recebe 64 fotos de mentira, com números sorteados entre 0 e 1, só para medir o
formato de cada camada. Os pesos também são sorteados: a rede não foi treinada.
"""
import numpy as np

# Um pedaço de 6 x 6 pixels do ombro de uma camiseta. 0 é o fundo, 3 é o tecido.
PEDACO = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 3, 3, 3],
    [0, 1, 3, 3, 3, 3],
    [0, 3, 3, 3, 3, 3],
    [0, 3, 3, 3, 3, 1],
    [0, 3, 3, 3, 1, 0],
], dtype=float)

# O kernel de borda vertical: a coluna da direita menos a da esquerda.
KERNEL = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
], dtype=float)


def convoluir(x, W, b, passo=1, borda=0):
    """x: (fotos, canais, altura, largura). W: (filtros, canais, k, k). b: (filtros,).

    Desliza cada filtro pela foto, de `passo` em `passo` pixels, depois de pôr `borda` pixels de zeros
    em volta. Cada posição é a soma de pesos x pixels mais o viés do filtro.
    """
    if borda:
        x = np.pad(x, ((0, 0), (0, 0), (borda, borda), (borda, borda)))
    k = W.shape[2]
    linhas = range(0, x.shape[2] - k + 1, passo)
    colunas = range(0, x.shape[3] - k + 1, passo)
    saida = np.empty((x.shape[0], W.shape[0], len(linhas), len(colunas)))
    for i, lin in enumerate(linhas):
        for j, col in enumerate(colunas):
            janela = x[:, :, lin:lin + k, col:col + k]
            saida[:, :, i, j] = np.tensordot(janela, W, axes=([1, 2, 3], [1, 2, 3])) + b
    return saida


def max_pool(x, k=2, passo=2):
    """Fica com o maior valor de cada janela k x k, de `passo` em `passo` pixels. Não tem pesos."""
    linhas = range(0, x.shape[2] - k + 1, passo)
    colunas = range(0, x.shape[3] - k + 1, passo)
    saida = np.empty((x.shape[0], x.shape[1], len(linhas), len(colunas)))
    for i, lin in enumerate(linhas):
        for j, col in enumerate(colunas):
            saida[:, :, i, j] = x[:, :, lin:lin + k, col:col + k].max(axis=(2, 3))
    return saida


def mostrar(mapa, recuo="    "):
    for linha in mapa:
        print(recuo + " ".join(f"{v:3.0f}" for v in linha))


def milhar(n):
    return f"{n:,}".replace(",", ".")


def por_cento(x):
    return f"{100 * x:.1f}%".replace(".", ",")


def um_canal(m):
    return m[np.newaxis, np.newaxis]


def parte1():
    print("1. O pedaço da foto, o kernel e o mapa (passo 1, sem borda)")
    print("  pedaço 6 x 6:")
    mostrar(PEDACO)
    print("  kernel 3 x 3:")
    mostrar(KERNEL)
    mapa = convoluir(um_canal(PEDACO), um_canal(KERNEL), np.zeros(1))[0, 0]
    print(f"  mapa {mapa.shape[0]} x {mapa.shape[1]}:")
    mostrar(mapa)
    janela = PEDACO[1:4, 0:3]
    print("  a linha 2, coluna 1 do mapa sai da janela das linhas 2 a 4, colunas 1 a 3 do pedaço:")
    mostrar(janela)
    direita, esquerda = int(janela[:, 2].sum()), int(janela[:, 0].sum())
    print(f"    coluna da direita x 1: {direita}; coluna da esquerda x (-1): {-esquerda};"
          f" coluna do meio x 0: 0; soma: {(janela * KERNEL).sum():.0f}")
    return mapa


def parte2():
    print("\n2. Passo e borda, no mesmo pedaço e no mesmo kernel")
    for passo, borda in [(2, 0), (1, 1), (2, 1)]:
        mapa = convoluir(um_canal(PEDACO), um_canal(KERNEL), np.zeros(1), passo=passo, borda=borda)[0, 0]
        print(f"  passo {passo}, borda {borda}: mapa {mapa.shape[0]} x {mapa.shape[1]}")
        mostrar(mapa)


def parte3(mapa):
    print("\n3. Max pooling 2 x 2, passo 2, no mapa da parte 1")
    print("  o mapa 4 x 4 depois da ReLU (negativo vira 0):")
    relu = np.maximum(mapa, 0)
    mostrar(relu)
    pool = max_pool(um_canal(relu))[0, 0]
    print(f"  depois do pooling: {pool.shape[0]} x {pool.shape[1]}")
    mostrar(pool)
    mapa5 = np.arange(1, 26, dtype=float).reshape(5, 5)
    pool5 = max_pool(um_canal(mapa5))[0, 0]
    print("  um mapa 5 x 5 (números de 1 a 25):")
    mostrar(mapa5)
    print(f"  depois do pooling: {pool5.shape[0]} x {pool5.shape[1]}")
    mostrar(pool5)


def parte4():
    print("\n4. Canais: uma foto com 3 canais e 4 filtros 3 x 3")
    gerador = np.random.default_rng(6)
    foto = gerador.random((1, 3, 6, 6))
    W = gerador.normal(0, 0.1, (4, 3, 3, 3))
    b = np.zeros(4)
    saida = convoluir(foto, W, b)
    print(f"  entrada {foto.shape[1:]} (canais, altura, largura); pesos W {W.shape}; vieses b {b.shape}")
    print(f"  saída {saida.shape[1:]}: um mapa por filtro")
    print(f"  parâmetros: W {W.size} + b {b.size} = {W.size + b.size}")


def rede_do_caio(borda, gerador, densa_entradas=None):
    """A rede da aula, camada por camada, com pesos sorteados. Retorna as linhas da tabela."""
    x = gerador.random((64, 1, 28, 28))
    linhas = [("entrada", x.shape[1:], 0)]
    W1, b1 = gerador.normal(0, 0.1, (32, 1, 3, 3)), np.zeros(32)
    x = np.maximum(convoluir(x, W1, b1, borda=borda), 0)
    linhas.append((f"conv 32 filtros 3 x 3, borda {borda}", x.shape[1:], W1.size + b1.size))
    x = max_pool(x)
    linhas.append(("max pool 2 x 2, passo 2", x.shape[1:], 0))
    W2, b2 = gerador.normal(0, 0.1, (64, 32, 3, 3)), np.zeros(64)
    x = np.maximum(convoluir(x, W2, b2, borda=borda), 0)
    linhas.append((f"conv 64 filtros 3 x 3, borda {borda}", x.shape[1:], W2.size + b2.size))
    x = max_pool(x)
    linhas.append(("max pool 2 x 2, passo 2", x.shape[1:], 0))
    x = x.reshape(x.shape[0], -1)
    linhas.append(("achatar", x.shape[1:], 0))
    entradas = densa_entradas or x.shape[1]
    W3, b3 = gerador.normal(0, 0.1, (entradas, 128)), np.zeros(128)
    x = np.maximum(np.dot(x, W3) + b3, 0)
    linhas.append((f"densa {entradas} -> 128", x.shape[1:], W3.size + b3.size))
    W4, b4 = gerador.normal(0, 0.1, (128, 10)), np.zeros(10)
    x = np.dot(x, W4) + b4
    linhas.append(("densa 128 -> 10", x.shape[1:], W4.size + b4.size))
    return linhas


def tabela(linhas):
    print(f"  {'camada':34} {'saída (canais, altura, largura)':>32} {'parâmetros':>11}")
    for nome, formato, params in linhas:
        formato = " x ".join(str(d) for d in formato)
        print(f"  {nome:34} {formato:>32} {milhar(params):>11}")
    total = sum(p for _, _, p in linhas)
    print(f"  {'total':34} {'':>32} {milhar(total):>11}")
    return total


def parte5():
    print("\n5. A rede do Caio, 64 fotos de 28 x 28, camada por camada")
    gerador = np.random.default_rng(6)
    print("  o plano do tutorial, com a densa esperando 7 x 7 x 64 = 3136 entradas, e borda 0:")
    try:
        rede_do_caio(0, gerador, densa_entradas=7 * 7 * 64)
    except ValueError as erro:
        print(f"    ValueError: {erro}")
    print("  borda 0 (a rede do Caio, com a densa corrigida):")
    total0 = tabela(rede_do_caio(0, np.random.default_rng(6)))
    print("  borda 1 (o tutorial):")
    total1 = tabela(rede_do_caio(1, np.random.default_rng(6)))
    return total0, total1


def parte6(total0, total1):
    print("\n6. Convolução contra camada densa, na mesma foto 28 x 28")
    densa = 784 * 128 + 128
    conv = 32 * (3 * 3 * 1) + 32
    mlp = densa + 128 * 10 + 10
    print(f"  densa 784 -> 128: 784 x 128 + 128 = {milhar(densa)}")
    print(f"  conv 32 filtros 3 x 3: 32 x (3 x 3 x 1) + 32 = {milhar(conv)}")
    print(f"  rede da aula «MLP de classificação», 784 -> 128 -> 10: {milhar(mlp)}")
    print(f"  rede do Caio, borda 0: {milhar(total0)}; borda 1: {milhar(total1)}")
    convs = 320 + 18496
    print(f"  na borda 0, as duas convoluções somam {milhar(convs)} ({por_cento(convs / total0)} do total);"
          f" a densa depois do achatar, {milhar(204928)} ({por_cento(204928 / total0)})")

if __name__ == "__main__":
    mapa = parte1()
    parte2()
    parte3(mapa)
    parte4()
    total0, total1 = parte5()
    parte6(total0, total1)
