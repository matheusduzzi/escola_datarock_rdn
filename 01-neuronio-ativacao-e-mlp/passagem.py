"""As contas da aula «Neurônio, ativação e MLP», do curso Redes neurais.

Rode de dentro da pasta da aula:  python3 passagem.py
Só usa o NumPy. Não tem sorteio: a saída sai igual toda vez.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True, floatmode="maxprec")

# A rede pequena da aula: 4 pixels de entrada, 3 neurônios com ReLU, 2 notas na saída.
W1 = np.array([[1, 2, 0, -1],
               [-1, 0, 2, 1],
               [0, -2, 1, -1]], dtype=float)
b1 = np.array([0, 0.5, 1], dtype=float)
W2 = np.array([[2, -1, 1],
               [-1, 2, 1]], dtype=float)
b2 = np.array([0, 0.5], dtype=float)
CLASSES = ["camiseta", "calça"]
FOTO = np.array([1, 0.5, 0, 1], dtype=float)


def relu(z):
    return np.maximum(z, 0)


def parametros(tamanhos):
    return [(entra, sai, entra * sai + sai) for entra, sai in zip(tamanhos[:-1], tamanhos[1:])]


print("1. Uma passagem à mão")
z1 = W1 @ FOTO + b1
h = relu(z1)
notas = W2 @ h + b2
print(f"  x = {FOTO}")
print(f"  W1 x + b1 = {z1}")
print(f"  h = ReLU(W1 x + b1) = {h}")
print(f"  notas = W2 h + b2 = {notas}")
print(f"  maior nota: {CLASSES[int(np.argmax(notas))]}")
print()

print("2. Um lote de duas fotos, uma por linha")
X = np.array([FOTO, [0, 0, 1, 1]], dtype=float)
H = relu(X @ W1.T + b1)
N = H @ W2.T + b2
print(f"  X tem formato {X.shape}, H tem {H.shape}, as notas têm {N.shape}")
print(f"  H =\n{H}")
print(f"  notas =\n{N}")
print()

print("3. Sem a ReLU, as duas camadas viram uma só")
sem = W2 @ (W1 @ FOTO + b1) + b2
W_unica = W2 @ W1
b_unico = W2 @ b1 + b2
print(f"  duas camadas sem ReLU: {sem}")
print(f"  W2 W1 =\n{W_unica}")
print(f"  W2 b1 + b2 = {b_unico}")
print(f"  uma camada só: {W_unica @ FOTO + b_unico}")
print(f"  iguais no lote também: {np.allclose(X @ W1.T @ W2.T + b1 @ W2.T + b2, X @ W_unica.T + b_unico)}")
print()

print("4. Parâmetros por camada (pesos + vieses)")
for tamanhos in ([4, 3, 2], [784, 10], [784, 64, 10], [784, 256, 10], [784, 32, 32, 32, 10]):
    camadas = parametros(tamanhos)
    partes = " + ".join(f"{p:,}".replace(",", ".") for _, _, p in camadas)
    total = sum(p for _, _, p in camadas)
    print(f"  {' -> '.join(map(str, tamanhos)):<28} camadas com peso: {len(camadas)}  {partes} = {total:,}".replace(",", "."))
print()

print("5. Aproximar x² entre 0 e 2 com n neurônios ReLU, pesos escolhidos à mão")
grade = np.linspace(0, 2, 4097)
for n in (1, 2, 4, 8, 32):
    d = 2 / n
    t = np.arange(n) * d                    # onde cada neurônio começa a subir
    inclinacao = t + (t + d)                # inclinação de x² em cada trecho de largura d
    c = np.diff(np.concatenate([[0.0], inclinacao]))   # peso de saída de cada neurônio
    rede = lambda x: np.sum(relu(np.subtract.outer(x, t)) * c, axis=1)
    erro_max = np.max(np.abs(rede(grade) - grade ** 2))
    em3 = rede(np.array([3.0]))[0]
    print(f"  n = {n:<3} parâmetros {3 * n + 1:<4} erro máximo entre 0 e 2: {erro_max:.6f}"
          f"   em x = 3: rede {em3:.4f}, x² = 9, erro {abs(em3 - 9):.4f}")
print()

print("6. Formato trocado que roda sem erro")
W = np.array([[1, 2, 0], [0, 1, -1], [2, 0, 1]], dtype=float)
L = np.array([[1, 0, 2], [0, 1, 1], [2, 1, 0]], dtype=float)   # 3 fotos de 3 pixels, uma por linha
print(f"  certo, L @ W.T:\n{L @ W.T}")
print(f"  trocado, W @ L:\n{W @ L}")
print(f"  iguais: {np.allclose(L @ W.T, W @ L)}")
