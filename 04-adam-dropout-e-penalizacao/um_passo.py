"""Um peso, alguns passos: o SGD, o Adam, a L2 e a L1 lado a lado, sem rede e sem fotos.

Rode de dentro da pasta da aula:  python3 um_passo.py
Roda na hora. Usa as mesmas classes SGD e Adam do rodada.py.
"""
import numpy as np

from rodada import SGD, Adam


def caminhar(otimizador, gradiente, passos):
    """Um peso que começa em 0,5 e recebe sempre o mesmo gradiente. Retorna o peso depois de cada passo."""
    rede = [{"W": np.array([0.5]), "b": np.array([0.0])}]
    pesos = []
    for _ in range(passos):
        otimizador.passo(rede, [{"W": np.array([gradiente]), "b": np.array([0.0])}])
        pesos.append(float(rede[0]["W"][0]))
    return pesos


print("1. O tamanho do primeiro passo, com taxa 0,05 no SGD e 0,001 no Adam")
print("  gradiente   passo do SGD   passo do Adam")
for g in (0.002, 0.2, 20.0):
    sgd = 0.5 - caminhar(SGD(0.05), g, 1)[0]
    adam = 0.5 - caminhar(Adam(0.001), g, 1)[0]
    print(f"  {g:>9}   {sgd:>12.6f}   {adam:>13.6f}")

print("\n2. O Adam com a taxa do SGD: dez passos com gradiente 0,002")
print("  taxa 0,05: peso depois de 1, 5 e 10 passos:",
      ", ".join(f"{caminhar(Adam(0.05), 0.002, n)[-1]:.4f}" for n in (1, 5, 10)))

print("\n3. Um peso sem gradiente da perda, só com a penalização, taxa 0,05")
print("  As duas puxam 0,00025 no primeiro passo. A L2 puxa uma fração do peso; a L1, sempre o mesmo tanto.")
print("  passos     L2 0,01     L1 0,005")
w2 = w1 = 0.5
for passo in range(1, 4711):
    w2 -= 0.05 * 0.01 * w2
    w1 -= 0.05 * 0.005 * np.sign(w1)
    if passo in (1, 157, 1000, 2000, 2001, 4710):
        print(f"  {passo:>6}   {w2:>9.5f}   {w1:>10.5f}")
