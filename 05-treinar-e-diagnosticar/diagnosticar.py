"""A aula «Treinar e diagnosticar», do começo ao fim. Rode de dentro da pasta da aula:

    python3 diagnosticar.py

Leva uns 25 segundos. Precisa das fotos em ../dados/fashion-mnist (python3 baixar.py lá dentro).
Cada parte da saída corresponde a uma seção da aula.
"""
import numpy as np

import rede as R

Xt, yt, Xv, yv = R.preparar()
BASE = {"camadas": [784, 128, 10], "taxa": 0.05, "inicio": "he", "lote": 64, "epocas": 3,
        "fotos_treino": 10000, "l2": 0.0, "escolher_epoca": False}
FUNDA = [784, 64, 64, 64, 64, 64, 10]


def treino(**mudancas):
    return R.treinar(dict(BASE, **mudancas), Xt, yt, Xv, yv)


print("1. A mesma rede com seis taxas: 784 -> 128 -> 10, 10000 fotos, 3 épocas, lote 64")
print("  taxa   perda validação: época 1    época 2    época 3   passo/peso na época 0 (camada 1)")
for taxa in [0.001, 0.01, 0.05, 0.2, 0.5, 1.0]:
    _, _, curva = treino(taxa=taxa)
    pv = [R.numero(c["perda_validacao"]) for c in curva[1:]] + ["parou"] * (4 - len(curva))
    print(f"  {taxa:<5}  {pv[0]:>24} {pv[1]:>10} {pv[2]:>10}   {R.cientifico(curva[0]['passo_por_peso'][0]):>8}")

print()
print("2. A taxa 0.5 por dentro")
_, _, curva = treino(taxa=0.5)
R.mostrar(curva)
print("  passo/peso por época, camada 1 e camada 2: "
      + "; ".join(" ".join(R.cientifico(v) for v in c["passo_por_peso"]) for c in curva))

print()
print("3. Cinco camadas escondidas de 64, pesos começando com desvio 0.01, taxa 0.05")
r0 = R.nova_rede(FUNDA, 0.01)
_, guardado = R.frente(Xv[:500], r0)
print("  tamanho médio da soma |z| em cada camada escondida, antes do treino: "
      + " ".join(R.cientifico(float(np.abs(guardado[2 * i + 1]).mean())) for i in range(5)))
rede_pequena, _, curva = treino(camadas=FUNDA, inicio=0.01)
R.mostrar(curva)
b = rede_pequena[4]["b"]
print(f"  depois de 3 épocas, viés da camada 5: {int((b < 0).sum())} de {len(b)} abaixo de zero; "
      f"o maior em módulo é {R.cientifico(float(np.abs(b).max()))}")

print()
print("4. A mesma rede funda com o início de He, desvio raiz(2 / entradas)")
r0 = R.nova_rede(FUNDA, "he")
_, guardado = R.frente(Xv[:500], r0)
print("  tamanho médio da soma |z| em cada camada escondida, antes do treino: "
      + " ".join(f"{float(np.abs(guardado[2 * i + 1]).mean()):.2f}" for i in range(5)))
_, _, curva = treino(camadas=FUNDA, inicio="he")
R.mostrar(curva)

print()
print("5. Decorar 100 fotos: 100 épocas só com as 100 primeiras fotos de treino")
print("  rede                              perda treino  acerto treino")
for nome, camadas, inicio in [("784 -> 128 -> 10, He", [784, 128, 10], "he"),
                              ("funda, desvio 0.01", FUNDA, 0.01),
                              ("funda, He", FUNDA, "he")]:
    _, _, curva = treino(camadas=camadas, inicio=inicio, epocas=100, fotos_treino=100)
    print(f"  {nome:<32}  {R.numero(curva[-1]['perda_treino']):>12}  {curva[-1]['acerto_treino']:>13.4f}")

print()
print("6. Só 2000 fotos de treino, 60 épocas, 784 -> 128 -> 10, taxa 0.05")
_, escolhida, curva = treino(fotos_treino=2000, epocas=60, escolher_epoca=True)
R.mostrar(curva, epocas=[0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60], gradiente=False)
ce, cu = curva[escolhida], curva[-1]
print(f"  menor perda de validação: época {escolhida}, {ce['perda_validacao']:.4f}")
print(f"  distância entre validação e treino na perda: época {escolhida} "
      f"{ce['perda_validacao'] - ce['perda_treino']:.4f}; época 60 {cu['perda_validacao'] - cu['perda_treino']:.4f}")

print()
print("7. Três correções para as 2000 fotos, comparadas na mesma validação")
print("  correção                                época  perda treino  perda validação  acerto validação")
linhas = [("nenhuma (última época)", cu["epoca"], cu),
          ("escolher a época pela validação", escolhida, ce)]
_, e, c = treino(fotos_treino=2000, epocas=60, escolher_epoca=True, l2=0.01)
linhas.append(("escolher a época, com l2 0.01", e, c[e]))
_, e, c = treino(fotos_treino=50000, epocas=10, escolher_epoca=True)
linhas.append(("50000 fotos, 10 épocas, escolher", e, c[e]))
for nome, e, c in linhas:
    print(f"  {nome:<38}  {e:>5}  {c['perda_treino']:>12.4f}  {c['perda_validacao']:>15.4f}  "
          f"{c['acerto_validacao']:>16.4f}")
