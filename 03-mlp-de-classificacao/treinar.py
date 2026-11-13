"""O ciclo da aula «MLP de classificação»: baselines, treino com validação, curvas e teste no fim.

Rode de dentro da pasta da aula, depois de baixar as fotos:  python3 treinar.py
Leva menos de um minuto num notebook recente. Escreve curva.csv nessa pasta.
O sorteio é fixo: a saída sai igual a cada rodada no mesmo computador.
"""
import numpy as np

import mlp

EPOCAS, TAXA, LOTE = 30, 0.05, 64

X, y, X_teste, y_teste = mlp.fotos()
i_treino, i_validacao = mlp.separar(len(X), 10000)
media, desvio = mlp.medidas_do_treino(X, i_treino)
X = (X - media) / desvio
X_teste = (X_teste - media) / desvio
Xt, yt, Xv, yv = X[i_treino], y[i_treino], X[i_validacao], y[i_validacao]

print("1. As três partes")
print(f"  treino {len(i_treino)} fotos, validação {len(i_validacao)}, teste {len(X_teste)}")
print(f"  fotos repetidas entre treino e validação: {len(np.intersect1d(i_treino, i_validacao))}")
print(f"  fotos de cada tipo na validação: menor {np.bincount(yv).min()}, maior {np.bincount(yv).max()}")
print(f"  passos por época com lote {LOTE}: {int(np.ceil(len(i_treino) / LOTE))}; "
      f"o último lote tem {len(i_treino) % LOTE or LOTE} fotos")

print("\n2. Baselines, na validação")
mais_comum = np.bincount(yt).argmax()
chute = float((yv == mais_comum).mean())
fm = mlp.foto_media(Xt, yt, Xv)
acerto_fm = float((fm == yv).mean())
print(f"  chutar sempre o tipo mais comum do treino ({mlp.CLASSES[mais_comum]}): acerto {chute:.4f}")
print(f"  foto média mais perto: acerto {acerto_fm:.4f}; camisa {mlp.acerto_por_tipo(yv, fm)[6]:.4f}")
linear = mlp.nova_rede([784, 10])
linear, ep_linear, curva_linear = mlp.ciclo(linear, X, y, i_treino, i_validacao, EPOCAS, TAXA, LOTE)
_, acerto_lin, prev_lin = mlp.avaliar(Xv, yv, linear)
print(f"  rede sem camada escondida, 784 -> 10 ({mlp.parametros(linear)} parâmetros), melhor época {ep_linear}: "
      f"perda {curva_linear[ep_linear]['perda_validacao']:.4f}, acerto {acerto_lin:.4f}; "
      f"camisa {mlp.acerto_por_tipo(yv, prev_lin)[6]:.4f}")

print(f"\n3. A MLP 784 -> 128 -> 10, {EPOCAS} épocas, taxa {TAXA}, lote {LOTE}")
rede = mlp.nova_rede([784, 128, 10])
print(f"  parâmetros: {mlp.parametros(rede)}")
melhor, ep, curva = mlp.ciclo(rede, X, y, i_treino, i_validacao, EPOCAS, TAXA, LOTE)
mlp.mostrar_curva(curva)
mlp.gravar_curva(curva, "curva.csv")
print("  curva gravada em curva.csv")

print("\n4. A época escolhida pela validação")
c = curva[ep]
print(f"  menor perda de validação: época {ep}, perda {c['perda_validacao']:.4f}, acerto {c['acerto_validacao']:.4f}")
print(f"  na época {EPOCAS}: perda de validação {curva[-1]['perda_validacao']:.4f}, "
      f"acerto {curva[-1]['acerto_validacao']:.4f}")
print(f"  distância entre treino e validação: época {ep} {c['perda_validacao'] - c['perda_treino']:.4f}; "
      f"época {EPOCAS} {curva[-1]['perda_validacao'] - curva[-1]['perda_treino']:.4f}")
ep_acerto = max(range(len(curva)), key=lambda e: (curva[e]["acerto_validacao"], -e))
print(f"  maior acerto de validação: época {ep_acerto}, {curva[ep_acerto]['acerto_validacao']:.4f}")
_, _, prev_v = mlp.avaliar(Xv, yv, melhor)
print(f"  camisa na validação: {mlp.acerto_por_tipo(yv, prev_v)[6]:.4f}")

print("\n5. Mais fotos de treino ajudam? Mesma rede, 20 épocas, melhor época pela validação")
print("  fotos de treino  melhor época  perda treino  perda validação  acerto validação")
for n in (2500, 5000, 10000, 25000, 50000):
    r = mlp.nova_rede([784, 128, 10])
    _, e, cv = mlp.ciclo(r, X, y, i_treino[:n], i_validacao, 20, TAXA, LOTE)
    print(f"  {n:>15}  {e:>12}  {cv[e]['perda_treino']:>12.4f}  {cv[e]['perda_validacao']:>15.4f}  "
          f"{cv[e]['acerto_validacao']:>16.4f}")

print("\n6. O teste, uma vez, com a rede da época escolhida")
perda_t, acerto_t, prev_t = mlp.avaliar(X_teste, y_teste, melhor)
fm_t = mlp.foto_media(Xt, yt, X_teste)
print(f"  MLP: perda {perda_t:.4f}, acerto {acerto_t:.4f}")
print(f"  foto média no mesmo teste: acerto {float((fm_t == y_teste).mean()):.4f}")
print("  acerto por tipo (MLP | foto média):")
for k, nome in enumerate(mlp.CLASSES):
    print(f"    {nome:<13} {mlp.acerto_por_tipo(y_teste, prev_t)[k]:.4f} | {mlp.acerto_por_tipo(y_teste, fm_t)[k]:.4f}")
i, j, n = mlp.maior_confusao(y_teste, prev_t)
print(f"  maior confusão da MLP: {n} fotos de {mlp.CLASSES[i]} viraram {mlp.CLASSES[j]}")
