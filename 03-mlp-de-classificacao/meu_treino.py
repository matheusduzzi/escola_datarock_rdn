"""Tarefa 2 do exercício: o seu ciclo nas fotos, com a rede 784 -> 256 -> 10 do Caio.

Rode de dentro da pasta da aula, depois de passar nos testes:  python3 meu_treino.py
Não mexa nesse arquivo: ele usa a sua treinar_com_paciencia, do meu_ciclo.py.
Leva menos de um minuto. Escreve minha_curva.csv nessa pasta.
"""
import numpy as np

import mlp
from meu_ciclo import treinar_com_paciencia

MAX_EPOCAS, PACIENCIA, TAXA, LOTE = 40, 5, 0.05, 64

X, y, X_teste, y_teste = mlp.fotos()
i_treino, i_validacao = mlp.separar(len(X), 10000, sorteio=1)
media, desvio = mlp.medidas_do_treino(X, i_treino)
X = (X - media) / desvio
X_teste = (X_teste - media) / desvio
Xt, yt, Xv, yv = X[i_treino], y[i_treino], X[i_validacao], y[i_validacao]
print(f"treino {len(i_treino)} fotos, validação {len(i_validacao)}, teste {len(X_teste)}")

print("\n1. Baselines, na validação")
mais_comum = np.bincount(yt).argmax()
print(f"  tipo mais comum do treino ({mlp.CLASSES[mais_comum]}): acerto {float((yv == mais_comum).mean()):.4f}")
fm = mlp.foto_media(Xt, yt, Xv)
print(f"  foto média: acerto {float((fm == yv).mean()):.4f}; camisa {mlp.acerto_por_tipo(yv, fm)[6]:.4f}")
linear, e_lin, c_lin = treinar_com_paciencia(mlp.nova_rede([784, 10]), X, y, i_treino, i_validacao,
                                             MAX_EPOCAS, PACIENCIA, TAXA, LOTE)
_, _, p_lin = mlp.avaliar(Xv, yv, linear)
print(f"  784 -> 10: melhor época {e_lin}, parou na {len(c_lin) - 1}; perda {c_lin[e_lin]['perda_validacao']:.4f}, "
      f"acerto {c_lin[e_lin]['acerto_validacao']:.4f}; camisa {mlp.acerto_por_tipo(yv, p_lin)[6]:.4f}")

print(f"\n2. A rede 784 -> 256 -> 10, até {MAX_EPOCAS} épocas, paciência {PACIENCIA}")
rede = mlp.nova_rede([784, 256, 10])
print(f"  parâmetros: {mlp.parametros(rede)}")
melhor, e, curva = treinar_com_paciencia(rede, X, y, i_treino, i_validacao, MAX_EPOCAS, PACIENCIA, TAXA, LOTE)
mlp.mostrar_curva(curva)
mlp.gravar_curva(curva, "minha_curva.csv")
print(f"  melhor época {e}, parou na {len(curva) - 1}; curva gravada em minha_curva.csv")
_, _, p_v = mlp.avaliar(Xv, yv, melhor)
print(f"  camisa na validação: {mlp.acerto_por_tipo(yv, p_v)[6]:.4f}")

print("\n3. O teste, uma vez")
perda_t, acerto_t, p_t = mlp.avaliar(X_teste, y_teste, melhor)
fm_t = mlp.foto_media(Xt, yt, X_teste)
print(f"  rede da época {e}: perda {perda_t:.4f}, acerto {acerto_t:.4f}; camisa {mlp.acerto_por_tipo(y_teste, p_t)[6]:.4f}")
print(f"  foto média: acerto {float((fm_t == y_teste).mean()):.4f}; camisa {mlp.acerto_por_tipo(y_teste, fm_t)[6]:.4f}")
i, j, n = mlp.maior_confusao(y_teste, p_t)
print(f"  maior confusão: {n} fotos de {mlp.CLASSES[i]} viraram {mlp.CLASSES[j]}")
