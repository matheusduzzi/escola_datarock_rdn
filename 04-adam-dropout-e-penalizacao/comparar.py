"""A comparação da aula «Adam, dropout e penalização»: uma mudança por vez, contra a mesma referência.

Rode de dentro da pasta da aula, depois de baixar as fotos:  python3 comparar.py
Leva uns 2 a 3 minutos num notebook recente. Escreve curvas.csv nessa pasta.

Todas as rodadas usam a mesma rede 784 -> 256 -> 10, as mesmas 10.000 fotos de treino e as mesmas
10.000 de validação, 30 épocas, lote 64 e a época escolhida pela menor perda de validação.
O sorteio é fixo: a saída sai igual a cada rodada no mesmo computador.
"""
import csv

import numpy as np

import rodada
from rodada import mlp

TAMANHOS, EPOCAS, LOTE, N_TREINO = [784, 256, 10], 30, 64, 10000

REFERENCIA = {"nome": "referência: SGD 0,05", "otimizador": "sgd", "taxa": 0.05}
PEDIDO_DO_CAIO = {"nome": "Caio: Adam + dropout + L2", "otimizador": "adam", "taxa": 0.001,
                  "dropout": 0.5, "l2": 1e-3}
OTIMIZACAO = [
    {"nome": "Adam 0,0003", "otimizador": "adam", "taxa": 0.0003},
    {"nome": "Adam 0,001", "otimizador": "adam", "taxa": 0.001},
    {"nome": "Adam 0,003", "otimizador": "adam", "taxa": 0.003},
    {"nome": "Adam 0,05 (taxa do SGD)", "otimizador": "adam", "taxa": 0.05},
]
REGULARIZACAO = [
    {"nome": "L2 0,001", "otimizador": "sgd", "taxa": 0.05, "l2": 1e-3},
    {"nome": "L2 0,01", "otimizador": "sgd", "taxa": 0.05, "l2": 1e-2},
    {"nome": "L1 0,0001", "otimizador": "sgd", "taxa": 0.05, "l1": 1e-4},
    {"nome": "L1 0,001", "otimizador": "sgd", "taxa": 0.05, "l1": 1e-3},
    {"nome": "dropout 0,2", "otimizador": "sgd", "taxa": 0.05, "dropout": 0.2},
    {"nome": "dropout 0,5", "otimizador": "sgd", "taxa": 0.05, "dropout": 0.5},
]


def num(x, casas=4):
    return "diverge" if not np.isfinite(x) else f"{x:.{casas}f}"


def tabela_curta(resumos, rotulo="rodada"):
    print(f"  {rotulo:<26}  melhor época  perda validação  acerto validação")
    for r in resumos:
        print(f"  {r['nome']:<26}  {r['melhor_epoca']:>12}  {num(r['perda_validacao']):>15}  "
              f"{num(r['acerto_validacao']):>16}")


X, y, X_teste, y_teste = mlp.fotos()
i_treino, i_validacao = mlp.separar(len(X), 10000)
i_treino = i_treino[:N_TREINO]
media, desvio = mlp.medidas_do_treino(X, i_treino)
X = (X - media) / desvio
X_teste = (X_teste - media) / desvio


def rodar(config, sorteio=0):
    return rodada.rodar(config, X, y, i_treino, i_validacao, TAMANHOS, EPOCAS, LOTE, sorteio)


print("1. O pedido do Caio: três mudanças de uma vez")
print(f"  rede {' -> '.join(map(str, TAMANHOS))}, {len(i_treino)} fotos de treino, {len(i_validacao)} de validação, "
      f"{EPOCAS} épocas, lote {LOTE}")
ref = rodar(REFERENCIA)
caio = rodar(PEDIDO_DO_CAIO)
tabela_curta([ref, caio])

print("\n2. Quanto muda só pelo sorteio: a referência três vezes")
refs = [ref] + [rodar(dict(REFERENCIA, nome=f"referência, sorteio {s}"), s) for s in (1, 2)]
refs[0] = dict(ref, nome="referência, sorteio 0")
tabela_curta(refs, "sorteio")
pv = [r["perda_validacao"] for r in refs]
av = [r["acerto_validacao"] for r in refs]
print(f"  faixa da perda de validação: {min(pv):.4f} a {max(pv):.4f} (largura {max(pv) - min(pv):.4f})")
print(f"  faixa do acerto de validação: {min(av):.4f} a {max(av):.4f} (largura {max(av) - min(av):.4f})")

print("\n3. Otimização: só o passo muda")
print("  rodada                      perda de treino depois da época    melhor  perda      acerto")
print("                                   1        3        5            época   validação  validação")
otim = [ref] + [rodar(c) for c in OTIMIZACAO]
for r in otim:
    c = r["curva"]
    t = [num(c[e]["perda_treino"]) if e < len(c) else "-" for e in (1, 3, 5)]
    print(f"  {r['nome']:<26}  {t[0]:>7}  {t[1]:>7}  {t[2]:>7}  {r['melhor_epoca']:>11}  "
          f"{num(r['perda_validacao']):>9}  {num(r['acerto_validacao']):>9}")

print("\n4. Regularização: o passo fica, a rede ganha um freio")
print("  rodada                      melhor  perda      acerto     na época 30:             tamanho  pesos")
print("                              época   validação  validação  treino  validação  dist.  dos W    perto de 0")
regs = [ref] + [rodar(c) for c in REGULARIZACAO]
for r in regs:
    f = r["curva"][-1]
    print(f"  {r['nome']:<26}  {r['melhor_epoca']:>6}  {num(r['perda_validacao']):>9}  {num(r['acerto_validacao']):>9}"
          f"  {f['perda_treino']:>6.4f}  {f['perda_validacao']:>9.4f}  {f['perda_validacao'] - f['perda_treino']:>5.2f}"
          f"  {r['norma']:>7.1f}  {r['zeros']:>9.1%}")

melhor = min(regs[1:], key=lambda r: r["perda_validacao"])
config_melhor = next(c for c in REGULARIZACAO if c["nome"] == melhor["nome"])
print(f"\n5. A menor perda de validação da parte 4 ({melhor['nome']}), com os mesmos três sorteios")
outras = [dict(melhor, nome=f"{melhor['nome']}, sorteio 0")] + [
    rodar(dict(config_melhor, nome=f"{melhor['nome']}, sorteio {s}"), s) for s in (1, 2)]
tabela_curta(outras, "sorteio")
pv2 = [r["perda_validacao"] for r in outras]
av2 = [r["acerto_validacao"] for r in outras]
print(f"  faixa da perda de validação: {min(pv2):.4f} a {max(pv2):.4f}; referência: {min(pv):.4f} a {max(pv):.4f}")
print(f"  faixa do acerto de validação: {min(av2):.4f} a {max(av2):.4f}; referência: {min(av):.4f} a {max(av):.4f}")
print(f"  as faixas da perda se cruzam: {'sim' if min(pv) <= max(pv2) else 'não'}")

print("\n6. O teste, uma vez, para as duas redes decididas antes (sorteio 0)")
for r in (ref, melhor):
    perda, acerto, prev = mlp.avaliar(X_teste, y_teste, r["rede"])
    print(f"  {r['nome']:<26}  perda {perda:.4f}  acerto {acerto:.4f}  camisa {mlp.acerto_por_tipo(y_teste, prev)[6]:.4f}")

with open("curvas.csv", "w", encoding="utf-8", newline="") as arquivo:
    gravar = csv.writer(arquivo, lineterminator="\n")
    gravar.writerow(["rodada", "epoca", "perda_treino", "perda_validacao"])
    for r in (ref, otim[2], next(x for x in regs if x["nome"] == "L2 0,01"),
              next(x for x in regs if x["nome"] == "dropout 0,5")):
        for c in r["curva"]:
            gravar.writerow([r["nome"], c["epoca"], f"{c['perda_treino']:.4f}", f"{c['perda_validacao']:.4f}"])
print("\n  curvas da referência, do Adam 0,001, do L2 0,01 e do dropout 0,5 gravadas em curvas.csv")
