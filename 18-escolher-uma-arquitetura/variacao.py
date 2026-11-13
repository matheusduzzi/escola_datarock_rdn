"""Quanto o acerto muda só por trocar o sorteio: a mesma rede, o mesmo orçamento, outros começos.

    python3 variacao.py mlp                    # a MLP da aula 08, taxa 0.05, sorteios 0 a 4
    python3 variacao.py caio --sorteios 3      # a CNN do Caio, taxa 0.02, sorteios 0 a 2

Usa o cnn.py da aula «CNN no Fashion MNIST», na pasta ../08-cnn-no-fashion-mnist/, e o orçamento comum
de lá: as mesmas 50.000 fotos de treino e 10.000 de validação, lote 64, 15 épocas, a taxa que a aula 08
escolheu para cada rede. Só muda o sorteio dos pesos iniciais e da ordem dos lotes. A divisão das fotos
fica a mesma, então a variação que aparece é a do treino, não a das fotos.
O sorteio 0 é o da aula 08: ele tem que repetir o número de lá. O teste não é aberto.
Rode de dentro da pasta da aula, depois de baixar as fotos.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

AULA_08 = Path(__file__).resolve().parent.parent / "08-cnn-no-fashion-mnist"
if not (AULA_08 / "cnn.py").exists():
    raise SystemExit("Não achei ../08-cnn-no-fashion-mnist/cnn.py. Esse programa usa a pasta da aula 08 do kit.")
sys.path.insert(0, str(AULA_08))
import cnn  # noqa: E402

TAXA_DA_AULA_08 = {"mlp": 0.05, "caio": 0.02}
CAMISETA, CAMISA = 0, 6

ap = argparse.ArgumentParser(description="Treina a mesma rede com sorteios diferentes e mostra a faixa.")
ap.add_argument("rede", choices=sorted(TAXA_DA_AULA_08))
ap.add_argument("--sorteios", type=int, default=5, help="quantos sorteios, a partir do 0 (de 2 a 10)")
args = ap.parse_args()
if not 2 <= args.sorteios <= 10:
    raise SystemExit("use de 2 a 10 sorteios: com 1 não há faixa para medir")

orc = dict(cnn.ORCAMENTO, taxa=TAXA_DA_AULA_08[args.rede])
Xt, yt, Xv, yv, _, _, _, _ = cnn.preparar(orc)
print(f"{cnn.NOMES[args.rede]}, taxa {orc['taxa']}: {len(Xt)} fotos de treino, {len(Xv)} de validação, "
      f"lote {orc['lote']}, {orc['epocas']} épocas; muda só o sorteio dos pesos e da ordem dos lotes")
print("  sorteio  época  perda validação  acerto validação  erros  camiseta  camisa  segundos", flush=True)
linhas = []
for s in range(args.sorteios):
    rede = cnn.nova_rede(cnn.PLANTAS[args.rede], sorteio=s)
    t0 = time.perf_counter()
    melhor, ep, curva = cnn.ciclo(rede, Xt, yt, Xv, yv, orc["epocas"], orc["taxa"], orc["lote"], sorteio=s)
    seg = time.perf_counter() - t0
    perda, acerto, prev, _ = cnn.avaliar(Xv, yv, melhor)
    por_tipo = cnn.acerto_por_tipo(yv, prev)
    erros = int((prev != yv).sum())
    linhas.append((acerto, erros, por_tipo[CAMISETA], por_tipo[CAMISA]))
    print(f"  {s:7d}  {ep:5d}  {perda:15.4f}  {acerto:16.4f}  {erros:5d}  {por_tipo[CAMISETA]:8.4f}  "
          f"{por_tipo[CAMISA]:6.4f}  {seg:8.0f}", flush=True)

a = np.array(linhas)
print(f"  acerto de validação: menor {a[:, 0].min():.4f}, maior {a[:, 0].max():.4f}, "
      f"distância {100 * (a[:, 0].max() - a[:, 0].min()):.2f} pontos ({int(a[:, 1].max() - a[:, 1].min())} fotos)")
print(f"  camiseta: de {a[:, 2].min():.4f} a {a[:, 2].max():.4f}; camisa: de {a[:, 3].min():.4f} a {a[:, 3].max():.4f}")
print("  (o sorteio 0 é o da aula 08; o teste não foi aberto)")
