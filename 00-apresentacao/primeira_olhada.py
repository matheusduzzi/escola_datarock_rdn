"""Primeira olhada nas fotos do catálogo: o número que uma rede precisa bater.

    python3 primeira_olhada.py

Antes, baixe as fotos: na pasta ../dados/fashion-mnist, rode python3 baixar.py.
Para ler as fotos de outra pasta: python3 primeira_olhada.py --fotos PASTA

Calcula a foto média de cada tipo de roupa nas 60.000 fotos de treino e manda cada foto de
teste para o tipo da foto média mais perto, pela distância entre os 784 pixels. Não treina nada.
Só usa o NumPy.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

FOTOS = Path(__file__).resolve().parent.parent / 'dados' / 'fashion-mnist'
sys.path.insert(0, str(FOTOS))
from baixar import CLASSES, carregar  # noqa: E402


def quantos(y):
    n = np.bincount(y)
    return f'{n.min()}' if n.min() == n.max() else f'{n.min()} a {n.max()}'


def pct(x):
    return f'{100 * x:.1f}'.replace('.', ',') + '%'


ap = argparse.ArgumentParser()
ap.add_argument('--fotos')
pasta = ap.parse_args().fotos or FOTOS
try:
    x_treino, y_treino, x_teste, y_teste = carregar(pasta)
except FileNotFoundError as erro:
    onde = '../dados/fashion-mnist' if pasta == FOTOS else pasta
    raise SystemExit(f'faltam as fotos em {onde}: {Path(erro.filename).name}\n'
                     'rode python3 baixar.py na pasta dados/fashion-mnist do kit e tente de novo')

treino = x_treino.reshape(len(x_treino), -1) / 255
teste = x_teste.reshape(len(x_teste), -1) / 255
n_tipos = len(CLASSES)

print('1. As fotos')
print(f'   treino: {len(treino)} fotos de {x_treino.shape[1]} x {x_treino.shape[2]} pixels, '
      f'{quantos(y_treino)} de cada tipo')
print(f'   teste: {len(teste)} fotos, {quantos(y_teste)} de cada tipo')
print(f'   chutar sempre o mesmo tipo acerta {pct(np.bincount(y_teste).max() / len(y_teste))} no teste')

medias = np.stack([treino[y_treino == k].mean(axis=0) for k in range(n_tipos)])
distancia = np.stack([((teste - media) ** 2).sum(axis=1) for media in medias], axis=1)
previsto = distancia.argmin(axis=1)

print('2. A foto média de cada tipo')
print(f'   {n_tipos} fotos médias de {treino.shape[1]} pixels, tiradas do treino: {medias.size} números, sem treino')
print(f'   acerto no teste: {100 * (previsto == y_teste).mean():.2f}%'.replace('.', ','))
print('   por tipo, do pior para o melhor:')
acerto = [(previsto[y_teste == k] == k).mean() for k in range(n_tipos)]
for k in np.argsort(acerto, kind='stable'):
    print(f'      {CLASSES[k]:<14}{pct(acerto[k]):>7}')

trocas = np.zeros((n_tipos, n_tipos), dtype=int)
np.add.at(trocas, (y_teste, previsto), 1)
np.fill_diagonal(trocas, 0)
print('3. As cinco trocas mais comuns no teste')
for i in np.argsort(-trocas, axis=None, kind='stable')[:5]:
    verdade, foi = divmod(int(i), n_tipos)
    print(f'   {CLASSES[verdade]} foi para {CLASSES[foi]}: {trocas[verdade, foi]} de {(y_teste == verdade).sum()}')
