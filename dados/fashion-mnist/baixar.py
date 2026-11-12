"""Baixa o Fashion MNIST do endereço oficial da Zalando Research e confere cada arquivo.

    python3 baixar.py                 # salva nesta pasta
    python3 baixar.py --destino PASTA # salva em outra pasta

Só usa a biblioteca padrão para baixar. O MD5 de cada arquivo é o publicado no README oficial
(github.com/zalandoresearch/fashion-mnist). Arquivo que já existe com o MD5 certo não é baixado de novo.

Para ler as fotos no NumPy, de dentro da pasta da aula:

    import sys; sys.path.insert(0, '../dados/fashion-mnist')
    from baixar import carregar
    x_treino, y_treino, x_teste, y_teste = carregar('../dados/fashion-mnist')
"""
import argparse
import gzip
import hashlib
import urllib.request
from pathlib import Path

ENDERECO = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
ARQUIVOS = {
    'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
    't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310',
}
CLASSES = ['camiseta/top', 'calça', 'pulôver', 'vestido', 'casaco', 'sandália', 'camisa', 'tênis', 'bolsa', 'bota']


def md5(caminho):
    return hashlib.md5(Path(caminho).read_bytes()).hexdigest()


def baixar(destino):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    for nome, esperado in ARQUIVOS.items():
        alvo = destino / nome
        if alvo.exists() and md5(alvo) == esperado:
            print(f'{nome}: já estava aqui, MD5 ok')
            continue
        urllib.request.urlretrieve(ENDERECO + nome, alvo)
        obtido = md5(alvo)
        if obtido != esperado:
            alvo.unlink()
            raise SystemExit(f'{nome}: MD5 {obtido} diferente do publicado ({esperado}); arquivo apagado, tente de novo')
        print(f'{nome}: baixado, {alvo.stat().st_size} bytes, MD5 ok')


def carregar(pasta):
    """Retorna x_treino (60000, 28, 28), y_treino (60000,), x_teste (10000, 28, 28), y_teste (10000,), em uint8."""
    import numpy as np
    pasta = Path(pasta)

    def ler(nome, desloca, formato):
        with gzip.open(pasta / nome, 'rb') as f:
            return np.frombuffer(f.read(), dtype=np.uint8, offset=desloca).reshape(formato)

    return (ler('train-images-idx3-ubyte.gz', 16, (-1, 28, 28)), ler('train-labels-idx1-ubyte.gz', 8, (-1,)),
            ler('t10k-images-idx3-ubyte.gz', 16, (-1, 28, 28)), ler('t10k-labels-idx1-ubyte.gz', 8, (-1,)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--destino', default=str(Path(__file__).resolve().parent))
    baixar(ap.parse_args().destino)
