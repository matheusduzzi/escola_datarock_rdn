# Fashion MNIST

70.000 fotos de roupas e calçados, em tons de cinza, com 28 por 28 pixels. São 60.000 para treino e 10.000 para teste, com 10 tipos e o mesmo número de fotos por tipo. As aulas 03, 08, 10 e 14 usam essas fotos.

As fotos não são da Loja Exemplo. Elas vêm de produtos reais do catálogo da Zalando, uma loja on-line europeia, e foram publicadas pela Zalando Research para quem estuda e compara modelos. O kit não traz as fotos: o `baixar.py` busca os quatro arquivos no endereço oficial e confere o MD5 de cada um.

## Baixar

```bash
python3 baixar.py
```

São quatro arquivos `.gz`, uns 30 MB no total. Cada linha da saída diz o nome, o tamanho e `MD5 ok`. Se o MD5 não bater, o arquivo é apagado e o programa para; rode de novo. Precisa de internet só nessa hora.

Para ler as fotos no NumPy, de dentro da pasta de uma aula:

```python
import sys
sys.path.insert(0, '../dados/fashion-mnist')
from baixar import CLASSES, carregar

x_treino, y_treino, x_teste, y_teste = carregar('../dados/fashion-mnist')
print(x_treino.shape, CLASSES[y_treino[0]])
```

## Os 10 tipos

| Número | Tipo (nome original) |
|---|---|
| 0 | camiseta/top (T-shirt/top) |
| 1 | calça (Trouser) |
| 2 | pulôver (Pullover) |
| 3 | vestido (Dress) |
| 4 | casaco (Coat) |
| 5 | sandália (Sandal) |
| 6 | camisa (Shirt) |
| 7 | tênis (Sneaker) |
| 8 | bolsa (Bag) |
| 9 | bota (Ankle boot) |

## De onde vem e licença

- Repositório oficial: <https://github.com/zalandoresearch/fashion-mnist>, com os endereços dos arquivos e o MD5 de cada um.
- Licença MIT, Copyright © 2017 Zalando SE. Ela permite usar, copiar e redistribuir, desde que o aviso de copyright e o da licença venham junto.
- Para citar: Han Xiao, Kashif Rasul e Roland Vollgraf, "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms", arXiv:1708.07747, 2017.
