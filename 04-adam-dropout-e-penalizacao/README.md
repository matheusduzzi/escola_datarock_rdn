# 04. Adam, dropout e penalização

Os arquivos da aula «Adam, dropout e penalização» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `um_passo.py` | Um peso só, sem rede e sem fotos: o primeiro passo do SGD e do Adam, o Adam com a taxa do SGD e a L2 ao lado da L1. Roda na hora. |
| `rodada.py` | Uma rodada de treino com o passo e o freio escolhidos: o Adam, o dropout e a penalização L2 e L1. Ele não roda sozinho; os outros dois arquivos usam ele. |
| `comparar.py` | A comparação da aula: a referência com três sorteios, as linhas de otimização, as de regularização, a vencedora com três sorteios e o teste no fim. Grava as curvas em `curvas.csv`. |
| `meu-plano.md` | Onde você faz o exercício: a tabela das mudanças do Caio e o plano nas oito partes. |

## Como rodar

Você precisa do Python com o NumPy, como no README do kit do curso, e das fotos do Fashion MNIST. Se ainda não baixou, baixe uma vez:

```bash
python3 ../dados/fashion-mnist/baixar.py
```

O `rodada.py` usa o `mlp.py` da aula «MLP de classificação», na pasta `03-mlp-de-classificacao`, ao lado dessa. As duas pastas vêm assim no kit. Não mude o nome delas.

Primeiro o peso sozinho:

```bash
python3 um_passo.py
```

Depois a comparação:

```bash
python3 comparar.py
```

Ela treina a rede 784 → 256 → 10 dezesseis vezes, com 10.000 fotos de treino cada vez. Leva uns 2 a 3 minutos num notebook recente. A saída tem seis partes, as mesmas das seções 6 a 9 da aula. O sorteio é fixo, então a saída sai igual a cada rodada no mesmo computador. Em outro computador, ou com outra versão do NumPy, os números do treino podem mudar um pouco nas últimas casas. A aula rodou com Python 3.9.6 e NumPy 2.0.2.

O exercício não pede para rodar nada. Ele é o plano que você escreve no `meu-plano.md`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
