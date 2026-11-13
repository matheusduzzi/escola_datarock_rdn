# 06. Convolução e dimensões

Os arquivos da aula «Convolução e dimensões» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `convolucao.py` | Tudo o que a aula mostra, partes 1 a 6: o pedaço da foto com o kernel, stride e padding, o pooling, os canais e a rede do Caio camada por camada. |
| `minhas_contas.py` | O exercício: a rede B e as três funções que você escreve. |
| `testar_minhas_contas.py` | Os testes do exercício. |
| `minhas-respostas.md` | A folha da sua tabela à mão e da resposta para a Bia. |

## Como rodar

Você só precisa do Python com o NumPy, como no README do kit do curso. Essa aula não usa as fotos do Fashion MNIST, o pandas nem o PyTorch.

```bash
python3 convolucao.py
```

Leva menos de um segundo. A saída é a das seções 2 a 7 da aula e sai igual toda vez. A parte 5 passa pela rede do Caio 64 fotos com números sorteados e pesos sorteados: ela mede o formato de cada camada, não treina nada.

```bash
python3 testar_minhas_contas.py
python3 minhas_contas.py
```

Enquanto as funções do `minhas_contas.py` não estiverem escritas, os dois param com `NotImplementedError`. É o esperado. Quando as suas funções estiverem certas, o teste mostra `ok: 10 testes passaram`. Se alguma conta estiver errada, ele mostra linhas `FALHOU` com o número que a sua função retornou e o da aula, e sai com código 1.

Os números grandes aparecem com ponto separando os milhares, como na aula: `18.496`. Os decimais, no código, saem com ponto: `5.5`.

Se aparecer `No module named 'numpy'`, o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
