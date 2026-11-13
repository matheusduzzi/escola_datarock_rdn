# 02. Backpropagation

Os arquivos da aula «Backpropagation» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `backprop.py` | As contas da aula: a perda de uma foto, a regra da cadeia em dois pesos, todos os gradientes do fim para o começo, a checagem por diferença finita, um passo de gradiente, um lote de duas fotos e a dobra da ReLU. |
| `meus_gradientes.py` | Onde você escreve as funções `retropropagar` e `gradiente_numerico` do exercício. A rede de camisa e casaco e a rede da aula já estão nele. |
| `gradientes_do_caio.py` | A função do Caio, que você examina na tarefa 3. Não mexa nela. |
| `testar_meus_gradientes.py` | Os testes das suas duas funções, com a rede da aula. |
| `minhas-respostas.md` | Onde você escreve as contas à mão e as respostas do exercício. |

## Como rodar

Você precisa do Python com o NumPy, como no README do kit do curso. Essa aula não usa o pandas, o PyTorch nem os arquivos de `dados/`.

```bash
python3 backprop.py
```

A saída é a mesma das seções 3 a 8 da aula. Ela sai igual toda vez: não tem sorteio. Só as diferenças relativas da parte 4 podem mudar nas últimas casas de um computador para outro, sempre muito abaixo de 0,0000001.

```bash
python3 testar_meus_gradientes.py
python3 meus_gradientes.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError: escreva retropropagar(x, classe, rede)`. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 10 testes passaram`. Se um teste falhar, a mensagem diz o que saiu e o que a aula tem.

Se aparecer `No module named 'numpy'`, o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
