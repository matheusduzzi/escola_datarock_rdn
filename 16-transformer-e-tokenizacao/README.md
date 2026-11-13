# 16. Transformer e tokenização

Os arquivos da aula «Transformer e tokenização» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `transformer.py` | Tudo o que a aula mostra, partes 1 a 8: a mensagem em palavras, caracteres e bytes, o vocabulário de palavras inteiras, o BPE treinado nas mensagens, as três montagens, o código de posição, a representação, o que cada modelo aprende e os parâmetros do BERT, do GPT-2 e do T5. |
| `minha_tokenizacao.py` | O exercício: o vocabulário WordPiece do Caio e a função `wordpiece` que você escreve. |
| `testar_minha_tokenizacao.py` | Os testes do exercício. |
| `minhas-respostas.md` | A folha das palavras quebradas à mão, dos pares, do mapa e da resposta para a coordenadora. |

## Como rodar

Você só precisa do Python com o NumPy, como no README do kit do curso. Essa aula não usa o PyTorch nem a biblioteca transformers: os modelos de verdade entram na aula 17. Os dois programas leem as mensagens em `../dados/mensagens/mensagens.csv`, a base do kit do curso.

```bash
python3 transformer.py
```

Leva uns 10 segundos, porque ele treina o BPE nas mensagens. A saída é a das seções 2 a 10 da aula e sai igual toda vez. Os vetores e os pesos da parte 6 são sorteados, sempre os mesmos, e não foram treinados. Os tamanhos da parte 8 vêm do `config.json` de cada modelo no Hugging Face.

```bash
python3 testar_minha_tokenizacao.py
python3 minha_tokenizacao.py
```

Enquanto a função `wordpiece` não estiver escrita, os dois param com `NotImplementedError`. É o esperado. Quando ela estiver certa, o teste mostra `ok: 11 testes passaram`. Se algum caso estiver errado, ele mostra linhas `FALHOU` com o que a sua função retornou e o esperado, e sai com código 1.

Os números da saída usam ponto como separador decimal, como o Python escreve: `11.3` é 11,3. Nos números grandes da parte 8, o ponto separa os milhares: `109.482.240`.

Se aparecer `No module named 'numpy'`, o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver. Se aparecer `No such file or directory` com `mensagens.csv`, a pasta da aula não está ao lado da pasta `dados/` do kit. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
