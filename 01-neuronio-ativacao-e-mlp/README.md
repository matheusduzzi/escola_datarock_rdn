# 01. Neurônio, ativação e MLP

Os arquivos da aula «Neurônio, ativação e MLP» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `passagem.py` | As contas da aula: a passagem direta da rede pequena, um lote de duas fotos, as duas camadas sem ReLU que viram uma só, os parâmetros de cinco redes, a aproximação de x² com neurônios ReLU e um formato trocado que roda sem erro. |
| `minha_passagem.py` | Onde você escreve as funções `passagem` e `contar_parametros` do exercício. A rede de calçados e as duas propostas já estão nele. |
| `testar_minha_passagem.py` | Os testes das suas duas funções, com a rede da aula. |
| `minhas-respostas.md` | Onde você escreve as contas à mão e as respostas do exercício. |

## Como rodar

Você precisa do Python com o NumPy, como no README do kit do curso. Essa aula não usa o pandas, o PyTorch nem os arquivos de `dados/`.

```bash
python3 passagem.py
```

A saída é a mesma das seções 5 a 8 e 10 da aula. Ela sai igual toda vez: não tem sorteio.

```bash
python3 testar_minha_passagem.py
python3 minha_passagem.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError: escreva passagem(x, camadas)`. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 9 testes passaram`. Se um teste falhar, a mensagem diz qual número saiu e qual a aula tem.

Se aparecer `No module named 'numpy'`, o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
