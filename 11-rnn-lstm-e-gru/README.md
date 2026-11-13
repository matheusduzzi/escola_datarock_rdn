# 11. RNN, LSTM e GRU

Os arquivos da aula «RNN, LSTM e GRU» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `recorrente.py` | Tudo o que a aula mostra, partes 1 a 7: a semana de venda, a RNN dia a dia, a contagem de passos, o que sobra do dia 1, um passo da LSTM, um passo da GRU e os parâmetros de cada uma. |
| `minha_gru.py` | O exercício: a GRU do Caio e a função `passo_gru` que você escreve. |
| `testar_minha_gru.py` | Os testes do exercício. |
| `minhas-respostas.md` | A folha do desenho, da contagem e da resposta para a gerente. |

## Como rodar

Você só precisa do Python com o NumPy, como no README do kit do curso. Essa aula lê a venda diária em `../dados/vendas/`, que já vem no kit. Ela não usa o PyTorch nem as fotos do Fashion MNIST.

```bash
python3 recorrente.py
```

Leva menos de um segundo. A saída é a das seções 2 a 7 da aula e sai igual toda vez. As células têm um número só de estado e pesos escolhidos à mão. Elas mostram o mecanismo e não foram treinadas.

```bash
python3 testar_minha_gru.py
python3 minha_gru.py
```

Enquanto a função `passo_gru` não estiver escrita, os dois param com `NotImplementedError`. É o esperado. Quando ela estiver certa, o teste mostra `ok: 9 testes passaram`. Se alguma conta estiver errada, ele mostra linhas `FALHOU` com o que a sua função retornou e o esperado, e sai com código 1.

Os números da saída usam ponto como separador decimal, como o Python escreve: `0.957`. Os pequenos aparecem em notação científica: `5.4e-02` é 0,054.

Se aparecer `No module named 'numpy'`, o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver. Se aparecer `No such file or directory` com `venda_diaria.csv`, a pasta da aula saiu de dentro do kit: os comandos precisam da pasta `dados/` ao lado. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
