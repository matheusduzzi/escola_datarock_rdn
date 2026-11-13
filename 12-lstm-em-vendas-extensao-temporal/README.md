# 12. LSTM em vendas: extensão temporal

Os arquivos da aula «LSTM em vendas: extensão temporal» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `comparar_lstm.py` | Tudo o que a aula mostra, partes 1 a 8, e as três rodadas da prática guiada (`--janela`, `--retreino`, `--estado`). |
| `serie.py` | A venda, a janela que a LSTM lê, as previsões simples e o calendário + AR(1) do curso Séries temporais, e as contas do erro. |
| `lstm.py` | A LSTM no NumPy: a ida, a volta pelo tempo, a checagem do gradiente e o treino com o Adam. |
| `experimento.py` | Monta os exemplos de cada corte, treina os sorteios, prevê as noites do teste e faz o teste do futuro. |
| `minha_comparacao.py` | O exercício: a configuração do Caio para a previsão de amanhã da tarefa 5. |
| `testar_minha_comparacao.py` | Os testes do exercício. |
| `minhas-respostas.md` | A folha da tabela e da resposta para a Bia. |

## Como rodar

Você só precisa do Python com o pandas e o NumPy, como no README do kit do curso. Essa aula lê a venda diária e o calendário em `../dados/vendas/`, que já vêm no kit. Ela não usa o PyTorch nem as fotos do Fashion MNIST.

```bash
python3 comparar_lstm.py
```

Leva uns 2 a 5 minutos, conforme a máquina: são 25 treinos de LSTM e 116 ajustes da regressão do calendário. Eu rodei com Python 3.9.6, pandas 2.3.3 e NumPy 2.0.2, num Mac com chip M4. A saída sai igual toda vez. Em outro computador, a última casa de um erro pode mudar.

As rodadas da prática guiada treinam só a LSTM com calendário:

```bash
python3 comparar_lstm.py --janela 56
python3 comparar_lstm.py --retreino mensal
python3 comparar_lstm.py --estado 4
```

O exercício:

```bash
python3 testar_minha_comparacao.py
python3 minha_comparacao.py
```

Com a configuração do Caio, o teste mostra linhas `FALHOU` e sai com código 1. É o esperado. Quando a configuração respeitar a tarefa 5 e o corte, aparece `ok: 8 testes passaram`. O `minha_comparacao.py` leva uns 15 a 40 segundos, conforme a máquina.

Os números das tabelas usam vírgula, como a aula. Os da linha `configuração` e do teste do futuro saem como o Python escreve, com ponto: `4.4717`.

Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. Se aparecer `No such file or directory` com `venda_diaria.csv`, a pasta da aula saiu de dentro do kit: os comandos precisam da pasta `dados/` ao lado. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
