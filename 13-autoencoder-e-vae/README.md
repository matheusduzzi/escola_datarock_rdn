# 13. Autoencoder e VAE

Os arquivos da aula «Autoencoder e VAE» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `autoencoder.py` | Tudo o que a aula mostra, partes 1 a 7: as duas perdas num pedaço de foto, o pixel cinza, o autoencoder de fotos 3 × 3, o sorteio de códigos, a KL por integral, a perda do VAE para vários σ e o erro de reconstrução como alarme. |
| `minhas_perdas.py` | O exercício: os dois casos do Caio e as três funções que você escreve. |
| `testar_minhas_perdas.py` | Os testes do exercício. |
| `minhas-respostas.md` | A folha das suas contas e da resposta para a Bia. |

## Como rodar

Você só precisa do Python com o NumPy, como no README do kit do curso. Essa aula não usa as fotos do Fashion MNIST, o pandas nem o PyTorch.

```bash
python3 autoencoder.py
```

Leva menos de um segundo. A saída é a das seções 2 a 8 da aula e sai igual toda vez. As 120 fotos de 3 × 3 pixels são geradas pelo próprio programa, com sorteios fixos.

```bash
python3 testar_minhas_perdas.py
python3 minhas_perdas.py
```

Enquanto as funções do `minhas_perdas.py` não estiverem escritas, os dois param com `NotImplementedError`. É o esperado. Quando as suas funções estiverem certas, o teste mostra `ok: 10 testes passaram`. Se alguma conta estiver errada, ele mostra linhas `FALHOU` com o número que a sua função retornou e o esperado, e sai com código 1.

Os números aparecem com vírgula, como na aula: `0,8181`. No código, os decimais são escritos com ponto: `0.8181`.

Se aparecer `No module named 'numpy'`, o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
