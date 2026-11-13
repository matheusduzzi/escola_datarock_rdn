# 15. Atenção Q/K/V

Os arquivos da aula «Atenção Q/K/V» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `atencao.py` | Tudo o que a aula mostra, partes 1 a 9: as três palavras, as projeções, as notas, os pesos, a saída, a divisão pela raiz de d_k, a máscara causal, a ordem das palavras e as dimensões do modelo base do artigo de 2017. |
| `minha_atencao.py` | O exercício: a camada de atenção do Caio, com duas cabeças, e a função `atencao` que você escreve. |
| `testar_minha_atencao.py` | Os testes do exercício. |
| `minhas-respostas.md` | A folha da tabela de dimensões, da conta à mão e da resposta para a coordenadora. |

## Como rodar

Você só precisa do Python com o NumPy, como no README do kit do curso. Essa aula não lê nenhuma base da pasta `dados/` e não usa o PyTorch.

```bash
python3 atencao.py
```

Leva menos de um segundo. A saída é a das seções 2 a 10 da aula e sai igual toda vez. Os números das palavras e os pesos das matrizes foram escolhidos à mão, para a conta caber no papel.

```bash
python3 testar_minha_atencao.py
python3 minha_atencao.py
```

Enquanto a função `atencao` não estiver escrita, os dois param com `NotImplementedError`. É o esperado. Quando ela estiver certa, o teste mostra `ok: 9 testes passaram`. Se alguma conta estiver errada, ele mostra linhas `FALHOU` com o que a sua função retornou e o esperado, e sai com código 1.

Os números da saída usam ponto como separador decimal, como o Python escreve: `0.673` é 0,673. A máscara aparece como `-inf`, menos infinito.

Se aparecer `No module named 'numpy'`, o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
