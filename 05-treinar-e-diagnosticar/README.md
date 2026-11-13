# 05. Treinar e diagnosticar

Os arquivos da aula «Treinar e diagnosticar» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `rede.py` | O ciclo de treino da aula «MLP de classificação», com as cinco medidas da aula em cada época. Os outros arquivos usam ele; ele não roda sozinho. |
| `diagnosticar.py` | Tudo o que a aula mostra, partes 1 a 7: as seis taxas, a rede funda que não aprende, o teste das 100 fotos e as 2.000 fotos. |
| `experimentar.py` | Treina qualquer configuração e mostra a curva com as medidas. É o que você usa na prática guiada. `python3 experimentar.py --help` mostra as opções. |
| `treino_do_caio.py` | O treino que o Caio mandou, do exercício. Não mexa nele. |
| `minha_correcao.py` | Onde você muda a configuração do Caio, um valor por vez. |
| `testar_minha_correcao.py` | Os testes do exercício. |
| `meu-registro.md` | A folha do seu registro de falhas e correções e da resposta para o Caio. |

## Como rodar

Você precisa do Python com o NumPy, como no README do kit do curso, e das fotos do Fashion MNIST em `../dados/fashion-mnist`. Se ainda não baixou, rode `python3 baixar.py` dentro de `../dados/fashion-mnist`. Essa aula não usa o pandas nem o PyTorch.

```bash
python3 diagnosticar.py
```

Leva uns 20 segundos. A saída é a das seções 3 a 7 da aula. Os sorteios são fixos, então ela sai igual toda vez no mesmo computador. Em outro computador, as últimas casas podem mudar um pouco.

```bash
python3 treino_do_caio.py
python3 minha_correcao.py
python3 testar_minha_correcao.py
```

Cada um leva uns 5 segundos. Enquanto o `minha_correcao.py` tiver a configuração do Caio, o teste mostra linhas `FALHOU` e sai com código 1. É o esperado. Quando a sua correção passa, aparece `ok: 10 testes passaram`.

Os números grandes aparecem como `1.2e+43`: um 12 seguido de 42 zeros. `nan` quer dizer que a conta deixou de ter número. O Python mostra ponto no lugar da vírgula.

Se aparecer `As fotos não estão em ../dados/fashion-mnist`, baixe as fotos como o README do kit do curso mostra. Se aparecer `No module named 'numpy'`, o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
