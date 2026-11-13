# 03. MLP de classificação

Os arquivos da aula «MLP de classificação» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `mlp.py` | As peças do ciclo de treino: as fotos, a divisão em treino e validação, a rede, os gradientes, o passo, a avaliação e o `ciclo`. Os outros arquivos usam ele. |
| `treinar.py` | Tudo o que a aula mostra: as três partes, os baselines, a curva da rede 784 → 128 → 10, a época escolhida, a curva com mais fotos e o teste. |
| `meu_ciclo.py` | Onde você escreve a função `treinar_com_paciencia` do exercício. |
| `testar_meu_ciclo.py` | Os testes da sua função. Eles não usam as fotos. |
| `meu_treino.py` | A tarefa 2 do exercício: treina a rede 784 → 256 → 10 com a sua função. Não mexa nele. |
| `minhas-respostas.md` | Onde você escreve as saídas e as respostas do exercício. |

## Como rodar

Você precisa do Python com o NumPy, como no README do kit do curso, e das fotos do Fashion MNIST. Se ainda não baixou, baixe uma vez:

```bash
python3 ../dados/fashion-mnist/baixar.py
```

O `baixar.py` salva as fotos na pasta dele, `../dados/fashion-mnist`, que é onde as aulas procuram.

Essa aula não usa o pandas nem outra biblioteca. A rede, os gradientes e o treino são todos no NumPy.

```bash
python3 treinar.py
```

Leva menos de um minuto num notebook recente. A saída é a das seções 2 a 8 da aula, e a curva vai para o `curva.csv`. O sorteio é fixo, então a saída sai igual a cada rodada no mesmo computador. Em outro computador, ou com outra versão do NumPy, os números do treino podem mudar um pouco nas últimas casas. A aula rodou com Python 3.9.6 e NumPy 2.0.2.

Se aparecer `As fotos não estão em ../dados/fashion-mnist`, rode o `baixar.py` como acima.

```bash
python3 testar_meu_ciclo.py
python3 meu_treino.py
```

Antes de você escrever a função, os dois param com `NotImplementedError: escreva treinar_com_paciencia(...)`. É o esperado. Quando a função estiver certa, o teste mostra `ok: 10 testes passaram`. Se um teste falhar, a mensagem diz o que saiu e o que era esperado. O `meu_treino.py` leva menos de um minuto e grava o `minha_curva.csv`.

Se aparecer `No module named 'numpy'`, o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
