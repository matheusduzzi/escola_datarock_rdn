# Minhas respostas: os gradientes de camisa e casaco, e o código do Caio

## Tarefa 1. À mão

Foto: x = (1; −1; 2), classe certa: camisa.

Camada escondida:

| Neurônio | Pesos × x | Mais o viés | Depois da ReLU |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

Notas (camisa, casaco):

Probabilidades:

Perda:

p − y:

dW2, linha camisa:

dW2, linha casaco:

db2:

Perda → h:

| Neurônio | Conta | Perda → h |
|---|---|---|
| 1 | | |
| 2 | | |
| 3 | | |

Máscara da ReLU:

Perda → z1:

dW1, linha 1:

dW1, linha 2:

dW1, linha 3:

db1:

Parâmetros com gradiente 0, e por quê:

## Tarefa 2. No NumPy

Saída de `python3 testar_meus_gradientes.py`:

```
```

Saída de `python3 meus_gradientes.py`:

```
```

Bateu com a tarefa 1? Se não bateu, onde estava a diferença?

## Tarefa 3. A resposta para o Caio

Gradientes errados:

Linha do `gradientes_do_caio.py`:

Resposta:
