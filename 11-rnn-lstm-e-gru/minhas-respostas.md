# Minhas respostas: a GRU do Caio

## Tarefa 1. À mão

### O desenho

Complete o esqueleto: o que entra em cada caixa, o que passa de uma caixa para a outra, de onde sai a previsão e quantos passos separam o h do dia 1 do último h.

```text
         x do dia 1          x do dia 2                    x do dia 60
        (    entradas)      (    entradas)                (    entradas)
             |                   |                             |
             v                   v                             v
h0 = 0 -> [ GRU ] --- h1 ---> [ GRU ] --- h2 ---> ... ---> [ GRU ] --- h60 ---> [      ] ---> previsão de
                                                                                              

passos entre h1 e h60:
```

### A contagem

| | Janela de 60 dias (conta e número) | Janela de 28 dias (conta e número) |
|---|---|---|
| Vezes que a célula roda | | |
| Passos entre o h do dia 1 e o último h | | |
| Blocos calculados na janela | | |
| Parâmetros da GRU | | |
| Parâmetros da densa | | |
| Total da rede | | |
| Total de uma LSTM com as mesmas entradas, estado e densa | | |
| O que sobra do h do dia 1, com update 0,9 todo dia e o h antigo só pelo update | | |

## Tarefa 2. No Python

Cole aqui a linha `ok` do `python3 testar_minha_gru.py`:

```text

```

Cole aqui a saída do `python3 minha_gru.py`:

```text

```

A saída bateu com a sua tabela da tarefa 1? Se não bateu, em que linha, e qual dos dois estava errado?

## Tarefa 3. A resposta para a gerente

(cinco a sete frases)
