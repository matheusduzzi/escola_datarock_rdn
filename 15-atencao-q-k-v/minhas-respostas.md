# Minhas respostas: a camada de atenção do Caio

## Tarefa 1. À mão

### As dimensões

Escreva o (linhas x colunas) de cada uma.

| Matriz | Dimensão |
|---|---|
| X | |
| W_Q inteira | |
| W_Q de uma cabeça | |
| Q de uma cabeça | |
| K de uma cabeça | |
| V de uma cabeça | |
| Q Kᵀ de uma cabeça | |
| Pesos de uma cabeça | |
| Saídas das duas cabeças lado a lado | |
| W_O | |
| Saída da camada | |

As notas são divididas por:

Pesos de W_Q, W_K, W_V e W_O juntos (com a conta):

### A linha do "não" na cabeça 1

Query do "não" (com a conta):

| Palavra | Key na cabeça 1 | Nota | Nota dividida | Peso sem máscara | Peso com máscara causal |
|---|---|---|---|---|---|
| meu | | | | | |
| pedido | | | | | |
| não | | | | | |
| chegou | | | | | |

Conta da softmax sem máscara:

Saída do "não" na cabeça 1, sem máscara (com a conta):

Saída do "não" na cabeça 1, com máscara causal (com a conta):

### O tamanho do artigo

Modelo base, frase de 50 tokens, uma cabeça:

| Matriz | Dimensão |
|---|---|
| Q | |
| V | |
| Pesos | |

As notas são divididas por:

Notas nas 8 cabeças (com a conta):

Notas por cabeça com 500 tokens (com a conta):

## Tarefa 2. No Python

Cole aqui a linha `ok` do `python3 testar_minha_atencao.py`:

```text

```

Cole aqui a saída do `python3 minha_atencao.py`:

```text

```

O que bateu com a tarefa 1 e o que não bateu (diga onde e qual dos dois estava errado):

## Tarefa 3. A resposta para a coordenadora

Cinco a sete frases:

