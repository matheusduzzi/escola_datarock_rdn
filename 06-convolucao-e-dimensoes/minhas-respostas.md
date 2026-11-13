# Minhas respostas: a rede B

## Tarefa 1. A tabela à mão (fotos 28 × 28)

| Camada | Conta do tamanho | Saída (canais × altura × largura) | Conta dos parâmetros | Parâmetros |
|---|---|---|---|---|
| Entrada | | 1 × 28 × 28 | | 0 |
| 1. Conv 16 filtros 5 × 5, stride 1, padding 2 | | | | |
| 2. Pooling 2 × 2, stride 2 | | | | |
| 3. Conv 32 filtros 3 × 3, stride 2, padding 1 | | | | |
| 4. Conv 64 filtros 3 × 3, stride 1, padding 0 | | | | |
| 5. Pooling 2 × 2, stride 2 | | | | |
| 6. Achatar | | | | |
| 7. Densa 10 | | | | |
| Total | | | | |

Números de cada mapa que o segundo pooling deixa de fora:

## Tarefa 2. No Python

Cole aqui a linha `ok` do `python3 testar_minhas_contas.py`:

```text

```

Cole aqui a saída do `python3 minhas_contas.py`:

```text

```

A tabela do Python bateu com a da tarefa 1? Se não bateu, em que linha, e qual dos dois estava errado?

## Tarefa 3. A resposta para a Bia

(quatro a seis frases)
