# Meu mapa: a rede R e a MobileNet

## Tarefa 1. A tabela da rede R (fotos 28 × 28)

| Camada | Conta do tamanho | Saída (canais × altura × largura) | Conta dos parâmetros | Parâmetros |
|---|---|---|---|---|
| Entrada | | 1 × 28 × 28 | | 0 |
| 1. Conv 32 filtros 3 × 3, stride 1, padding 1 | | | | |
| 2. Pooling 2 × 2, stride 2 | | | | |
| 3a. Conv 32 filtros 3 × 3, stride 1, padding 1 | | | | |
| 3b. Conv 32 filtros 3 × 3, stride 1, padding 1 | | | | |
| Atalho do bloco A | | | | |
| 4a. Conv 64 filtros 3 × 3, stride 2, padding 1 | | | | |
| 4b. Conv 64 filtros 3 × 3, stride 1, padding 1 | | | | |
| Atalho do bloco B | | | | |
| 5. Média global | | | | |
| 6. Densa 10 | | | | |
| Total | | | | |

O atalho do bloco A pode ser só x? Por quê?

O atalho do bloco B pode ser só x? Por quê?

Com achatar + densa de 10 no lugar da média global: parâmetros da densa e total da rede.

## Tarefa 2. A coluna da MobileNet

| | MobileNet, 2017 |
|---|---|
| Problema que atacou | |
| Ideia | |
| O que o artigo mostrou | |
| O que custou | |

A convolução 4b da rede R, separável em profundidade:

| Parte | Conta dos parâmetros | Parâmetros |
|---|---|---|
| 3 × 3, um filtro por canal | | |
| 1 × 1 que mistura os canais | | |
| Total separável | | |
| 4b comum (da tarefa 1) | | |

## Tarefa 3. A resposta para a Bia

O lado do mapa na ResNet-50, fotos 28 × 28:

| Passo | Conta do tamanho | Lado |
|---|---|---|
| Convolução 7 × 7, stride 2, padding 3 | | |
| Pooling 3 × 3, stride 2, padding 1 | | |
| Etapa 1 | | |
| Etapa 2 | | |
| Etapa 3 | | |
| Etapa 4 | | |

Parâmetros da ResNet-50 com a densa de 10 classes:

A resposta (cinco a oito frases):

