# Diagnóstico: por onde começar em Redes neurais

A equipe de catálogo da Loja Exemplo me pediu uma rede neural para separar as fotos de roupa por tipo. Antes de montar a primeira camada, eu refiz cinco contas que toda rede faz por dentro. Multiplicar matriz por vetor, derivar a perda e transformar notas em probabilidades. Ler uma curva de treino. E medir o que uma previsão simples já acerta. As cinco tarefas abaixo são essas contas.

Elas pedem o que vem dos cursos anteriores: produto de matrizes, derivada e gradiente, NumPy sem loop e com teste. E também treino, validação e teste e o corte no tempo para vendas. O diagnóstico não é prova e não tem nota. Ele mostra por onde você começa. Anote na [folha de respostas](minhas-respostas.md) a conta que fez, o comando que rodou e a saída que apareceu. Só depois compare com as respostas.

As tarefas 1, 2 e 4 são de lápis e papel ou calculadora. As tarefas 3 e 5 rodam de dentro da pasta `00-apresentacao/`:

```bash
cd 00-apresentacao
python3 tarefa5.py
```

No Windows, troque `python3` por `py`.

## Tarefa 1. Uma passagem por uma rede pequena

Uma rede recebe duas entradas, tem uma camada escondida com três neurônios e uma saída. A camada escondida usa ReLU: valor negativo vira 0, o resto passa igual.

```text
x = (1, 2)

W = |  1  -1 |      b = ( 0, -1, 1)
    |  2   0 |
    |  0   1 |

h = ReLU(W x + b)
saída = v · h + c,   com v = (1, 2, -1) e c = 0,5
```

1. Calcule `W x + b`, depois `h` e a saída.
2. Quantos números essa rede aprende no treino, contando pesos e vieses?
3. Uma foto de 28 por 28 pixels entra como 784 números. A rede tem uma camada escondida de 128 neurônios e uma saída de 10 classes. Quantos números ela aprende?
4. No treino, as fotos entram em lotes de 32. A matriz de entrada tem formato (32, 784). Que formato sai da camada escondida?
5. Tire a ReLU e deixe `h = W x + b`. A rede ainda consegue aprender alguma coisa que um modelo linear não aprende? Por quê?

## Tarefa 2. Um passo de gradiente

Um modelo de um peso só prevê `ŷ = w · x`. A perda de um exemplo é `L = (ŷ - y)²`. O exemplo tem `x = 2` e `y = 3`, e o peso começa em `w = 0,5`.

1. Calcule `ŷ` e a perda.
2. Escreva `L` como `u²`, com `u = w · x - y`. Pela regra da cadeia, quanto vale `dL/dw`?
3. Dê um passo de gradiente, `w_novo = w - taxa · dL/dw`, com taxa 0,1. Qual o novo peso e a nova perda?
4. Repita o passo a partir de `w = 0,5` com taxa 0,25 e com taxa 0,5. O que acontece com a perda em cada caso?
5. Que peso zera a perda? Por que uma taxa grande demais nunca chega nele?

## Tarefa 3. Das notas às probabilidades

A última camada de uma rede de classificação dá uma nota por classe. A softmax transforma as notas em probabilidades, e a log-loss mede o quanto a rede errou. Em `tarefa3.py`, escreva duas funções, sem loop em Python:

| Função | O que retorna |
|---|---|
| `softmax(z)` | as probabilidades, do mesmo formato das notas `z` (n linhas, k classes): `exp(z)` dividido pela soma de `exp(z)` da linha; cada linha soma 1; `z` pode chegar como lista de listas |
| `log_loss(p, y)` | a média de `-log` da probabilidade que a rede deu à classe certa `y` de cada linha, com 1e-12 no lugar de uma probabilidade 0 |

Depois rode:

```bash
python3 tarefa3.py
python3 testar_tarefa3.py
```

Enquanto as funções não existem, os dois comandos param com `NotImplementedError`. Anote na folha:

1. As três linhas que `tarefa3.py` mostrou.
2. Qual das três fotos pesa mais na log-loss? Por quê, se a rede deu a maior probabilidade a uma classe em todas?
3. `np.exp(1000)` dá `inf`. Como a sua `softmax` lida com notas como 1000, 1001 e 1002 sem mudar o resultado?
4. Por que a log-loss precisa do piso 1e-12?

## Tarefa 4. A curva de treino

Eu treinei uma rede pequena com 40 exemplos de duas classes. Seis desses 40 rótulos estão trocados, como acontece quando alguém marca errado. A validação tem 2.000 exemplos sorteados da mesma regra, sem rótulo trocado. A perda é a log-loss. A cada época, a rede vê os 40 exemplos uma vez.

| Época | Perda no treino | Perda na validação | Acerto no treino | Acerto na validação |
|---|---|---|---|---|
| 0 | 0,693 | 0,693 | 45,0% | 50,6% |
| 100 | 0,448 | 0,328 | 82,5% | 88,4% |
| 300 | 0,267 | 0,371 | 90,0% | 84,0% |
| 1.000 | 0,091 | 0,517 | 97,5% | 82,5% |
| 3.000 | 0,013 | 0,828 | 100% | 81,8% |
| 10.000 | 0,001 | 1,236 | 100% | 82,6% |
| 30.000 | 0,000 | 1,467 | 100% | 82,3% |

1. Na época 0, as duas perdas valem 0,693. De onde vem esse número?
2. A partir de que época a rede piora na validação? Que época você guardaria?
3. A rede chega a 100% de acerto no treino. O que isso quer dizer, se 6 dos 40 rótulos estão errados?
4. Da época 3.000 à 30.000, o acerto na validação quase não muda, mas a perda na validação quase dobra. O que a rede está fazendo com as probabilidades?
5. Eu também tenho 2.000 exemplos de teste. Por que eu não uso o teste para escolher a época?

## Tarefa 5. O que uma rede de vendas precisa bater

Rode `python3 tarefa5.py`. Ele lê dois anos de venda diária da loja física em `../dados/vendas/` e mede o erro de três previsões simples em novembro e dezembro de 2025. A previsão de cada dia sai na noite anterior, depois que a loja fecha, e só usa o que já se sabia nessa hora. Anote a saída.

1. Qual das três previsões uma rede de vendas precisa bater? Por quanto?
2. Por que "o mesmo dia da semana passada" erra menos que "ontem"?
3. Uma colega cortou a série em janelas de 14 dias, embaralhou as janelas e separou 20% delas para validar. A rede dela errou 12 pedidos por dia. Por que esse número não serve para comparar com a saída do `tarefa5.py`?
4. A equipe pediu a previsão de amanhã mais cedo: hoje às 18h, com a loja aberta até as 22h. Nesse horário, quais destas informações a rede pode usar: os pedidos de hoje, os pedidos de ontem, a promoção marcada para amanhã no calendário, o feriado de amanhã? E as duas primeiras previsões do `tarefa5.py` continuam valendo?

## Depois das tarefas

Compare a folha com as respostas na aula e veja a tabela da apresentação do curso. Ela diz que aula estudar para cada tarefa em que você travou.

Se precisar de ajuda, deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
