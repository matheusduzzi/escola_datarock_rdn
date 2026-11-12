# Kit do curso Redes neurais

Os arquivos que você usa nas aulas: o diagnóstico e o programa que baixa as fotos do Fashion MNIST. E também a venda diária da loja física da Loja Exemplo e as mensagens ao atendimento da mesma loja.

## Como o kit está organizado

| Pasta | O que tem |
|---|---|
| `00-apresentacao/` | O [diagnóstico](00-apresentacao/diagnostico.md), os programas das tarefas 3 e 5, o teste da tarefa 3, a [folha de respostas](00-apresentacao/minhas-respostas.md), a [folha da ordem de estudo](00-apresentacao/minha-ordem.md) e a `primeira_olhada.py`, que mede a foto média nas fotos baixadas. |
| `dados/fashion-mnist/` | O `baixar.py`, que busca as 70.000 fotos no endereço oficial e confere cada arquivo. Ver [dados/fashion-mnist/LEIA-ME.md](dados/fashion-mnist/LEIA-ME.md). |
| `dados/vendas/` | Dois anos de venda diária da loja física e o calendário de feriados e promoções, os mesmos de Séries temporais. Ver [dados/vendas/LEIA-ME.md](dados/vendas/LEIA-ME.md). |
| `dados/mensagens/` | 1.800 mensagens ao atendimento, as mesmas de ML supervisionado. Ver [dados/mensagens/LEIA-ME.md](dados/mensagens/LEIA-ME.md). |
| `NN-titulo/` | Uma pasta por aula, com o número e o título da aula. Elas entram no kit junto com as aulas. |

Os comandos das aulas rodam de dentro da pasta da aula. Quando uma aula usa uma base compartilhada, ela diz o caminho, por exemplo `../dados/vendas/venda_diaria.csv`.

## Do que você precisa

- Python 3.9 ou mais novo, com o pandas e o NumPy. É o mesmo ambiente de ML supervisionado; se você ainda não preparou o seu, a aula 38. Dependências e ambientes, de Python para dados, mostra como.
- Um editor de código e um terminal.
- Internet para baixar as fotos uma vez e para instalar as bibliotecas.

As redes do curso são pequenas e treinam no processador do computador, sem placa de vídeo. A biblioteca de redes neurais, a de modelos de texto e a de gráficos entram na primeira aula que usa cada uma. Essa aula traz a versão, o Python mínimo e o comando de instalação.

Na pasta do kit, instale o que o diagnóstico usa:

```bash
python3 -m pip install -r requirements.txt
```

E veja se funcionou:

```bash
cd 00-apresentacao
python3 tarefa5.py
```

A primeira linha da saída começa com `dias no calendário: 731`; o resto é a tarefa 5 do diagnóstico. Se aparecer `No module named 'pandas'`, o `pip` instalou em outro Python. Rode `python3 -m pip install pandas` com o mesmo `python3` que você usa para rodar o programa. No Windows, troque `python3` por `py`.

Depois baixe as fotos:

```bash
cd ../dados/fashion-mnist
python3 baixar.py
```

Cada um dos quatro arquivos termina com `MD5 ok`.

Para gerar de novo a venda e as mensagens, rode `python3 gerar.py` dentro da pasta de cada uma. As saídas são:

| Pasta | Saída de `python3 gerar.py` |
|---|---|
| `dados/vendas/` | `venda_diaria=722 dias_no_calendario=731 sem_registro=9 calendario=762` |
| `dados/mensagens/` | `mensagens=1800 atraso=669 troca=495 pagamento=362 outro=274` |

Os geradores só usam a biblioteca padrão e saem iguais toda vez.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.

## Licença

A licença de uso e redistribuição do kit ainda vai ser definida pela Escola Datarock. Até lá, use os arquivos para estudar o curso. A Loja Exemplo é inventada: a venda e as mensagens foram geradas pelos programas do próprio kit.

As fotos do Fashion MNIST não fazem parte do kit. Elas são da Zalando SE, sob licença MIT, e o `baixar.py` busca cada arquivo no endereço oficial. O aviso da licença e a forma de citar estão em [dados/fashion-mnist/LEIA-ME.md](dados/fashion-mnist/LEIA-ME.md).

## Aulas com arquivos neste kit

| Pasta | Aula |
|---|---|
| [00-apresentacao/](00-apresentacao/) | Apresentação |
