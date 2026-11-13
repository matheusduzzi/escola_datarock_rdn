# 18. Escolher uma arquitetura

Os arquivos da aula «Escolher uma arquitetura» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `variacao.py` | Treina a mesma rede da aula 08 com sorteios diferentes e mostra quanto o acerto muda. `python3 variacao.py --help` mostra as opções. |
| `ficha-de-defesa.md` | A ficha da defesa, com os doze itens. É a entrega do exercício. |

## Do que você precisa

- O Python com o NumPy, como no README do kit do curso.
- As fotos do Fashion MNIST em `../dados/fashion-mnist`. Se ainda não baixou, rode `python3 baixar.py` dentro de `../dados/fashion-mnist`.
- A pasta da aula «CNN no Fashion MNIST», `../08-cnn-no-fashion-mnist`. O `variacao.py` usa o `cnn.py` de lá e o mesmo orçamento comum.
- Para preencher a ficha das fotos, as saídas das aulas 03, 08 e 10. Se você não guardou, rode de novo os comandos de cada aula, nas pastas `../03-mlp-de-classificacao`, `../08-cnn-no-fashion-mnist` e `../10-transferencia-na-pratica`. Para a ficha da venda, a pasta `../12-lstm-em-vendas-extensao-temporal`.

## Rodar

```bash
python3 variacao.py mlp
```

Leva uns 2 minutos. A primeira linha da tabela, o sorteio 0, repete a validação da MLP na aula 08: perda 0.2957, acerto 0.8931, 1069 erros.

```bash
python3 variacao.py caio --sorteios 3
```

Leva uns 30 minutos numa máquina livre, porque a CNN do Caio treina 3 vezes com o orçamento todo. Com outros programas abertos, pode passar de uma hora. O sorteio 0 repete a validação da CNN do Caio na aula 08: perda 0.2673, acerto 0.9015, 985 erros. Se a primeira linha sair diferente, a pasta da aula 08 não é a do kit; baixe de novo.

O teste não é aberto pelo `variacao.py`. As fotos de treino e de validação são as mesmas da aula 08 em todos os sorteios. O que muda é o sorteio dos pesos iniciais e da ordem dos lotes.

Os segundos mudam com o computador e com o que estiver aberto. A perda, o acerto e os erros saem iguais no mesmo computador e podem mudar nas últimas casas em outro.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
