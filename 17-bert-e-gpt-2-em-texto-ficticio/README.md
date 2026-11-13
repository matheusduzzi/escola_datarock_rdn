# 17. BERT e GPT-2 em texto

Os arquivos da aula «BERT e GPT-2 em texto» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `baixar_modelos.py` | Baixa o BERT em inglês, o BERT multilíngue e o GPT-2 do Hugging Face, numa versão fixa, e confere o `SHA-256` de cada arquivo. |
| `modelos.py` | O BERT e o GPT-2 escritos no NumPy: lê os pesos sem o PyTorch, quebra o texto em tokens como os originais, faz a ida e escreve um token de cada vez. Os outros arquivos usam ele; ele não roda sozinho. |
| `textos.py` | As contas da aula, em seis partes: `modelos`, `tokens`, `lacuna`, `escrever`, `assunto` e `custo`. `python3 textos.py --help` mostra as opções. |
| `julho.csv` | As 40 mensagens de julho que o Diego separou e marcou, 10 de cada assunto. É o mesmo arquivo da aula «Naive Bayes na prática» do curso ML supervisionado. |
| `meu_resumo.py` | Onde você escreve a função `resumir` do exercício. O resto já está pronto. |
| `testar_meu_resumo.py` | Os testes da sua função. |
| `minhas-respostas.md` | A folha do exercício. |

## Do que você precisa

- O Python 3.9 ou mais novo com o NumPy, como no README do kit do curso. Essa aula não usa o PyTorch nem a biblioteca transformers.
- As mensagens em `../dados/mensagens/mensagens.csv`, que já vêm no kit do curso.
- Uns 1,7 GB livres no disco e internet uma vez, para os três modelos. Na memória, o maior deles ocupa uns 1,1 GB.
- Paciência com o tempo: a aula toda roda em uns 30 minutos no processador, sem placa de vídeo.

## Os modelos

```bash
python3 baixar_modelos.py
```

Cada linha termina com `SHA-256 ok`. Os arquivos vão para `modelos/` e não são baixados de novo depois. Se um `SHA-256` não bater, o arquivo é apagado e o programa para; rode de novo. Para baixar um só, passe o nome: `python3 baixar_modelos.py gpt2`.

O kit não traz os pesos: você baixa do Hugging Face, sem conta. As licenças estão no cartão de cada modelo: Apache 2.0 no [BERT em inglês](https://huggingface.co/google-bert/bert-base-uncased) e no [BERT multilíngue](https://huggingface.co/google-bert/bert-base-multilingual-cased), MIT no [GPT-2](https://huggingface.co/openai-community/gpt2). Os cartões também trazem os limites de cada modelo. Para usar numa empresa, a decisão é de quem responde pelo produto.

## Como rodar

Na ordem da aula. Eu rodei com Python 3.9.6 e NumPy 2.0.2, num Mac com chip M4, com outros programas abertos.

```bash
python3 textos.py modelos      # uns 20 segundos
python3 textos.py tokens       # uns 2 segundos
python3 textos.py lacuna       # uns 12 minutos
python3 textos.py escrever     # uns 2 minutos
python3 textos.py assunto      # uns 10 minutos na primeira vez
python3 textos.py custo        # uns 20 segundos
```

O `assunto` guarda os vetores das mensagens em `meus-resultados/`. Na segunda vez, ele leva segundos. Para tirar os vetores de novo, use `python3 textos.py assunto --refazer`. O `custo` usa os tempos que o `assunto` mediu, então rode o `assunto` antes.

Não há sorteio: os tokens, os palpites, as perdas e os acertos saem iguais a cada rodada no mesmo computador. Os segundos mudam com o que mais está aberto. Rode com o computador parado: dois programas juntos dividem o processador. Em outro computador, ou com outra versão do NumPy, as últimas casas das chances e das perdas podem mudar um pouco.

## O exercício

```bash
python3 testar_meu_resumo.py
python3 meu_resumo.py                                    # o GPT-2, uns 3 minutos
python3 meu_resumo.py --modelo bert-base-multilingual-cased
```

Antes de você escrever a função `resumir`, os três param com `NotImplementedError: escreva resumir(estados, mascara, jeito)`. É o esperado. Os testes não precisam dos modelos. Quando a função estiver certa, aparece `ok: 10 testes passaram`. Se um teste falhar, a mensagem diz qual e o que saiu.

## Se der erro

- `Não achei model.safetensors ... Rode: python3 baixar_modelos.py ...`: baixe os modelos.
- `Não achei as mensagens`: rode de dentro da pasta da aula, com a pasta `dados/` na raiz do kit.
- `SHA-256 ... diferente do esperado`: o download veio pela metade ou o arquivo mudou. Rode `python3 baixar_modelos.py` de novo.
- `MemoryError` ou o computador travando: feche outros programas. O BERT multilíngue precisa de uns 1,1 GB livres.
- `No module named 'numpy'`: o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver.

O Python mostra ponto no lugar da vírgula: `0.1073` é 0,1073. No GPT-2, os tokens aparecem como o tokenizador guarda: `Ġ` é o espaço e `Ã£` são os dois bytes do "ã". No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
