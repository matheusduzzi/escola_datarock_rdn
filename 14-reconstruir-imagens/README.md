# 14. Reconstruir imagens

Os arquivos da aula «Reconstruir imagens» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `redes.py` | As peças da aula: as fotos, a mesma divisão da aula 03, o autoencoder, o VAE, a perda, o Adam e o ciclo de treino. Os outros arquivos usam ele; ele não roda sozinho. |
| `por_dentro.py` | Abre as duas redes com uma foto: o caminho, a perda termo a termo, o sorteio, o gradiente e o erro de uma foto escura. |
| `comparar.py` | Compara as duas redes na validação: a foto refeita, onde elas erram, o que sai de um código sorteado e quantos números o VAE usa. |
| `alarme.py` | Usa o erro de reconstrução como alarme: o limiar na validação, o falso alarme por tipo no teste e quanto ele pega de cada defeito. |
| `treinar.py` | Treina uma das duas redes com o orçamento comum e grava os pesos em `meus-pesos/`. `python3 treinar.py --help` mostra as opções. |
| `pesos/` | O autoencoder e o VAE já treinados, com código de 16 números. |
| `meu_defeito.py` | A função do exercício, para você escrever. |
| `testar_meu_defeito.py` | Os testes da função do exercício. |
| `minhas-respostas.md` | A folha do exercício. |

## Do que você precisa

O Python com o NumPy, como no README do kit do curso, e as fotos do Fashion MNIST em `../dados/fashion-mnist`. Se ainda não baixou, rode `python3 baixar.py` dentro de `../dados/fashion-mnist`.

As grades de fotos usam o matplotlib, como na aula «CNN no Fashion MNIST». Sem ele, o resto roda igual e só as grades não são gravadas. Para instalar:

```bash
python3 -m pip install matplotlib
```

A aula rodou com Python 3.9.6, NumPy 2.0.2 e matplotlib 3.9.4, num Mac M4. As duas redes são escritas no NumPy, como as das aulas 03 e 08. Elas não usam o PyTorch.

## Como rodar

```bash
python3 por_dentro.py
python3 comparar.py
python3 alarme.py
```

Cada um leva alguns segundos e usa os pesos de `pesos/`, então você não precisa treinar nada para seguir a aula. O `comparar.py` grava `reconstrucoes-16.png` e `sorteios-16.png`; o `alarme.py` grava `defeitos-16.png`.

Para treinar você mesmo:

```bash
python3 treinar.py autoencoder
python3 treinar.py vae
python3 comparar.py --pesos meus-pesos
```

Cada treino leva de um a dois minutos no processador. Com `--codigo 2`, as duas redes treinam com um código de 2 números, como o exercício pede. O `comparar.py` e o `alarme.py` também recebem `--codigo` e `--pesos`. Os sorteios são fixos: no mesmo computador, os pesos saem iguais a cada rodada. Em outro computador, ou com outra versão do NumPy, as últimas casas podem mudar um pouco.

O Python mostra ponto no lugar da vírgula em alguns números: `3.7e-07` é 0,00000037.

Se aparecer `As fotos não estão em ../dados/fashion-mnist`, baixe as fotos como o README do kit do curso mostra. Se aparecer `Não achei vae-codigo-2.npz`, treine antes a rede com esse código. Se aparecer `No module named 'numpy'`, o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
