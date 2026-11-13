# 10. Transferência na prática

Os arquivos da aula «Transferência na prática» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `baixar_pesos.py` | Baixa os pesos da ResNet-18 pronta, do endereço oficial do PyTorch, e confere o `SHA-256`. |
| `resnet.py` | A ResNet-18 escrita no NumPy: lê o arquivo de pesos sem o PyTorch, prepara as fotos, faz a ida pela rede e a ida e a volta da etapa 4. Os outros arquivos usam ele; ele não roda sozinho. |
| `experimento.py` | O experimento da aula, em partes: `rede`, `tamanho`, `treinar` e `comparar`. `python3 experimento.py --help` mostra as opções. |
| `minhas-respostas.md` | A folha do exercício. |

## Do que você precisa

- O Python com o NumPy, como no README do kit do curso.
- As fotos do Fashion MNIST em `../dados/fashion-mnist`. Se ainda não baixou, rode `python3 baixar.py` dentro de `../dados/fashion-mnist`.
- A pasta da aula «CNN no Fashion MNIST», `../08-cnn-no-fashion-mnist`, com os pesos da CNN do Caio em `pesos/`. O `experimento.py` usa o `cnn.py` dela e compara com essa rede.
- Uns 45 MB livres e internet uma vez, para os pesos da ResNet-18.

Essa aula não usa o PyTorch. A ResNet-18 é a mesma arquitetura do torchvision, com o mesmo arquivo de pesos, escrita no NumPy como a CNN da aula 08.

## Os pesos da ResNet-18

```bash
python3 baixar_pesos.py
```

A saída termina com `SHA-256 ok`. O arquivo vai para `pesos/resnet18-f37072fd.pth` e não é baixado de novo depois. Se o `SHA-256` não bater, o arquivo é apagado e o programa para; rode de novo.

O kit não traz os pesos: você baixa do PyTorch. A documentação do torchvision avisa que os pesos prontos podem ter termos de uso próprios, que vêm do conjunto de fotos em que foram treinados (aqui, o ImageNet). Para estudar com eles, como nessa aula, leia o aviso na [página de modelos do torchvision](https://docs.pytorch.org/vision/stable/models.html). Para usar numa empresa, a decisão é de quem responde pelo produto.

## Como rodar

Na ordem da aula. Eu rodei com Python 3.9.6 e NumPy 2.0.2, num Mac com chip M4.

```bash
python3 experimento.py rede        # segundos
python3 experimento.py tamanho     # uns 5 minutos
python3 experimento.py treinar     # uns 20 minutos
python3 experimento.py comparar    # uns 2 minutos
```

O `treinar` grava o que treinou em `meus-resultados/fotos-50000/`, e o `comparar` lê de lá. A CNN do Caio não é treinada de novo: ela vem de `../08-cnn-no-fashion-mnist/pesos/`, treinada no mesmo orçamento. Rode com o computador parado: dois treinos juntos dividem o processador e os dois ficam mais lentos.

Os números de perda e acerto saem iguais a cada rodada no mesmo computador. Os segundos mudam com o que mais está aberto. Em outro computador, ou com outra versão do NumPy, as últimas casas da perda e do acerto podem mudar um pouco.

Só depois de escolher a rede pela validação:

```bash
python3 experimento.py comparar --teste
```

## O exercício

```bash
python3 experimento.py treinar --fotos 1000     # uns 4 minutos
python3 experimento.py comparar --fotos 1000    # menos de 1 minuto
```

Com `--fotos`, o `treinar` usa só as primeiras fotos de treino da divisão da aula 03 e treina também a CNN do Caio do zero com essas mesmas fotos. A validação continua com as 10.000 de sempre. Os resultados vão para `meus-resultados/fotos-1000/`.

## Se der erro

- `Não achei pesos/resnet18-f37072fd.pth`: rode `python3 baixar_pesos.py`.
- `As fotos não estão em ../dados/fashion-mnist`: baixe as fotos como o README do kit do curso mostra.
- `Não achei ../08-cnn-no-fashion-mnist/cnn.py`: essa aula precisa da pasta da aula 08 ao lado, como no kit.
- `o arquivo pede ..., que não é peça de tensor: parei`: o arquivo em `pesos/` não é o dos pesos. Apague e rode `python3 baixar_pesos.py`.
- `No module named 'numpy'`: o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver.

O Python mostra ponto no lugar da vírgula: `0.9015` é 90,15%. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
