# 07. AlexNet, VGG e ResNet

Os arquivos da aula «AlexNet, VGG e ResNet» do curso Redes neurais. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `evolucao.py` | Tudo o que a aula mostra, partes 1 a 5: onde ficam os parâmetros de quatro redes, três 3 × 3 contra uma 7 × 7, o tamanho do mapa na VGG16 e na ResNet-50, o atalho com e sem pesos e o atalho que muda o formato. |
| `meu-mapa.md` | A folha do exercício: a tabela da rede R, a coluna da MobileNet e a resposta para a Bia. |

## Como rodar

Você só precisa do Python com o NumPy, como no README do kit do curso. Essa aula não usa as fotos do Fashion MNIST, o pandas nem o PyTorch, e não baixa nenhuma rede.

```bash
python3 evolucao.py
```

Leva menos de um segundo, e a saída é igual toda vez. As redes da parte 1 estão descritas camada por camada dentro do arquivo, do jeito que a biblioteca torchvision monta cada uma. Os totais batem com os da documentação do torchvision. A parte 4 sorteia pesos, sempre com o mesmo sorteio, para mostrar o atalho: nenhuma rede é treinada.

O exercício é feito à mão, no `meu-mapa.md`. O `evolucao.py` não resolve o exercício, mas você pode abrir o arquivo e ver como cada conta da aula foi feita. Para fazer uma conta, use o Python como calculadora:

```bash
python3 -c "print(128 * (3 * 3 * 64) + 128)"
```

Os números grandes da saída aparecem com ponto separando os milhares, como na aula: `18.496`. Os números muito pequenos ou muito grandes aparecem como `1,42e-15`, que quer dizer 1,42 vezes 10 elevado a -15.

Se aparecer `No module named 'numpy'`, o NumPy foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
