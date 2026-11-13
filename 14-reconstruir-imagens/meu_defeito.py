"""O defeito do exercício: a etiqueta branca que a Bia achou em fotos do catálogo.

Escreva a função etiqueta. Depois rode, de dentro da pasta da aula:
  python3 testar_meu_defeito.py
  python3 alarme.py
"""
import numpy as np


def etiqueta(fotos, linha, coluna, lado=6):
    """Retorna uma cópia das fotos com um quadrado branco (pixels 1,0) de lado x lado pixels.

    fotos: matriz (fotos, 784), cada linha é uma foto 28 x 28 em fila, pixels entre 0 e 1.
    linha, coluna: o canto de cima à esquerda do quadrado, na foto 28 x 28.
    lado: pelo menos 1.
    Se fotos não tiver 784 colunas, se lado for menor que 1 ou se o quadrado não couber inteiro
    na foto, ValueError. As fotos que entram não mudam.
    """
    raise NotImplementedError("escreva a função etiqueta")
