"""Exercício da aula «MLP de classificação»: o ciclo de treino que para sozinho.

Escreva a função treinar_com_paciencia. Depois rode, de dentro da pasta da aula:

    python3 testar_meu_ciclo.py
    python3 meu_treino.py

Enquanto a função não existe, os dois comandos param com NotImplementedError.
"""
import numpy as np

import mlp


def treinar_com_paciencia(rede, X, y, i_treino, i_validacao, max_epocas, paciencia, taxa, lote, sorteio=0):
    """Treina e para quando a perda de validação passa `paciencia` épocas seguidas sem melhorar.

    Retorna (melhor_rede, melhor_epoca, curva), como o mlp.ciclo da aula:
    - curva é uma lista com uma linha por época, começando na época 0 (antes do primeiro passo).
      Cada linha é o dicionário de mlp.linha_da_curva.
    - melhor_epoca é a época de menor perda de validação. Empate não conta como melhora.
    - melhor_rede é uma cópia da rede nessa época (mlp.copiar), que não muda quando o treino continua.
    - o treino para depois de `paciencia` épocas seguidas sem melhora, ou em max_epocas.

    Use mlp.uma_epoca para cada época e mlp.linha_da_curva para cada linha da curva, como o
    mlp.ciclo faz. Crie o sorteio da ordem das fotos uma vez só, antes da primeira época.

    Levante ValueError, sem treinar nada, quando:
    - max_epocas ou paciencia forem menores que 1;
    - i_validacao estiver vazio;
    - alguma foto estiver ao mesmo tempo em i_treino e em i_validacao.
    """
    raise NotImplementedError("escreva treinar_com_paciencia(rede, X, y, i_treino, i_validacao, ...)")
