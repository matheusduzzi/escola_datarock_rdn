"""Tarefa 3 do diagnóstico: das notas da rede às probabilidades e à perda.

    python3 tarefa3.py
    python3 testar_tarefa3.py

Escreva as duas funções sem loop em Python, só com operações do NumPy.
"""
import numpy as np


def softmax(z):
    """Recebe as notas z (n linhas, k classes) e retorna as probabilidades: cada linha soma 1.

    Precisa funcionar com notas grandes, como 1000, sem dar nan nem inf.
    """
    raise NotImplementedError('escreva softmax')


def log_loss(p, y):
    """Recebe as probabilidades p (n x k) e as classes certas y (n inteiros de 0 a k-1).

    Retorna a média de -log(probabilidade da classe certa). Uma probabilidade 0 na classe certa
    não pode virar infinito: troque por 1e-12 antes do log.
    """
    raise NotImplementedError('escreva log_loss')


if __name__ == '__main__':
    # Três fotos de produto e três classes: 0 camiseta, 1 calça, 2 tênis.
    z = np.array([[2.0, 1.0, 0.1],
                  [0.5, 0.5, 3.0],
                  [1.0, 3.0, 0.0]])
    y = np.array([0, 2, 0])
    p = softmax(z)
    print('probabilidades:', np.round(p, 3).tolist())
    print('log-loss:', round(float(log_loss(p, y)), 4))
    print('notas grandes:', np.round(softmax(np.array([[1000.0, 1001.0, 1002.0]])), 3).tolist())
