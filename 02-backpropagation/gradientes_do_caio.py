"""A função que o Caio escreveu com ajuda de uma IA para calcular os gradientes da rede.

Não mexa nesse arquivo: ele é o código que você vai examinar na tarefa 3 do exercício.
"""
import numpy as np


def gradientes_do_caio(x, classe, rede):
    X = np.atleast_2d(np.asarray(x, dtype=float))
    classes = np.atleast_1d(classe)
    n = len(X)
    z1 = X @ rede["W1"].T + rede["b1"]
    h = np.maximum(z1, 0)
    notas = h @ rede["W2"].T + rede["b2"]
    notas = notas - notas.max(axis=1, keepdims=True)
    p = np.exp(notas) / np.exp(notas).sum(axis=1, keepdims=True)
    y = np.eye(rede["W2"].shape[0])[classes]
    d_notas = (p - y) / n
    d_h = d_notas @ rede["W2"]
    d_z1 = d_h
    return {
        "W1": d_z1.T @ X,
        "b1": d_z1.sum(axis=0),
        "W2": d_notas.T @ h,
        "b2": d_notas.sum(axis=0),
    }
