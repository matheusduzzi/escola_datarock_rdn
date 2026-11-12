"""Testes da tarefa 3. Rode de dentro da pasta 00-apresentacao:

    python3 testar_tarefa3.py

Para testar outro arquivo com as mesmas funções: python3 testar_tarefa3.py caminho/do/arquivo.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

arquivo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'tarefa3.py'
spec = importlib.util.spec_from_file_location('tarefa3', arquivo)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

falhas = []


def teste(nome, condicao):
    print(('ok    ' if condicao else 'FALHA ') + nome)
    if not condicao:
        falhas.append(nome)


z = np.array([[2.0, 1.0, 0.1], [0.5, 0.5, 3.0], [1.0, 3.0, 0.0]])
with np.errstate(all='ignore'):
    p = mod.softmax(z)
    teste('softmax mantém o formato (3, 3)', np.shape(p) == (3, 3))
    teste('cada linha soma 1', np.allclose(np.sum(p, axis=1), 1))
    teste('probabilidades entre 0 e 1', bool(np.all((np.asarray(p) >= 0) & (np.asarray(p) <= 1))))
    teste('a maior nota tem a maior probabilidade', np.array_equal(np.argmax(p, axis=1), [0, 2, 1]))
    teste('valor da primeira linha', np.allclose(p[0], np.exp([2.0, 1.0, 0.1]) / np.exp([2.0, 1.0, 0.1]).sum()))
    g = mod.softmax(np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1000.0, -1000.0]]))
    teste('notas grandes ou muito negativas não dão nan nem inf', bool(np.all(np.isfinite(g))))
    teste('somar a mesma constante a uma linha não muda a probabilidade', np.allclose(g[0], mod.softmax(np.array([[0.0, 1.0, 2.0]]))[0]))
    teste('aceita lista de listas', np.allclose(mod.softmax([[0.0, 0.0]]), [[0.5, 0.5]]))
    y = np.array([0, 2, 0])
    perda = mod.log_loss(p, y)
    esperado = -np.mean(np.log([p[0, 0], p[1, 2], p[2, 0]]))
    teste('log_loss usa só a probabilidade da classe certa', np.isclose(perda, esperado))
    teste('log_loss de quem acerta com certeza é 0', np.isclose(mod.log_loss(np.array([[1.0, 0.0]]), np.array([0])), 0))
    zero = mod.log_loss(np.array([[1.0, 0.0], [0.5, 0.5]]), np.array([1, 0]))
    teste('probabilidade 0 na classe certa dá número finito', bool(np.isfinite(zero)))
    teste('o piso é 1e-12', np.isclose(zero, (-np.log(1e-12) - np.log(0.5)) / 2))

print(f'{12 - len(falhas)} de 12 testes passaram')
sys.exit(1 if falhas else 0)
