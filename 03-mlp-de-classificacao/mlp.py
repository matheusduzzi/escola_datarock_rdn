"""As peças do ciclo de treino da aula «MLP de classificação».

Fotos do Fashion MNIST, separação em treino e validação, a rede, os gradientes, o passo,
a avaliação e o ciclo de treino com a curva por época. O treinar.py e o meu_treino.py usam
este arquivo. Ele não roda sozinho.

A rede é uma lista de camadas. Cada camada é um dicionário com W no formato (saídas, entradas)
e b, como nas aulas 01 e 02. Todas as camadas, menos a última, passam pela ReLU.
"""
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
DADOS = AQUI.parent / "dados" / "fashion-mnist"
sys.path.insert(0, str(DADOS))
from baixar import CLASSES, carregar  # noqa: E402

# O NumPy 2.0 com a biblioteca Accelerate do Mac avisa "overflow in matmul" sem ter overflow.
# Os avisos ficam desligados; quem pega infinito ou NaN de verdade é o avaliar(), que para o treino.
np.seterr(divide="ignore", over="ignore", invalid="ignore")


def fotos(pasta=DADOS):
    """As 60.000 fotos de treino e as 10.000 de teste, com cada pixel entre 0 e 1."""
    try:
        xt, yt, xs, ys = carregar(pasta)
    except FileNotFoundError:
        raise SystemExit("As fotos não estão em ../dados/fashion-mnist. Rode python3 baixar.py dentro dessa pasta.")
    return (xt.reshape(len(xt), -1) / 255.0, yt.astype(np.int64),
            xs.reshape(len(xs), -1) / 255.0, ys.astype(np.int64))


def separar(n, n_validacao, sorteio=0):
    """Embaralha os números de 0 a n - 1 e separa as primeiras n_validacao posições para a validação."""
    if not 0 < n_validacao < n:
        raise ValueError(f"n_validacao tem que ficar entre 1 e {n - 1}: veio {n_validacao}")
    ordem = np.random.default_rng(sorteio).permutation(n)
    return ordem[n_validacao:], ordem[:n_validacao]


def medidas_do_treino(X, i_treino):
    """Média de cada pixel e desvio de todos os pixels, só nas fotos de treino.

    Use (X - media) / desvio em treino, validação e teste. Validação e teste não entram na conta.
    """
    return X[i_treino].mean(axis=0), float(X[i_treino].std())


def nova_rede(tamanhos, sorteio=0):
    """Pesos sorteados numa normal com desvio raiz(2 / entradas), viés zero.

    tamanhos = [784, 128, 10] é uma camada escondida de 128 neurônios; [784, 10] é sem camada escondida.
    """
    rng = np.random.default_rng(sorteio)
    return [{"W": rng.normal(0, np.sqrt(2 / ent), size=(sai, ent)), "b": np.zeros(sai)}
            for ent, sai in zip(tamanhos[:-1], tamanhos[1:])]


def copiar(rede):
    return [{k: v.copy() for k, v in camada.items()} for camada in rede]


def parametros(rede):
    return sum(c["W"].size + c["b"].size for c in rede)


def frente(X, rede):
    """Passagem direta. Retorna as notas e o que o caminho de trás precisa."""
    guardado = [X]
    a = X
    for i, c in enumerate(rede):
        z = a @ c["W"].T + c["b"]
        if i < len(rede) - 1:
            guardado.append(z)
            a = np.maximum(z, 0)
            guardado.append(a)
        else:
            a = z
    return a, guardado


def log_probabilidades(notas):
    z = notas - notas.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def gradientes(X, y, rede):
    """Backpropagation da aula 02 num lote: gradiente da log-loss média de cada camada."""
    n = len(X)
    notas, guardado = frente(X, rede)
    d = np.exp(log_probabilidades(notas))
    d[np.arange(n), y] -= 1
    d /= n                                          # perda -> notas, média do lote
    g = [None] * len(rede)
    for i in range(len(rede) - 1, -1, -1):
        entrada = guardado[2 * i]                   # o que entrou na camada i
        g[i] = {"W": d.T @ entrada, "b": d.sum(axis=0)}
        if i > 0:
            d = (d @ rede[i]["W"]) * (guardado[2 * i - 1] > 0)  # volta e passa pela ReLU
    return g


def passo(rede, g, taxa):
    for c, gc in zip(rede, g):
        c["W"] -= taxa * gc["W"]
        c["b"] -= taxa * gc["b"]


def avaliar(X, y, rede, lote=10000):
    """Perda média, acerto e a classe prevista de cada foto."""
    perdas, previstas = [], []
    for i in range(0, len(X), lote):
        notas, _ = frente(X[i:i + lote], rede)
        lp = log_probabilidades(notas)
        perdas.append(-lp[np.arange(len(lp)), y[i:i + lote]])
        previstas.append(notas.argmax(axis=1))
    perdas, previstas = np.concatenate(perdas), np.concatenate(previstas)
    if not np.isfinite(perdas).all():
        raise FloatingPointError("a perda virou infinito ou NaN: a taxa de aprendizado está alta demais?")
    return float(perdas.mean()), float((previstas == y).mean()), previstas


def uma_epoca(rede, X, y, taxa, lote, rng):
    """Uma passada por todas as fotos de treino, em lotes, numa ordem nova a cada época."""
    if lote < 1:
        raise ValueError(f"o lote tem que ter pelo menos 1 foto: veio {lote}")
    ordem = rng.permutation(len(X))
    passos = 0
    for i in range(0, len(X), lote):
        idx = ordem[i:i + lote]
        passo(rede, gradientes(X[idx], y[idx], rede), taxa)
        passos += 1
    return passos


def linha_da_curva(epoca, rede, X, y, i_treino, i_validacao):
    pt, at, _ = avaliar(X[i_treino], y[i_treino], rede)
    pv, av, _ = avaliar(X[i_validacao], y[i_validacao], rede)
    return {"epoca": epoca, "perda_treino": pt, "acerto_treino": at, "perda_validacao": pv, "acerto_validacao": av}


def ciclo(rede, X, y, i_treino, i_validacao, epocas, taxa, lote, sorteio=0):
    """Treina por um número fixo de épocas e guarda a cópia da rede na época de menor perda de validação.

    Retorna (melhor_rede, melhor_epoca, curva). A curva começa na época 0, antes do primeiro passo.
    """
    rng = np.random.default_rng(sorteio)
    curva = [linha_da_curva(0, rede, X, y, i_treino, i_validacao)]
    melhor_rede, melhor_epoca = copiar(rede), 0
    for epoca in range(1, epocas + 1):
        uma_epoca(rede, X[i_treino], y[i_treino], taxa, lote, rng)
        curva.append(linha_da_curva(epoca, rede, X, y, i_treino, i_validacao))
        if curva[-1]["perda_validacao"] < curva[melhor_epoca]["perda_validacao"]:
            melhor_rede, melhor_epoca = copiar(rede), epoca
    return melhor_rede, melhor_epoca, curva


def foto_media(X_treino, y_treino, X, k=10):
    """Classifica cada foto pela classe cuja foto média, no treino, está mais perto."""
    centros = np.stack([X_treino[y_treino == c].mean(axis=0) for c in range(k)])
    d = (X ** 2).sum(1)[:, None] - 2 * X @ centros.T + (centros ** 2).sum(1)[None, :]
    return d.argmin(axis=1)


def acerto_por_tipo(y, previstas, k=10):
    return [float((previstas[y == c] == c).mean()) for c in range(k)]


def maior_confusao(y, previstas, k=10):
    m = np.zeros((k, k), dtype=int)
    np.add.at(m, (y, previstas), 1)
    np.fill_diagonal(m, 0)
    i, j = np.unravel_index(m.argmax(), m.shape)
    return int(i), int(j), int(m[i, j])


def gravar_curva(curva, caminho):
    linhas = ["epoca,perda_treino,acerto_treino,perda_validacao,acerto_validacao"]
    for c in curva:
        linhas.append(f"{c['epoca']},{c['perda_treino']:.4f},{c['acerto_treino']:.4f},"
                      f"{c['perda_validacao']:.4f},{c['acerto_validacao']:.4f}")
    Path(caminho).write_text("\n".join(linhas) + "\n", encoding="utf-8")


def mostrar_curva(curva):
    print("  época  perda treino  acerto treino  perda validação  acerto validação")
    for c in curva:
        print(f"  {c['epoca']:>5}  {c['perda_treino']:>12.4f}  {c['acerto_treino']:>13.4f}  "
              f"{c['perda_validacao']:>15.4f}  {c['acerto_validacao']:>16.4f}")
