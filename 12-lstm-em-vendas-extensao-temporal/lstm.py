"""Uma LSTM pequena no NumPy, com a volta pelo tempo e o Adam. Aula «LSTM em vendas: extensão temporal».

Este arquivo não imprime nada. As contas da célula são as da aula «RNN, LSTM e GRU» e as mesmas
da documentação do torch.nn.LSTM, na mesma ordem dos blocos (entrada i, esquecer f, valor g, saída o):

    i = sigmoide(W_ii x + b_ii + W_hi h + b_hi)      f = sigmoide(W_if x + b_if + W_hf h + b_hf)
    g = tanh(W_ig x + b_ig + W_hg h + b_hg)          o = sigmoide(W_io x + b_io + W_ho h + b_ho)
    c = f * c_anterior + i * g                       h = o * tanh(c)

Depois do último dia da janela, uma densa lê o h e o calendário do dia previsto e dá um número:
a venda do dia previsto, no log, centrada e dividida.
"""
import numpy as np


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


def nova_rede(entradas, estado, calendario, sorteio):
    """Pesos começando como no PyTorch: uniforme entre −1/raiz(estado) e +1/raiz(estado)."""
    rng = np.random.default_rng(sorteio)
    k = 1.0 / np.sqrt(estado)
    u = lambda *forma: rng.uniform(-k, k, size=forma)
    kd = 1.0 / np.sqrt(estado + calendario)
    return {"W_ih": u(4 * estado, entradas), "W_hh": u(4 * estado, estado),
            "b_ih": u(4 * estado), "b_hh": u(4 * estado),
            "w": rng.uniform(-kd, kd, size=estado + calendario), "b": rng.uniform(-kd, kd, size=1)}


def parametros(rede):
    return int(sum(v.size for v in rede.values()))


def frente(rede, X, C):
    """X: (exemplos, dias, entradas); C: (exemplos, calendário). Retorna a previsão e o que a volta precisa."""
    n, T, _ = X.shape
    H = rede["W_hh"].shape[0] // 4
    h = np.zeros((n, H))
    c = np.zeros((n, H))
    passos = []
    with np.errstate(all="ignore"):              # o NumPy do macOS avisa à toa no produto de matrizes
        for t in range(T):
            a = X[:, t] @ rede["W_ih"].T + rede["b_ih"] + h @ rede["W_hh"].T + rede["b_hh"]
            i, f = sigmoide(a[:, :H]), sigmoide(a[:, H:2 * H])
            g, o = np.tanh(a[:, 2 * H:3 * H]), sigmoide(a[:, 3 * H:])
            c_ant, h_ant = c, h
            c = f * c_ant + i * g
            tc = np.tanh(c)
            h = o * tc
            passos.append((X[:, t], h_ant, c_ant, i, f, g, o, tc))
        z = np.concatenate([h, C], axis=1)
        y = z @ rede["w"] + rede["b"][0]
    return y, (passos, z)


def perda(rede, X, C, y):
    prev, _ = frente(rede, X, C)
    return float(np.mean((prev - y) ** 2))


def gradientes(rede, X, C, y):
    """A perda (erro quadrático médio) e o gradiente de cada peso, pela volta pelo tempo."""
    prev, (passos, z) = frente(rede, X, C)
    n = len(y)
    H = rede["W_hh"].shape[0] // 4
    d_prev = 2.0 * (prev - y) / n
    g = {k: np.zeros_like(v) for k, v in rede.items()}
    with np.errstate(all="ignore"):
        g["w"] = z.T @ d_prev
        g["b"] = np.array([d_prev.sum()])
        dh = np.outer(d_prev, rede["w"][:H])
        dc = np.zeros_like(dh)
        for x, h_ant, c_ant, i, f, gg, o, tc in reversed(passos):
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            di, df, dg = dc * gg, dc * c_ant, dc * i
            da = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - gg ** 2), do * o * (1 - o)], axis=1)
            g["W_ih"] += da.T @ x
            g["W_hh"] += da.T @ h_ant
            g["b_ih"] += da.sum(axis=0)
            g["b_hh"] += da.sum(axis=0)
            dh = da @ rede["W_hh"]
            dc = dc * f
    return float(np.mean((prev - y) ** 2)), g


def checar_gradiente(rede, X, C, y, quantos=12, eps=1e-4, sorteio=0):
    """Compara o gradiente da volta com a diferença finita em alguns pesos sorteados. Retorna o maior erro relativo."""
    _, g = gradientes(rede, X, C, y)
    rng = np.random.default_rng(sorteio)
    pior = 0.0
    for nome in rede:
        for _ in range(quantos):
            idx = tuple(rng.integers(0, s) for s in rede[nome].shape)
            guardado = rede[nome][idx]
            rede[nome][idx] = guardado + eps
            mais = perda(rede, X, C, y)
            rede[nome][idx] = guardado - eps
            menos = perda(rede, X, C, y)
            rede[nome][idx] = guardado
            numerico = (mais - menos) / (2 * eps)
            pior = max(pior, abs(numerico - g[nome][idx]) / max(1e-8, abs(numerico) + abs(g[nome][idx])))
    return pior


def treinar(rede, treino, validacao, epocas=300, taxa=0.01):
    """Adam com todos os exemplos de treino em cada passo. Guarda a rede da época de menor perda de validação.

    treino e validacao: (X, C, y). Retorna a melhor rede, a época dela e a curva [(época, treino, validação)].
    """
    m = {k: np.zeros_like(v) for k, v in rede.items()}
    v = {k: np.zeros_like(v) for k, v in rede.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    melhor = (perda(rede, *validacao), 0, {k: x.copy() for k, x in rede.items()})
    curva = [(0, perda(rede, *treino), melhor[0])]
    for ep in range(1, epocas + 1):
        p_treino, g = gradientes(rede, *treino)
        for k in rede:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            rede[k] -= taxa * (m[k] / (1 - b1 ** ep)) / (np.sqrt(v[k] / (1 - b2 ** ep)) + eps)
        p_val = perda(rede, *validacao)
        curva.append((ep, p_treino, p_val))
        if p_val < melhor[0]:
            melhor = (p_val, ep, {k: x.copy() for k, x in rede.items()})
    return melhor[2], melhor[1], curva
