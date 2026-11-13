"""Uma rodada de treino com o passo e a penalização escolhidos, para a aula «Adam, dropout e penalização».

Usa as peças do mlp.py da aula «MLP de classificação» (pasta ../03-mlp-de-classificacao): as fotos,
a divisão, a rede, a avaliação e a cópia. Acrescenta o que essa aula ensina:

- o passo do Adam, ao lado do passo simples (SGD) da aula 03;
- o dropout nas camadas escondidas, só durante o treino;
- a penalização L2 e a L1 nos pesos W (os vieses b ficam fora).

Este arquivo não roda sozinho. O comparar.py usa ele.
"""
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "03-mlp-de-classificacao"))
import mlp  # noqa: E402


def frente_com_dropout(X, rede, p, rng):
    """Passagem direta de treino. Com p > 0, cada neurônio escondido é desligado com chance p.

    Quem sobra é multiplicado por 1 / (1 - p). Assim, na avaliação, a rede roda inteira e sem
    nenhuma conta a mais: é o que o mlp.avaliar faz.
    """
    guardado, mascaras = [X], []
    a = X
    for i, c in enumerate(rede):
        z = a @ c["W"].T + c["b"]
        if i < len(rede) - 1:
            guardado.append(z)
            a = np.maximum(z, 0)
            if p > 0:
                m = (rng.random(a.shape) >= p) / (1 - p)
                a = a * m
                mascaras.append(m)
            else:
                mascaras.append(None)
            guardado.append(a)
        else:
            a = z
    return a, guardado, mascaras


def gradientes(X, y, rede, p=0.0, l2=0.0, l1=0.0, rng=None):
    """Gradiente de: log-loss média do lote + l2/2 * soma(W²) + l1 * soma(|W|).

    É a backpropagation da aula 02. O dropout entra pela máscara; a penalização só soma um termo
    no gradiente de cada W.
    """
    n = len(X)
    notas, guardado, mascaras = frente_com_dropout(X, rede, p, rng)
    d = np.exp(mlp.log_probabilidades(notas))
    d[np.arange(n), y] -= 1
    d /= n
    g = [None] * len(rede)
    for i in range(len(rede) - 1, -1, -1):
        entrada = guardado[2 * i]
        gW = d.T @ entrada
        if l2:
            gW = gW + l2 * rede[i]["W"]
        if l1:
            gW = gW + l1 * np.sign(rede[i]["W"])
        g[i] = {"W": gW, "b": d.sum(axis=0)}
        if i > 0:
            d = d @ rede[i]["W"]
            if mascaras[i - 1] is not None:
                d = d * mascaras[i - 1]
            d = d * (guardado[2 * i - 1] > 0)
    return g


class SGD:
    """O passo da aula 03: peso novo = peso − taxa × gradiente."""

    def __init__(self, taxa):
        self.taxa = taxa

    def passo(self, rede, g):
        mlp.passo(rede, g, self.taxa)


class Adam:
    """O passo do Adam (Kingma e Ba, 2015), com os valores padrão do artigo.

    Para cada peso, guarda a média dos gradientes recentes (m) e a média dos quadrados (v).
    O passo é taxa × m / (raiz(v) + eps), com a correção das duas médias nos primeiros passos.
    """

    def __init__(self, taxa, beta1=0.9, beta2=0.999, eps=1e-8):
        self.taxa, self.beta1, self.beta2, self.eps = taxa, beta1, beta2, eps
        self.t, self.m, self.v = 0, None, None

    def passo(self, rede, g):
        if self.m is None:
            self.m = [{k: np.zeros_like(c[k]) for k in c} for c in rede]
            self.v = [{k: np.zeros_like(c[k]) for k in c} for c in rede]
        self.t += 1
        for c, gc, m, v in zip(rede, g, self.m, self.v):
            for k in c:
                m[k] = self.beta1 * m[k] + (1 - self.beta1) * gc[k]
                v[k] = self.beta2 * v[k] + (1 - self.beta2) * gc[k] ** 2
                m_hat = m[k] / (1 - self.beta1 ** self.t)
                v_hat = v[k] / (1 - self.beta2 ** self.t)
                c[k] -= self.taxa * m_hat / (np.sqrt(v_hat) + self.eps)


def otimizador(nome, taxa):
    if nome == "sgd":
        return SGD(taxa)
    if nome == "adam":
        return Adam(taxa)
    raise ValueError(f"otimizador tem que ser 'sgd' ou 'adam': veio {nome!r}")


def pesos(rede):
    """Tamanho dos pesos W: a raiz da soma dos quadrados e a fração de pesos perto de zero."""
    W = np.concatenate([c["W"].ravel() for c in rede])
    return float(np.sqrt((W ** 2).sum())), float((np.abs(W) < 1e-3).mean())


def rodar(config, X, y, i_treino, i_validacao, tamanhos, epocas, lote, sorteio=0):
    """Treina uma rede com a config e retorna o resumo da melhor época pela perda de validação.

    config: {"nome", "otimizador": "sgd" ou "adam", "taxa", "dropout", "l2", "l1"}.
    O sorteio fixa os pesos iniciais, a ordem das fotos e as máscaras do dropout.
    A perda da curva é só a log-loss, medida sem dropout e sem a penalização,
    para que todas as rodadas fiquem na mesma régua.
    """
    p = config.get("dropout", 0.0)
    if not 0 <= p < 1:
        raise ValueError(f"o dropout tem que ficar entre 0 e 1, sem chegar a 1: veio {p}")
    l2, l1 = config.get("l2", 0.0), config.get("l1", 0.0)
    if l2 < 0 or l1 < 0:
        raise ValueError("a penalização não pode ser negativa")
    rede = mlp.nova_rede(tamanhos, sorteio=sorteio)
    opt = otimizador(config["otimizador"], config["taxa"])
    rng_ordem = np.random.default_rng(sorteio)
    rng_drop = np.random.default_rng(1000 + sorteio)
    Xt, yt = X[i_treino], y[i_treino]
    curva = [mlp.linha_da_curva(0, rede, X, y, i_treino, i_validacao)]
    melhor_rede, melhor_epoca = mlp.copiar(rede), 0
    for epoca in range(1, epocas + 1):
        ordem = rng_ordem.permutation(len(Xt))
        for i in range(0, len(Xt), lote):
            idx = ordem[i:i + lote]
            opt.passo(rede, gradientes(Xt[idx], yt[idx], rede, p, l2, l1, rng_drop))
        try:
            curva.append(mlp.linha_da_curva(epoca, rede, X, y, i_treino, i_validacao))
        except FloatingPointError:
            curva.append({"epoca": epoca, "perda_treino": float("inf"), "acerto_treino": float("nan"),
                          "perda_validacao": float("inf"), "acerto_validacao": float("nan")})
            break
        if curva[-1]["perda_validacao"] < curva[melhor_epoca]["perda_validacao"]:
            melhor_rede, melhor_epoca = mlp.copiar(rede), epoca
    c = curva[melhor_epoca]
    norma, zeros = pesos(melhor_rede)
    return {"nome": config["nome"], "melhor_epoca": melhor_epoca, "perda_treino": c["perda_treino"],
            "perda_validacao": c["perda_validacao"], "acerto_validacao": c["acerto_validacao"],
            "distancia": c["perda_validacao"] - c["perda_treino"],
            "norma": norma, "zeros": zeros, "epocas_treinadas": len(curva) - 1,
            "curva": curva, "rede": melhor_rede}
