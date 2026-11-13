"""As peças da aula «Reconstruir imagens»: o autoencoder, o VAE e o mesmo treino para os dois.

Os outros arquivos da pasta usam este. Ele não roda sozinho.

As duas redes têm a mesma planta, com camadas densas como nas aulas 01 a 03:
  codificador   784 pixels -> 256 (ReLU) -> código
  decodificador código -> 256 (ReLU) -> 784 (sigmoide: a probabilidade de cada pixel estar aceso)
No autoencoder, o código é um ponto: CODIGO números. No VAE, o codificador devolve o centro (mu) e o
log da variância (log_var) de cada número do código, e o código é sorteado: z = mu + sigma x eps.

A perda é a da aula «Autoencoder e VAE», somada nos 784 pixels e nos números do código, e com média
nas fotos do lote:
  autoencoder  Bernoulli
  VAE          Bernoulli + KL até N(0, 1)
As fotos entram com cada pixel entre 0 e 1, sem centralizar: a perda de Bernoulli pede isso.
"""
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
DADOS = AQUI.parent / "dados" / "fashion-mnist"
PESOS = AQUI / "pesos"
sys.path.insert(0, str(DADOS))
from baixar import CLASSES, carregar  # noqa: E402

# O NumPy 2.0 com a biblioteca Accelerate do Mac avisa "overflow in matmul" sem ter overflow.
# Os avisos ficam desligados; quem pega infinito ou NaN de verdade é o treino, que para.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

OCULTA = 256
# O orçamento comum: o que fica igual para as duas redes.
ORCAMENTO = {"fotos_treino": 50000, "fotos_validacao": 10000, "sorteio_divisao": 0, "lote": 128,
             "epocas": 20, "taxa": 0.001, "sorteio_pesos": 0, "sorteio_ordem": 0, "sorteio_eps": 0}
NOMES = {"autoencoder": "Autoencoder", "vae": "VAE"}


def fotos(pasta=DADOS):
    """As 60.000 fotos de treino e as 10.000 de teste em (fotos, 784), cada pixel entre 0 e 1."""
    try:
        xt, yt, xs, ys = carregar(pasta)
    except FileNotFoundError:
        raise SystemExit("As fotos não estão em ../dados/fashion-mnist. Rode python3 baixar.py dentro dessa pasta.")
    return (xt.reshape(-1, 784) / 255.0, yt.astype(np.int64), xs.reshape(-1, 784) / 255.0, ys.astype(np.int64))


def separar(n, n_validacao, sorteio=0):
    """A mesma divisão da aula 03: embaralha 0..n-1 e as primeiras n_validacao posições vão para a validação."""
    if not 0 < n_validacao < n:
        raise ValueError(f"n_validacao tem que ficar entre 1 e {n - 1}: veio {n_validacao}")
    ordem = np.random.default_rng(sorteio).permutation(n)
    return ordem[n_validacao:], ordem[:n_validacao]


def preparar(orcamento=ORCAMENTO):
    """Retorna Xt, yt, Xv, yv, X_teste, y_teste: treino, validação e teste, com a divisão da aula 03."""
    X, y, X_teste, y_teste = fotos()
    i_treino, i_validacao = separar(len(X), orcamento["fotos_validacao"], orcamento["sorteio_divisao"])
    i_treino = i_treino[:orcamento["fotos_treino"]]
    return X[i_treino], y[i_treino], X[i_validacao], y[i_validacao], X_teste, y_teste


# ---------------------------------------------------------------- as redes

def nova_rede(tipo, codigo, sorteio=0):
    """Pesos iniciais. ReLU depois: regra de He, desvio raiz(2 / entradas). Sem ReLU: raiz(1 / entradas)."""
    if tipo not in NOMES:
        raise ValueError(f"tipo tem que ser autoencoder ou vae: veio {tipo!r}")
    if not 1 <= codigo <= 256:
        raise ValueError(f"o código tem que ter de 1 a 256 números: veio {codigo}")
    rng = np.random.default_rng(sorteio)

    def densa(saidas, entradas, ganho):
        return rng.normal(0, np.sqrt(ganho / entradas), (saidas, entradas)), np.zeros(saidas)

    rede = {}
    rede["W1"], rede["b1"] = densa(OCULTA, 784, 2)
    rede["Wm"], rede["bm"] = densa(codigo, OCULTA, 1)
    if tipo == "vae":
        rede["Wv"], rede["bv"] = densa(codigo, OCULTA, 1)
    rede["W3"], rede["b3"] = densa(OCULTA, codigo, 2)
    rede["W4"], rede["b4"] = densa(784, OCULTA, 1)
    return rede


def tipo_de(rede):
    return "vae" if "Wv" in rede else "autoencoder"


def parametros(rede):
    return int(sum(v.size for v in rede.values()))


def relu(a):
    return np.maximum(a, 0.0)


def sigmoide(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))


def codificar(rede, x):
    """Autoencoder: retorna o código. VAE: retorna (mu, log_var)."""
    h = relu(np.dot(x, rede["W1"].T) + rede["b1"])
    mu = np.dot(h, rede["Wm"].T) + rede["bm"]
    if "Wv" not in rede:
        return mu
    return mu, np.dot(h, rede["Wv"].T) + rede["bv"]


def decodificar(rede, z):
    """Retorna a probabilidade de cada pixel estar aceso: a foto refeita, entre 0 e 1."""
    h = relu(np.dot(z, rede["W3"].T) + rede["b3"])
    return sigmoide(np.dot(h, rede["W4"].T) + rede["b4"])


def refazer(rede, x):
    """A foto refeita. No VAE, a partir do centro mu, sem sorteio: é a reconstrução de comparar."""
    c = codificar(rede, x)
    return decodificar(rede, c[0] if isinstance(c, tuple) else c)


def erro_quadrado(x, x_rec):
    """Soma dos erros ao quadrado de cada foto, nos 784 pixels: a régua para comparar as duas redes."""
    return ((x - x_rec) ** 2).sum(axis=1)


def kl_por_numero(mu, log_var):
    """A KL entre N(mu, sigma²) e N(0, 1), número por número do código (a fórmula da aula 13)."""
    return 0.5 * (mu ** 2 + np.exp(log_var) - 1.0 - log_var)


def bernoulli_logits(x, a):
    """-log da probabilidade de Bernoulli, somada nos pixels, a partir dos logits a (antes da sigmoide).

    É log(1 + e^a) - x * a, escrito de um jeito que não estoura com |a| grande.
    """
    return (np.maximum(a, 0) - x * a + np.log1p(np.exp(-np.abs(a)))).sum(axis=1)


def passada(rede, x, eps=None):
    """Ida e volta num lote. Retorna (perda média por foto, reconstrução média, KL média, gradientes).

    No VAE, eps é o sorteio de N(0, 1) com o formato de mu. Sem eps, o VAE usa z = mu (só para medir).
    """
    n = len(x)
    a1 = np.dot(x, rede["W1"].T) + rede["b1"]
    h1 = relu(a1)
    mu = np.dot(h1, rede["Wm"].T) + rede["bm"]
    vae = "Wv" in rede
    if vae:
        log_var = np.dot(h1, rede["Wv"].T) + rede["bv"]
        sigma = np.exp(0.5 * log_var)
        e = np.zeros_like(mu) if eps is None else eps
        z = mu + sigma * e
    else:
        z = mu
    a3 = np.dot(z, rede["W3"].T) + rede["b3"]
    h3 = relu(a3)
    a4 = np.dot(h3, rede["W4"].T) + rede["b4"]
    rec = bernoulli_logits(x, a4)
    kl = kl_por_numero(mu, log_var).sum(axis=1) if vae else np.zeros(n)
    perda = float((rec + kl).mean())

    g = {}
    d4 = (sigmoide(a4) - x) / n
    g["W4"], g["b4"] = np.dot(d4.T, h3), d4.sum(axis=0)
    d3 = np.dot(d4, rede["W4"]) * (a3 > 0)
    g["W3"], g["b3"] = np.dot(d3.T, z), d3.sum(axis=0)
    dz = np.dot(d3, rede["W3"])
    if vae:
        dmu = dz + mu / n
        dlv = dz * e * 0.5 * sigma + 0.5 * (np.exp(log_var) - 1.0) / n
        g["Wv"], g["bv"] = np.dot(dlv.T, h1), dlv.sum(axis=0)
        dh1 = np.dot(dmu, rede["Wm"]) + np.dot(dlv, rede["Wv"])
    else:
        dmu = dz
        dh1 = np.dot(dmu, rede["Wm"])
    g["Wm"], g["bm"] = np.dot(dmu.T, h1), dmu.sum(axis=0)
    d1 = dh1 * (a1 > 0)
    g["W1"], g["b1"] = np.dot(d1.T, x), d1.sum(axis=0)
    return perda, float(rec.mean()), float(kl.mean()), g


class Adam:
    """O Adam da aula «Adam, dropout e penalização», com os valores de sempre: 0,9, 0,999 e 1e-8."""

    def __init__(self, taxa, b1=0.9, b2=0.999, eps=1e-8):
        self.taxa, self.b1, self.b2, self.eps, self.t, self.m, self.v = taxa, b1, b2, eps, 0, {}, {}

    def passo(self, rede, g):
        self.t += 1
        for k in rede:
            self.m[k] = self.b1 * self.m.get(k, 0) + (1 - self.b1) * g[k]
            self.v[k] = self.b2 * self.v.get(k, 0) + (1 - self.b2) * g[k] ** 2
            m = self.m[k] / (1 - self.b1 ** self.t)
            v = self.v[k] / (1 - self.b2 ** self.t)
            rede[k] = rede[k] - self.taxa * m / (np.sqrt(v) + self.eps)


def avaliar(rede, x, sorteio=0, lote=2000):
    """Perda média por foto, reconstrução média e KL média num conjunto, com um sorteio de eps fixo."""
    rng = np.random.default_rng(sorteio)
    soma = np.zeros(3)
    for i in range(0, len(x), lote):
        xb = x[i:i + lote]
        eps = rng.standard_normal((len(xb), rede["Wm"].shape[0])) if "Wv" in rede else None
        p, r, k, _ = passada(rede, xb, eps)
        soma += np.array([p, r, k]) * len(xb)
    return tuple(float(s) for s in soma / len(x))


def treinar(tipo, codigo, Xt, Xv, orcamento=ORCAMENTO, falar=print):
    """Treina com o orçamento comum. Guarda os pesos da época de menor perda de validação.

    Retorna (melhores pesos, curva): curva tem uma linha por época com
    (época, perda de treino, perda de validação, reconstrução de validação, KL de validação).
    """
    rede = nova_rede(tipo, codigo, orcamento["sorteio_pesos"])
    otim = Adam(orcamento["taxa"])
    ordem_rng = np.random.default_rng(orcamento["sorteio_ordem"])
    eps_rng = np.random.default_rng(orcamento["sorteio_eps"])
    lote, curva, melhor, melhor_v = orcamento["lote"], [], None, np.inf
    for epoca in range(1, orcamento["epocas"] + 1):
        t0 = time.time()
        ordem = ordem_rng.permutation(len(Xt))
        soma = 0.0
        for i in range(0, len(Xt), lote):
            xb = Xt[ordem[i:i + lote]]
            eps = eps_rng.standard_normal((len(xb), codigo)) if tipo == "vae" else None
            p, _, _, g = passada(rede, xb, eps)
            if not np.isfinite(p):
                raise SystemExit(f"a perda virou {p} na época {epoca}: o treino parou")
            otim.passo(rede, g)
            soma += p * len(xb)
        pv, rv, kv = avaliar(rede, Xv)
        curva.append((epoca, soma / len(Xt), pv, rv, kv))
        if pv < melhor_v:
            melhor_v, melhor = pv, {k: v.copy() for k, v in rede.items()}
        falar(f"  época {epoca:>2}: treino {soma / len(Xt):8.2f}   validação {pv:8.2f}"
              f"   (reconstrução {rv:7.2f}, KL {kv:6.2f})   {time.time() - t0:4.1f} s")
    return melhor, curva


def arquivo_pesos(tipo, codigo, pasta=PESOS):
    return Path(pasta) / f"{tipo}-codigo-{codigo}.npz"


def gravar(rede, curva, tipo, codigo, pasta, orcamento=ORCAMENTO):
    pasta = Path(pasta)
    pasta.mkdir(exist_ok=True)
    np.savez(arquivo_pesos(tipo, codigo, pasta), curva=np.array(curva),
             orcamento=np.array([orcamento[k] for k in sorted(orcamento)], dtype=float),
             **{k: v.astype(np.float32) for k, v in rede.items()})


def ler(tipo, codigo, pasta=PESOS, orcamento=ORCAMENTO):
    """Os pesos gravados e a curva. Recusa pesos de outro orçamento ou de outra rede."""
    arq = arquivo_pesos(tipo, codigo, pasta)
    if not arq.exists():
        raise SystemExit(f"Não achei {arq.name} em {Path(pasta).name}/. Treine antes: "
                         f"python3 treinar.py {tipo} --codigo {codigo}")
    d = np.load(arq)
    esperado = np.array([orcamento[k] for k in sorted(orcamento)], dtype=float)
    if not np.array_equal(d["orcamento"], esperado):
        raise SystemExit(f"{arq.name} foi treinado com outro orçamento; a comparação recusa.")
    rede = {k: d[k].astype(np.float64) for k in d.files if k not in ("curva", "orcamento")}
    if tipo_de(rede) != tipo or rede["Wm"].shape[0] != codigo:
        raise SystemExit(f"{arq.name} não é um {tipo} com código de {codigo} números.")
    return rede, d["curva"]
