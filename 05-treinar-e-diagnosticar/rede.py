"""As peças de treino da aula «Treinar e diagnosticar», com os instrumentos de diagnóstico.

É o mesmo ciclo da aula «MLP de classificação» (fotos do Fashion MNIST, rede em camadas, backpropagation
da aula 02, passo de gradiente, curva por época), com três medidas a mais em cada época:

- a norma do gradiente de cada camada, no fim da época, nas 500 primeiras fotos de treino;
- a razão entre o tamanho do passo e o tamanho dos pesos, por camada;
- a fração de neurônios apagados de cada camada escondida (ReLU zero em todas as fotos de validação).

O diagnosticar.py, o treino_do_caio.py e o testar_minha_correcao.py usam este arquivo. Ele não roda sozinho.

Uma configuração de treino é um dicionário:

    {"camadas": [784, 128, 10],   # tamanhos, da entrada à saída
     "inicio": "he",               # "he" (desvio raiz(2 / entradas)) ou um número (desvio fixo)
     "taxa": 0.05, "lote": 64, "epocas": 10,
     "fotos_treino": 10000,        # quantas fotos de treino usar (as primeiras do sorteio)
     "l2": 0.0,                    # penalização dos pesos; 0 desliga
     "escolher_epoca": True}       # True: devolve a época de menor perda de validação; False: a última
"""
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).absolute().parent
DADOS = AQUI.parent / "dados" / "fashion-mnist"
sys.path.insert(0, str(DADOS))
from baixar import CLASSES, carregar  # noqa: E402

# O NumPy 2.0 com a biblioteca Accelerate do Mac avisa "overflow in matmul" sem ter overflow.
# Os avisos ficam desligados; quem pega infinito ou NaN de verdade é o treinar(), que para e anota.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

CHAVES = ("camadas", "inicio", "taxa", "lote", "epocas", "fotos_treino", "l2", "escolher_epoca")


def preparar(n_validacao=10000, sorteio=0, pasta=DADOS):
    """Fotos de treino e validação, já centralizadas com a média e o desvio só do treino.

    Mesma divisão da aula «MLP de classificação»: o sorteio 0 separa 10.000 das 60.000 fotos para a
    validação. As fotos de teste não são carregadas aqui.
    """
    try:
        xt, yt, _, _ = carregar(pasta)
    except FileNotFoundError:
        raise SystemExit("As fotos não estão em ../dados/fashion-mnist. Rode python3 baixar.py dentro dessa pasta.")
    X = xt.reshape(len(xt), -1) / 255.0
    y = yt.astype(np.int64)
    ordem = np.random.default_rng(sorteio).permutation(len(X))
    i_val, i_tre = ordem[:n_validacao], ordem[n_validacao:]
    media, desvio = X[i_tre].mean(axis=0), float(X[i_tre].std())
    X = (X - media) / desvio
    return X[i_tre], y[i_tre], X[i_val], y[i_val]


def conferir_config(config):
    falta = [k for k in CHAVES if k not in config]
    sobra = [k for k in config if k not in CHAVES]
    if falta or sobra:
        raise ValueError(f"configuração com chaves faltando {falta} ou a mais {sobra}")
    c = config["camadas"]
    if len(c) < 2 or c[0] != 784 or c[-1] != 10 or any(int(t) != t or t < 1 for t in c):
        raise ValueError(f"camadas tem que começar em 784 e terminar em 10: veio {c}")
    if not (config["inicio"] == "he" or (isinstance(config["inicio"], (int, float)) and config["inicio"] > 0)):
        raise ValueError(f'inicio tem que ser "he" ou um desvio positivo: veio {config["inicio"]!r}')
    if not config["taxa"] > 0:
        raise ValueError(f"a taxa tem que ser positiva: veio {config['taxa']}")
    if int(config["lote"]) != config["lote"] or config["lote"] < 1:
        raise ValueError(f"o lote tem que ter pelo menos 1 foto: veio {config['lote']}")
    if int(config["epocas"]) != config["epocas"] or config["epocas"] < 1:
        raise ValueError(f"epocas tem que ser um inteiro positivo: veio {config['epocas']}")
    if config["l2"] < 0:
        raise ValueError(f"l2 não pode ser negativo: veio {config['l2']}")
    if not isinstance(config["escolher_epoca"], bool):
        raise ValueError("escolher_epoca tem que ser True ou False")


def nova_rede(camadas, inicio="he", sorteio=0):
    """Pesos sorteados numa normal; viés zero. inicio="he" usa desvio raiz(2 / entradas)."""
    rng = np.random.default_rng(sorteio)
    rede = []
    for ent, sai in zip(camadas[:-1], camadas[1:]):
        desvio = np.sqrt(2 / ent) if inicio == "he" else float(inicio)
        rede.append({"W": rng.normal(0, desvio, size=(sai, ent)), "b": np.zeros(sai)})
    return rede


def copiar(rede):
    return [{k: v.copy() for k, v in c.items()} for c in rede]


def frente(X, rede):
    """Passagem direta. Retorna as notas e as entradas e somas de cada camada (para a volta)."""
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


def gradientes(X, y, rede, l2=0.0):
    """Backpropagation da aula 02 num lote. Com l2 > 0, soma l2 × W ao gradiente de cada W."""
    n = len(X)
    notas, guardado = frente(X, rede)
    d = np.exp(log_probabilidades(notas))
    d[np.arange(n), y] -= 1
    d /= n
    g = [None] * len(rede)
    for i in range(len(rede) - 1, -1, -1):
        g[i] = {"W": d.T @ guardado[2 * i] + l2 * rede[i]["W"], "b": d.sum(axis=0)}
        if i > 0:
            d = (d @ rede[i]["W"]) * (guardado[2 * i - 1] > 0)
    return g


def avaliar(X, y, rede, lote=10000):
    """Perda média e acerto. A perda pode sair infinita ou NaN; quem decide o que fazer é quem chama."""
    perdas, certas = [], 0
    for i in range(0, len(X), lote):
        notas, _ = frente(X[i:i + lote], rede)
        lp = log_probabilidades(notas)
        perdas.append(-lp[np.arange(len(lp)), y[i:i + lote]])
        certas += int((notas.argmax(axis=1) == y[i:i + lote]).sum())
    return float(np.concatenate(perdas).mean()), certas / len(X)


def norma(m):
    return float(np.sqrt((m * m).sum()))


def apagados(X, rede):
    """Fração de neurônios de cada camada escondida que dão 0 (ReLU) em todas as fotos de X."""
    _, guardado = frente(X, rede)
    return [float((guardado[2 * i + 2] <= 0).all(axis=0).mean()) for i in range(len(rede) - 1)]


def treinar(config, Xt, yt, Xv, yv, sorteio=0):
    """Treina com a configuração e retorna (rede escolhida, época escolhida, curva).

    Cada linha da curva tem perda e acerto de treino e de validação, a fração de apagados (nas fotos de
    validação) e, no fim da época, a norma do gradiente e a razão passo/peso de cada camada, medidas
    nas 500 primeiras fotos de treino. A época 0 é antes do primeiro passo. Se a perda virar infinito ou NaN, o treino para e a última linha fica com
    "parou": True.
    """
    conferir_config(config)
    n = int(config["fotos_treino"])
    if not 0 < n <= len(Xt):
        raise ValueError(f"fotos_treino tem que ficar entre 1 e {len(Xt)}: veio {n}")
    Xt, yt = Xt[:n], yt[:n]
    rede = nova_rede(config["camadas"], config["inicio"], sorteio)
    rng = np.random.default_rng(sorteio + 1)
    taxa, lote, l2 = config["taxa"], int(config["lote"]), config["l2"]

    def linha(epoca):
        pt, at = avaliar(Xt, yt, rede)
        pv, av = avaliar(Xv, yv, rede)
        g = gradientes(Xt[:500], yt[:500], rede, l2)
        reg = {"epoca": epoca, "perda_treino": pt, "acerto_treino": at, "perda_validacao": pv,
               "acerto_validacao": av, "apagados": apagados(Xv, rede),
               "norma_gradiente": [norma(gc["W"]) for gc in g],
               "passo_por_peso": [taxa * norma(gc["W"]) / max(norma(c["W"]), 1e-300) for gc, c in zip(g, rede)],
               "parou": not (np.isfinite(pt) and np.isfinite(pv))}
        return reg

    curva = [linha(0)]
    melhor, melhor_epoca = copiar(rede), 0
    for epoca in range(1, int(config["epocas"]) + 1):
        ordem = rng.permutation(n)
        for i in range(0, n, lote):
            idx = ordem[i:i + lote]
            g = gradientes(Xt[idx], yt[idx], rede, l2)
            for c, gc in zip(rede, g):
                c["W"] -= taxa * gc["W"]
                c["b"] -= taxa * gc["b"]
        curva.append(linha(epoca))
        if curva[-1]["parou"]:
            break
        if curva[-1]["perda_validacao"] < curva[melhor_epoca]["perda_validacao"]:
            melhor, melhor_epoca = copiar(rede), epoca
    if not config["escolher_epoca"]:
        melhor, melhor_epoca = rede, curva[-1]["epoca"]
    return melhor, melhor_epoca, curva


def numero(v, casas=4):
    """Número para a tabela: NaN e infinito aparecem como texto, o resto com casas fixas."""
    if not np.isfinite(v):
        return "nan" if np.isnan(v) else "inf"
    if abs(v) >= 1000:
        return f"{v:.1e}"
    return f"{v:.{casas}f}"


def cientifico(v):
    return f"{v:.1e}" if np.isfinite(v) else ("nan" if np.isnan(v) else "inf")


def mostrar(curva, epocas=None, gradiente=True):
    """Imprime a curva. epocas limita às épocas listadas; gradiente=False esconde as medidas por camada."""
    cab = "  época  perda treino  acerto treino  perda validação  acerto validação"
    if gradiente:
        cab += "  norma do gradiente por camada  apagados"
    print(cab)
    for r in curva:
        if epocas is not None and r["epoca"] not in epocas and not r["parou"]:
            continue
        s = (f"  {r['epoca']:>5}  {numero(r['perda_treino']):>12}  {r['acerto_treino']:>13.4f}  "
             f"{numero(r['perda_validacao']):>15}  {r['acerto_validacao']:>16.4f}")
        if gradiente:
            ng = " ".join(cientifico(v) for v in r["norma_gradiente"])
            ap = " ".join(f"{v:.2f}" for v in r["apagados"])
            s += f"  {ng:>29}  {ap}"
        print(s)


def resumo(curva, escolhida):
    """As linhas que um registro de falhas precisa: época 0, época escolhida, última, apagados e pulos."""
    c0, ce, cu = curva[0], curva[escolhida], curva[-1]
    print(f"  perda na época 0: treino {numero(c0['perda_treino'])}, validação {numero(c0['perda_validacao'])}")
    print("  norma do gradiente na época 0, por camada: " + " ".join(cientifico(v) for v in c0["norma_gradiente"]))
    print("  passo/peso na época 0, por camada: " + " ".join(cientifico(v) for v in c0["passo_por_peso"]))
    if cu["parou"]:
        print(f"  o treino parou na época {cu['epoca']}: a perda virou {numero(cu['perda_treino'])}")
    pior = max(range(len(curva)), key=lambda i: max(curva[i]["apagados"]) if curva[i]["apagados"] else 0)
    if curva[pior]["apagados"]:
        print(f"  maior fração de apagados: {max(curva[pior]['apagados']):.2f}, na época {curva[pior]['epoca']}")
    print(f"  época escolhida: {ce['epoca']}; perda treino {numero(ce['perda_treino'])}, "
          f"validação {numero(ce['perda_validacao'])}, acerto validação {ce['acerto_validacao']:.4f}")
    print(f"  última época: {cu['epoca']}; perda treino {numero(cu['perda_treino'])}, acerto treino "
          f"{cu['acerto_treino']:.4f}, perda validação {numero(cu['perda_validacao'])}, acerto validação "
          f"{cu['acerto_validacao']:.4f}")
    subidas = [curva[i]["perda_validacao"] - curva[i - 1]["perda_validacao"] for i in range(2, len(curva))]
    subidas = [s for s in subidas if np.isfinite(s)]
    if subidas and max(subidas) > 0:
        print(f"  maior subida da perda de validação de uma época para a seguinte: {max(subidas):.4f}")
    elif subidas:
        print("  a perda de validação não subiu de uma época para a seguinte")
