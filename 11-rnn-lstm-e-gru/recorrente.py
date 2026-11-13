"""Aula «RNN, LSTM e GRU»: tudo o que a aula mostra, partes 1 a 7.

Rode de dentro da pasta da aula:  python3 recorrente.py
Usa só o NumPy e a venda diária em ../dados/vendas/venda_diaria.csv.

As células daqui têm um número só de estado e pesos escolhidos à mão, para a conta caber no papel.
Elas não foram treinadas: mostram o mecanismo. Quem escolhe os pesos de verdade é o treino, na aula 12.
"""
import csv
from pathlib import Path

import numpy as np

VENDA = Path(__file__).resolve().parent.parent / "dados" / "vendas" / "venda_diaria.csv"
CORTE = "2025-11-01"


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def ler_venda():
    with open(VENDA, encoding="utf-8") as f:
        linhas = [(r["data"], int(r["pedidos"])) for r in csv.DictReader(f) if r["data"] < CORTE]
    return linhas


def escala(pedidos):
    """O número que entra na rede: (pedidos - 100) / 50. 150 pedidos viram 1, 100 viram 0, 50 viram -1."""
    return (pedidos - 100) / 50


def br(data):
    a, m, d = data.split("-")
    return f"{d}/{m}"


# ---------------------------------------------------------------- as três células, com um número de estado

RNN_W, RNN_U, RNN_B = 0.5, 1.0, 0.0          # h = tanh(w * h_anterior + u * x + b)

# LSTM: (peso de x, peso de h, viés). Aqui os portões e o valor novo só olham o dia, para a conta
# caber no papel e a memória c só passar pelo portão de esquecer. Numa LSTM treinada, todos olham h também.
LSTM = {"f": (0.5, 0.0, 3.0),     # esquecer: perto de 1 guarda a memória
        "i": (1.0, 0.0, -1.0),    # entrada: quanto do valor novo entra
        "g": (1.0, 0.0, 0.0),     # valor novo (tanh)
        "o": (0.5, 0.0, 1.0)}     # saída: quanto da memória vira estado

# GRU: (peso de x, peso de h, viés); r e z são portões, n é o valor novo
GRU = {"r": (1.0, 0.0, 0.0),      # reset: quanto do estado anterior entra no valor novo
       "z": (0.5, 0.0, 2.0),      # update: perto de 1 mantém o estado anterior
       "n": (1.0, 0.5, 0.0)}


def passo_rnn(x, h, w=RNN_W):
    return np.tanh(w * h + RNN_U * x + RNN_B)


def passo_lstm(x, h, c, detalhe=False):
    p = {k: a * x + b * h + v for k, (a, b, v) in LSTM.items()}
    f, i, o = sigmoide(p["f"]), sigmoide(p["i"]), sigmoide(p["o"])
    g = np.tanh(p["g"])
    c_novo = f * c + i * g
    h_novo = o * np.tanh(c_novo)
    if detalhe:
        return h_novo, c_novo, {"f": f, "i": i, "g": g, "o": o}
    return h_novo, c_novo


def passo_gru(x, h, detalhe=False):
    r = sigmoide(GRU["r"][0] * x + GRU["r"][1] * h + GRU["r"][2])
    z = sigmoide(GRU["z"][0] * x + GRU["z"][1] * h + GRU["z"][2])
    a, b, v = GRU["n"]
    n = np.tanh(a * x + v + r * (b * h))
    h_novo = (1 - z) * n + z * h
    if detalhe:
        return h_novo, {"r": r, "z": z, "n": n}
    return h_novo


class Contador:
    """Conta quantas vezes a célula roda."""

    def __init__(self, passo):
        self.passo, self.vezes = passo, 0

    def __call__(self, *args, **kw):
        self.vezes += 1
        return self.passo(*args, **kw)


def ultimo_estado(celula, xs, w=RNN_W, empurrao=0.0):
    """Roda a janela inteira. O empurrão soma um pouco na memória logo depois do dia 1:
    no h da RNN e da GRU, no c da LSTM. Retorna a memória do último dia (h, ou c na LSTM)."""
    h, c = 0.0, 0.0
    for t, x in enumerate(xs):
        if celula == "rnn":
            h = passo_rnn(x, h, w)
        elif celula == "lstm":
            h, c = passo_lstm(x, h, c)
        else:
            h = passo_gru(x, h)
        if t == 0:
            if celula == "lstm":
                c += empurrao
            else:
                h += empurrao
    return c if celula == "lstm" else h


def sobra_do_dia_1(celula, xs, w=RNN_W, eps=1e-3):
    """Empurro a memória do dia 1 em +eps e -eps e vejo quanto a memória do último dia muda, por unidade."""
    return (ultimo_estado(celula, xs, w, eps) - ultimo_estado(celula, xs, w, -eps)) / (2 * eps)


def sobra_rnn_pela_conta(xs, w=RNN_W):
    """Regra da cadeia do h do dia 1 até o último h: a cada passo seguinte, fator w * (1 - h^2)."""
    h, total, fatores = 0.0, 1.0, []
    for t, x in enumerate(xs):
        h = passo_rnn(x, h, w)
        if t > 0:
            fator = w * (1 - h ** 2)
            fatores.append(fator)
            total *= fator
    return total, fatores


def fmt(v, casas=3):
    return f"{v:.{casas}f}"


def cient(v):
    if v == 0:
        return "0 (abaixo do que o computador vê)"
    return f"{v:.1e}"


def main():
    venda = ler_venda()

    print("1. A semana antes do corte e o número que entra na rede")
    semana = venda[-7:]
    print("  dia     pedidos   x = (pedidos - 100) / 50")
    for data, p in semana:
        print(f"  {br(data)}  {p:9d}   {escala(p):24.2f}")
    print()

    print("2. A RNN de um número, dia a dia: h = tanh(0.5 x h anterior + 1.0 x x)")
    print("  dia     x      0.5 x h anterior   1.0 x x    soma     h")
    h = 0.0
    for data, p in semana:
        x = escala(p)
        soma = RNN_W * h + RNN_U * x + RNN_B
        h_novo = np.tanh(soma)
        print(f"  {br(data)}  {x:5.2f}  {RNN_W * h:17.3f}  {RNN_U * x:9.2f}  {soma:7.3f}  {h_novo:6.3f}")
        h = h_novo
    print(f"  a previsão sai do último h, {h:.3f}, por uma camada densa (um peso e um viés)")
    print()

    print("3. Contagem de passos: a mesma célula, janelas de 7, 28 e 60 dias")
    print("  janela   vezes que a célula roda   passos entre o dia 1 e o último estado   pesos da célula")
    for t in (7, 28, 60):
        cont = Contador(passo_rnn)
        h = 0.0
        for _, p in venda[-t:]:
            h = cont(escala(p), h)
        print(f"  {t:6d}   {cont.vezes:23d}   {t - 1:38d}   {3:15d}")
    print(f"  as 60 linhas vão de {br(venda[-60][0])} a {br(venda[-1][0])}; um dia de setembro não tem linha de venda")
    print()

    print("4. Quanto da memória do dia 1 chega no último dia")
    xs60 = [escala(p) for _, p in venda[-60:]]
    print("  sobra = quanto a memória do último dia muda quando a do dia 1 muda 1 (empurro 0.001 para cima e para baixo)")
    print("  a memória é o h na RNN e na GRU e o c na LSTM")
    print("  célula                          7 dias      28 dias     60 dias")
    linhas = [("RNN, w = 0.5", "rnn", 0.5), ("RNN, w = 0.95", "rnn", 0.95),
              ("LSTM, esquecer perto de 0.95", "lstm", RNN_W), ("GRU, update perto de 0.88", "gru", RNN_W)]
    for nome, cel, w in linhas:
        vals = [sobra_do_dia_1(cel, xs60[-t:], w) for t in (7, 28, 60)]
        print(f"  {nome:30s}  " + "  ".join(f"{cient(v):10s}" for v in vals).rstrip())
    print("  RNN pela conta dos fatores w x (1 - h x h), sem empurrar nada:")
    for w in (0.5, 0.95):
        vals = [sobra_rnn_pela_conta(xs60[-t:], w)[0] for t in (7, 28, 60)]
        fatores = sobra_rnn_pela_conta(xs60, w)[1]
        print(f"  RNN, w = {w:<4}                   " + "  ".join(f"{v:.1e}   " for v in vals)
              + f"  fator típico por passo: {float(np.exp(np.mean(np.log(fatores)))):.3f}")
    for nome, cel, chave in (("esquecer da LSTM", "lstm", "f"), ("update da GRU", "gru", "z")):
        vals = []
        h, c = 0.0, 0.0
        for x in xs60:
            if cel == "lstm":
                h, c, d = passo_lstm(x, h, c, detalhe=True)
            else:
                h, d = passo_gru(x, h, detalhe=True)
            vals.append(d[chave])
        print(f"  {nome} nos 60 dias: de {min(vals):.3f} a {max(vals):.3f}; "
              f"produto dos 59 depois do dia 1: {np.prod(vals[1:]):.3f}")
    print()

    print("5. Um passo da LSTM, à mão: o dia 31/10, depois de 30/10")
    h, c = 0.0, 0.0
    for _, p in semana[:-1]:
        h, c = passo_lstm(escala(p), h, c)
    x = escala(semana[-1][1])
    print(f"  entra: x = {x:.2f}, h anterior = {h:.3f}, c anterior = {c:.3f}")
    h_novo, c_novo, d = passo_lstm(x, h, c, detalhe=True)
    nomes = {"f": "esquecer f = sigmoide(0.5 x x + 3)", "i": "entrada  i = sigmoide(1.0 x x - 1)",
             "g": "valor    g = tanh(1.0 x x)", "o": "saída    o = sigmoide(0.5 x x + 1)"}
    for k in ("f", "i", "g", "o"):
        a, b, v = LSTM[k]
        soma = a * x + b * h + v
        print(f"  {nomes[k]:38s} soma {soma:6.3f}  ->  {d[k]:.3f}")
    print(f"  c = f x c anterior + i x g = {d['f']:.3f} x {c:.3f} + {d['i']:.3f} x {d['g']:.3f} = {c_novo:.3f}")
    print(f"  h = o x tanh(c) = {d['o']:.3f} x {np.tanh(c_novo):.3f} = {h_novo:.3f}")
    print()

    print("6. Um passo da GRU, à mão: o mesmo dia 31/10")
    h = 0.0
    for _, p in semana[:-1]:
        h = passo_gru(escala(p), h)
    print(f"  entra: x = {x:.2f}, h anterior = {h:.3f}")
    h_novo, d = passo_gru(x, h, detalhe=True)
    print(f"  reset  r = sigmoide(1.0 x x)                 -> {d['r']:.3f}")
    print(f"  update z = sigmoide(0.5 x x + 2)             -> {d['z']:.3f}")
    print(f"  valor  n = tanh(1.0 x x + r x 0.5 x h)       -> {d['n']:.3f}")
    print(f"  h = (1 - z) x n + z x h anterior = {1 - d['z']:.3f} x {d['n']:.3f} + {d['z']:.3f} x {h:.3f} = {h_novo:.3f}")
    print()

    print("7. Parâmetros com o formato do PyTorch: 3 entradas por dia, estado de 32")
    entradas, estado = 3, 32
    print("  célula   blocos   pesos de x      pesos de h       vieses       total   + densa 32 -> 1   total da rede")
    for nome, blocos in (("RNN", 1), ("LSTM", 4), ("GRU", 3)):
        W_ih = np.zeros((blocos * estado, entradas))
        W_hh = np.zeros((blocos * estado, estado))
        b_ih = np.zeros(blocos * estado)
        b_hh = np.zeros(blocos * estado)
        total = W_ih.size + W_hh.size + b_ih.size + b_hh.size
        densa = estado * 1 + 1
        print(f"  {nome:6s}   {blocos:6d}   {str(W_ih.shape):13s}   {str(W_hh.shape):14s}   2 x {blocos * estado:<6d}"
              f"   {total:5d}   {densa:15d}   {total + densa:13d}")
    print("  os parâmetros não dependem do tamanho da janela: 7, 28 ou 60 dias usam os mesmos pesos")


if __name__ == "__main__":
    main()
