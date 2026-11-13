"""Aula «Transformer e tokenização» do curso Redes neurais: tudo o que a aula mostra, partes 1 a 8.

    python3 transformer.py

Só usa o NumPy e a biblioteca padrão. Lê as mensagens ao atendimento em ../dados/mensagens/mensagens.csv.
O tokenizador BPE da parte 3 é treinado aqui, nas mensagens de janeiro a maio. Os vetores das palavras
e os pesos da parte 6 são sorteados (sempre os mesmos) e não foram treinados: eles mostram o mecanismo.
Os tamanhos da parte 8 vêm do config.json publicado de cada modelo no Hugging Face.
"""
import collections
import csv
import re
import unicodedata
from pathlib import Path

import numpy as np

CSV = Path(__file__).resolve().parent.parent / "dados" / "mensagens" / "mensagens.csv"
CORTE = "2026-06-01"          # janeiro a maio treinam o vocabulário; junho fica de fora para testar
FRASE = "meu pedido ainda não chegou. pedido 47868"
INICIO = "▁"                  # marca o começo de palavra, como no SentencePiece


def ler():
    with open(CSV, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def palavras(texto):
    """Separa palavras e sinais: 'chegou.' vira 'chegou' e '.'."""
    return re.findall(r"\w+|[^\w\s]", texto)


def pedacos_com_inicio(texto):
    """Como palavras(), mas marca com ▁ o que vem depois de um espaço (ou abre o texto)."""
    return [(INICIO if m.start() == 0 or texto[m.start() - 1].isspace() else "") + m.group(0)
            for m in re.finditer(r"\w+|[^\w\s]", texto)]


def sem_acento(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def mil(n):
    return f"{n:,}".replace(",", ".")


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


# ---------------------------------------------------------------- BPE (Sennrich e outros, 2016)
def treinar_bpe(contagem, n_fusoes):
    """Começa com cada palavra quebrada em caracteres e junta, a cada passo, o par vizinho mais frequente."""
    pedacos = {p: simbolos_iniciais(p) for p in contagem}
    fusoes = []
    for _ in range(n_fusoes):
        pares = collections.Counter()
        for p, simbolos in pedacos.items():
            for a, b in zip(simbolos, simbolos[1:]):
                pares[(a, b)] += contagem[p]
        if not pares:
            break
        melhor = min(pares, key=lambda par: (-pares[par], par))   # empate: ordem alfabética
        fusoes.append((melhor, pares[melhor]))
        pedacos = {p: juntar(s, melhor) for p, s in pedacos.items()}
    return fusoes


def simbolos_iniciais(p):
    """'▁chegou' vira ▁ c h e g o u; '.' vira só ."""
    return (INICIO,) + tuple(p[1:]) if p.startswith(INICIO) else tuple(p)


def juntar(simbolos, par):
    saida, i = [], 0
    while i < len(simbolos):
        if i + 1 < len(simbolos) and (simbolos[i], simbolos[i + 1]) == par:
            saida.append(simbolos[i] + simbolos[i + 1])
            i += 2
        else:
            saida.append(simbolos[i])
            i += 1
    return tuple(saida)


def tokenizar_bpe(texto, fusoes, base):
    """Aplica as fusões na ordem em que foram aprendidas. Caractere fora da base vira <unk>."""
    tokens = []
    for p in pedacos_com_inicio(texto):
        tokens.extend(tokenizar_pedaco(p, fusoes, base))
    return tokens


def tokenizar_pedaco(p, fusoes, base):
    s = tuple(c if c in base or c == INICIO else "<unk>" for c in simbolos_iniciais(p))
    for par, _ in fusoes:
        s = juntar(s, par)
    return list(s)


# ---------------------------------------------------------------- atenção de uma cabeça (aula 15)
def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def atencao(X, W_Q, W_K, W_V, causal=False):
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    notas = Q @ K.T / np.sqrt(K.shape[1])
    if causal:
        notas = np.where(np.tril(np.ones_like(notas)) == 1, notas, -np.inf)
    return softmax(notas) @ V


def posicao(n, d_model):
    """Código de posição do artigo de 2017: seno nas colunas pares, cosseno nas ímpares."""
    pos = np.arange(n)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    angulo = pos / 10000 ** (i / d_model)
    pe = np.zeros((n, d_model))
    pe[:, 0::2] = np.sin(angulo)
    pe[:, 1::2] = np.cos(angulo)
    return pe


# ---------------------------------------------------------------- partes
def parte1():
    print("1. Três jeitos de quebrar a mesma mensagem")
    print(f"  mensagem: {FRASE}")
    p = palavras(FRASE)
    print(f"  por palavra    {len(p):>3} tokens: " + " | ".join(p))
    print(f"  por caractere  {len(FRASE):>3} tokens (o espaço conta)")
    print(f"  por byte       {len(FRASE.encode('utf-8')):>3} tokens (UTF-8)")
    for w in ("nao", "não", "NÃO"):
        print(f"  {w:<4} {len(w)} caracteres, {len(w.encode('utf-8'))} bytes: "
              + " ".join(f"{b:02x}" for b in w.encode("utf-8")))


def parte2(linhas):
    print("\n2. Um vocabulário de palavras inteiras")
    treino = [r["texto"] for r in linhas if r["data"] < CORTE]
    junho = [r["texto"] for r in linhas if r["data"] >= CORTE]
    print(f"  mensagens: {len(linhas)}; janeiro a maio {len(treino)}; junho {len(junho)}")
    print("  jeito de escrever          palavras distintas   palavras de junho fora do vocabulário   delas, números")
    for nome, f in (("como chegou", lambda s: s), ("minúsculas", str.lower),
                    ("minúsculas e sem acento", lambda s: sem_acento(s.lower()))):
        vocab = {f(w) for t in treino for w in palavras(t)}
        junho_p = [f(w) for t in junho for w in palavras(t)]
        fora = [w for w in junho_p if w not in vocab]
        numeros = sum(w.isdigit() for w in fora)
        print(f"  {nome:<26} {len(vocab):>9}           {len(fora):>4} de {len(junho_p)} ({pct(len(fora), len(junho_p))})"
              f"            {numeros:>4}")
    todas = collections.Counter(w for r in linhas for w in palavras(r["texto"]))
    print("  o mesmo 'não' escrito de quatro jeitos: "
          + ", ".join(f"{w} {todas[w]}" for w in ("não", "nao", "NÃO", "NAO")))
    n_num = sum(1 for w in todas if w.isdigit() and len(w) == 5)
    print(f"  números de pedido distintos (5 dígitos) nas {len(linhas)} mensagens: {n_num}, cada um aparece "
          f"{'uma vez' if all(todas[w] == 1 for w in todas if w.isdigit() and len(w) == 5) else 'mais de uma vez'}")


def parte3(linhas):
    print("\n3. BPE: juntar os pares mais frequentes, treinado em janeiro a maio")
    treino = [r["texto"] for r in linhas if r["data"] < CORTE]
    junho = [r["texto"] for r in linhas if r["data"] >= CORTE]
    contagem = collections.Counter(w for t in treino for w in pedacos_com_inicio(t))
    base = {c for w in contagem for c in w if c != INICIO}
    fusoes = treinar_bpe(contagem, 800)
    print(f"  base: {len(base)} caracteres diferentes + o {INICIO} de começo de palavra")
    print("  as 10 primeiras fusões (par -> novo token, vezes que o par apareceu)")
    for k, ((a, b), c) in enumerate(fusoes[:10], 1):
        print(f"    {k:>2}. {a} + {b} -> {a + b:<8} {c:>5}")
    print("  a mensagem com 0, 50, 200 e 800 fusões")
    for n in (0, 50, 200, 800):
        t = tokenizar_bpe(FRASE, fusoes[:n], base)
        print(f"    {n:>3} fusões, {len(t):>2} tokens: " + " ".join(t))
    print("  junho, que o tokenizador não viu")
    print("  fusões   vocabulário   tokens por mensagem   palavras de junho em 1 token   <unk>")
    for n in (0, 50, 200, 800):
        toks = [tokenizar_bpe(t, fusoes[:n], base) for t in junho]
        pal = [w for t in junho for w in pedacos_com_inicio(t)]
        inteiras = sum(len(tokenizar_pedaco(w, fusoes[:n], base)) == 1 for w in pal)
        unk = sum(t.count("<unk>") for t in toks)
        print(f"  {n:>6}   {len(base) + 1 + n:>11}   {sum(map(len, toks)) / len(toks):>19.1f}   "
              f"{inteiras:>5} de {len(pal)} ({pct(inteiras, len(pal))})   {unk:>5}")
    for w in ("47868", "chegou", "CHEGOU", "reembolsaram"):
        print(f"  {w:<13}-> " + " ".join(tokenizar_bpe(w, fusoes, base)))
    return fusoes, base


def parte4():
    print("\n4. O mesmo bloco, três montagens: quem pode olhar para quem")
    ent = ["meu", "pedido", "não", "chegou", "."]
    sai = ["atraso", "</s>"]
    n, m = len(ent), len(sai)
    def grade(titulo, linhas_rot, colunas_rot, pode):
        print(f"  {titulo}")
        print("    " + " " * 8 + "".join(f"{c:>8}" for c in colunas_rot))
        for i, r in enumerate(linhas_rot):
            print(f"    {r:<8}" + "".join(f"{'x' if pode(i, j) else '.':>8}" for j in range(len(colunas_rot))))
    grade("encoder (BERT, e a entrada do T5): cada token olha para todos", ent, ent, lambda i, j: True)
    grade("decoder (GPT-2): cada token olha para ele e para os anteriores", ent, ent, lambda i, j: j <= i)
    grade("T5, saída olhando a entrada (atenção cruzada): cada token da saída olha a entrada toda",
          sai, ent, lambda i, j: True)
    grade("T5, saída olhando a própria saída: só para trás", sai, sai, lambda i, j: j <= i)
    print(f"  pares que a atenção calcula, com n = {n} na entrada e m = {m} na saída")
    print(f"    encoder        n x n            = {n * n}")
    print(f"    decoder        n x (n + 1) / 2  = {n * (n + 1) // 2}")
    print(f"    T5 entrada     n x n            = {n * n}")
    print(f"    T5 cruzada     m x n            = {m * n}")
    print(f"    T5 saída       m x (m + 1) / 2  = {m * (m + 1) // 2}")


def parte5():
    print("\n5. O código de posição do artigo de 2017, com d_model = 4")
    pe = posicao(3, 4)
    print("  posição   sin(pos)   cos(pos)   sin(pos/100)   cos(pos/100)")
    for p, linha in enumerate(pe):
        print(f"  {p:>7}" + "".join(f"{v:11.3f}" for v in linha))
    # a frase e os pesos da aula 15, agora com a posição somada
    X = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]], dtype=float)
    W_Q = np.array([[1, 0], [0, 1], [0, 0], [0, 1]], dtype=float)
    W_K = np.array([[0, 1], [0, 0], [1, 0], [0, 1]], dtype=float)
    W_V = np.array([[1, 0], [0, -2], [1, 0], [0, 1]], dtype=float)
    nomes = ["pedido", "não", "chegou"]
    ordem = [2, 1, 0]
    print("  a frase da aula 15 nas duas ordens, saída do 'não' (está no meio nas duas)")
    for titulo, soma in (("sem posição", 0), ("com posição", 1)):
        s1 = atencao(X + soma * pe, W_Q, W_K, W_V)
        s2 = atencao(X[ordem] + soma * pe, W_Q, W_K, W_V)
        igual = np.allclose(s1[1], s2[1])
        print(f"    {titulo}: {' '.join(nomes)} -> [{s1[1][0]:.3f}, {s1[1][1]:.3f}]   "
              f"{' '.join(nomes[i] for i in ordem)} -> [{s2[1][0]:.3f}, {s2[1][1]:.3f}]   iguais: {igual}")


def parte6():
    print("\n6. A representação de cada token depende do que vem depois?")
    rng = np.random.default_rng(16)
    d = 8
    vocab = ["meu", "pedido", "já", "não", "chegou"]
    E = {w: rng.standard_normal(d) for w in vocab}
    W_Q, W_K, W_V = (rng.standard_normal((d, d)) for _ in range(3))
    a = ["meu", "pedido", "já", "chegou"]
    b = ["meu", "pedido", "não", "chegou"]
    Xa = np.array([E[w] for w in a]) + posicao(4, d)
    Xb = np.array([E[w] for w in b]) + posicao(4, d)
    print(f"  frase A: {' '.join(a)}   frase B: {' '.join(b)}   (só o 3º token muda)")
    print("  quanto a saída de cada token muda de A para B (distância entre os dois vetores)")
    print("  token     encoder   decoder")
    enc = np.linalg.norm(atencao(Xa, W_Q, W_K, W_V) - atencao(Xb, W_Q, W_K, W_V), axis=1)
    dec = np.linalg.norm(atencao(Xa, W_Q, W_K, W_V, True) - atencao(Xb, W_Q, W_K, W_V, True), axis=1)
    for k in range(4):
        rot = a[k] if a[k] == b[k] else f"{a[k]}/{b[k]}"
        print(f"  {rot:<9}{enc[k]:8.3f}  {dec[k]:8.3f}")


def parte7():
    print("\n7. O que cada modelo aprende a prever, na mesma mensagem")
    t = ["meu", "pedido", "ainda", "não", "chegou"]
    print("  mensagem: " + " ".join(t) + f"   ({len(t)} tokens)")
    print("  BERT (palavra escondida): entrada [CLS] meu pedido [MASK] não chegou [SEP]")
    print("    alvo: ainda, na posição do [MASK]; 1 previsão, na posição que eu escondi (o BERT sorteia 15% das posições)")
    print("  GPT-2 (próximo token): cada posição prevê a seguinte")
    pares = list(zip(t, t[1:] + ["<fim>"]))
    print("    " + "   ".join(f"{x} -> {y}" for x, y in pares) + f"   ({len(pares)} previsões)")
    print("  T5 (trecho escondido): entrada meu <extra_id_0> não chegou")
    print("    alvo: <extra_id_0> pedido ainda <extra_id_1>   (2 palavras escondidas, 4 tokens no alvo)")


def parte8():
    print("\n8. Onde ficam os parâmetros: BERT base, GPT-2 e T5 base, pelo config.json de cada um")
    H, F = 768, 3072
    # BERT base: vocab 30522, 512 posições, 2 tipos de frase, 12 camadas
    b_emb = 30522 * H + 512 * H + 2 * H + 2 * H
    camada = 4 * (H * H + H) + 2 * H + (H * F + F) + (F * H + H) + 2 * H
    b_tot = b_emb + 12 * camada + (H * H + H)
    # GPT-2: vocab 50257, 1024 posições, 12 camadas; a saída reaproveita a tabela de tokens
    g_emb = 50257 * H + 1024 * H
    g_tot = g_emb + 12 * camada + 2 * H
    # T5 base: vocab 32128, sem tabela de posição (posição relativa: 32 faixas x 12 cabeças), 12 + 12 camadas
    t_emb = 32128 * H
    t_enc = 4 * H * H + 2 * H * F + 2 * H
    t_dec = 8 * H * H + 2 * H * F + 3 * H
    t_tot = t_emb + 12 * t_enc + 12 * t_dec + 2 * (32 * 12) + 2 * H
    print("  modelo     camadas   tokens no vocabulário   tabela de tokens   uma camada   total         tabela/total")
    print(f"  BERT base  12        {mil(30522):>21}   {mil(30522 * H):>16}   {mil(camada):>10}   {mil(b_tot):>11}   "
          f"{round(100 * 30522 * H / b_tot):>11}%")
    print(f"  GPT-2      12        {mil(50257):>21}   {mil(50257 * H):>16}   {mil(camada):>10}   {mil(g_tot):>11}   "
          f"{round(100 * 50257 * H / g_tot):>11}%")
    print(f"  T5 base    12 + 12   {mil(32128):>21}   {mil(32128 * H):>16}   {mil(t_enc):>10}   {mil(t_tot):>11}   "
          f"{round(100 * 32128 * H / t_tot):>11}%")
    print(f"  uma camada do decoder do T5 (com a atenção cruzada): {mil(t_dec)}")
    print(f"  tabela de posição: BERT 512 x 768 = {mil(512 * H)}; GPT-2 1024 x 768 = {mil(1024 * H)}; "
          f"T5 32 x 12 por pilha = {32 * 12}")


if __name__ == "__main__":
    linhas = ler()
    parte1()
    parte2(linhas)
    parte3(linhas)
    parte4()
    parte5()
    parte6()
    parte7()
    parte8()
