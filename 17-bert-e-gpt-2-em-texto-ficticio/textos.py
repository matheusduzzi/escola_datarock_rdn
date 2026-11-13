"""BERT e GPT-2 nas mensagens do atendimento, em partes. De dentro da pasta da aula:

    python3 textos.py modelos     # os três modelos, lidos sem o PyTorch, e a ida conferida
    python3 textos.py tokens      # as 1800 mensagens em tokens, nos três tokenizadores
    python3 textos.py lacuna      # a palavra escondida: BERT em inglês e BERT multilíngue
    python3 textos.py escrever    # o GPT-2 continuando texto em português e em inglês
    python3 textos.py assunto     # marcar o assunto: vetor congelado e regressão logística, contra o Naive Bayes
    python3 textos.py custo       # a ficha: o que cada um fez e quanto custou

Lê ../dados/mensagens/mensagens.csv (raiz do kit) e julho.csv (dessa pasta). Os modelos ficam em
modelos/, baixados uma vez por baixar_modelos.py. Não há sorteio: tudo sai igual a cada rodada,
menos os segundos, que dependem do computador e do que mais está aberto.
"""
import argparse
import csv
import json
import math
import platform
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

import modelos as m

AQUI = Path(__file__).resolve().parent
MENSAGENS = AQUI.parent / "dados" / "mensagens" / "mensagens.csv"
JULHO = AQUI / "julho.csv"
RESULTADOS = AQUI / "meus-resultados"
ASSUNTOS = ["atraso", "troca", "pagamento", "outro"]
NOMES = ["bert-base-uncased", "bert-base-multilingual-cased", "gpt2"]
CURTO = {"bert-base-uncased": "BERT inglês", "bert-base-multilingual-cased": "BERT multilíngue", "gpt2": "GPT-2"}
LAMBDAS = [0.0001, 0.001, 0.01, 0.1, 1.0]

# As frases que o cartão de cada modelo e o blog do Hugging Face publicam, com o que eles mostram.
CARTAO_BERT = {
    "bert-base-uncased": [("fashion", 0.1073106899857521), ("role", 0.08774490654468536), ("new", 0.05338378623127937)],
    "bert-base-multilingual-cased": [("model", 0.10182085633277893), ("world", 0.052126359194517136),
                                     ("data", 0.048930276185274124)],
}
BLOG_GPT2 = ("I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with my dog. "
             "I'm not sure if I'll ever be able to walk with my dog.\n\nI'm not sure")

# Cinco mensagens de junho e a tradução delas, para medir o GPT-2 no mesmo conteúdo nas duas línguas.
PARES = [
    ("meu pedido ainda não chegou", "my order still has not arrived"),
    ("quero trocar o tamanho da camiseta", "I want to exchange the size of the t-shirt"),
    ("o boleto venceu, preciso da segunda via", "the bank slip expired, I need a new copy"),
    ("não consigo pagar com cartão", "I can't pay with my card"),
    ("qual o horário do atendimento?", "what are the customer service hours?"),
]
LACUNAS = ["meu pedido ainda não [MASK].", "quero [MASK] o tamanho da camiseta.",
           "o boleto [MASK], preciso da segunda via.", "The capital of France is [MASK]."]
RASCUNHOS = ["Cliente: meu pedido ainda não chegou.\nAtendente:",
             "Customer: my order still has not arrived.\nAgent:"]


def mil(n):
    return f"{n:,}".replace(",", ".")


def ler_csv(caminho, rotulo):
    if not caminho.exists():
        raise SystemExit(f"Não achei {rotulo} em {caminho}. Rode de dentro da pasta da aula, "
                         "com a pasta dados/ na raiz do kit, como no README.")
    with open(caminho, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def carregar():
    linhas = ler_csv(MENSAGENS, "as mensagens")
    julho = ler_csv(JULHO, "o julho.csv")
    return linhas, julho


def carregar_modelo(nome):
    return m.GPT2(nome) if nome == "gpt2" else m.Bert(nome)


def cabecalho(n, titulo):
    print(f"{n}. {titulo}")


# ---------------------------------------------------------------- parte 1

def parte_modelos():
    cabecalho(1, "Os três modelos, lidos sem o PyTorch")
    print(f"  {'modelo':30s} {'arquivo':>9s}  {'SHA-256':8s} {'parâmetros':>12s}  {'carregar':>8s}  licença")
    info = {}
    for nome in NOMES:
        pasta = m.pasta_do_modelo(nome)
        arq = pasta / "model.safetensors"
        ok = all(m.sha256(pasta / a) == h for a, h in m.MODELOS[nome]["arquivos"].items())
        t0 = time.perf_counter()
        mod = carregar_modelo(nome)
        seg = time.perf_counter() - t0
        info[nome] = mod
        print(f"  {nome:30s} {arq.stat().st_size / 1e6:6.1f} MB  {'ok' if ok else 'ERRADO':8s} "
              f"{mil(mod.parametros()):>12s}  {seg:6.1f} s  {m.MODELOS[nome]['licenca']}")
        if not ok:
            raise SystemExit(f"{nome}: o SHA-256 de um arquivo não bate. Apague modelos/{nome} e rode baixar_modelos.py")
    g = info["gpt2"]
    todos = m.contar_parametros(g.tensores)
    print(f"  gpt2: o arquivo tem {mil(todos)} números; {mil(todos - g.parametros())} são as 12 máscaras "
          "causais (1024 x 1024), guardadas junto e que não aprendem")
    for nome in NOMES[:2]:
        b = info[nome]
        cabeca = m.contar_parametros({k: v for k, v in b.tensores.items() if k.startswith("cls.")})
        print(f"  {nome}: {mil(cabeca)} são da cabeça do pré-treino; o encoder tem {mil(b.parametros() - cabeca)}")
    print("  a ida conferida com o exemplo publicado de cada modelo:")
    for nome in NOMES[:2]:
        palpites = info[nome].adivinhar("Hello I'm a [MASK] model.", 3)[0]
        publicado = CARTAO_BERT[nome]
        igual = all(a == b and abs(p - q) < 1e-4 for (a, p), (b, q) in zip(palpites, publicado))
        print(f"    {nome:30s} Hello I'm a [MASK] model.  "
              + "  ".join(f"{a} {p:.4f}" for a, p in palpites)
              + f"   cartão do modelo: " + "  ".join(f"{a} {p:.4f}" for a, p in publicado)
              + f"   iguais: {igual}")
    t0 = time.perf_counter()
    texto = g.escrever("I enjoy walking with my cute dog", 40)
    seg = time.perf_counter() - t0
    print(f"    gpt2  I enjoy walking with my cute dog + 40 tokens, sempre o mais provável: "
          f"igual ao texto do blog: {texto == BLOG_GPT2} ({seg:.1f} s)")


# ---------------------------------------------------------------- parte 2

def tokenizadores():
    return {nome: (m.TokenizadorGPT2(m.pasta_do_modelo(nome)) if nome == "gpt2" else
                   m.TokenizadorBert(m.pasta_do_modelo(nome), minusculas=(nome == "bert-base-uncased")))
            for nome in NOMES}


def palavras_e_sinais(texto):
    return re.findall(r"\w+|[^\w\s]", texto)


def parte_tokens():
    linhas, _ = carregar()
    textos = [l["texto"] for l in linhas]
    toks = tokenizadores()
    cabecalho(2, f"As {len(textos)} mensagens em tokens (sem os tokens especiais do BERT)")
    print(f"  {'jeito de quebrar':40s} {'tokens':>7s} {'por mensagem':>13s} {'maior':>6s} {'[UNK]':>6s} "
          f"{'pares da atenção por mensagem':>30s}")
    jeitos = [("palavras e sinais", palavras_e_sinais)] + [(f"{n} ({'BPE de bytes' if n == 'gpt2' else 'WordPiece'})",
                                                           toks[n].pedacos) for n in NOMES]
    for rotulo, f in jeitos:
        tam = [len(f(t)) for t in textos]
        unk = sum(p == "[UNK]" for t in textos for p in f(t)) if "WordPiece" in rotulo else 0
        print(f"  {rotulo:40s} {sum(tam):7d} {np.mean(tam):13.1f} {max(tam):6d} {unk:6d} "
              f"{np.mean(np.square(tam)):30.0f}")
    exemplo = "meu pedido ainda não chegou. pedido 47868"
    print(f"  a mesma mensagem: {exemplo}")
    for n in NOMES:
        p = toks[n].pedacos(exemplo)
        print(f"    {n:30s} {len(p):2d}: {' '.join(p)}")
    print("    (no GPT-2, Ġ é o espaço e 'Ã£' são os 2 bytes do 'ã', mostrados pela tabela de bytes)")
    junho = [l["texto"] for l in linhas if l["data"].startswith("2026-06-")]
    print(f"  palavras de letras em junho que viram um token só ({len(junho)} mensagens):")
    for n in NOMES:
        tok = toks[n]
        total = inteiras = 0
        for t in junho:
            if n == "gpt2":
                for pedaco in m.pre_tokenizar_gpt2(t):
                    if pedaco.strip().isalpha():
                        total += 1
                        letras = "".join(tok.letras[b] for b in pedaco.encode("utf-8"))
                        inteiras += len(tok.bpe(letras)) == 1
            else:
                for w in tok.palavras(t):
                    if w.isalpha():
                        total += 1
                        inteiras += len(tok.wordpiece(w)) == 1
        print(f"    {n:30s} {inteiras:5d} de {total} ({100 * inteiras / total:.1f}%), "
              f"{np.mean([len(tok.pedacos(t)) for t in junho]):.1f} tokens por mensagem de junho")
    for w in ["não", "pedido", "devolução", "pagamento", "entrega"]:
        print(f"    {w:10s} " + "   ".join(f"{CURTO[n]}: {' '.join(toks[n].pedacos(w if n != 'gpt2' else ' ' + w))}"
                                       for n in NOMES))


# ---------------------------------------------------------------- parte 3

def parte_lacuna():
    linhas, _ = carregar()
    junho = [l["texto"] for l in linhas if l["data"].startswith("2026-06-")]
    cabecalho(3, "A palavra escondida: BERT em inglês e BERT multilíngue")
    berts = {n: m.Bert(n) for n in NOMES[:2]}
    for frase in LACUNAS:
        print(f"  {frase}")
        for n, b in berts.items():
            palpites = b.adivinhar(frase, 3)[0]
            print(f"    {CURTO[n]:17s} " + " | ".join(f"{a} {p:.3f}" for a, p in palpites))
    print(f"  junho: cada palavra de letras escondida uma vez, nas {len(junho)} mensagens")
    print(f"  {'modelo':30s} {'palavras':>8s} {'cabem num token':>16s} {'1º palpite certo':>17s} "
          f"{'certo entre os 5':>17s} {'segundos':>9s}")
    for n, b in berts.items():
        tok = b.tok
        variantes, alvos, total = [], [], 0
        for t in junho:
            ps = tok.palavras(t)
            for i, w in enumerate(ps):
                if not w.isalpha():
                    continue
                total += 1
                if len(tok.wordpiece(w)) != 1:
                    continue
                ids = [tok.cls]
                for j, x in enumerate(ps):
                    ids += [tok.mask] if j == i else [tok.vocab[p] for p in tok.wordpiece(x)]
                variantes.append(ids + [tok.sep])
                alvos.append(tok.vocab[w])
        t0 = time.perf_counter()
        certo1 = certo5 = 0
        for quais, ids, mascara, est in m.idas_em_lotes(b, variantes):
            for k, q in enumerate(quais):
                pos = variantes[q].index(tok.mask)
                notas = b.notas_da_palavra(est[k, pos])
                cinco = np.argsort(-notas, kind="stable")[:5]
                certo1 += cinco[0] == alvos[q]
                certo5 += alvos[q] in cinco
        seg = time.perf_counter() - t0
        print(f"  {n:30s} {total:8d} {len(variantes):9d} ({100 * len(variantes) / total:4.1f}%) "
              f"{certo1:8d} ({100 * certo1 / total:4.1f}%) {certo5:8d} ({100 * certo5 / total:4.1f}%) {seg:9.0f}")
    print("  (palavra que vira mais de um token não pode sair num [MASK] só: conta como erro)")


# ---------------------------------------------------------------- parte 4

def parte_escrever():
    linhas, _ = carregar()
    junho = [l["texto"] for l in linhas if l["data"].startswith("2026-06-")]
    g = m.GPT2()
    cabecalho(4, "O GPT-2 escrevendo")
    print("  a surpresa do GPT-2 com cada token (a perda do pré-treino, em nats; quanto menor, mais ele esperava)")
    print(f"  {'texto':44s} {'tokens':>7s} {'por token':>10s} {'por caractere':>14s}")

    def medir(textos):
        perdas = [g.perda_por_token(t) for t in textos]
        tokens = sum(len(p) for p in perdas)
        soma = sum(float(p.sum()) for p in perdas)
        return tokens, soma / tokens, soma / sum(len(t) for t in textos)
    for rotulo, textos in [(f"as {len(junho)} mensagens de junho", junho),
                           ("as 5 mensagens, em português", [a for a, _ in PARES]),
                           ("as mesmas 5, traduzidas para o inglês", [b for _, b in PARES])]:
        n, pt, pc = medir(textos)
        print(f"  {rotulo:44s} {n:7d} {pt:10.2f} {pc:14.2f}")
    print("  um rascunho de resposta, sempre o token mais provável, 30 tokens:")
    for inicio in RASCUNHOS:
        t0 = time.perf_counter()
        texto = g.escrever(inicio, 30)
        seg = time.perf_counter() - t0
        print(f"    começo: {inicio!r}")
        print(f"    saiu:   {texto[len(inicio):]!r}")
        print(f"    {seg / 30 * 1000:.0f} ms por token escrito, com as keys e os values guardados")


# ---------------------------------------------------------------- parte 5

def palavras_nb(texto):
    """A limpeza do Naive Bayes de ML supervisionado: minúsculas, sem acento, só letras."""
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z]+", t)


def naive_bayes(treino, alpha=1.0):
    n_msgs = Counter(l["assunto"] for l in treino)
    vezes = {a: Counter() for a in ASSUNTOS}
    for l in treino:
        vezes[l["assunto"]].update(palavras_nb(l["texto"]))
    vocab = {p for a in ASSUNTOS for p in vezes[a]}
    tam = {a: sum(vezes[a].values()) for a in ASSUNTOS}
    total = sum(n_msgs.values())

    def prever(texto):
        logs = []
        for a in ASSUNTOS:
            s = math.log(n_msgs[a] / total)
            for p in palavras_nb(texto):
                if p in vocab:
                    s += math.log((vezes[a][p] + alpha) / (tam[a] + alpha * len(vocab)))
            logs.append(s)
        return ASSUNTOS[int(np.argmax(logs))]
    return prever


def logistica(X, y, lam, passos=500, taxa=0.5):
    """Regressão logística de 4 assuntos (softmax), passo de gradiente com todas as linhas, pesos começando em zero."""
    n, d = X.shape
    W = np.zeros((d, len(ASSUNTOS)))
    b = np.zeros(len(ASSUNTOS))
    Y = np.eye(len(ASSUNTOS))[y]
    for _ in range(passos):
        P = m.softmax(X @ W + b)
        G = (P - Y) / n
        W -= taxa * (X.T @ G + lam * W)
        b -= taxa * G.sum(0)
    return W, b


def padronizar(X, linhas_treino):
    media = X[linhas_treino].mean(0)
    desvio = X[linhas_treino].std(0) + 1e-6
    return (X - media) / desvio


def vetores(nome, textos, refazer=False):
    """A média dos vetores da última camada de cada mensagem, guardada em meus-resultados/."""
    RESULTADOS.mkdir(exist_ok=True)
    arquivo = RESULTADOS / f"vetores-{nome}.npz"
    chave = m.sha256(MENSAGENS) + m.sha256(JULHO) + m.sha256(AQUI / "modelos.py")
    if arquivo.exists() and not refazer:
        guardado = np.load(arquivo)
        if str(guardado["chave"]) == chave:
            return guardado["media"], float(guardado["segundos"]), int(guardado["tokens"])
    mod = carregar_modelo(nome)
    listas = [mod.tok.codificar(t) for t in textos]
    X = np.zeros((len(textos), 768), dtype=np.float32)
    t0 = time.perf_counter()
    for quais, ids, mascara, est in m.idas_em_lotes(mod, listas):
        X[quais] = m.media_dos_tokens(est, mascara)
    seg = time.perf_counter() - t0
    tokens = sum(len(l) for l in listas)
    np.savez(arquivo, media=X, segundos=seg, tokens=tokens, chave=chave)
    return X, seg, tokens


def classificar(X, y, mes):
    """Escolhe o lambda em maio (treino jan a abr), refaz com jan a maio e mede junho e julho."""
    X = X.astype(np.float64)
    treino, maio = mes <= 4, mes == 5
    Xs = padronizar(X, treino)
    em_maio = {}
    for lam in LAMBDAS:
        W, b = logistica(Xs[treino], y[treino], lam)
        em_maio[lam] = int((np.argmax(Xs[maio] @ W + b, 1) == y[maio]).sum())
    lam = max(LAMBDAS, key=lambda l: (em_maio[l], l))        # empate: o lambda maior, que segura mais
    ate_maio = mes <= 5
    Xs = padronizar(X, ate_maio)
    W, b = logistica(Xs[ate_maio], y[ate_maio], lam)
    return lam, em_maio[lam], np.argmax(Xs @ W + b, 1)


def resultados_assunto(refazer=False, imprimir=True):
    linhas, julho = carregar()
    todas = linhas + [dict(l, data="2026-07-01") for l in julho]
    textos = [l["texto"] for l in todas]
    y = np.array([ASSUNTOS.index(l["assunto"]) for l in todas])
    mes = np.array([int(l["data"][5:7]) for l in todas])
    jun, jul = mes == 6, mes == 7
    t0 = time.perf_counter()
    prever = naive_bayes([l for l in linhas if not l["data"].startswith("2026-06-")])
    nb = np.array([ASSUNTOS.index(prever(t)) for t in textos])
    seg_nb = time.perf_counter() - t0
    tabela = [("Naive Bayes de ML supervisionado", "-", None, None, nb, seg_nb, None)]
    for nome in NOMES:
        X, seg, tokens = vetores(nome, textos, refazer)
        lam, certo_maio, prev = classificar(X, y, mes)
        tabela.append((nome, lam, certo_maio, int((mes == 5).sum()), prev, seg, tokens))
    if imprimir:
        cabecalho(5, "Marcar o assunto: a média dos vetores congelados e uma regressão logística, contra o Naive Bayes")
        print("  treino de janeiro a abril e lambda escolhido em maio; depois treino de janeiro a maio; junho e julho uma vez")
        print(f"  {'modelo':33s} {'lambda':>7s} {'maio':>9s} {'junho':>16s} {'julho (40 difíceis)':>20s} "
              f"{'segundos':>9s} {'ms por mensagem':>16s}")
        for nome, lam, cm, nm, prev, seg, tokens in tabela:
            cj, cl = int((prev[jun] == y[jun]).sum()), int((prev[jul] == y[jul]).sum())
            maio = f"{cm} de {nm}" if cm is not None else "-"
            print(f"  {nome:33s} {lam!s:>7s} {maio:>9s} {cj:5d} de {jun.sum()} {100 * cj / jun.sum():5.1f}% "
                  f"{cl:7d} de 40 {100 * cl / 40:5.1f}% {seg:9.2f} {1000 * seg / len(textos):16.2f}")
        print(f"  segundos: a ida pelas {len(textos)} mensagens (o Naive Bayes: treinar e prever todas)")
        certo = {t[0]: t[4][jul] == y[jul] for t in tabela}
        base = certo["Naive Bayes de ML supervisionado"]
        for nome in NOMES:
            so_nb = int((base & ~certo[nome]).sum())
            so_ele = int((~base & certo[nome]).sum())
            print(f"  julho, as mesmas 40: só o Naive Bayes acerta {so_nb:2d}; só o {CURTO[nome]} acerta {so_ele:2d}")
        erradas = [i for i in np.flatnonzero(jul) if all(t[4][i] != y[i] for t in tabela)]
        print(f"  julho: {len(erradas)} mensagens que os quatro erram:")
        for i in erradas:
            print(f"    «{textos[i]}» ({ASSUNTOS[y[i]]}): " +
                  ", ".join(f"{'NB' if t[0].startswith('Naive') else CURTO[t[0]]} {ASSUNTOS[t[4][i]]}" for t in tabela))
    return tabela, y, mes


# ---------------------------------------------------------------- parte 6

def parte_custo():
    tabela, y, mes = resultados_assunto(imprimir=False)
    cabecalho(6, "O custo de cada um, nesse computador")
    print(f"  {platform.machine()}, Python {platform.python_version()}, NumPy {np.__version__}, só o processador")
    print(f"  {'modelo':33s} {'parâmetros':>12s} {'arquivo':>9s} {'na memória':>11s} {'ida, 1840 mensagens':>20s} "
          f"{'tokens':>7s}")
    for nome, lam, cm, nm, prev, seg, tokens in tabela:
        if nome.startswith("Naive"):
            print(f"  {nome:33s} {'-':>12s} {'-':>9s} {'-':>11s} {seg:18.2f} s {'-':>7s}")
            continue
        mod = carregar_modelo(nome)
        memoria = sum(v.nbytes for v in mod.w.values()) + mod.saida.nbytes
        print(f"  {nome:33s} {mil(mod.parametros()):>12s} "
              f"{(m.pasta_do_modelo(nome) / 'model.safetensors').stat().st_size / 1e6:6.0f} MB "
              f"{memoria / 1e6:8.0f} MB {seg:18.0f} s {tokens:7d}")
    print("  (ida: o tempo medido na parte 5; na memória: os pesos copiados do arquivo, mais a tabela de tokens virada para a saída)")


PARTES = {"modelos": parte_modelos, "tokens": parte_tokens, "lacuna": parte_lacuna,
          "escrever": parte_escrever, "assunto": None, "custo": parte_custo}


def main(argv=None):
    p = argparse.ArgumentParser(description="BERT e GPT-2 nas mensagens do atendimento, em partes.")
    p.add_argument("parte", choices=list(PARTES))
    p.add_argument("--refazer", action="store_true", help="refaz os vetores do assunto, mesmo se já estão guardados")
    a = p.parse_args(argv)
    if a.parte == "assunto":
        resultados_assunto(refazer=a.refazer)
    else:
        PARTES[a.parte]()


if __name__ == "__main__":
    main()
