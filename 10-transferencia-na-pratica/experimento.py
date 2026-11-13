"""O experimento da aula «Transferência na prática»: a CNN do Caio do zero contra a ResNet-18 pronta.

    python3 experimento.py rede                   # parte 1: a rede pronta, conferida e contada (segundos)
    python3 experimento.py tamanho                # parte 2: o tamanho da foto, com a rede congelada (uns 5 minutos)
    python3 experimento.py treinar                # parte 3: congelar e adaptar nas 50.000 fotos (uns 20 minutos)
    python3 experimento.py comparar               # parte 4: lado a lado com a CNN do Caio, na validação (uns 2 minutos)
    python3 experimento.py comparar --teste       # parte 5: o teste, uma vez, depois da escolha
    python3 experimento.py treinar --fotos 1000   # o exercício: as três redes com 1.000 fotos de treino
    python3 experimento.py comparar --fotos 1000

O orçamento comum é o da aula «CNN no Fashion MNIST»: as mesmas 50.000 fotos de treino e 10.000 de
validação, lote 64, 15 épocas, as taxas 0.05 e 0.02, fica a taxa e a época de menor perda de validação.
Com 50.000 fotos, a CNN do Caio vem pronta de ../08-cnn-no-fashion-mnist/pesos/. Com menos fotos,
o treinar.py treina ela do zero aqui, com as mesmas fotos das outras.

Precisa do NumPy, das fotos em ../dados/fashion-mnist, da pasta ../08-cnn-no-fashion-mnist e dos pesos
da ResNet-18 em pesos/ (python3 baixar_pesos.py). Os resultados vão para meus-resultados/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import resnet

AQUI = Path(__file__).resolve().parent
AULA08 = AQUI.parent / "08-cnn-no-fashion-mnist"
if not (AULA08 / "cnn.py").exists():
    raise SystemExit("Não achei ../08-cnn-no-fashion-mnist/cnn.py. Essa aula usa a pasta da aula 08 do kit.")
sys.path.insert(0, str(AULA08))
import cnn  # noqa: E402

LADO = 64                         # o tamanho escolhido na parte 2; os pesos prontos pedem 224
TAXAS = cnn.TAXAS                 # (0.05, 0.02), as mesmas da aula 08
LOTE = cnn.ORCAMENTO["lote"]      # 64
EPOCAS = cnn.ORCAMENTO["epocas"]  # 15
SO_CABECA = 3                     # adaptar: 3 épocas só com a cabeça, depois solta a etapa 4
DIVIDE_TAXA = 10                  # a etapa 4, quando solta, aprende com a taxa da cabeça dividida por 10
VALIDACAO = cnn.ORCAMENTO["fotos_validacao"]
RESULTADOS = AQUI / "meus-resultados"
NOMES = {"caio": "CNN do Caio, do zero", "congelar": "ResNet-18, congelar", "adaptar": "ResNet-18, adaptar"}


def pct(x, casas=1):
    return f"{100 * x:.{casas}f}%".replace(".", ",")


def milhar(n):
    return f"{n:,}".replace(",", ".")


def fotos_brutas(n_treino):
    """As fotos em 0..255, com a divisão da aula 03 (a mesma do cnn.py)."""
    try:
        X, y, Xs, ys = resnet.carregar(resnet.DADOS)
    except FileNotFoundError:
        raise SystemExit("As fotos não estão em ../dados/fashion-mnist. Rode python3 baixar.py dentro dessa pasta.")
    i_treino, i_validacao = cnn.separar(len(X), VALIDACAO, cnn.ORCAMENTO["sorteio_divisao"])
    i_treino = i_treino[:n_treino]
    return X[i_treino], y[i_treino].astype(np.int64), X[i_validacao], y[i_validacao].astype(np.int64), Xs, ys.astype(np.int64)


def multiplicacoes_resnet(lado, saidas=10):
    """Multiplicações por foto na ida pela ResNet-18, com a cabeça de 10 saídas."""
    total = 0
    h = (lado + 6 - 7) // 2 + 1
    total += h * h * 64 * 3 * 49                     # entrada: conv 7 x 7, 64 filtros, 3 canais
    h = (h + 2 - 3) // 2 + 1                         # max pooling
    canais = 64
    for etapa, saida in enumerate([64, 128, 256, 512], start=1):
        for bloco in range(2):
            passo = 2 if (etapa > 1 and bloco == 0) else 1
            ho = (h + 2 - 3) // passo + 1
            total += ho * ho * saida * canais * 9 + ho * ho * saida * saida * 9
            if passo == 2:
                total += ho * ho * saida * canais     # atalho 1 x 1
            h, canais = ho, saida
    return total + 512 * saidas


# ---------------------------------------------------------------- parte 1

def parte_rede():
    if not resnet.PESOS.exists():
        raise SystemExit("Não achei pesos/resnet18-f37072fd.pth. Rode python3 baixar_pesos.py nessa pasta.")
    print(f"{resnet.PESOS.name}: SHA-256 {'ok' if resnet.sha256(resnet.PESOS) == resnet.SHA256 else 'DIFERENTE'}")
    if resnet.sha256(resnet.PESOS) != resnet.SHA256:
        raise SystemExit("O arquivo de pesos não é o oficial. Apague pesos/ e rode python3 baixar_pesos.py.")
    p = resnet.ler_pth()
    partes = resnet.contar(p)
    medias = sum(1 for k in p if k.endswith("running_mean"))
    print("1. A rede pronta: ResNet-18 do torchvision, pesos do ImageNet (IMAGENET1K_V1)")
    print(f"  lido sem o PyTorch: {len(p)} tensores; o leitor só aceita as peças de um tensor")
    total = sum(partes.values())
    print(f"  parâmetros no arquivo: {milhar(total)}")
    for nome, n in partes.items():
        print(f"    {nome:<14} {milhar(n):>10}")
    corpo = total - partes["densa de 1000"]
    cab = 512 * 10 + 10
    print(f"  a densa de 1000 tipos do ImageNet sai; entra a cabeça nova, 512 x 10 + 10 = {milhar(cab)}")
    print(f"  congelar: só a cabeça aprende, {milhar(cab)} parâmetros; "
          f"adaptar: etapa 4 e cabeça, {milhar(partes['etapa 4'] + cab)} de {milhar(corpo + cab)}")
    print(f"  camadas de batch normalization: {medias}, ficam em modo de avaliação (médias do ImageNet paradas)")
    print("  o lado do mapa depois da entrada, do pooling e das etapas 2, 3 e 4:")
    c = resnet.Corpo(p)
    for lado in (28, LADO, 224):
        print(f"    foto {lado:>3} x {lado:<3} -> {' -> '.join(map(str, c.tamanhos(lado)))}   "
              f"multiplicações por foto: {milhar(multiplicacoes_resnet(lado))}")
    print(f"  CNN do Caio (aula 08), foto 28 x 28: multiplicações por foto: {milhar(cnn.multiplicacoes(cnn.PLANTAS['caio']))}")


# ---------------------------------------------------------------- treino da cabeça e da etapa 4

def mapas_etapa3(corpo, fotos, lote=200):
    return corpo.em_lotes(fotos, LADO, etapa_final=3, lote=lote, media=False)


def saidas_etapa4(e4, mapas, lote=500):
    return np.concatenate([e4.ida(mapas[i:i + lote]) for i in range(0, len(mapas), lote)])


def medir(cab, F, y):
    lp = resnet.log_probabilidades(cab.notas(F))
    perdas = -lp[np.arange(len(y)), y]
    if not np.isfinite(perdas).all():
        raise FloatingPointError("a perda virou infinito ou NaN: a taxa de aprendizado está alta demais?")
    return float(perdas.mean()), float((lp.argmax(axis=1) == y).mean()), lp.argmax(axis=1)


def treinar_opcao(opcao, taxa, M3t, yt, M3v, yv, e0, Ft0, mostrar=True):
    """Congelar: só a cabeça, nas 15 épocas. Adaptar: 3 épocas só a cabeça, depois a etapa 4 junto.

    Retorna a cabeça e a etapa 4 da época de menor perda de validação, a época e a curva.
    """
    e4 = e0.copia() if opcao == "adaptar" else e0
    cab = resnet.Cabeca(Ft0, sorteio=cnn.ORCAMENTO["sorteio_pesos"])
    rng = np.random.default_rng(cnn.ORCAMENTO["sorteio_ordem"])
    Ft, Fv = Ft0, saidas_etapa4(e0, M3v)
    pt, at, _ = medir(cab, Ft, yt)
    pv, av, _ = medir(cab, Fv, yv)
    curva = [{"epoca": 0, "perda_treino": pt, "acerto_treino": at, "perda_validacao": pv, "acerto_validacao": av,
              "segundos": 0.0, "solta": False, "mudou": 0.0}]
    if mostrar:
        print("    época  o que aprende       perda treino  acerto treino  perda validação  acerto validação  "
              "etapa 4 mudou  segundos")
    melhor = (pv, 0, cab.copia(), e4.copia() if opcao == "adaptar" else None)
    for epoca in range(1, EPOCAS + 1):
        solta = opcao == "adaptar" and epoca > SO_CABECA
        ordem = rng.permutation(len(yt))
        t0 = time.perf_counter()
        for i in range(0, len(yt), LOTE):
            j = ordem[i:i + LOTE]
            if solta:
                guardado = {}
                f = e4.ida(M3t[j], guardado)
            else:
                f = Ft[j]
            d = resnet.erro_das_notas(cab.notas(f), yt[j])
            gW, gb, df = cab.volta(f, d)
            cab.W -= taxa * gW
            cab.b -= taxa * gb
            if solta:
                e4.passo_sgd(e4.volta(df, guardado), taxa / DIVIDE_TAXA)
        segundos = time.perf_counter() - t0
        if solta:
            Ft, Fv = saidas_etapa4(e4, M3t), saidas_etapa4(e4, M3v)
        pt, at, _ = medir(cab, Ft, yt)
        pv, av, _ = medir(cab, Fv, yv)
        mudou = e4.mudou(e0) if opcao == "adaptar" else 0.0
        curva.append({"epoca": epoca, "perda_treino": pt, "acerto_treino": at, "perda_validacao": pv,
                      "acerto_validacao": av, "segundos": segundos, "solta": solta, "mudou": mudou})
        if mostrar:
            print(f"    {epoca:>5}  {'etapa 4 e cabeça' if solta else 'só a cabeça':<18} {pt:>13.4f} {at:>14.4f} "
                  f"{pv:>16.4f} {av:>17.4f} {pct(mudou, 2):>14} {segundos:>9.1f}", flush=True)
        if pv < melhor[0]:
            melhor = (pv, epoca, cab.copia(), e4.copia() if opcao == "adaptar" else None)
    return melhor, curva


def guardar(pasta, nome, cab, e4, **info):
    dados = {"cab_W": cab.W, "cab_b": cab.b, "cab_media": cab.media, "cab_desvio": cab.desvio}
    if e4 is not None:
        for c in e4.W:
            dados[f"{c}.W"], dados[f"{c}.g"], dados[f"{c}.beta"] = e4.W[c], e4.g[c], e4.beta[c]
    np.savez(pasta / f"{nome}.npz", **dados, info=json.dumps(info))


def abrir(pasta, nome, p):
    caminho = pasta / f"{nome}.npz"
    if not caminho.exists():
        raise SystemExit(f"Não achei {caminho.relative_to(AQUI)}. Rode antes python3 experimento.py treinar"
                         + ("" if pasta.name == "fotos-50000" else f" --fotos {pasta.name[6:]}") + ".")
    with np.load(caminho) as a:
        cab = resnet.Cabeca.__new__(resnet.Cabeca)
        cab.W, cab.b, cab.media, cab.desvio = a["cab_W"], a["cab_b"], a["cab_media"], a["cab_desvio"]
        e4 = resnet.Etapa4(p)
        if "layer4.0.conv1.W" in a.files:
            for c in e4.W:
                e4.W[c], e4.g[c], e4.beta[c] = a[f"{c}.W"], a[f"{c}.g"], a[f"{c}.beta"]
        info = json.loads(str(a["info"]))
    return cab, e4, info


# ---------------------------------------------------------------- parte 2

def parte_tamanho():
    n_t, n_v = 5000, 2000
    xt, yt, xv, yv, _, _ = fotos_brutas(n_t)
    xv, yv = xv[:n_v], yv[:n_v]
    corpo = resnet.Corpo(resnet.ler_pth())
    print("2. O tamanho da foto, com a rede congelada (só a cabeça aprende)")
    print(f"  {n_t} fotos de treino e {n_v} de validação, tiradas das mesmas da aula 08; 15 épocas, lote 64,"
          " taxas 0.05 e 0.02; fica a de menor perda de validação")
    print("    foto                    ms por foto na ida   perda validação   acerto validação")
    for lado, normalizar in ((32, True), (LADO, True), (112, True), (224, True), (LADO, False)):
        t0 = time.perf_counter()
        Ft = corpo.em_lotes(xt, lado, lote=100 if lado < 224 else 25, normalizar=normalizar)
        Fv = corpo.em_lotes(xv, lado, lote=100 if lado < 224 else 25, normalizar=normalizar)
        ms = (time.perf_counter() - t0) / (n_t + n_v) * 1000
        resultados = []
        for taxa in TAXAS:
            cab = resnet.Cabeca(Ft)
            rng = np.random.default_rng(0)
            melhor = None
            for _ in range(EPOCAS):
                ordem = rng.permutation(n_t)
                for i in range(0, n_t, LOTE):
                    j = ordem[i:i + LOTE]
                    gW, gb, _ = cab.volta(Ft[j], resnet.erro_das_notas(cab.notas(Ft[j]), yt[j]))
                    cab.W -= taxa * gW
                    cab.b -= taxa * gb
                pv, av, _ = medir(cab, Fv, yv)
                if melhor is None or pv < melhor[0]:
                    melhor = (pv, av)
            resultados.append(melhor)
        pv, av = min(resultados)
        rotulo = f"{lado:>3} x {lado:<3}" + ("" if normalizar else ", sem normalizar")
        print(f"    {rotulo:<22}  {ms:>18.2f}  {pv:>16.4f}  {pct(av, 2):>17}", flush=True)


# ---------------------------------------------------------------- parte 3

def parte_treinar(n_fotos):
    pasta = RESULTADOS / f"fotos-{n_fotos}"
    pasta.mkdir(parents=True, exist_ok=True)
    xt, yt, xv, yv, _, _ = fotos_brutas(n_fotos)
    p = resnet.ler_pth()
    corpo, e0 = resnet.Corpo(p), resnet.Etapa4(p)
    resumo = {"fotos_treino": len(yt), "fotos_validacao": len(yv), "lado": LADO, "lote": LOTE, "epocas": EPOCAS,
              "taxas": list(TAXAS), "so_cabeca": SO_CABECA, "divide_taxa": DIVIDE_TAXA, "redes": {}}
    print(f"3. Congelar e adaptar a ResNet-18 pronta: {len(yt)} fotos de treino, {len(yv)} de validação, "
          f"foto {LADO} x {LADO}")
    print(f"  orçamento comum: lote {LOTE}, {EPOCAS} épocas, taxas {' e '.join(map(str, TAXAS))}; "
          "fica a taxa e a época de menor perda de validação")
    print(f"  adaptar: {SO_CABECA} épocas só com a cabeça; depois a etapa 4 aprende junto, com a taxa / {DIVIDE_TAXA}")
    t0 = time.perf_counter()
    M3t, M3v = mapas_etapa3(corpo, xt), mapas_etapa3(corpo, xv)
    ida_etapa3 = time.perf_counter() - t0
    t0 = time.perf_counter()
    Ft0 = saidas_etapa4(e0, M3t)
    ida_etapa4 = time.perf_counter() - t0
    resumo["ida_ate_etapa3_s"], resumo["ida_etapa4_treino_s"] = ida_etapa3, ida_etapa4
    print(f"  uma vez, antes do treino: a ida até a etapa 3 em {len(yt) + len(yv)} fotos, {ida_etapa3:.0f} s; "
          f"a etapa 4 nas fotos de treino, {ida_etapa4:.0f} s", flush=True)
    for opcao in ("congelar", "adaptar"):
        for taxa in TAXAS:
            print(f"  {NOMES[opcao]}, taxa {taxa}:")
            (pv, ep, cab, e4), curva = treinar_opcao(opcao, taxa, M3t, yt, M3v, yv, e0, Ft0)
            nome = f"{opcao}-taxa-{taxa}"
            guardar(pasta, nome, cab, e4, epoca=ep, taxa=taxa, perda_validacao=pv)
            resumo["redes"][nome] = {"epoca": ep, "perda_validacao": pv, "curva": curva}
            print(f"    menor perda de validação: época {ep}, perda {pv:.4f}, "
                  f"acerto {curva[ep]['acerto_validacao']:.4f}", flush=True)
    if n_fotos < cnn.ORCAMENTO["fotos_treino"]:
        orc = dict(cnn.ORCAMENTO, fotos_treino=n_fotos)
        Xt, Yt, Xv, Yv, _, _, _, _ = cnn.preparar(orc)
        for taxa in TAXAS:
            print(f"  {NOMES['caio']}, taxa {taxa}, com as mesmas {n_fotos} fotos:")
            rede = cnn.nova_rede(cnn.PLANTAS["caio"], orc["sorteio_pesos"])
            melhor, ep, curva = cnn.ciclo(rede, Xt, Yt, Xv, Yv, EPOCAS, taxa, LOTE, orc["sorteio_ordem"], mostrar=True)
            cnn.salvar(melhor, pasta / cnn.arquivo("caio", taxa), epoca=ep, epocas=EPOCAS, taxa=taxa,
                       fotos_treino=n_fotos, perda_validacao=round(curva[ep]["perda_validacao"], 4))
            resumo["redes"][f"caio-taxa-{taxa}"] = {"epoca": ep, "perda_validacao": curva[ep]["perda_validacao"],
                                                    "curva": curva}
            print(f"    menor perda de validação: época {ep}, perda {curva[ep]['perda_validacao']:.4f}, "
                  f"acerto {curva[ep]['acerto_validacao']:.4f}", flush=True)
    (pasta / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  resultados em meus-resultados/fotos-{n_fotos}/")


# ---------------------------------------------------------------- partes 4 e 5

def escolher(resumo, opcao):
    nomes = [f"{opcao}-taxa-{t}" for t in TAXAS]
    return min(nomes, key=lambda n: (resumo["redes"][n]["perda_validacao"], -float(n.rsplit("-", 1)[1])))


def caio_escolhido(n_fotos, pasta):
    """A CNN do Caio do zero, com a regra da aula 08: das duas taxas, a de menor perda de validação."""
    origem = AULA08 / "pesos" if n_fotos == cnn.ORCAMENTO["fotos_treino"] else pasta
    abertas = []
    for taxa in TAXAS:
        rede, info = cnn.abrir("caio", taxa, origem)
        if (info["epocas"], info["fotos_treino"]) != (EPOCAS, n_fotos):
            raise SystemExit(f"{cnn.arquivo('caio', taxa)} em {origem.name}/ foi treinado com {info['fotos_treino']} "
                             f"fotos e {info['epocas']} épocas; o experimento pede {n_fotos} e {EPOCAS}.")
        abertas.append((info["perda_validacao"], -taxa, rede, info))
    _, _, rede, info = min(abertas, key=lambda t: (t[0], t[1]))
    return rede, info


EMPATE = 0.005                    # meio ponto de acerto, umas 50 fotos em 10.000


def escolher_rede(validacao, mult):
    """A regra escrita antes de rodar: a menor perda de validação; se outra rede acerta a menos de meio
    ponto dela, as duas empatam e fica a que faz menos multiplicações por foto (mais barata de prever)."""
    melhor = min(validacao, key=lambda k: validacao[k][0])
    empate = [k for k in validacao if k != melhor and abs(validacao[k][1] - validacao[melhor][1]) < EMPATE]
    return min([melhor] + empate, key=lambda k: mult[k]), empate


def cronometro(funcao, vezes):
    funcao()                                               # a primeira vez aquece a memória
    t0 = time.perf_counter()
    for _ in range(vezes):
        funcao()
    return (time.perf_counter() - t0) / vezes


def parte_comparar(n_fotos, teste):
    pasta = RESULTADOS / f"fotos-{n_fotos}"
    if not (pasta / "resumo.json").exists():
        raise SystemExit(f"Não achei meus-resultados/fotos-{n_fotos}/resumo.json. Rode antes python3 experimento.py "
                         "treinar" + ("" if n_fotos == cnn.ORCAMENTO["fotos_treino"] else f" --fotos {n_fotos}") + ".")
    resumo = json.loads((pasta / "resumo.json").read_text(encoding="utf-8"))
    p = resnet.ler_pth()
    corpo, e0 = resnet.Corpo(p), resnet.Etapa4(p)
    xt, yt, xv, yv, xs, ys = fotos_brutas(n_fotos)
    Xt, Yt, Xv, Yv, Xs, Ys, _, _ = cnn.preparar(dict(cnn.ORCAMENTO, fotos_treino=n_fotos))

    caio, info_caio = caio_escolhido(n_fotos, pasta)
    redes = {"caio": (caio, info_caio["taxa"], info_caio["epoca"])}
    for opcao in ("congelar", "adaptar"):
        nome = escolher(resumo, opcao)
        cab, e4, info = abrir(pasta, nome, p)
        redes[opcao] = ((cab, e4), info["taxa"], info["epoca"])

    M3v = mapas_etapa3(corpo, xv)
    validacao = {"caio": cnn.avaliar(Xv, Yv, caio)[:3]}
    for opcao in ("congelar", "adaptar"):
        cab, e4 = redes[opcao][0]
        validacao[opcao] = medir(cab, saidas_etapa4(e4, M3v), yv)

    print(f"4. Lado a lado: {len(yt)} fotos de treino, as mesmas {len(yv)} de validação")
    print("  rede                         taxa  época  perda validação  acerto validação  erros")
    for k in ("caio", "congelar", "adaptar"):
        perda, acerto, prev = validacao[k]
        print(f"  {NOMES[k]:<27} {redes[k][1]:>5} {redes[k][2]:>6} {perda:>16.4f} {pct(acerto, 2):>17} "
              f"{int((prev != yv).sum()):>6}")
    if n_fotos == cnn.ORCAMENTO["fotos_treino"]:
        print("  (a CNN do Caio vem de ../08-cnn-no-fashion-mnist/pesos/, treinada no mesmo orçamento)")

    print("  acerto por tipo (do zero | congelar | adaptar):")
    for c, nome in enumerate(resnet.CLASSES):
        linha = " | ".join(pct(float((validacao[k][2][yv == c] == c).mean())) for k in ("caio", "congelar", "adaptar"))
        print(f"    {nome:<13} {linha}")
    a, b = validacao["adaptar"][2] == yv, validacao["caio"][2] == yv
    print(f"  as mesmas fotos: só adaptar acerta {int((a & ~b).sum())}; só a CNN do Caio acerta {int((~a & b).sum())}; "
          f"as duas erram {int((~a & ~b).sum())}")

    # o custo, medido agora, do mesmo jeito para as três redes
    cab_c, e4_c = redes["congelar"][0]
    cab_a, e4_a = redes["adaptar"][0]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(yt), LOTE)
    M3_lote = mapas_etapa3(corpo, xt[idx])
    F_lote = e0.ida(M3_lote)

    def passo_caio():
        cnn.gradientes(Xt[idx], Yt[idx], caio)

    def passo_cabeca():
        cab_c.volta(F_lote, resnet.erro_das_notas(cab_c.notas(F_lote), yt[idx]))

    def passo_etapa4():
        g = {}
        f = e4_a.ida(M3_lote, g)
        _, _, df = cab_a.volta(f, resnet.erro_das_notas(cab_a.notas(f), yt[idx]))
        e4_a.volta(df, g)

    n_ida = 1000
    ida = cronometro(lambda: mapas_etapa3(corpo, xt[:n_ida]), 1) / n_ida
    M3_500 = mapas_etapa3(corpo, xt[:500])
    ida4 = cronometro(lambda: saidas_etapa4(e0, M3_500), 1) / 500
    passos = len(range(0, len(yt), LOTE))
    s_caio, s_cab, s_e4 = (cronometro(f, 20) * passos for f in (passo_caio, passo_cabeca, passo_etapa4))
    n_uma_vez = len(yt) + len(yv)
    uma_vez = {"caio": 0.0, "congelar": (ida + ida4) * n_uma_vez, "adaptar": ida * n_uma_vez + ida4 * len(yt)}
    por_epoca = {"caio": s_caio, "congelar": s_cab, "adaptar": s_e4}
    orcamento = {"caio": len(TAXAS) * EPOCAS * s_caio, "congelar": uma_vez["congelar"] + len(TAXAS) * EPOCAS * s_cab,
                 "adaptar": uma_vez["adaptar"] + len(TAXAS) * (SO_CABECA * s_cab + (EPOCAS - SO_CABECA) * s_e4)}
    prever = {"caio": cronometro(lambda: cnn.avaliar(Xv[:2000], Yv[:2000], caio), 1) * 5,
              "congelar": cronometro(lambda: corpo.em_lotes(xv[:2000], LADO), 1) * 5}
    prever["adaptar"] = prever["congelar"]
    treinam = {"caio": cnn.parametros(caio), "congelar": cab_c.parametros(),
               "adaptar": cab_a.parametros() + e4_a.parametros()}
    mult = {"caio": cnn.multiplicacoes(cnn.PLANTAS["caio"]), "congelar": multiplicacoes_resnet(LADO),
            "adaptar": multiplicacoes_resnet(LADO)}
    print(f"  o custo, medido agora nessa máquina ({passos} passos por época; tempos em segundos):")
    print("  rede                         treinam  multiplicações   uma vez antes   uma época   orçamento todo"
          "   prever 10.000")
    print("                                             por foto    do treino                (2 taxas)        fotos")
    for k in ("caio", "congelar", "adaptar"):
        print(f"  {NOMES[k]:<27} {milhar(treinam[k]):>9} {milhar(mult[k]):>15} {uma_vez[k]:>14.0f} "
              f"{por_epoca[k]:>11.1f} {orcamento[k]:>16.0f} {prever[k]:>14.1f}")
    print("  (uma época: 20 passos medidos, vezes os passos da época; prever: 2.000 fotos medidas, vezes 5;"
          " sem as avaliações de cada época)")

    escolhida, empate = escolher_rede(validacao, mult)
    print(f"  a regra: fica a menor perda de validação ({NOMES[min(validacao, key=lambda k: validacao[k][0])]}); "
          "se outra acerta a menos de meio ponto dela, empata, e fica a de menos multiplicações por foto")
    print(f"  escolhida: {NOMES[escolhida]}" + (f" (empate com {', '.join(NOMES[k] for k in empate)})" if empate else ""))

    if teste:
        print(f"5. O teste, uma vez: {NOMES[escolhida]} (a escolhida) e a CNN do Caio")
        perda_caio, acerto_caio, prev_caio = cnn.avaliar(Xs, Ys, caio)[:3]
        linhas = [("caio", perda_caio, acerto_caio, prev_caio)]
        if escolhida != "caio":
            cab, e4 = redes[escolhida][0]
            perda, acerto, prev = medir(cab, saidas_etapa4(e4, mapas_etapa3(corpo, xs)), ys)
            linhas.append((escolhida, perda, acerto, prev))
        for k, perda, acerto, prev in linhas:
            print(f"  {NOMES[k]:<27} perda {perda:.4f}  acerto {pct(acerto, 2)}  erros {int((prev != ys).sum())} "
                  f"de {len(ys)}")
            print("    " + "; ".join(f"{resnet.CLASSES[c]} {pct(float((prev[ys == c] == c).mean()))}" for c in (0, 2, 4, 6)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A CNN do Caio do zero contra a ResNet-18 pronta.")
    ap.add_argument("parte", choices=["rede", "tamanho", "treinar", "comparar"])
    ap.add_argument("--fotos", type=int, default=cnn.ORCAMENTO["fotos_treino"],
                    help="fotos de treino, de 100 a 50000 (padrão: 50000, o orçamento da aula 08)")
    ap.add_argument("--teste", action="store_true", help="com comparar: o teste, uma vez; use só depois de escolher")
    args = ap.parse_args()
    if not 100 <= args.fotos <= cnn.ORCAMENTO["fotos_treino"]:
        raise SystemExit("--fotos tem que ficar entre 100 e 50000")
    if args.teste and args.parte != "comparar":
        raise SystemExit("--teste só vale com comparar, depois da escolha pela validação")
    if not resnet.PESOS.exists():
        raise SystemExit("Não achei pesos/resnet18-f37072fd.pth. Rode python3 baixar_pesos.py nessa pasta.")
    {"rede": parte_rede, "tamanho": parte_tamanho,
     "treinar": lambda: parte_treinar(args.fotos), "comparar": lambda: parte_comparar(args.fotos, args.teste)}[args.parte]()
