"""Tudo o que a aula «AlexNet, VGG e ResNet» mostra, partes 1 a 5.

Rode de dentro da pasta da aula:  python3 evolucao.py
Só precisa do NumPy. Não baixa nenhuma rede nem foto: as quatro redes da parte 1 estão descritas aqui,
camada por camada, como a biblioteca torchvision monta cada uma. A parte 4 sorteia pesos, sempre com o
mesmo sorteio, para mostrar o que o atalho da ResNet faz com o sinal e com o gradiente. Nada é treinado.
"""
import numpy as np


def fmt(n):
    """18496 vira 18.496, como na aula."""
    return f"{n:,}".replace(",", ".")


def pct(parte, todo):
    return f"{100 * parte / todo:.1f}".replace(".", ",") + "%"


# ---------------------------------------------------------------------------------------------
# Parte 1. As quatro redes, camada por camada.
# Cada camada é uma tupla. ("conv", canais_entrada, filtros, k, com_vies) ou ("bn", canais) ou
# ("densa", entradas, neuronios). O BN (batch normalization) aprende 2 números por canal.
# ---------------------------------------------------------------------------------------------

def conv(c_in, filtros, k, vies=True):
    return ("conv", c_in, filtros, k, vies)


def densa(entradas, neuronios):
    return ("densa", entradas, neuronios)


def bn(canais):
    return ("bn", canais)


def parametros(camada):
    tipo = camada[0]
    if tipo == "conv":
        _, c_in, filtros, k, vies = camada
        return filtros * (k * k * c_in) + (filtros if vies else 0)
    if tipo == "densa":
        _, entradas, neuronios = camada
        return entradas * neuronios + neuronios
    if tipo == "bn":
        return 2 * camada[1]
    raise ValueError(tipo)


def alexnet():
    """A AlexNet como o torchvision monta (uma GPU só; o artigo dividia a rede em duas GPUs)."""
    return [conv(3, 64, 11), conv(64, 192, 5), conv(192, 384, 3), conv(384, 256, 3), conv(256, 256, 3),
            densa(256 * 6 * 6, 4096), densa(4096, 4096), densa(4096, 1000)]


VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


def vgg16():
    """Configuração D do artigo: só convoluções 3 x 3, cinco poolings e três densas."""
    camadas, c = [], 3
    for item in VGG16:
        if item != "M":
            camadas.append(conv(c, item, 3))
            c = item
    return camadas + [densa(512 * 7 * 7, 4096), densa(4096, 4096), densa(4096, 1000)]


def resnet(blocos, gargalo):
    """ResNet-18 (gargalo=False, blocos 2 2 2 2) e ResNet-50 (gargalo=True, blocos 3 4 6 3).

    Como no artigo e no torchvision: convolução sem viés seguida de BN, atalho com projeção 1 x 1 só no
    primeiro bloco de cada etapa que muda o formato, média global e uma densa de 1000.
    Devolve as camadas e quantas camadas com peso ficam no caminho principal (o número do nome).
    """
    camadas = [conv(3, 64, 7, False), bn(64)]
    com_peso = 1
    c = 64
    for etapa, n in enumerate(blocos):
        largura = 64 * 2 ** etapa
        saida = largura * 4 if gargalo else largura
        for i in range(n):
            if gargalo:
                caminho = [conv(c, largura, 1, False), bn(largura), conv(largura, largura, 3, False),
                           bn(largura), conv(largura, saida, 1, False), bn(saida)]
                com_peso += 3
            else:
                caminho = [conv(c, largura, 3, False), bn(largura), conv(largura, largura, 3, False), bn(largura)]
                com_peso += 2
            camadas += caminho
            if i == 0 and (c != saida or etapa > 0):
                camadas += [conv(c, saida, 1, False), bn(saida)]  # projeção do atalho
            c = saida
    camadas.append(densa(c, 1000))
    return camadas, com_peso + 1


def parte1():
    print("1. Onde ficam os parâmetros (as redes como o torchvision monta)")
    r18, n18 = resnet([2, 2, 2, 2], False)
    r50, n50 = resnet([3, 4, 6, 3], True)
    redes = [("AlexNet", alexnet(), 8), ("VGG16", vgg16(), 16), ("ResNet-18", r18, n18), ("ResNet-50", r50, n50)]
    print(f"  {'rede':10} {'camadas com peso':>16} {'convoluções e BN':>17} {'densas':>12} {'total':>12} "
          f"{'nas densas':>11}")
    for nome, camadas, n in redes:
        total = sum(parametros(c) for c in camadas)
        dens = sum(parametros(c) for c in camadas if c[0] == "densa")
        print(f"  {nome:10} {n:>16} {fmt(total - dens):>17} {fmt(dens):>12} {fmt(total):>12} {pct(dens, total):>11}")
    vgg = vgg16()
    primeira = parametros(vgg[13])
    print(f"  a primeira densa da VGG16 (7 x 7 x 512 = {fmt(512 * 49)} entradas -> 4096): {fmt(primeira)}, "
          f"{pct(primeira, sum(parametros(c) for c in vgg))} da rede")
    atalhos = [c for c in r18 if c[0] == "conv" and c[3] == 1]
    print(f"  atalhos com projeção 1 x 1 na ResNet-18: {len(atalhos)}, com "
          f"{fmt(sum(parametros(c) for c in atalhos))} parâmetros (sem contar o BN deles); "
          f"os outros {8 - len(atalhos)} atalhos não têm nenhum")
    print()


# ---------------------------------------------------------------------------------------------
# Parte 2. Três convoluções 3 x 3 contra uma 7 x 7.
# ---------------------------------------------------------------------------------------------

def espalhar(campo, k):
    """Uma convolução k x k com todos os pesos 1 e borda (k - 1) / 2: cada pixel passa a enxergar
    os vizinhos. Serve para ver até onde chega a influência de um pixel só."""
    b = (k - 1) // 2
    x = np.pad(campo, b)
    saida = np.zeros_like(campo)
    for i in range(campo.shape[0]):
        for j in range(campo.shape[1]):
            saida[i, j] = x[i:i + k, j:j + k].sum()
    return saida


def lado_tocado(campo):
    linhas = np.nonzero(campo.any(axis=1))[0]
    return int(linhas[-1] - linhas[0] + 1)


def parte2():
    print("2. Três convoluções 3 x 3 contra uma 7 x 7")
    campo = np.zeros((15, 15))
    campo[7, 7] = 1
    print("  um pixel aceso no meio de um campo 15 x 15; quantos pixels da saída ele alcança:")
    x = campo
    for camada in (1, 2, 3):
        x = espalhar(x, 3)
        nome = "convolução" if camada == 1 else "convoluções"
        print(f"    depois de {camada} {nome} 3 x 3: {lado_tocado(x)} x {lado_tocado(x)}")
    print(f"    depois de 1 convolução 7 x 7: {lado_tocado(espalhar(campo, 7))} x {lado_tocado(espalhar(campo, 7))}")
    c = 64
    tres = 3 * parametros(conv(c, c, 3))
    uma = parametros(conv(c, c, 7))
    print(f"  com {c} canais na entrada e na saída:")
    print(f"    três 3 x 3: 3 x ({c} x (3 x 3 x {c}) + {c}) = {fmt(tres)} parâmetros, 3 ReLU")
    print(f"    uma 7 x 7: {c} x (7 x 7 x {c}) + {c} = {fmt(uma)} parâmetros, 1 ReLU")
    print(f"    só os pesos: 27 x {c} x {c} = {fmt(27 * c * c)} contra 49 x {c} x {c} = {fmt(49 * c * c)} "
          f"({pct(49 * c * c - 27 * c * c, 27 * c * c)} a mais na 7 x 7)")
    print()


# ---------------------------------------------------------------------------------------------
# Parte 3. O tamanho do mapa ao longo da VGG16 e da ResNet-50, em fotos 224 x 224.
# ---------------------------------------------------------------------------------------------

def lado(n, k, passo=1, borda=0):
    return (n + 2 * borda - k) // passo + 1


def parte3():
    print("3. O tamanho do mapa, fotos 224 x 224 com 3 canais")
    n, c, linha = 224, 3, []
    for item in VGG16:
        if item == "M":
            n = lado(n, 2, 2)
            linha.append(f"{c} x {n} x {n}")
        else:
            c = item  # 3 x 3 com borda 1: o lado não muda
    print("  VGG16, depois de cada um dos 5 poolings: " + "; ".join(linha))
    print(f"  VGG16, achatar: 512 x 7 x 7 = {fmt(512 * 49)} números para a primeira densa")
    n = lado(224, 7, 2, 3)
    etapas = [f"entrada da rede, convolução 7 x 7 passo 2: 64 x {n} x {n}"]
    n = lado(n, 3, 2, 1)
    etapas.append(f"pooling 3 x 3 passo 2: 64 x {n} x {n}")
    c = 256
    etapas.append(f"etapa 1: {c} x {n} x {n}")
    for etapa in (2, 3, 4):
        n = lado(n, 3, 2, 1)
        c *= 2
        etapas.append(f"etapa {etapa}: {c} x {n} x {n}")
    print("  ResNet-50:")
    for e in etapas:
        print("    " + e)
    print(f"  ResNet-50, média global: os {c} mapas de {n} x {n} viram {c} números, um por mapa")
    print()


# ---------------------------------------------------------------------------------------------
# Parte 4. O atalho: x + F(x).
# Cada bloco é F(x) = W2 . ReLU(W1 . x), com 16 números entrando e saindo.
# Rede simples: x passa a ser F(x). Rede com atalho: x passa a ser x + F(x). Mesmos pesos nas duas.
# ---------------------------------------------------------------------------------------------

LARGURA = 16


def sortear_pesos(blocos, desvio, gerador):
    return [(gerador.normal(0, desvio, (LARGURA, LARGURA)), gerador.normal(0, desvio, (LARGURA, LARGURA)))
            for _ in range(blocos)]


def passar(x, pesos, atalho):
    """Passa x por todos os blocos e volta pelo caminho do gradiente (a regra da cadeia da aula 02).

    A conta de erro é a soma das saídas. Devolve a saída e o gradiente que chega na entrada da rede.
    """
    guardados = []
    for W1, W2 in pesos:
        z = W1 @ x
        f = W2 @ np.maximum(z, 0)
        guardados.append(z)
        x = x + f if atalho else f
    g = np.ones(LARGURA)
    for (W1, W2), z in zip(reversed(pesos), reversed(guardados)):
        g_f = W1.T @ ((W2.T @ g) * (z > 0))
        g = g + g_f if atalho else g_f
    return x, g


def virgula(v):
    """6.4 vira 6,40; números muito pequenos ou muito grandes saem como 2,17e-03."""
    texto = f"{v:.2f}" if 0.01 <= abs(v) < 1000 or v == 0 else f"{v:.2e}"
    return texto.replace(".", ",")


def norma(v):
    return float(np.sqrt((v ** 2).sum()))


def parte4():
    print("4. O atalho: x + F(x)")
    x = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 0.0, 2.0, -0.5, 1.5, -3.0, 0.25, 1.0, -1.5, 2.5, -0.25, 0.75])
    zeros = [(np.zeros((LARGURA, LARGURA)), np.zeros((LARGURA, LARGURA)))]
    print("  um bloco com todos os pesos em zero:")
    for rotulo, v in (("entrada:   ", x), ("sem atalho:", passar(x, zeros, False)[0] + 0.0),
                      ("com atalho:", passar(x, zeros, True)[0])):
        print("    " + rotulo + " ".join(f"{virgula(n):>6}" for n in v[:6]) + " ...")
    resumo = {}
    for desvio, profundidades in ((0.05, (2, 10, 50)), (0.25, (50,))):
        print(f"  a mesma entrada (tamanho {virgula(norma(x))}) por uma pilha de blocos, "
              f"pesos sorteados com desvio {virgula(desvio)}:")
        print(f"    {'blocos':>6} {'saída sem atalho':>17} {'saída com atalho':>17} "
              f"{'gradiente na entrada, sem atalho':>33} {'com atalho':>11}")
        for blocos in profundidades:
            pesos = sortear_pesos(blocos, desvio, np.random.default_rng(7))
            s_sem, g_sem = passar(x, pesos, False)
            s_com, g_com = passar(x, pesos, True)
            resumo[(desvio, blocos)] = (norma(s_sem), norma(s_com), norma(g_sem), norma(g_com))
            print(f"    {blocos:>6} {virgula(norma(s_sem)):>17} {virgula(norma(s_com)):>17} "
                  f"{virgula(norma(g_sem)):>33} {virgula(norma(g_com)):>11}")
    print(f"  (o gradiente que sai da conta de erro tem tamanho {virgula(norma(np.ones(LARGURA)))}; "
          f"e-15 quer dizer vezes 10 elevado a -15)")
    print()
    return resumo


# ---------------------------------------------------------------------------------------------
# Parte 5. Quando o formato muda, o atalho precisa de uma projeção 1 x 1.
# ---------------------------------------------------------------------------------------------

def parte5():
    print("5. Atalho quando o bloco muda o formato")
    n, c_in, c_out = 56, 64, 128
    saida = lado(n, 3, 2, 1)
    print(f"  entrada do bloco: {c_in} x {n} x {n}; caminho principal com convolução 3 x 3 passo 2: "
          f"{c_out} x {saida} x {saida}")
    print(f"  somar {c_in} x {n} x {n} com {c_out} x {saida} x {saida}: não dá, os formatos são diferentes")
    proj = lado(n, 1, 2, 0)
    print(f"  projeção 1 x 1, {c_out} filtros, passo 2: {c_out} x {proj} x {proj}; "
          f"parâmetros {c_out} x (1 x 1 x {c_in}) + {c_out} = {fmt(parametros(conv(c_in, c_out, 1)))}")
    print(f"  caminho principal: {c_out} x (3 x 3 x {c_in}) + {c_out} = {fmt(parametros(conv(c_in, c_out, 3)))} "
          f"e {c_out} x (3 x 3 x {c_out}) + {c_out} = {fmt(parametros(conv(c_out, c_out, 3)))}")


if __name__ == "__main__":
    parte1()
    parte2()
    parte3()
    parte4()
    parte5()
