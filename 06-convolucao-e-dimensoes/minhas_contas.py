"""Exercício da aula «Convolução e dimensões»: as contas da rede B.

Escreva as três funções. O resto do arquivo já está pronto: ele usa as suas funções para montar a
tabela da rede B, camada por camada. Rode de dentro da pasta da aula:

    python3 testar_minhas_contas.py
    python3 minhas_contas.py

Antes de você escrever as funções, os dois comandos param com NotImplementedError.
"""

# A rede B que a colega mandou para o Caio. Cada camada é uma tupla:
#   ("conv", filtros, k, passo, borda)   convolução com filtros k x k e ReLU
#   ("pool", k, passo)                   max pooling k x k
#   ("densa", neuronios)                 camada densa; antes dela, o mapa é achatado
REDE_B = [
    ("conv", 16, 5, 1, 2),
    ("pool", 2, 2),
    ("conv", 32, 3, 2, 1),
    ("conv", 64, 3, 1, 0),
    ("pool", 2, 2),
    ("densa", 10),
]


def dimensao_saida(n, k, passo=1, borda=0):
    """Altura (ou largura) da saída de uma convolução ou de um pooling, para uma entrada n x n.

    Retorna um número inteiro. Para com ValueError se passo for menor que 1, se borda for negativa
    ou se o kernel não couber na entrada com a borda (k maior que n + 2 x borda).
    """
    raise NotImplementedError("escreva dimensao_saida")


def parametros_conv(canais, filtros, k):
    """Quantos pesos e vieses uma convolução aprende: filtros k x k sobre uma entrada com `canais` canais.

    Para com ValueError se algum dos três for menor que 1.
    """
    raise NotImplementedError("escreva parametros_conv")


def parametros_densa(entradas, neuronios):
    """Quantos pesos e vieses uma camada densa aprende. Para com ValueError se algum for menor que 1."""
    raise NotImplementedError("escreva parametros_densa")


# Daqui para baixo já está pronto. Não precisa mexer.

def percorrer(rede, n, canais):
    """Passa pela rede com as suas funções. Retorna uma linha por camada: (nome, saída, parâmetros)."""
    linhas = [("entrada", f"{canais} x {n} x {n}", 0)]
    achatado = None
    for camada in rede:
        if camada[0] == "conv":
            _, filtros, k, passo, borda = camada
            params = parametros_conv(canais, filtros, k)
            n, canais = dimensao_saida(n, k, passo, borda), filtros
            linhas.append((f"conv {filtros} filtros {k} x {k}, passo {passo}, borda {borda}",
                           f"{canais} x {n} x {n}", params))
        elif camada[0] == "pool":
            _, k, passo = camada
            n = dimensao_saida(n, k, passo)
            linhas.append((f"max pool {k} x {k}, passo {passo}", f"{canais} x {n} x {n}", 0))
        elif camada[0] == "densa":
            entradas = achatado if achatado is not None else canais * n * n
            if achatado is None:
                linhas.append(("achatar", str(entradas), 0))
            neuronios = camada[1]
            achatado = neuronios
            linhas.append((f"densa {entradas} -> {neuronios}", str(neuronios),
                           parametros_densa(entradas, neuronios)))
        else:
            raise ValueError(f"camada desconhecida: {camada[0]}")
    return linhas


def milhar(numero):
    return f"{numero:,}".replace(",", ".")


def mostrar(rede, n, canais):
    linhas = percorrer(rede, n, canais)
    print(f"  {'camada':40} {'saída':>14} {'parâmetros':>11}")
    for nome, saida, params in linhas:
        print(f"  {nome:40} {saida:>14} {milhar(params):>11}")
    total = sum(p for _, _, p in linhas)
    print(f"  {'total':40} {'':>14} {milhar(total):>11}")
    return total


if __name__ == "__main__":
    print("Rede B, fotos 28 x 28 com 1 canal")
    mostrar(REDE_B, 28, 1)
    print("\nRede B, fotos 56 x 56 com 1 canal")
    mostrar(REDE_B, 56, 1)
