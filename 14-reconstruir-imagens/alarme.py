"""O erro de reconstrução como alarme: o que ele pega e o que deixa passar.

Rode de dentro da pasta da aula:
  python3 alarme.py                          os pesos do kit, código de 16 números
  python3 alarme.py --codigo 2 --pesos meus-pesos
O limiar sai da validação. Só depois o alarme vê o teste, sem mexer e com defeitos feitos a partir dele.
Se a função etiqueta do meu_defeito.py estiver escrita, a parte 4 testa a etiqueta do exercício.
Grava defeitos-<codigo>.png se o matplotlib estiver instalado.
"""
import argparse

import numpy as np

from comparar import br, grade
from redes import AQUI, CLASSES, erro_quadrado, ler, preparar, refazer

PERCENTIL = 99


def defeitos(x):
    """Os defeitos da parte 3, cada um feito a partir das mesmas fotos de teste."""
    rng = np.random.default_rng(3)
    fotos = x.reshape(-1, 28, 28)
    deslocada = np.zeros_like(fotos)
    deslocada[:, :, 8:] = fotos[:, :, :-8]
    outra = x[rng.permutation(len(x))]
    return {
        "sem mexer": x,
        "negativo (1 - pixel)": 1.0 - x,
        "ruído (desvio 0,2)": np.clip(x + rng.normal(0, 0.2, x.shape), 0, 1),
        "girada 90 graus": np.rot90(fotos, 1, axes=(1, 2)).reshape(len(x), 784),
        "8 pixels para a direita": deslocada.reshape(len(x), 784),
        "duas fotos sobrepostas": np.maximum(x, outra),
        "escura (30% do brilho)": 0.3 * x,
        "vazia (tudo preto)": np.zeros_like(x),
    }


def main():
    ap = argparse.ArgumentParser(description="Testa o erro de reconstrução como alarme.")
    ap.add_argument("--codigo", type=int, default=16)
    ap.add_argument("--pesos", default="pesos", help="pasta dos pesos (padrão: pesos, os do kit)")
    a = ap.parse_args()
    pasta = AQUI / a.pesos
    redes = {"autoencoder": ler("autoencoder", a.codigo, pasta)[0], "VAE": ler("vae", a.codigo, pasta)[0]}
    _, _, Xv, _, Xs, ys = preparar()

    print(f"1. O limiar, na validação: o erro que só {100 - PERCENTIL}% das fotos boas passam (código de {a.codigo})")
    limiar = {}
    for nome, rede in redes.items():
        e = erro_quadrado(Xv, refazer(rede, Xv))
        limiar[nome] = float(np.percentile(e, PERCENTIL))
        print(f"  {nome:<12} limiar {br(limiar[nome])}   (mediana da validação {br(np.median(e))})")

    print("\n2. As 10.000 fotos de teste, sem mexer: quanto o alarme apita, por tipo")
    erro_teste = {n: erro_quadrado(Xs, refazer(r, Xs)) for n, r in redes.items()}
    print("  tipo             autoencoder      VAE")
    for c, nome in enumerate(CLASSES):
        m = ys == c
        print(f"  {nome:<14} " + "   ".join(f"{br(100 * (erro_teste[n][m] > limiar[n]).mean(), 1):>8}%" for n in redes))
    print("  total          " + "   ".join(f"{br(100 * (erro_teste[n] > limiar[n]).mean(), 1):>8}%" for n in redes))

    print("\n3. As mesmas 10.000 fotos de teste com um defeito: quanto o alarme pega")
    print("  defeito                    autoencoder pega   mediana do erro      VAE pega   mediana do erro")
    casos = defeitos(Xs)
    for nome, x in casos.items():
        linha = f"  {nome:<26}"
        for n, rede in redes.items():
            e = erro_quadrado(x, refazer(rede, x))
            linha += f" {br(100 * (e > limiar[n]).mean(), 1):>15}%   {br(np.median(e)):>15}"
        print(linha)
    exemplo = [0, 1, 2, 3, 4, 5]
    curtos = ["original", "negativo", "ruído", "girada", "deslocada", "sobreposta", "escura", "vazia"]
    grade([casos[k][exemplo] for k in casos], curtos, f"defeitos-{a.codigo}.png",
          "As mesmas 6 fotos de teste com cada defeito")

    print("\n4. A etiqueta do exercício (meu_defeito.py)")
    try:
        from meu_defeito import etiqueta
        lugares = {"etiqueta no canto": (0, 0), "etiqueta no meio": (11, 11)}
        for nome, (lin, col) in lugares.items():
            x = etiqueta(Xs, lin, col)
            linha = f"  {nome:<26}"
            for n, rede in redes.items():
                e = erro_quadrado(x, refazer(rede, x))
                linha += f" {br(100 * (e > limiar[n]).mean(), 1):>15}%   {br(np.median(e)):>15}"
            print(linha)
    except NotImplementedError:
        print("  a função etiqueta ainda não foi escrita: veja o exercício")


if __name__ == "__main__":
    main()
