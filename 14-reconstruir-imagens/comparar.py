"""Compara o autoencoder e o VAE nas mesmas fotos de validação: o que cada um refaz e o que sai de um sorteio.

Rode de dentro da pasta da aula:
  python3 comparar.py                          os pesos do kit, código de 16 números
  python3 comparar.py --codigo 2 --pesos meus-pesos
Grava reconstrucoes-<codigo>.png e sorteios-<codigo>.png se o matplotlib estiver instalado.
"""
import argparse

import numpy as np

from redes import AQUI, CLASSES, ORCAMENTO, codificar, decodificar, erro_quadrado, kl_por_numero, ler, parametros, \
    preparar, refazer


def br(v, casas=2):
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def grade(linhas, rotulos, arquivo, titulo):
    """Uma linha de fotos 28 x 28 por item de linhas. Sem o matplotlib, só avisa."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  (sem o matplotlib: {arquivo} não foi gravado)")
        return
    n_l, n_c = len(linhas), len(linhas[0])
    fig, eixos = plt.subplots(n_l, n_c, figsize=(n_c * 1.1, n_l * 1.25), squeeze=False)
    for i, (fila, rot) in enumerate(zip(linhas, rotulos)):
        for j, foto in enumerate(fila):
            ax = eixos[i][j]
            ax.imshow(np.clip(foto, 0, 1).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(rot, fontsize=8)
    fig.suptitle(titulo, fontsize=9)
    fig.tight_layout()
    fig.savefig(AQUI / arquivo, dpi=110)
    plt.close(fig)
    print(f"  grade gravada em {arquivo}")


def mais_perto(a, b, lote=500, posicao=False):
    """Para cada linha de a, a menor distância (euclidiana) até uma linha de b. Com posicao, também qual linha."""
    dist, onde = [], []
    b2 = (b ** 2).sum(axis=1)
    for i in range(0, len(a), lote):
        q = a[i:i + lote]
        d2 = (q ** 2).sum(axis=1)[:, None] + b2[None, :] - 2 * np.dot(q, b.T)
        onde.append(d2.argmin(axis=1))
        dist.append(np.sqrt(np.maximum(d2.min(axis=1), 0)))
    return (np.concatenate(dist), np.concatenate(onde)) if posicao else np.concatenate(dist)


def main():
    ap = argparse.ArgumentParser(description="Compara o autoencoder e o VAE na validação.")
    ap.add_argument("--codigo", type=int, default=16)
    ap.add_argument("--pesos", default="pesos", help="pasta dos pesos (padrão: pesos, os do kit)")
    a = ap.parse_args()
    pasta = AQUI / a.pesos
    ae, curva_ae = ler("autoencoder", a.codigo, pasta)
    vae, curva_vae = ler("vae", a.codigo, pasta)
    Xt, yt, Xv, yv, _, _ = preparar()

    print(f"1. As duas redes, código de {a.codigo} números, no mesmo orçamento")
    print(f"  {len(Xt)} fotos de treino, {len(Xv)} de validação; lote {ORCAMENTO['lote']}, "
          f"{ORCAMENTO['epocas']} épocas, Adam {ORCAMENTO['taxa']}; época escolhida pela perda de validação")
    print("  rede          parâmetros   época   perda de validação   reconstrução   KL")
    for nome, rede, curva in (("autoencoder", ae, curva_ae), ("VAE", vae, curva_vae)):
        e = curva[int(np.argmin(curva[:, 2]))]
        n_par = f"{parametros(rede):,}".replace(",", ".")
        print(f"  {nome:<12}  {n_par:>10}   {int(e[0]):>5}   {br(e[2]):>18}   {br(e[3]):>12}   {br(e[4])}")

    rec_ae, rec_vae = refazer(ae, Xv), refazer(vae, Xv)
    err_ae, err_vae = erro_quadrado(Xv, rec_ae), erro_quadrado(Xv, rec_vae)
    print("\n2. A foto refeita: soma dos erros ao quadrado por foto, na validação")
    print(f"  autoencoder: média {br(err_ae.mean())}, mediana {br(np.median(err_ae))}")
    print(f"  VAE (a partir de mu): média {br(err_vae.mean())}, mediana {br(np.median(err_vae))}")
    print(f"  fotos em que o VAE refaz melhor que o autoencoder: {int((err_vae < err_ae).sum())} de {len(Xv)}")
    print("  tipo             fotos   autoencoder   VAE    VAE / autoencoder")
    for c, nome in enumerate(CLASSES):
        m = yv == c
        print(f"  {nome:<14} {int(m.sum()):>7}   {br(err_ae[m].mean()):>11}   {br(err_vae[m].mean()):>5}"
              f"   {br(err_vae[m].mean() / err_ae[m].mean()):>6}")

    print("\n3. Onde o autoencoder mais erra: as 100 fotos de maior erro na validação (1%)")
    pior = np.argsort(-err_ae)
    contagem = np.bincount(yv[pior[:100]], minlength=10)
    for c in np.argsort(-contagem):
        if contagem[c]:
            print(f"  {CLASSES[c]:<14} {contagem[c]:>3} de 100   (o tipo tem {int((yv == c).sum())} das 10.000 fotos)")
    print("  as 8 piores:")
    for i in pior[:8]:
        print(f"    foto {i:>5} da validação  {CLASSES[yv[i]]:<14} autoencoder {br(err_ae[i]):>6}   VAE {br(err_vae[i]):>6}")
    tipicas = [int(np.argsort(np.abs(err_ae - np.median(err_ae)))[k]) for k in range(4)]
    print("  4 fotos de erro mediano: " + ", ".join(f"foto {i} ({CLASSES[yv[i]]}, {br(err_ae[i])})" for i in tipicas))
    escolha = tipicas + [int(i) for i in pior[:4]]
    grade([Xv[escolha], rec_ae[escolha], rec_vae[escolha]], ["original", "autoencoder", "VAE"],
          f"reconstrucoes-{a.codigo}.png", "4 fotos de erro mediano e as 4 de maior erro do autoencoder")

    print("\n4. Sortear um código de N(0, 1) e decodificar: 64 sorteios para cada rede")
    z_ae = codificar(ae, Xt)
    mu_vae, lv_vae = codificar(vae, Xt)
    print("  os códigos das fotos de treino, número por número:")
    print(f"    autoencoder: média de {br(z_ae.mean(axis=0).min())} a {br(z_ae.mean(axis=0).max())}, "
          f"desvio de {br(z_ae.std(axis=0).min())} a {br(z_ae.std(axis=0).max())}")
    print(f"    VAE (mu):    média de {br(mu_vae.mean(axis=0).min())} a {br(mu_vae.mean(axis=0).max())}, "
          f"desvio de {br(mu_vae.std(axis=0).min())} a {br(mu_vae.std(axis=0).max())}")
    z = np.random.default_rng(1).standard_normal((64, a.codigo))
    amostra = np.random.default_rng(2).choice(len(Xt), 10000, replace=False)
    viz_ae = mais_perto(z, z_ae[amostra])
    viz_vae = mais_perto(z, mu_vae[amostra])
    viz_treino_ae = mais_perto(z_ae[amostra[:500]], z_ae[amostra[500:]])
    viz_treino_vae = mais_perto(mu_vae[amostra[:500]], mu_vae[amostra[500:]])
    print("  distância do sorteio ao código de treino mais perto (mediana; 10.000 fotos de treino):")
    print(f"    autoencoder: {br(np.median(viz_ae))}   (entre duas fotos de treino: {br(np.median(viz_treino_ae))})")
    print(f"    VAE:         {br(np.median(viz_vae))}   (entre duas fotos de treino: {br(np.median(viz_treino_vae))})")
    f_ae, f_vae = decodificar(ae, z), decodificar(vae, z)
    d_ae, p_ae = mais_perto(f_ae, Xv, posicao=True)
    d_vae, p_vae = mais_perto(f_vae, Xv, posicao=True)
    d_ae, d_vae = d_ae ** 2, d_vae ** 2
    print("  erro ao quadrado da foto sorteada até a foto de validação mais parecida (mediana):")
    print(f"    autoencoder: {br(np.median(d_ae))}   VAE: {br(np.median(d_vae))}"
          f"   (a reconstrução mediana erra {br(np.median(err_ae))} e {br(np.median(err_vae))})")
    print("  tipo da foto de validação mais parecida, nos 64 sorteios:")
    for nome, p in (("autoencoder", p_ae), ("VAE", p_vae)):
        n = np.bincount(yv[p], minlength=10)
        print(f"    {nome:<12} " + ", ".join(f"{CLASSES[c]} {n[c]}" for c in np.argsort(-n, kind="stable") if n[c]))
    cinza = lambda f: float(((f > 0.15) & (f < 0.85)).mean())  # noqa: E731
    print(f"  pixels entre 0,15 e 0,85 (meio-tom): fotos de validação {br(100 * cinza(Xv), 1)}%, "
          f"sorteios do autoencoder {br(100 * cinza(f_ae), 1)}%, do VAE {br(100 * cinza(f_vae), 1)}%")
    grade([f_ae[:12], f_vae[:12]], ["autoencoder", "VAE"], f"sorteios-{a.codigo}.png",
          "12 códigos sorteados de N(0, 1), os mesmos para as duas redes")

    print("\n5. Quantos números do código o VAE usa: KL média por número, na validação")
    mu_v, lv_v = codificar(vae, Xv)
    kl = kl_por_numero(mu_v, lv_v).mean(axis=0)
    ordem = np.argsort(-kl)
    print("  " + "  ".join(br(kl[j]) for j in ordem))
    usados = int((kl > 0.1).sum())
    print(f"  números com KL acima de 0,1: {usados} de {a.codigo}; KL total por foto {br(kl.sum())}")
    sig = np.exp(0.5 * lv_v).mean(axis=0)
    print(f"  sigma médio dos números usados: de {br(sig[kl > 0.1].min())} a {br(sig[kl > 0.1].max())}"
          if usados else "  nenhum número usado")
    if usados < a.codigo:
        print(f"  sigma médio dos outros: de {br(sig[kl <= 0.1].min())} a {br(sig[kl <= 0.1].max())}")


if __name__ == "__main__":
    main()
