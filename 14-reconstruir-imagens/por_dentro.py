"""As duas redes por dentro, com uma foto de validação e os pesos do kit.

Rode de dentro da pasta da aula:  python3 por_dentro.py
"""
import numpy as np

from comparar import br
from redes import CLASSES, bernoulli_logits, codificar, decodificar, erro_quadrado, kl_por_numero, ler, \
    nova_rede, passada, preparar, refazer

ae, _ = ler("autoencoder", 16)
vae, _ = ler("vae", 16)
_, _, Xv, yv, _, _ = preparar()
i = 1748  # uma foto de erro mediano na validação
x = Xv[i:i + 1]
print(f"A foto {i} da validação: {CLASSES[yv[i]]}, {int((x > 0).sum())} pixels acesos de 784")

print("\n1. O caminho de uma foto")
z = codificar(ae, x)
print(f"  autoencoder: 784 pixels -> {ae['W1'].shape[0]} -> código {z.shape[1]} -> {ae['W3'].shape[0]} -> 784")
mu, lv = codificar(vae, x)
print(f"  VAE:         784 pixels -> {vae['W1'].shape[0]} -> mu {mu.shape[1]} e log_var {lv.shape[1]}"
      f" -> z {mu.shape[1]} -> {vae['W3'].shape[0]} -> 784")
print("  código do autoencoder: " + "  ".join(br(v) for v in z[0, :6]) + "  ...")
print("  mu do VAE:             " + "  ".join(br(v) for v in mu[0, :6]) + "  ...")
print("  sigma do VAE:          " + "  ".join(br(v) for v in np.exp(0.5 * lv[0, :6])) + "  ...")

print("\n2. A perda dessa foto, termo por termo")
logit = lambda p: np.log(np.clip(p, 1e-12, 1)) - np.log(np.clip(1 - p, 1e-12, 1))  # noqa: E731
rec_ae = float(bernoulli_logits(x, logit(decodificar(ae, z)))[0])
rec_vae = float(bernoulli_logits(x, logit(decodificar(vae, mu)))[0])
kl = kl_por_numero(mu, lv)[0]
piso = float(bernoulli_logits(x, logit(x))[0])
print(f"  autoencoder: Bernoulli {br(rec_ae)}")
print(f"  VAE: Bernoulli {br(rec_vae)} (a partir de mu) + KL {br(kl.sum())} = {br(rec_vae + kl.sum())}")
print("  KL de cada número do código: " + "  ".join(br(v) for v in np.sort(kl)[::-1]))
print(f"  piso: a mesma perda de Bernoulli com a foto refeita igual à original dá {br(piso)}")

print("\n3. O sorteio do VAE: z = mu + sigma x eps, três vezes, e a foto refeita a partir de mu")
rng = np.random.default_rng(5)
for k in range(3):
    eps = rng.standard_normal(mu.shape)
    zz = mu + np.exp(0.5 * lv) * eps
    print(f"  sorteio {k + 1}: erro ao quadrado {br(erro_quadrado(x, decodificar(vae, zz))[0])}")
print(f"  a partir de mu: {br(erro_quadrado(x, refazer(vae, x))[0])};"
      f" autoencoder: {br(erro_quadrado(x, refazer(ae, x))[0])}")

print("\n4. O gradiente: a conta da volta contra a diferença finita, em 5 fotos e pesos sorteados")
rng = np.random.default_rng(0)
xb = Xv[:5]
for tipo in ("autoencoder", "vae"):
    r = nova_rede(tipo, 4, 0)
    eps = rng.standard_normal((5, 4)) if tipo == "vae" else None
    _, _, _, g = passada(r, xb, eps)
    pior = 0.0
    for k in sorted(r):
        for _ in range(4):
            pos = tuple(int(rng.integers(0, s)) for s in r[k].shape)
            antes = r[k][pos]
            r[k][pos] = antes + 1e-5
            mais = passada(r, xb, eps)[0]
            r[k][pos] = antes - 1e-5
            menos = passada(r, xb, eps)[0]
            r[k][pos] = antes
            fd = (mais - menos) / 2e-5
            pior = max(pior, abs(fd - g[k][pos]) / max(abs(fd) + abs(g[k][pos]), 1e-12))
    print(f"  {tipo:<12} maior diferença relativa em {4 * len(r)} pesos: {pior:.1e}")

print("\n5. O erro soma os pixels: a mesma foto com menos brilho")
for b in (1.0, 0.6, 0.3, 0.1):
    xx = b * x
    print(f"  brilho {br(100 * b, 0):>3}%: soma dos pixels {br(xx.sum(), 1):>6}   erro do autoencoder"
          f" {br(erro_quadrado(xx, refazer(ae, xx))[0])}   do VAE {br(erro_quadrado(xx, refazer(vae, xx))[0])}")
