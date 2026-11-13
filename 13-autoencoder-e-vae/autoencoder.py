"""Tudo o que a aula «Autoencoder e VAE» mostra, partes 1 a 7.

Rode de dentro da pasta da aula:  python3 autoencoder.py
Só usa o NumPy. As fotos de 3 x 3 pixels são geradas aqui mesmo e saem iguais toda vez.
"""
import numpy as np

np.set_printoptions(suppress=True)
g = np.random.default_rng(13)


def br(v, casas=3):
    """Número com vírgula, como no texto da aula."""
    return f"{v:.{casas}f}".replace(".", ",")


def grade(v):
    """Uma foto de 9 pixels em 3 linhas."""
    linhas = []
    for i in range(3):
        linhas.append("      " + "  ".join(f"{p:5.2f}".replace(".", ",") for p in v[3 * i:3 * i + 3]))
    return "\n".join(linhas)


# ---------------------------------------------------------------- parte 1
print("1. Um pedaço de foto, duas reconstruções e duas perdas")
x = np.array([0.0, 1.0, 1.0, 0.0])
recs = {"A": np.array([0.01, 0.99, 0.99, 0.45]), "B": np.array([0.2, 0.8, 0.8, 0.2])}
print("  pedaço (fundo, tecido, tecido, fundo):", "  ".join(br(p, 2) for p in x))
for nome, r in recs.items():
    # probabilidade de Bernoulli de cada pixel: r se o pixel é 1, (1 - r) se é 0
    prob = np.where(x == 1, r, 1 - r)
    soma_quad = float(((x - r) ** 2).sum())
    print(f"  reconstrução {nome}:", "  ".join(br(p, 2) for p in r))
    print(f"    soma dos erros ao quadrado: {br(soma_quad, 4)}")
    print(f"    probabilidade de cada pixel (Bernoulli): {'  '.join(br(p, 2) for p in prob)}")
    print(f"    -log do produto das probabilidades: {br(float(-np.log(np.prod(prob))), 3)}")
print("  quem ganha: erro ao quadrado prefere B; Bernoulli prefere A")

# ---------------------------------------------------------------- parte 2
print()
print("2. Um pixel cinza (0,5) e as duas perdas")
print("  reconstrução   erro ao quadrado   -log Bernoulli")
cinza = 0.5
for r in (0.1, 0.3, 0.5, 0.7, 0.9):
    # com o pixel em 0,5, a perda de Bernoulli é a média de -log r e -log (1 - r)
    bern = -(cinza * np.log(r) + (1 - cinza) * np.log(1 - r))
    print(f"  {br(r, 1):>12}   {br((cinza - r) ** 2, 4):>16}   {br(float(bern), 4):>14}")
print(f"  o menor valor de -log Bernoulli é {br(float(np.log(2)), 4)} (log de 2), com a reconstrução perfeita")

# ---------------------------------------------------------------- dados das fotos 3 x 3
CAMISETA = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0], dtype=float)
CALCA = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1], dtype=float)
BOLSA = np.array([0, 1, 0, 1, 1, 1, 1, 1, 1], dtype=float)
N = 120
tipo = np.repeat([0, 1], N // 2)
brilho = g.uniform(0.6, 1.0, N)
fotos = np.where(tipo[:, None] == 0, CAMISETA, CALCA) * brilho[:, None] + g.normal(0, 0.03, (N, 9))

# ---------------------------------------------------------------- parte 3
print()
print("3. Um autoencoder linear: 9 pixels -> 2 números -> 9 pixels")
D, J = 9, 2
We = g.normal(0, 0.1, (J, D)); be = np.zeros(J)
Wd = g.normal(0, 0.1, (D, J)); bd = np.zeros(D)


def erro_medio(rec):
    """Soma dos erros ao quadrado por foto, média nas fotos."""
    return float(((fotos - rec) ** 2).sum(axis=1).mean())


print(f"  {N} fotos de treino: {N // 2} camisetas e {N // 2} calças, brilho entre 0,6 e 1,0")
print(f"  erro antes do treino: {br(erro_medio(np.dot(np.dot(fotos, We.T) + be, Wd.T) + bd), 4)}")
taxa = 0.05
for passo in range(4000):
    z = np.dot(fotos, We.T) + be
    rec = np.dot(z, Wd.T) + bd
    d_rec = 2 * (rec - fotos) / N
    gWd = np.dot(d_rec.T, z); gbd = d_rec.sum(0)
    d_z = np.dot(d_rec, Wd)
    gWe = np.dot(d_z.T, fotos); gbe = d_z.sum(0)
    We -= taxa * gWe; be -= taxa * gbe; Wd -= taxa * gWd; bd -= taxa * gbd
codigos = np.dot(fotos, We.T) + be
rec = np.dot(codigos, Wd.T) + bd
erro_treinado = erro_medio(rec)
print(f"  erro depois de 4000 passos: {br(erro_treinado, 4)} por foto")
print("  códigos de 4 fotos (os 2 números do meio):")
for i in (0, 1, N // 2, N // 2 + 1):
    print(f"    {'camiseta' if tipo[i] == 0 else 'calça':8}  z = ({br(codigos[i, 0], 2)}; {br(codigos[i, 1], 2)})")
fx = [codigos[tipo == t] for t in (0, 1)]
for t, nome in ((0, "camisetas"), (1, "calças")):
    c = fx[t]
    print(f"    {nome:9} z1 de {br(c[:, 0].min(), 2)} a {br(c[:, 0].max(), 2)}; "
          f"z2 de {br(c[:, 1].min(), 2)} a {br(c[:, 1].max(), 2)}")

print("  o mesmo autoencoder com o código multiplicado por 10 e o decodificador dividido por 10:")
codigos10 = codigos * 10
rec10 = np.dot(codigos10, (Wd / 10).T) + bd
print(f"    erro: {br(erro_medio(rec10), 4)} por foto (igual: {np.allclose(rec, rec10)})")
print(f"    código da 1ª camiseta: ({br(codigos10[0, 0], 2)}; {br(codigos10[0, 1], 2)})")

# ---------------------------------------------------------------- parte 4
print()
print("4. Sortear um código e decodificar")
sorteio = g.normal(0, 1, (3, J))
for versao, cod, W in (("código original", codigos, Wd), ("código x 10", codigos10, Wd / 10)):
    print(f"  {versao}:")
    for k, zk in enumerate(sorteio):
        perto = float(np.sqrt(((cod - zk) ** 2).sum(1)).min())
        foto = np.dot(W, zk) + bd
        d_cam = float(((foto - CAMISETA * 0.8) ** 2).sum())
        d_cal = float(((foto - CALCA * 0.8) ** 2).sum())
        print(f"    z = ({br(zk[0], 2)}; {br(zk[1], 2)}), código de treino mais perto a {br(perto, 2)}")
        print(grade(foto))
        print(f"      distância à camiseta média: {br(d_cam, 2)}; à calça média: {br(d_cal, 2)}")
viz = np.sqrt(((codigos[:, None, :] - codigos[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(viz, np.inf)
print(f"  distância típica entre um código de treino e o vizinho mais perto: {br(float(np.median(viz.min(1))), 2)} "
      f"(original) e {br(float(np.median(viz.min(1))) * 10, 2)} (x 10)")

# ---------------------------------------------------------------- parte 5
print()
print("5. A divergência KL entre N(mu, sigma²) e N(0, 1), por integral numérica")
print("     mu   sigma   KL")
for mu, s in ((0.0, 1.0), (1.0, 0.5), (2.0, 1.0), (0.0, 0.1), (0.0, 3.0)):
    t = np.linspace(mu - 12 * s, mu + 12 * s, 400001)
    q = np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    logq = -0.5 * ((t - mu) / s) ** 2 - np.log(s * np.sqrt(2 * np.pi))
    logp = -0.5 * t ** 2 - np.log(np.sqrt(2 * np.pi))
    kl = float(np.sum(q * (logq - logp)) * (t[1] - t[0]))
    print(f"  {br(mu, 1):>5}   {br(s, 1):>5}   {br(kl, 4)}")
for versao, cod in (("original", codigos), ("x 10", codigos10)):
    print(f"  só o custo do centro (mu²/2, somado nas 2 dimensões), média das fotos, código {versao}: "
          f"{br(float((cod ** 2).sum(1).mean() / 2), 2)}")

# ---------------------------------------------------------------- parte 6
print()
print("6. A perda do VAE numa foto: o decodificador da parte 3, mu = código da 1ª camiseta, vários sigma")
S = 0.1                         # desvio do ruído de cada pixel na hipótese gaussiana
foto0 = fotos[0]; mu0 = codigos[0]
eps = g.normal(0, 1, (20000, J))
print(f"  hipótese do decodificador: cada pixel = reconstrução + ruído gaussiano de desvio {br(S, 1)}")
print(f"  reconstrução = média de (soma dos erros ao quadrado) / (2 x {br(S, 1)}²) em 20000 sorteios z = mu + sigma x eps")
print("  total = peso x reconstrução + KL; peso 1 é a soma nos pixels, 1/9 é a média nos 9 pixels,")
print("  1/784 é o peso que a média daria numa foto de 28 x 28")
print("          sigma   reconstrução      KL   peso 1   peso 1/9   peso 1/784")
linhas = []
for s in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
    z = mu0 + s * eps
    fotos_z = np.dot(z, Wd.T) + bd
    recon = float((((foto0 - fotos_z) ** 2).sum(1) / (2 * S ** 2)).mean())
    kl = float(0.5 * (mu0 ** 2 + s ** 2 - 1 - np.log(s ** 2)).sum())
    linhas.append((f"sigma {br(s, 2)}", recon, kl))
z = 0 + 1.0 * eps
recon = float((((foto0 - (np.dot(z, Wd.T) + bd)) ** 2).sum(1) / (2 * S ** 2)).mean())
linhas.append(("mu 0, sigma 1", recon, 0.0))
for nome, recon, kl in linhas:
    print(f"  {nome:>13}   {br(recon, 2):>12}   {br(kl, 2):>5}   {br(recon + kl, 2):>6}   "
          f"{br(recon / 9 + kl, 2):>8}   {br(recon / 784 + kl, 2):>10}")
for peso, rot in ((1, "1"), (9, "1/9"), (784, "1/784")):
    print(f"  menor total com peso {rot}: {min(linhas, key=lambda l: l[1] / peso + l[2])[0]}")

# ---------------------------------------------------------------- parte 7
print()
print("7. Erro de reconstrução como alarme: quatro fotos novas no autoencoder da parte 3")
novas = {"camiseta comum (brilho 0,8)": CAMISETA * 0.8,
         "camiseta e calça sobrepostas": 0.5 * CAMISETA + 0.5 * CALCA,
         "camiseta quase apagada (brilho 0,1)": CAMISETA * 0.1,
         "bolsa (forma que ele nunca viu)": BOLSA * 0.8}
erros_treino = ((fotos - rec) ** 2).sum(1)
print(f"  erro nas fotos de treino: mediana {br(float(np.median(erros_treino)), 4)}, maior {br(float(erros_treino.max()), 4)}")
for nome, f in novas.items():
    z = np.dot(We, f) + be
    r = np.dot(Wd, z) + bd
    print(f"  {nome:36} erro {br(float(((f - r) ** 2).sum()), 4)}")
