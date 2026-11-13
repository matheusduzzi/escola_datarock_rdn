"""O exercício da aula «Autoencoder e VAE»: as duas perdas do VAE do Caio.

Escreva as três funções marcadas com NotImplementedError. O resto do arquivo já está pronto:
ele usa as suas funções nos dois casos que o Caio escreveu à mão para testar a perda.

Rode de dentro da pasta da aula:
    python3 testar_minhas_perdas.py
    python3 minhas_perdas.py
"""
import numpy as np

LIMITE = 1e-6   # a reconstrução fica entre 0,000001 e 0,999999 antes do log


def kl_normal(mu, log_var):
    """KL entre N(mu, sigma²) e N(0, 1), somada nas dimensões do código.

    mu e log_var (o log de sigma²) têm o mesmo formato: (J,) para uma foto ou (n, J) para n fotos.
    Retorna um número para uma foto ou um array (n,) para n fotos.
    Para com ValueError se os formatos forem diferentes.
    """
    raise NotImplementedError("escreva kl_normal")


def reconstrucao_bernoulli(x, x_rec):
    """-log da probabilidade de Bernoulli de cada pixel, somada nos pixels.

    x são os pixels da foto, entre 0 e 1; x_rec é a reconstrução, do mesmo formato.
    Antes do log, a reconstrução fica entre LIMITE e 1 - LIMITE.
    Retorna um número para uma foto ou um array (n,) para n fotos.
    Para com ValueError se os formatos forem diferentes ou se algum pixel de x sair de [0, 1].
    """
    raise NotImplementedError("escreva reconstrucao_bernoulli")


def perda_vae(x, x_rec, mu, log_var, peso_kl=1.0):
    """Reconstrução de Bernoulli + peso_kl x KL, por foto.

    Para com ValueError se peso_kl for negativo (e nos casos das duas funções acima).
    """
    raise NotImplementedError("escreva perda_vae")


# ------------------------------------------------------------------ daqui para baixo já está pronto
# Os dois casos que o Caio escreveu à mão: um recorte de 4 pixels da mesma foto, duas
# reconstruções e duas saídas do codificador (2 números no código).
X = [0.0, 0.9, 1.0, 0.2]
CASOS = {
    "caso 1": {"x_rec": [0.1, 0.7, 0.95, 0.3], "mu": [0.6, -1.2], "log_var": [-2.0, -0.5]},
    "caso 2": {"x_rec": [0.3, 0.5, 0.5, 0.4], "mu": [0.05, -0.1], "log_var": [-0.05, 0.0]},
}


def br(v, casas=4):
    return f"{float(v):.{casas}f}".replace(".", ",")


def main():
    print("Recorte da foto:", "  ".join(br(p, 2) for p in X))
    for nome, c in CASOS.items():
        print(f"\n{nome}: reconstrução {'  '.join(br(p, 2) for p in c['x_rec'])}; "
              f"mu ({br(c['mu'][0], 2)}; {br(c['mu'][1], 2)}); log_var ({br(c['log_var'][0], 2)}; {br(c['log_var'][1], 2)})")
        for j in range(2):
            print(f"  KL do número {j + 1}:         {br(kl_normal(c['mu'][j:j + 1], c['log_var'][j:j + 1]))}")
        print(f"  KL somada:              {br(kl_normal(c['mu'], c['log_var']))}")
        print(f"  reconstrução Bernoulli: {br(reconstrucao_bernoulli(X, c['x_rec']))}")
        print(f"  perda do VAE (peso 1):  {br(perda_vae(X, c['x_rec'], c['mu'], c['log_var']))}")
    lote_x = np.array([X, X])
    lote_rec = np.array([CASOS["caso 1"]["x_rec"], CASOS["caso 2"]["x_rec"]])
    lote_mu = np.array([CASOS["caso 1"]["mu"], CASOS["caso 2"]["mu"]])
    lote_lv = np.array([CASOS["caso 1"]["log_var"], CASOS["caso 2"]["log_var"]])
    por_foto = perda_vae(lote_x, lote_rec, lote_mu, lote_lv)
    print(f"\nOs dois casos num lote: perda por foto {'  '.join(br(p) for p in por_foto)} (formato {por_foto.shape})")


if __name__ == "__main__":
    main()
