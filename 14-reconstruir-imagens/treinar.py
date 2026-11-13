"""Treina o autoencoder ou o VAE com o orçamento comum e grava os pesos em meus-pesos/.

Rode de dentro da pasta da aula:
  python3 treinar.py autoencoder
  python3 treinar.py vae
  python3 treinar.py vae --codigo 2
"""
import argparse
import time

from redes import AQUI, NOMES, ORCAMENTO, gravar, nova_rede, parametros, preparar, treinar


def main():
    ap = argparse.ArgumentParser(description="Treina uma das duas redes da aula com o orçamento comum.")
    ap.add_argument("rede", choices=sorted(NOMES))
    ap.add_argument("--codigo", type=int, default=16, help="quantos números tem o código (padrão 16)")
    ap.add_argument("--pasta", default="meus-pesos", help="onde gravar (padrão meus-pesos)")
    a = ap.parse_args()

    Xt, _, Xv, _, _, _ = preparar()
    print(f"{NOMES[a.rede]}, código de {a.codigo} números, {parametros(nova_rede(a.rede, a.codigo)):,} parâmetros"
          .replace(",", "."))
    print(f"  {len(Xt)} fotos de treino, {len(Xv)} de validação, lote {ORCAMENTO['lote']}, "
          f"{ORCAMENTO['epocas']} épocas, Adam com taxa {ORCAMENTO['taxa']}")
    t0 = time.time()
    rede, curva = treinar(a.rede, a.codigo, Xt, Xv)
    melhor = min(curva, key=lambda c: c[2])
    gravar(rede, curva, a.rede, a.codigo, AQUI / a.pasta)
    print(f"  época escolhida pela validação: {melhor[0]} (perda {melhor[2]:.2f})")
    print(f"  gravado em {a.pasta}/{a.rede}-codigo-{a.codigo}.npz, {time.time() - t0:.0f} s no total")


if __name__ == "__main__":
    main()
