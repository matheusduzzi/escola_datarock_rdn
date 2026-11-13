"""Treina uma configuração e mostra a curva com os instrumentos da aula. Rode de dentro da pasta da aula:

    python3 experimentar.py                       # 784 -> 128 -> 10, He, taxa 0.05, lote 64, 3 épocas, 10000 fotos
    python3 experimentar.py --taxa 0.2 --lote 128
    python3 experimentar.py --camadas 784,64,64,10 --inicio 0.01 --epocas 5
    python3 experimentar.py --fotos 2000 --epocas 60 --escolher --l2 0.01

Mude uma coisa por vez e anote o número antes e depois.
"""
import argparse

import rede as R


def inicio(texto):
    if texto == "he":
        return "he"
    try:
        return float(texto)
    except ValueError:
        raise SystemExit(f'configuração recusada: --inicio aceita "he" ou um número, por exemplo 0.01 (veio {texto!r})')


ap = argparse.ArgumentParser(description="Treina uma configuração e mostra a curva com os instrumentos da aula.")
ap.add_argument("--camadas", default="784,128,10", help="tamanhos separados por vírgula (padrão 784,128,10)")
ap.add_argument("--inicio", default="he", help='"he" ou um desvio, por exemplo 0.01 (padrão he)')
ap.add_argument("--taxa", type=float, default=0.05)
ap.add_argument("--lote", type=int, default=64)
ap.add_argument("--epocas", type=int, default=3)
ap.add_argument("--fotos", type=int, default=10000, help="fotos de treino (padrão 10000, máximo 50000)")
ap.add_argument("--l2", type=float, default=0.0)
ap.add_argument("--escolher", action="store_true", help="guardar a época de menor perda de validação")
a = ap.parse_args()

config = {"camadas": [int(t) for t in a.camadas.split(",")], "inicio": inicio(a.inicio), "taxa": a.taxa, "lote": a.lote,
          "epocas": a.epocas, "fotos_treino": a.fotos, "l2": a.l2, "escolher_epoca": a.escolher}
try:
    R.conferir_config(config)
except ValueError as erro:
    raise SystemExit(f"configuração recusada: {erro}")
Xt, yt, Xv, yv = R.preparar()
_, escolhida, curva = R.treinar(config, Xt, yt, Xv, yv)
print("configuração:", config)
R.mostrar(curva)
R.resumo(curva, escolhida)
