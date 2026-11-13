"""O treino que o Caio mandou para a Bia. Rode de dentro da pasta da aula:

    python3 treino_do_caio.py

Não mude este arquivo. As suas correções vão no minha_correcao.py.
"""
import rede as R

CAIO = {"camadas": [784, 256, 256, 10], "inicio": 1.0, "taxa": 0.4, "lote": 64, "epocas": 40,
        "fotos_treino": 5000, "l2": 0.0, "escolher_epoca": False}

if __name__ == "__main__":
    Xt, yt, Xv, yv = R.preparar()
    _, escolhida, curva = R.treinar(CAIO, Xt, yt, Xv, yv)
    print("configuração do Caio:", CAIO)
    R.mostrar(curva)
    R.resumo(curva, escolhida)
