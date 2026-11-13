"""A sua correção do treino do Caio. Rode de dentro da pasta da aula:

    python3 minha_correcao.py           # treina a sua configuração e mostra a curva
    python3 testar_minha_correcao.py    # roda os testes do exercício

O ponto de partida é a configuração do Caio, igual à do treino_do_caio.py. Mude um valor por vez,
rode de novo e anote no meu-registro.md o que apareceu antes e depois.

Fica como o Caio pediu: camadas [784, 256, 256, 10], fotos_treino 5000 e no máximo 40 épocas.
Pode mudar: inicio, taxa, lote (de 16 a 256), epocas, l2 e escolher_epoca.
"""
import rede as R

CONFIG = {"camadas": [784, 256, 256, 10], "inicio": 1.0, "taxa": 0.4, "lote": 64, "epocas": 40,
          "fotos_treino": 5000, "l2": 0.0, "escolher_epoca": False}

if __name__ == "__main__":
    Xt, yt, Xv, yv = R.preparar()
    _, escolhida, curva = R.treinar(CONFIG, Xt, yt, Xv, yv)
    print("configuração:", CONFIG)
    R.mostrar(curva)
    R.resumo(curva, escolhida)
