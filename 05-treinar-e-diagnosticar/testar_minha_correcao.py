"""Testes do exercício da aula «Treinar e diagnosticar». Rode de dentro da pasta da aula:

    python3 testar_minha_correcao.py

Treina a CONFIG do minha_correcao.py uma vez (uns 5 segundos) e confere o resultado. Enquanto a
configuração for a do Caio, os testes falham e o programa sai com código 1. Os testes olham o treino,
não o seu registro: o meu-registro.md é a outra metade do exercício.
"""
import sys

import numpy as np

import rede as R

falhas = []


def teste(nome, cond, detalhe):
    if not cond:
        falhas.append(f"FALHOU {nome}: {detalhe}")


try:
    from minha_correcao import CONFIG
except Exception as erro:  # arquivo com erro de sintaxe ou sem CONFIG
    print(f"FALHOU importar_minha_correcao: {erro!r}")
    sys.exit(1)

try:
    R.conferir_config(CONFIG)
except ValueError as erro:
    print(f"FALHOU configuracao_valida: {erro}")
    sys.exit(1)

teste("mesmas_camadas_do_caio", list(CONFIG["camadas"]) == [784, 256, 256, 10],
      f"o Caio quer a rede 784 -> 256 -> 256 -> 10; veio {CONFIG['camadas']}")
teste("mesmas_fotos_do_caio", CONFIG["fotos_treino"] == 5000, f"fotos_treino tem que ser 5000; veio {CONFIG['fotos_treino']}")
teste("no_maximo_40_epocas", CONFIG["epocas"] <= 40, f"no máximo 40 épocas; veio {CONFIG['epocas']}")
teste("lote_entre_16_e_256", 16 <= CONFIG["lote"] <= 256, f"lote de 16 a 256; veio {CONFIG['lote']}")
if falhas:
    print("\n".join(falhas))
    sys.exit(1)

Xt, yt, Xv, yv = R.preparar()
_, escolhida, curva = R.treinar(CONFIG, Xt, yt, Xv, yv)
c0 = curva[0]
teste("perda_da_epoca_0_perto_do_chute", np.isfinite(c0["perda_treino"]) and c0["perda_treino"] <= 3.0,
      f"antes do primeiro passo a perda de treino deu {R.numero(c0['perda_treino'])}; com 10 tipos, perto de 2,3 é o esperado, "
      f"e acima de 3 o início dos pesos já está errado")
parou = [c["epoca"] for c in curva if c["parou"]]
teste("perda_sempre_finita", not parou, f"a perda virou infinito ou NaN na época {parou[0] if parou else ''}")
pior = max(max(c["apagados"]) for c in curva)
teste("poucos_neuronios_apagados", pior <= 0.10,
      f"em alguma época, {pior:.2f} dos neurônios de uma camada ficaram apagados em todas as fotos de validação (máximo 0,10)")
teste("epoca_escolhida_pela_validacao", CONFIG["escolher_epoca"] is True,
      "escolher_epoca está False: a rede que vai para a Bia é a da última época")
if not parou:
    menor = min(range(len(curva)), key=lambda i: curva[i]["perda_validacao"])
    teste("epoca_escolhida_e_a_de_menor_perda", escolhida == menor,
          f"a época escolhida foi {escolhida}, mas a menor perda de validação está na época {menor}")
    pv = curva[escolhida]["perda_validacao"]
    teste("perda_de_validacao_ate_0_52", pv <= 0.52, f"perda de validação na época escolhida: {pv:.4f} (máximo 0,52)")

if falhas:
    print("\n".join(falhas))
    sys.exit(1)
ce = curva[escolhida]
print(f"ok: 10 testes passaram; época {escolhida}, perda de validação {ce['perda_validacao']:.4f}, "
      f"acerto de validação {ce['acerto_validacao']:.4f}")
