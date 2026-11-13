"""Exercício da aula «BERT e GPT-2 em texto»: um vetor por mensagem, de dois jeitos.

Escreva a função resumir() e rode, de dentro da pasta da aula:

    python3 testar_meu_resumo.py               # os testes da sua função
    python3 meu_resumo.py                      # o GPT-2, uns 3 minutos
    python3 meu_resumo.py --modelo bert-base-multilingual-cased

O programa passa as 1.800 mensagens e as 40 de julho pelo modelo, faz um vetor por mensagem com
a sua função (primeiro e último token) e com a média da aula, e mede cada um no assunto, do mesmo
jeito da parte 5 do textos.py.
"""
import argparse
import time

import numpy as np

import modelos as m
import textos


def resumir(estados, mascara, jeito):
    """Um vetor por mensagem, tirado dos vetores da última camada.

    estados: array (mensagens, tokens, d) com os vetores de cada token.
    mascara: array (mensagens, tokens) de True/False; True nos tokens de verdade. O preenchimento
             fica à direita: depois do primeiro False de uma linha, só vem False.
    jeito:   "primeiro" -> o vetor do primeiro token de cada mensagem;
             "ultimo"   -> o vetor do último token de verdade de cada mensagem.

    Retorna um array novo (mensagens, d): mexer nele depois não muda estados. A função também não
    muda estados nem mascara.

    Para com ValueError quando:
      - jeito não é "primeiro" nem "ultimo";
      - estados não tem 3 dimensões, ou mascara não tem o formato (mensagens, tokens) de estados;
      - mascara não é de True/False (dtype bool);
      - alguma mensagem não tem nenhum token de verdade;
      - alguma linha tem True depois de um False (buraco ou preenchimento à esquerda).
    """
    raise NotImplementedError("escreva resumir(estados, mascara, jeito)")


def main(argv=None):
    p = argparse.ArgumentParser(description="Primeiro token, último token e média, no assunto.")
    p.add_argument("--modelo", default="gpt2", choices=textos.NOMES)
    a = p.parse_args(argv)
    linhas, julho = textos.carregar()
    todas = linhas + [dict(l, data="2026-07-01") for l in julho]
    textos_ = [l["texto"] for l in todas]
    y = np.array([textos.ASSUNTOS.index(l["assunto"]) for l in todas])
    mes = np.array([int(l["data"][5:7]) for l in todas])
    mod = textos.carregar_modelo(a.modelo)
    listas = [mod.tok.codificar(t) for t in textos_]
    jeitos = {"primeiro": np.zeros((len(todas), 768), np.float32),
              "ultimo": np.zeros((len(todas), 768), np.float32),
              "media": np.zeros((len(todas), 768), np.float32)}
    t0 = time.perf_counter()
    for quais, ids, mascara, est in m.idas_em_lotes(mod, listas):
        jeitos["primeiro"][quais] = resumir(est, mascara, "primeiro")
        jeitos["ultimo"][quais] = resumir(est, mascara, "ultimo")
        jeitos["media"][quais] = m.media_dos_tokens(est, mascara)
    print(f"{a.modelo}: ida pelas {len(todas)} mensagens em {time.perf_counter() - t0:.0f} s")
    if a.modelo != "gpt2":
        print("  (no BERT, o primeiro token é o [CLS] e o último é o [SEP])")
    jun, jul = mes == 6, mes == 7
    print(f"  {'vetor':10s} {'lambda':>7s} {'maio':>10s} {'junho':>16s} {'julho (40 difíceis)':>20s}")
    for jeito, X in jeitos.items():
        lam, cm, prev = textos.classificar(X, y, mes)
        cj, cl = int((prev[jun] == y[jun]).sum()), int((prev[jul] == y[jul]).sum())
        print(f"  {jeito:10s} {lam!s:>7s} {cm:4d} de {int((mes == 5).sum())} {cj:5d} de {jun.sum()} "
              f"{100 * cj / jun.sum():5.1f}% {cl:7d} de 40 {100 * cl / 40:5.1f}%")


if __name__ == "__main__":
    main()
