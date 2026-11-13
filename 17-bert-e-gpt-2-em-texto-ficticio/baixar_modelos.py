"""Baixa os três modelos da aula do Hugging Face, na versão fixa, e confere o SHA-256 de cada arquivo.

    python3 baixar_modelos.py                 # os três, uns 1,7 GB
    python3 baixar_modelos.py gpt2            # só um: bert-base-uncased, bert-base-multilingual-cased ou gpt2

Os arquivos vão para modelos/<nome>/ e não são baixados de novo depois. Se um SHA-256 não bater,
o arquivo é apagado e o programa para. Precisa de internet só nessa hora; não precisa de conta.
"""
import sys

import modelos

nomes = sys.argv[1:] or list(modelos.MODELOS)
for nome in nomes:
    if nome not in modelos.MODELOS:
        raise SystemExit(f"Não conheço {nome}. Use: {', '.join(modelos.MODELOS)}")
for nome in nomes:
    for linha in modelos.baixar(nome):
        print(linha)
