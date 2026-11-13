"""Testes do exercício da aula «Transformer e tokenização». Rode de dentro da pasta da aula:

    python3 testar_minha_tokenizacao.py

Os testes usam vocabulários pequenos, montados para cada caso, e o exemplo da aula.
Eles não usam as palavras da tarefa 1: essa conta é sua.
"""
import sys
import traceback

from minha_tokenizacao import wordpiece


def teste_palavra_inteira():
    v = {"pedido", "p", "##e", "##d", "##i", "##o"}
    r = wordpiece("pedido", v)
    assert r == ["pedido"], f"{r}; esperado ['pedido']"


def teste_exemplo_da_aula():
    v = {"cheg", "c", "##h", "##e", "##g", "##a", "##r", "##ar"}
    r = wordpiece("chegar", v)
    assert r == ["cheg", "##ar"], f"chegar: {r}; esperado ['cheg', '##ar']"
    v2 = {"un", "##aff", "##able", "u", "##n", "##a", "##f", "##b", "##l", "##e"}
    r = wordpiece("unaffable", v2)
    assert r == ["un", "##aff", "##able"], f"unaffable: {r}; esperado ['un', '##aff', '##able']"


def teste_maior_pedaco_primeiro():
    v = {"pag", "paga", "p", "##a", "##g", "##r", "##m", "##ar", "##am"}
    r = wordpiece("pagaram", v)
    assert r == ["paga", "##r", "##am"], f"pagaram: {r}; esperado ['paga', '##r', '##am']"


def teste_continuacao_com_cerquilha():
    # "ou" existe, mas só como começo de palavra; no meio da palavra vale "##o" e "##u"
    v = {"pag", "ou", "##o", "##u"}
    r = wordpiece("pagou", v)
    assert r == ["pag", "##o", "##u"], f"pagou: {r}; esperado ['pag', '##o', '##u']"


def teste_primeira_peca_sem_cerquilha():
    # "##pag" não serve para o começo da palavra
    v = {"##pag", "pa", "##g", "##ou"}
    r = wordpiece("pagou", v)
    assert r == ["pa", "##g", "##ou"], f"pagou: {r}; esperado ['pa', '##g', '##ou']"


def teste_caractere_fora_vira_unk_inteiro():
    v = {"p", "##a", "##o"}
    r = wordpiece("pao", v)
    assert r == ["p", "##a", "##o"], f"pao: {r}; esperado ['p', '##a', '##o']"
    r = wordpiece("pão", v)
    assert r == ["[UNK]"], f"pão: {r}; esperado ['[UNK]'] (a palavra inteira, uma peça só)"
    r = wordpiece("ãp", v)
    assert r == ["[UNK]"], f"ãp: {r}; esperado ['[UNK]']"


def teste_unk_escolhido():
    r = wordpiece("pão", {"p", "##a", "##o"}, unk="<?>")
    assert r == ["<?>"], f"pão com unk='<?>': {r}; esperado ['<?>']"


def teste_limite_de_caracteres():
    v = {"a", "##a"}
    r = wordpiece("a" * 200, v)
    assert len(r) == 200 and r[0] == "a" and r[1] == "##a", f"200 letras: {len(r)} peças; esperado 200"
    r = wordpiece("a" * 201, v)
    assert r == ["[UNK]"], f"201 letras: {r[:3]}...; esperado ['[UNK]']"
    r = wordpiece("aaa", v, max_caracteres=3)
    assert r == ["a", "##a", "##a"], f"3 letras com limite 3: {r}; esperado ['a', '##a', '##a']"
    r = wordpiece("aaaa", v, max_caracteres=3)
    assert r == ["[UNK]"], f"4 letras com limite 3: {r}; esperado ['[UNK]']"


def teste_palavra_vazia():
    r = wordpiece("", {"a"})
    assert r == [], f"palavra vazia: {r}; esperado []"


def teste_espaco_para():
    for w in ("meu pedido", " ", "a\tb"):
        try:
            wordpiece(w, {"a", "##b", "meu", "pedido"})
        except ValueError:
            continue
        raise AssertionError(f"{w!r} não parou com ValueError")


def teste_nao_mexe_no_vocab():
    v = {"cheg", "##ou", "c"}
    copia = set(v)
    r = wordpiece("chegou", v)
    assert v == copia, "a função mudou o vocabulário"
    assert isinstance(r, list) and all(isinstance(x, str) for x in r), f"retorno {r!r}; esperado lista de str"


TESTES = [teste_palavra_inteira, teste_exemplo_da_aula, teste_maior_pedaco_primeiro, teste_continuacao_com_cerquilha,
          teste_primeira_peca_sem_cerquilha, teste_caractere_fora_vira_unk_inteiro, teste_unk_escolhido,
          teste_limite_de_caracteres, teste_palavra_vazia, teste_espaco_para, teste_nao_mexe_no_vocab]


if __name__ == "__main__":
    falhas = 0
    for t in TESTES:
        try:
            t()
        except NotImplementedError:
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:  # noqa: BLE001
            falhas += 1
            print(f"FALHOU {t.__name__[6:]}: {type(e).__name__}: {e}")
    if falhas:
        print(f"{falhas} de {len(TESTES)} testes falharam")
        sys.exit(1)
    print(f"ok: {len(TESTES)} testes passaram")
