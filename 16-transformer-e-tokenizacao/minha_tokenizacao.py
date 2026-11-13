"""Exercício da aula «Transformer e tokenização»: o tokenizador WordPiece do Caio.

    python3 testar_minha_tokenizacao.py
    python3 minha_tokenizacao.py

Você só escreve a função wordpiece. O resto do arquivo já está pronto.
Lê as mensagens ao atendimento em ../dados/mensagens/mensagens.csv.
"""
import csv
import re
import unicodedata
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "dados" / "mensagens" / "mensagens.csv"

# O vocabulário pequeno do Caio: 141 peças (o, a, e já estão nas letras). "##" marca a peça que continua uma palavra.
ESPECIAIS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
SINAIS = [",", ".", "!", "?", "-"]
LETRAS = list("abcdefghijklmnopqrstuvwxyz0123456789")
INTEIRAS = ["o", "a", "e", "de", "do", "da", "no", "na", "um", "em", "eu", "que", "por", "favor", "oi", "ola",
            "boa", "bom", "dia", "tarde", "noite", "meu", "pedido", "produto", "entrega", "chegou", "nao", "ainda",
            "quero", "veio", "faz", "dias", "urgente", "obrigado", "aguardo", "retorno", "preciso", "ajuda",
            "rastreio", "para", "com", "troca", "reembolso", "pagamento", "boleto", "cartao"]
COMECOS = ["cheg", "entreg", "pag", "atras", "devol", "compr", "receb"]
CONTINUACOES = ["##ou", "##ar", "##ada", "##ado", "##ei", "##am", "##s", "##ucao", "##er", "##eu"]
VOCAB = set(ESPECIAIS + SINAIS + LETRAS + ["##" + c for c in LETRAS] + INTEIRAS + COMECOS + CONTINUACOES)
MAX_CARACTERES = 200      # o mesmo limite do tokenizador original do BERT


def wordpiece(palavra, vocab, unk="[UNK]", max_caracteres=MAX_CARACTERES):
    """Quebra UMA palavra em peças do vocabulário, pelo maior pedaço primeiro. Retorna uma lista de str.

    - Começa do início da palavra e pega o maior pedaço que está no vocabulário.
    - Toda peça depois da primeira é procurada com "##" na frente.
    - Se em algum ponto nenhum pedaço serve, a palavra inteira vira [unk] (uma peça só).
    - Palavra com mais de max_caracteres caracteres também vira [unk].
    - Palavra vazia retorna lista vazia.
    - Palavra com espaço dentro não é uma palavra: pare com ValueError.
    """
    raise NotImplementedError("escreva a função wordpiece")


# ---------------------------------------------------------------------------------------------
# Daqui para baixo já está pronto.

def sem_acento(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


JEITOS = {
    "como chegou": lambda s: s,
    "minúsculas": str.lower,
    "minúsculas e sem acento": lambda s: sem_acento(s.lower()),
}


def tokenizar(texto, jeito):
    """Prepara o texto, separa palavras e sinais e passa cada pedaço pela sua função."""
    pecas = []
    for p in re.findall(r"\w+|[^\w\s]", JEITOS[jeito](texto)):
        pecas.extend(wordpiece(p, VOCAB))
    return pecas


def ler():
    with open(CSV, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    linhas = ler()
    print(f"O tokenizador do Caio: {len(VOCAB)} peças no vocabulário, {len(linhas)} mensagens")
    print("  jeito de preparar          peças por mensagem   palavras que viraram [UNK]")
    for jeito in JEITOS:
        pecas = [tokenizar(r["texto"], jeito) for r in linhas]
        n_pal = sum(len(re.findall(r"\w+|[^\w\s]", r["texto"])) for r in linhas)
        unk = sum(p.count("[UNK]") for p in pecas)
        print(f"  {jeito:<26} {sum(map(len, pecas)) / len(pecas):>10.1f}           "
              f"{unk:>5} de {n_pal} ({100 * unk / n_pal:.1f}%)")
    print("\nCinco mensagens, nos três jeitos")
    for r in linhas[:4] + [x for x in linhas if x["id_mensagem"] == "31"]:
        print(f"  {r['texto']}")
        for jeito in JEITOS:
            print(f"    {jeito:<24} " + " ".join(tokenizar(r["texto"], jeito)))
    print("\nAs palavras da tarefa 1")
    for w in ("entregaram", "atrasado", "devolução", "devolucao", "PEDIDO", "47868"):
        print(f"  {w:<11} -> " + " ".join(wordpiece(w, VOCAB)))
