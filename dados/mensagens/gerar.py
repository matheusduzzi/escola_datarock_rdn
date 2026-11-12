"""Gera as mensagens de clientes ao atendimento da Loja Exemplo usadas nas aulas de Naive Bayes.

Só biblioteca padrão (Python 3.9 ou mais novo). Semente fixa: rodar duas vezes dá o mesmo
arquivo, byte a byte.

    python3 gerar.py                 # escreve mensagens.csv nesta pasta
    python3 gerar.py --saida /tmp/x  # escreve em outra pasta

Uma linha é a primeira mensagem de um atendimento aberto entre 2026-01-01 e 2026-06-30. O
rótulo `assunto` é o que a pessoa do atendimento marcou ao fechar: atraso, troca, pagamento ou
outro. As mensagens não têm nome, telefone, e-mail ou documento.

Nas aulas, você treina e avalia primeiro e só depois lê este programa para comparar.
"""
import argparse
import csv
import datetime
import pathlib
import random

SEMENTE = 1508
N = 1800
INICIO = datetime.date(2026, 1, 1)
DIAS = 181
ASSUNTOS = [("atraso", 38), ("troca", 27), ("pagamento", 20), ("outro", 15)]

ABERTURAS = ["", "", "oi, ", "olá, ", "bom dia, ", "boa tarde, ", "boa noite, ", "por favor, "]
FRASES = {
    "atraso": [
        "meu pedido ainda não chegou", "o prazo de entrega já passou", "cadê meu pedido",
        "o rastreio está parado faz {d} dias", "a entrega está atrasada", "comprei faz {d} dias e nada",
        "o pedido não saiu para entrega", "a transportadora não atualiza o rastreio",
        "era para chegar ontem", "quando meu pedido vai chegar",
    ],
    "troca": [
        "quero trocar o tamanho", "veio a cor errada", "o produto chegou quebrado",
        "quero devolver o produto", "veio um item diferente do que eu pedi", "a peça não serviu",
        "como faço a troca", "o produto veio com defeito", "quero o reembolso da devolução",
        "a caixa chegou amassada e o produto riscado",
    ],
    "pagamento": [
        "o pix não foi reconhecido", "o boleto venceu", "cobraram duas vezes no cartão",
        "o pagamento ficou pendente", "não consigo pagar com cartão", "quero a segunda via do boleto",
        "o desconto do cupom não entrou", "o estorno ainda não apareceu na fatura",
        "a compra foi recusada no cartão", "paguei e o pedido continua aguardando pagamento",
    ],
    "outro": [
        "vocês têm loja física", "qual o horário do atendimento", "quero mudar meu endereço de cadastro",
        "vocês vendem para empresa", "o produto tem garantia", "quero cancelar o recebimento de e-mails",
        "tem esse produto em outra cor", "como uso o cupom de aniversário", "quero elogiar o atendimento",
        "vocês entregam no sábado",
    ],
}
COMPLEMENTOS = ["", "", "", ". preciso de ajuda", ". aguardo retorno", ". obrigado", ". é urgente",
                "!!", "?", ". pedido {p}"]
# Mensagens com palavra de outro assunto: o rótulo é o pedido principal da pessoa.
MISTURAS = [
    ("atraso", "não quero trocar nada, só saber quando chega"),
    ("atraso", "paguei no pix faz {d} dias e o pedido não chegou"),
    ("troca", "chegou atrasado e ainda veio errado, quero trocar"),
    ("troca", "o pagamento foi aprovado mas o produto veio com defeito"),
    ("pagamento", "o pedido chegou mas cobraram duas vezes"),
    ("pagamento", "quero trocar a forma de pagamento do boleto para pix"),
    ("outro", "não é sobre entrega nem pagamento, vocês têm vaga de emprego"),
]
SEM_ACENTO = str.maketrans("áâãàéêíóôõúç", "aaaaeeiooouc")


def _sorteio(rng, opcoes):
    return rng.choices([o[0] for o in opcoes], weights=[o[1] for o in opcoes])[0]


def _texto(rng, frase):
    return frase.replace("{d}", str(rng.randint(2, 15))).replace("{p}", str(rng.randint(10000, 99999)))


def gerar(saida):
    rng = random.Random(SEMENTE)
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for i in range(1, N + 1):
        data = INICIO + datetime.timedelta(days=rng.randrange(DIAS))
        canal = rng.choices(["chat", "app", "email"], weights=[50, 30, 20])[0]
        if rng.random() < 0.08:
            assunto, frase = rng.choice(MISTURAS)
        else:
            assunto = _sorteio(rng, ASSUNTOS)
            frase = rng.choice(FRASES[assunto])
            if rng.random() < 0.25:
                frase += " e " + rng.choice(FRASES[assunto])
        texto = rng.choice(ABERTURAS) + _texto(rng, frase) + _texto(rng, rng.choice(COMPLEMENTOS))
        if rng.random() < 0.3:
            texto = texto.translate(SEM_ACENTO)
        if rng.random() < 0.15:
            texto = texto.upper()
        linhas.append({"id_mensagem": i, "data": data.isoformat(), "canal": canal, "texto": texto, "assunto": assunto})
    linhas.sort(key=lambda l: (l["data"], l["id_mensagem"]))
    for novo, l in enumerate(linhas, start=1):
        l["id_mensagem"] = novo
    with open(saida / "mensagens.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(linhas)
    contagem = {a: sum(1 for l in linhas if l["assunto"] == a) for a, _ in ASSUNTOS}
    return {"mensagens": len(linhas), **contagem}


def main():
    parser = argparse.ArgumentParser(description="Gera mensagens.csv da Loja Exemplo.")
    parser.add_argument("--saida", default=str(pathlib.Path(__file__).resolve().parent),
                        help="pasta de saída (padrão: esta pasta)")
    resumo = gerar(pathlib.Path(parser.parse_args().saida))
    print(" ".join("%s=%s" % (k, v) for k, v in resumo.items()))


if __name__ == "__main__":
    main()
