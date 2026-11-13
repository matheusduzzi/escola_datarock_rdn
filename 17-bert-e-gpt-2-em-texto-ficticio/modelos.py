"""BERT e GPT-2 prontos, escritos no NumPy. Os outros arquivos da pasta usam este.

O que tem aqui:
  MODELOS             os três modelos da aula, com o endereço fixo de cada arquivo e o SHA-256
  baixar(nome)        baixa os arquivos de um modelo do Hugging Face e confere o SHA-256
  ler_safetensors()   lê os pesos do arquivo .safetensors, sem o PyTorch
  TokenizadorBert     o WordPiece do BERT, com a limpeza do texto (minúsculas e acento no uncased)
  TokenizadorGPT2     o BPE de bytes do GPT-2
  Bert                a ida pelo encoder e a cabeça que adivinha a palavra escondida
  GPT2                a ida pelo decoder, a perda do próximo token e a escrita, um token de cada vez
  media_dos_tokens()  a média dos vetores de uma mensagem, sem as posições de preenchimento

Os vetores andam no formato (mensagens, tokens, 768), como na aula «Atenção Q/K/V».
"""
import hashlib
import json
import struct
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np

# O NumPy 2.0 com a biblioteca Accelerate do Mac avisa "overflow in matmul" sem ter overflow,
# como na aula «Transferência na prática». Os avisos ficam desligados; quem pega infinito ou NaN
# de verdade é conferir_finito(), chamada depois de cada ida.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

AQUI = Path(__file__).resolve().parent
PASTA_MODELOS = AQUI / "modelos"

# Cada arquivo vem de uma versão fixa do repositório no Hugging Face (o commit no endereço).
# Assim o arquivo é sempre o mesmo, mesmo que o repositório mude depois.
MODELOS = {
    "bert-base-uncased": {
        "repositorio": "google-bert/bert-base-uncased",
        "versao": "86b5e0934494bd15c9632b12f734a8a67f723594",
        "licenca": "Apache-2.0",
        "arquivos": {
            "config.json": "7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20",
            "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
            "model.safetensors": "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3",
        },
    },
    "bert-base-multilingual-cased": {
        "repositorio": "google-bert/bert-base-multilingual-cased",
        "versao": "3f076fdb1ab68d5b2880cb87a0886f315b8146f8",
        "licenca": "Apache-2.0",
        "arquivos": {
            "config.json": "4d10c2021c6b5fbe8b70698abc6ccc6691dc28d1bae9732f7fa1087782feba7e",
            "vocab.txt": "fe0fda7c425b48c516fc8f160d594c8022a0808447475c1a7c6d6479763f310c",
            "model.safetensors": "876f584f15ebf14887dec17539c114bb99a032e96b9e72507a51c41e205337fc",
        },
    },
    "gpt2": {
        "repositorio": "openai-community/gpt2",
        "versao": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "licenca": "MIT",
        "arquivos": {
            "config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
            "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
            "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
            "model.safetensors": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
        },
    },
}


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def endereco(nome, arquivo):
    m = MODELOS[nome]
    return f"https://huggingface.co/{m['repositorio']}/resolve/{m['versao']}/{arquivo}"


def baixar(nome, pasta=PASTA_MODELOS):
    """Baixa os arquivos que faltam e confere o SHA-256 de cada um. Retorna as linhas do que fez."""
    destino = Path(pasta) / nome
    destino.mkdir(parents=True, exist_ok=True)
    linhas = []
    for arquivo, esperado in MODELOS[nome]["arquivos"].items():
        alvo = destino / arquivo
        if alvo.exists() and sha256(alvo) == esperado:
            linhas.append(f"{nome}/{arquivo}: já estava aqui, SHA-256 ok")
            continue
        parcial = alvo.with_name(alvo.name + ".parcial")
        urllib.request.urlretrieve(endereco(nome, arquivo), parcial)
        obtido = sha256(parcial)
        if obtido != esperado:
            parcial.unlink()
            raise SystemExit(f"{nome}/{arquivo}: SHA-256 {obtido} diferente do esperado; "
                             "arquivo apagado, tente de novo")
        parcial.replace(alvo)
        linhas.append(f"{nome}/{arquivo}: baixado, {alvo.stat().st_size} bytes, SHA-256 ok")
    return linhas


def pasta_do_modelo(nome, pasta=PASTA_MODELOS):
    destino = Path(pasta) / nome
    faltam = [a for a in MODELOS[nome]["arquivos"] if not (destino / a).exists()]
    if faltam:
        raise SystemExit(f"Não achei {', '.join(faltam)} em {destino}. "
                         f"Rode: python3 baixar_modelos.py {nome}")
    return destino


# ---------------------------------------------------------------- ler o .safetensors sem o PyTorch

def ler_safetensors(caminho):
    """Lê um arquivo .safetensors: 8 bytes com o tamanho do cabeçalho, o cabeçalho em JSON e os números.

    O arquivo só tem números e um JSON que diz onde cada tensor começa e termina. Ler não roda
    código nenhum, ao contrário do pickle do .pth da aula «Transferência na prática».
    Retorna um dicionário nome -> array float32, lido direto do disco (np.memmap).
    """
    caminho = Path(caminho)
    total = caminho.stat().st_size
    with open(caminho, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > total - 8:
            raise ValueError(f"{caminho.name}: cabeçalho de {n} bytes não cabe no arquivo")
        cabecalho = json.loads(f.read(n))
    inicio = 8 + n
    dados = np.memmap(caminho, dtype=np.uint8, mode="r")
    tensores = {}
    fim_anterior = 0
    for nome, info in sorted(((k, v) for k, v in cabecalho.items() if k != "__metadata__"),
                             key=lambda kv: kv[1]["data_offsets"][0]):
        if info["dtype"] != "F32":
            raise ValueError(f"{nome}: tipo {info['dtype']}, esse leitor só aceita F32")
        a, b = info["data_offsets"]
        formato = tuple(info["shape"])
        if a != fim_anterior or b - a != 4 * int(np.prod(formato, dtype=np.int64)) or inicio + b > total:
            raise ValueError(f"{nome}: posição {a}..{b} não bate com o formato {formato}")
        fim_anterior = b
        tensores[nome] = dados[inicio + a:inicio + b].view(np.float32).reshape(formato)
    if inicio + fim_anterior != total:
        raise ValueError(f"{caminho.name}: sobram {total - inicio - fim_anterior} bytes que nenhum tensor usa")
    return tensores


# ---------------------------------------------------------------- tokenizadores

def _e_pontuacao(c):
    o = ord(c)
    if 33 <= o <= 47 or 58 <= o <= 64 or 91 <= o <= 96 or 123 <= o <= 126:
        return True
    return unicodedata.category(c).startswith("P")


def _e_cjk(o):
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2A6DF
            or 0x2A700 <= o <= 0x2B73F or 0x2B740 <= o <= 0x2B81F or 0x2B820 <= o <= 0x2CEAF
            or 0xF900 <= o <= 0xFAFF or 0x2F800 <= o <= 0x2FA1F)


class TokenizadorBert:
    """O tokenizador do BERT: limpa o texto, separa palavras e sinais e quebra cada palavra pelo WordPiece."""

    def __init__(self, pasta, minusculas):
        linhas = (Path(pasta) / "vocab.txt").read_text(encoding="utf-8").split("\n")
        if linhas and linhas[-1] == "":
            linhas.pop()
        self.vocab = {t: i for i, t in enumerate(linhas)}
        self.tokens = linhas
        self.minusculas = minusculas
        self.unk, self.cls, self.sep, self.mask, self.pad = (
            self.vocab[t] for t in ("[UNK]", "[CLS]", "[SEP]", "[MASK]", "[PAD]"))

    def palavras(self, texto):
        """Limpeza e separação, antes do WordPiece (o BasicTokenizer do BERT)."""
        limpo = []
        for c in texto:
            o = ord(c)
            if o == 0 or o == 0xFFFD or (unicodedata.category(c).startswith("C") and c not in "\t\n\r"):
                continue
            if c in " \t\n\r" or unicodedata.category(c) == "Zs":
                limpo.append(" ")
            elif _e_cjk(o):
                limpo.append(f" {c} ")
            else:
                limpo.append(c)
        saida = []
        for palavra in "".join(limpo).split():
            if self.minusculas:
                palavra = palavra.lower()
                palavra = "".join(c for c in unicodedata.normalize("NFD", palavra)
                                  if unicodedata.category(c) != "Mn")
            atual = ""
            for c in palavra:
                if _e_pontuacao(c):
                    if atual:
                        saida.append(atual)
                    saida.append(c)
                    atual = ""
                else:
                    atual += c
            if atual:
                saida.append(atual)
        return saida

    def wordpiece(self, palavra):
        if len(palavra) > 100:
            return ["[UNK]"]
        pecas, inicio = [], 0
        while inicio < len(palavra):
            fim = len(palavra)
            achou = None
            while inicio < fim:
                peca = palavra[inicio:fim] if inicio == 0 else "##" + palavra[inicio:fim]
                if peca in self.vocab:
                    achou = peca
                    break
                fim -= 1
            if achou is None:
                return ["[UNK]"]
            pecas.append(achou)
            inicio = fim
        return pecas

    def pedacos(self, texto):
        return [p for palavra in self.palavras(texto) for p in self.wordpiece(palavra)]

    def codificar(self, texto):
        """[CLS] + os pedaços + [SEP], em números do vocabulário."""
        partes = []
        for p in texto.split("[MASK]"):
            partes.append([self.vocab[t] for t in self.pedacos(p)])
        ids = partes[0]
        for resto in partes[1:]:
            ids = ids + [self.mask] + resto
        return [self.cls] + ids + [self.sep]


def _bytes_para_letras():
    """A tabela do GPT-2 que troca cada um dos 256 bytes por um caractere que se vê na tela."""
    visiveis = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    letras = visiveis[:]
    n = 0
    for b in range(256):
        if b not in visiveis:
            visiveis.append(b)
            letras.append(256 + n)
            n += 1
    return dict(zip(visiveis, (chr(c) for c in letras)))


def _classe(c):
    cat = unicodedata.category(c)
    if cat[0] == "L":
        return "L"
    if cat[0] == "N":
        return "N"
    if c.isspace():
        return "S"
    return "O"


def pre_tokenizar_gpt2(texto):
    """Separa o texto como a expressão regular do GPT-2:
    's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+
    O re do Python não tem \\p{L}; aqui cada alternativa é testada na mesma ordem, à mão."""
    saida, i, n = [], 0, len(texto)
    classes = [_classe(c) for c in texto]
    while i < n:
        if texto[i] == "'":
            achou = None
            for c in ("s", "t", "re", "ve", "m", "ll", "d"):
                if texto.startswith(c, i + 1):
                    achou = "'" + c
                    break
            if achou:
                saida.append(achou)
                i += len(achou)
                continue
        j = i + 1 if texto[i] == " " and i + 1 < n else i
        tipo = classes[j]
        if tipo != "S":
            k = j
            while k < n and classes[k] == tipo:
                k += 1
            saida.append(texto[i:k])
            i = k
            continue
        # sobrou espaço: \s+(?!\S) e depois \s+
        k = i
        while k < n and classes[k] == "S":
            k += 1
        if k < n and k - i > 1:
            k -= 1
        saida.append(texto[i:k])
        i = k
    return saida


class TokenizadorGPT2:
    """O BPE de bytes do GPT-2: o texto vira bytes, os bytes viram letras e as fusões juntam pares."""

    def __init__(self, pasta):
        pasta = Path(pasta)
        self.vocab = json.loads((pasta / "vocab.json").read_text(encoding="utf-8"))
        self.tokens = {i: t for t, i in self.vocab.items()}
        linhas = (pasta / "merges.txt").read_text(encoding="utf-8").split("\n")
        fusoes = [tuple(l.split()) for l in linhas[1:] if l and not l.startswith("#version")]
        self.posto = {par: i for i, par in enumerate(fusoes)}
        self.letras = _bytes_para_letras()
        self.bytes = {c: b for b, c in self.letras.items()}
        self.fim = self.vocab["<|endoftext|>"]
        self._memoria = {}

    def bpe(self, pedaco):
        if pedaco in self._memoria:
            return self._memoria[pedaco]
        simbolos = list(pedaco)
        while len(simbolos) > 1:
            pares = [(self.posto.get((a, b), 1 << 30), k) for k, (a, b) in enumerate(zip(simbolos, simbolos[1:]))]
            melhor, _ = min(pares)
            if melhor == 1 << 30:
                break
            a, b = next((simbolos[k], simbolos[k + 1]) for p, k in pares if p == melhor)
            novo, k = [], 0
            while k < len(simbolos):
                if k < len(simbolos) - 1 and simbolos[k] == a and simbolos[k + 1] == b:
                    novo.append(a + b)
                    k += 2
                else:
                    novo.append(simbolos[k])
                    k += 1
            simbolos = novo
        self._memoria[pedaco] = simbolos
        return simbolos

    def pedacos(self, texto):
        saida = []
        for pedaco in pre_tokenizar_gpt2(texto):
            letras = "".join(self.letras[b] for b in pedaco.encode("utf-8"))
            saida.extend(self.bpe(letras))
        return saida

    def codificar(self, texto):
        return [self.vocab[p] for p in self.pedacos(texto)]

    def decodificar(self, ids):
        letras = "".join(self.tokens[i] for i in ids)
        return bytes(self.bytes[c] for c in letras).decode("utf-8", errors="replace")


# ---------------------------------------------------------------- as contas que os dois usam

def normalizar(x, peso, vies, eps):
    media = x.mean(-1, keepdims=True)
    var = ((x - media) ** 2).mean(-1, keepdims=True)
    return (x - media) / np.sqrt(var + eps) * peso + vies


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _erf(x):
    """erf pela fórmula 7.1.26 de Abramowitz e Stegun: erro abaixo de 1,5e-7, menor que o do float32."""
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-a * a)
    return s * y


def gelu_erf(x):
    """A GELU do BERT (hidden_act "gelu"): x vezes a chance de uma normal ficar abaixo de x."""
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def gelu_tanh(x):
    """A GELU do GPT-2 (activation_function "gelu_new"): a mesma curva, pela fórmula com tanh."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def atencao(q, k, v, cabecas, bloqueio):
    """A atenção da aula 15 com várias cabeças. bloqueio: True onde a linha NÃO pode olhar a coluna."""
    n, t, d = q.shape
    dc = d // cabecas

    def separar(x):
        return x.reshape(n, t, cabecas, dc).transpose(0, 2, 1, 3)
    q, k, v = separar(q), separar(k), separar(v)
    notas = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dc)
    notas = np.where(bloqueio[:, None, :, :], np.float32(-1e9), notas)
    pesos = softmax(notas)
    return (pesos @ v).transpose(0, 2, 1, 3).reshape(n, t, d)


def empilhar(listas, preencher):
    """Junta mensagens de tamanhos diferentes numa matriz, completando à direita. Retorna ids e máscara."""
    t = max(len(l) for l in listas)
    ids = np.full((len(listas), t), preencher, dtype=np.int64)
    mascara = np.zeros((len(listas), t), dtype=bool)
    for i, l in enumerate(listas):
        ids[i, :len(l)] = l
        mascara[i, :len(l)] = True
    return ids, mascara


def idas_em_lotes(modelo, listas, lote=32):
    """Passa as mensagens pelo modelo em lotes de tamanho parecido, para preencher pouco.

    Gera (posições na lista original, ids, máscara, vetores) a cada lote."""
    ordem = sorted(range(len(listas)), key=lambda i: (len(listas[i]), i))
    preencher = modelo.tok.pad if hasattr(modelo.tok, "pad") else modelo.tok.fim
    for inicio in range(0, len(ordem), lote):
        quais = ordem[inicio:inicio + lote]
        ids, mascara = empilhar([listas[i] for i in quais], preencher)
        yield quais, ids, mascara, modelo.ida(ids, mascara)


def media_dos_tokens(estados, mascara):
    """A média dos vetores de cada mensagem, contando só as posições com token de verdade."""
    m = mascara[:, :, None].astype(estados.dtype)
    return (estados * m).sum(1) / m.sum(1)


def conferir_finito(x, onde):
    if not np.isfinite(x).all():
        raise FloatingPointError(f"{onde}: apareceu infinito ou NaN")
    return x


def contar_parametros(tensores, ignorar=()):
    return sum(int(np.prod(t.shape)) for nome, t in tensores.items()
               if not any(nome.endswith(s) for s in ignorar))


# ---------------------------------------------------------------- BERT

class Bert:
    def __init__(self, nome, pasta=PASTA_MODELOS):
        p = pasta_do_modelo(nome, pasta)
        self.nome = nome
        self.config = json.loads((p / "config.json").read_text(encoding="utf-8"))
        self.tensores = ler_safetensors(p / "model.safetensors")
        self.tok = TokenizadorBert(p, minusculas=(nome == "bert-base-uncased"))
        w = self.tensores
        prefixo = "bert." if "bert.embeddings.word_embeddings.weight" in w else ""
        # Os pesos vão para a memória (np.array copia): multiplicar direto do disco é bem mais lento.
        self.w = {k[len(prefixo):] if k.startswith(prefixo) else k: np.array(v) for k, v in w.items()}
        self.saida = np.ascontiguousarray(self.w["embeddings.word_embeddings.weight"].T)
        self.cabecas = self.config["num_attention_heads"]
        self.camadas = self.config["num_hidden_layers"]
        self.eps = self.config["layer_norm_eps"]

    def _peso(self, nome):
        # O arquivo usa gamma/beta nas normalizações (nome antigo) ou weight/bias.
        if nome in self.w:
            return self.w[nome]
        return self.w[nome.replace(".weight", ".gamma").replace(".bias", ".beta")]

    def _densa(self, x, nome):
        return x @ self.w[nome + ".weight"].T + self.w[nome + ".bias"]

    def _ln(self, x, nome):
        return normalizar(x, self._peso(nome + ".weight"), self._peso(nome + ".bias"), self.eps)

    def ida(self, ids, mascara):
        """Os vetores da última camada: (mensagens, tokens, 768)."""
        n, t = ids.shape
        x = (self.w["embeddings.word_embeddings.weight"][ids]
             + self.w["embeddings.position_embeddings.weight"][:t][None]
             + self.w["embeddings.token_type_embeddings.weight"][0])
        x = self._ln(x, "embeddings.LayerNorm")
        bloqueio = np.broadcast_to(~mascara[:, None, :], (n, t, t))
        for c in range(self.camadas):
            a = f"encoder.layer.{c}."
            q = self._densa(x, a + "attention.self.query")
            k = self._densa(x, a + "attention.self.key")
            v = self._densa(x, a + "attention.self.value")
            y = self._densa(atencao(q, k, v, self.cabecas, bloqueio), a + "attention.output.dense")
            x = self._ln(x + y, a + "attention.output.LayerNorm")
            y = self._densa(gelu_erf(self._densa(x, a + "intermediate.dense")), a + "output.dense")
            x = self._ln(x + y, a + "output.LayerNorm")
        return conferir_finito(x, self.nome)

    def notas_da_palavra(self, estados):
        """A cabeça do pré-treino: de um vetor para uma nota por token do vocabulário."""
        h = gelu_erf(self._densa(estados, "cls.predictions.transform.dense"))
        h = self._ln(h, "cls.predictions.transform.LayerNorm")
        # A saída usa a mesma tabela de tokens da entrada, como no GPT-2 (aula 16, seção 10).
        return h @ self.saida + self.w["cls.predictions.bias"]

    def adivinhar(self, texto, quantos=5):
        """Para cada [MASK] do texto, os tokens mais prováveis e a chance de cada um."""
        ids = self.tok.codificar(texto)
        x = self.ida(np.array([ids]), np.ones((1, len(ids)), dtype=bool))[0]
        saida = []
        for pos in [i for i, t in enumerate(ids) if t == self.tok.mask]:
            p = softmax(self.notas_da_palavra(x[pos]))
            melhores = np.argsort(-p, kind="stable")[:quantos]
            saida.append([(self.tok.tokens[i], float(p[i])) for i in melhores])
        return saida

    def parametros(self):
        return contar_parametros(self.tensores)


# ---------------------------------------------------------------- GPT-2

class GPT2:
    def __init__(self, nome="gpt2", pasta=PASTA_MODELOS):
        p = pasta_do_modelo(nome, pasta)
        self.nome = nome
        self.config = json.loads((p / "config.json").read_text(encoding="utf-8"))
        self.tensores = ler_safetensors(p / "model.safetensors")
        self.tok = TokenizadorGPT2(p)
        # Os pesos vão para a memória (np.array copia), sem as máscaras guardadas no arquivo.
        self.w = {k: np.array(v) for k, v in self.tensores.items() if not k.endswith(".attn.bias")}
        self.saida = np.ascontiguousarray(self.w["wte.weight"].T)
        self.cabecas = self.config["n_head"]
        self.camadas = self.config["n_layer"]
        self.eps = self.config["layer_norm_epsilon"]

    def _ln(self, x, nome):
        return normalizar(x, self.w[nome + ".weight"], self.w[nome + ".bias"], self.eps)

    def _densa(self, x, nome):
        # No GPT-2 o peso já vem como (entrada, saída): é x @ W, sem transpor.
        return x @ self.w[nome + ".weight"] + self.w[nome + ".bias"]

    def ida(self, ids, mascara, guardado=None):
        """Os vetores depois da última normalização: (mensagens, tokens, 768).

        guardado: as keys e os values dos tokens anteriores, camada por camada. Com ele, a ida
        só faz as contas dos tokens novos, que olham para os antigos sem refazer nada. Ao escrever,
        cada token novo custa uma ida de um token só. A lista é atualizada no lugar.
        """
        n, t = ids.shape
        antes = 0 if not guardado else guardado[0][0].shape[2]
        x = self.w["wte.weight"][ids] + self.w["wpe.weight"][antes:antes + t][None]
        causal = np.triu(np.ones((t, antes + t), dtype=bool), antes + 1)
        mascara_toda = np.concatenate([np.ones((n, antes), dtype=bool), mascara], axis=1)
        bloqueio = causal[None] | ~mascara_toda[:, None, :]
        dc = x.shape[-1] // self.cabecas
        for c in range(self.camadas):
            a = f"h.{c}."
            qkv = self._densa(self._ln(x, a + "ln_1"), a + "attn.c_attn")
            q, k, v = (z.reshape(n, t, self.cabecas, dc).transpose(0, 2, 1, 3) for z in np.split(qkv, 3, axis=-1))
            if guardado is not None:
                if len(guardado) > c:
                    k = np.concatenate([guardado[c][0], k], axis=2)
                    v = np.concatenate([guardado[c][1], v], axis=2)
                    guardado[c] = (k, v)
                else:
                    guardado.append((k, v))
            notas = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dc)
            notas = np.where(bloqueio[:, None, :, :], np.float32(-1e9), notas)
            y = (softmax(notas) @ v).transpose(0, 2, 1, 3).reshape(n, t, -1)
            x = x + self._densa(y, a + "attn.c_proj")
            h = gelu_tanh(self._densa(self._ln(x, a + "ln_2"), a + "mlp.c_fc"))
            x = x + self._densa(h, a + "mlp.c_proj")
        return conferir_finito(self._ln(x, "ln_f"), self.nome)

    def notas_do_proximo(self, estados):
        return estados @ self.saida

    def perda_por_token(self, texto):
        """Quanto o GPT-2 se surpreende com cada token do texto, dado o que veio antes (em nats)."""
        ids = [self.tok.fim] + self.tok.codificar(texto)
        x = self.ida(np.array([ids]), np.ones((1, len(ids)), dtype=bool))[0]
        notas = self.notas_do_proximo(x[:-1])
        notas = notas - notas.max(-1, keepdims=True)
        logp = notas - np.log(np.exp(notas).sum(-1, keepdims=True))
        return -logp[np.arange(len(ids) - 1), ids[1:]]

    def escrever(self, texto, novos=40):
        """Continua o texto escolhendo sempre o token mais provável (busca gulosa)."""
        ids = self.tok.codificar(texto)
        guardado = []
        x = self.ida(np.array([ids]), np.ones((1, len(ids)), dtype=bool), guardado)[0, -1]
        for _ in range(novos):
            ids.append(int(np.argmax(self.notas_do_proximo(x))))
            x = self.ida(np.array([[ids[-1]]]), np.ones((1, 1), dtype=bool), guardado)[0, -1]
        return self.tok.decodificar(ids)

    def parametros(self):
        # O arquivo guarda também 12 máscaras causais (attn.bias, 1024 x 1024). Elas não aprendem.
        return contar_parametros(self.tensores, ignorar=(".attn.bias",))
