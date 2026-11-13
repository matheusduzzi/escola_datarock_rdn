"""A ResNet-18 pronta, com os pesos do ImageNet, escrita no NumPy. Os outros arquivos da pasta usam este.

O que tem aqui:
  baixar_pesos()   baixa o arquivo oficial do PyTorch e confere o SHA-256
  ler_pth()        lê o arquivo .pth sem o PyTorch, aceitando só as peças de um tensor
  preparar()       aumenta as fotos de 28 x 28, repete o cinza nos três canais e normaliza como o ImageNet
  corpo()          a ida pela rede pronta, com o batch normalization em modo de avaliação
  Etapa4           a etapa 4 com ida e volta, para adaptar: pesos e batch normalization que aprendem
  Cabeca           a densa nova de 512 para 10, com ida e volta

As fotos andam no formato (fotos, altura, largura, canais), como no cnn.py da aula «CNN no Fashion MNIST».
"""
import hashlib
import io
import pickle
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

AQUI = Path(__file__).resolve().parent
DADOS = AQUI.parent / "dados" / "fashion-mnist"
sys.path.insert(0, str(DADOS))
from baixar import CLASSES, carregar  # noqa: E402,F401

# O NumPy 2.0 com a biblioteca Accelerate do Mac avisa "overflow in matmul" sem ter overflow.
# Os avisos ficam desligados; quem pega infinito ou NaN de verdade é avaliar() em experimento.py.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

ENDERECO = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
PESOS = AQUI / "pesos" / "resnet18-f37072fd.pth"

# O preparo dos pesos do ImageNet, na documentação do torchvision (IMAGENET1K_V1)
MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EPS = 1e-5                     # o eps do BatchNorm2d do PyTorch


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar_pesos(destino=PESOS):
    destino = Path(destino)
    if destino.exists() and sha256(destino) == SHA256:
        return f"{destino.name}: já estava aqui, SHA-256 ok"
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(".parcial")
    urllib.request.urlretrieve(ENDERECO, temporario)
    obtido = sha256(temporario)
    if obtido != SHA256:
        temporario.unlink()
        raise SystemExit(f"{destino.name}: SHA-256 {obtido} diferente do esperado; arquivo apagado, tente de novo")
    temporario.replace(destino)
    return f"{destino.name}: baixado, {destino.stat().st_size} bytes, SHA-256 ok"


# ---------------------------------------------------------------- ler o .pth sem o PyTorch

class _Armazem:
    """Marca o tipo de número de um bloco de dados do .pth."""
    TIPOS = {"FloatStorage": np.float32, "LongStorage": np.int64}

    def __init__(self, nome):
        self.dtype = self.TIPOS[nome]


def _montar_tensor(armazem, deslocamento, formato, passos, *resto):
    dados, dtype = armazem
    plano = np.frombuffer(dados, dtype=dtype)
    if not formato:
        return plano[deslocamento].copy()
    item = plano.itemsize
    return np.lib.stride_tricks.as_strided(plano[deslocamento:], shape=tuple(formato),
                                           strides=tuple(p * item for p in passos)).copy()


class _Leitor(pickle.Unpickler):
    """Um .pth é um pickle, e um pickle pode mandar rodar qualquer função. Esse leitor só aceita
    as peças de um dicionário de tensores; qualquer outra coisa para a leitura."""
    PERMITIDO = {("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2"),
                 ("torch._utils", "_rebuild_parameter"), ("torch", "FloatStorage"), ("torch", "LongStorage")}

    def __init__(self, arquivo, zipado, prefixo):
        super().__init__(arquivo)
        self.zipado, self.prefixo = zipado, prefixo

    def find_class(self, modulo, nome):
        if (modulo, nome) not in self.PERMITIDO:
            raise pickle.UnpicklingError(f"o arquivo pede {modulo}.{nome}, que não é peça de tensor: parei")
        if nome == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        if nome == "_rebuild_tensor_v2":
            return _montar_tensor
        if nome == "_rebuild_parameter":
            return lambda dados, *resto: dados
        return _Armazem(nome)

    def persistent_load(self, pid):
        _, armazem, chave, _, _ = pid
        return self.zipado.read(f"{self.prefixo}/data/{chave}"), armazem.dtype


def ler_pth(caminho=PESOS):
    """Retorna o dicionário nome -> array do arquivo de pesos."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise SystemExit(f"Não achei {caminho.name} em pesos/. Rode python3 baixar_pesos.py nessa pasta.")
    with zipfile.ZipFile(caminho) as z:
        pkl = next(n for n in z.namelist() if n.endswith("/data.pkl"))
        prefixo = pkl[: -len("/data.pkl")]
        return dict(_Leitor(io.BytesIO(z.read(pkl)), z, prefixo).load())


# ---------------------------------------------------------------- preparo das fotos

def matriz_aumento(lado_novo, lado=28):
    """Interpolação bilinear com centro de pixel, a do Resize do torchvision ao aumentar."""
    A = np.zeros((lado_novo, lado), dtype=np.float32)
    for i in range(lado_novo):
        src = min(max((i + 0.5) * lado / lado_novo - 0.5, 0.0), lado - 1.0)
        a = int(np.floor(src))
        b = min(a + 1, lado - 1)
        A[i, a] += 1.0 - (src - a)
        A[i, b] += src - a
    return A


def preparar(fotos_uint8, lado, normalizar=True):
    """(n, 28, 28) em 0..255 -> (n, lado, lado, 3): aumenta, repete o cinza nos três canais e normaliza.

    Com normalizar=False, os pixels ficam entre 0 e 1, sem a média e o desvio do ImageNet: é o erro
    que a parte 2 do experimento.py mede. A rede roda igual, sem aviso nenhum.
    """
    x = fotos_uint8.astype(np.float32) / 255.0
    if lado != x.shape[1]:
        A = matriz_aumento(lado, x.shape[1])
        x = np.einsum("ij,njk,lk->nil", A, x, A, optimize=True)
    x = np.clip(x, 0.0, 1.0)[..., None]
    if not normalizar:
        return np.repeat(x, 3, axis=-1).astype(np.float32)
    return ((x - MEDIA) / DESVIO).astype(np.float32)


# ---------------------------------------------------------------- as contas de uma convolução

def janelas(x, k, passo, borda):
    """Cada janela k x k vira uma linha (canal, linha, coluna), a mesma ordem do W do PyTorch."""
    if borda:
        x = np.pad(x, ((0, 0), (borda, borda), (borda, borda), (0, 0)))
    n, h, w, c = x.shape
    ho, wo = (h - k) // passo + 1, (w - k) // passo + 1
    v = sliding_window_view(x, (k, k), axis=(1, 2))[:, ::passo, ::passo][:, :ho, :wo]
    return v.reshape(n * ho * wo, c * k * k), (n, ho, wo)


def conv(x, W, passo, borda, b=None):
    f = W.shape[0]
    cols, (n, ho, wo) = janelas(x, W.shape[2], passo, borda)
    z = cols @ W.reshape(f, -1).T
    if b is not None:
        z += b
    return z.reshape(n, ho, wo, f), cols


def conv_volta(d, cols, W, forma, passo, borda):
    """Gradiente do W e o erro que volta para a entrada: cada linha das janelas devolve sua parte."""
    f, c, k, _ = W.shape
    n, ho, wo, _ = d.shape
    dz = d.reshape(-1, f)
    dW = (dz.T @ cols).reshape(W.shape)
    dcols = (dz @ W.reshape(f, -1)).reshape(n, ho, wo, c, k, k)
    _, h, w, _ = forma
    volta = np.zeros((n, h + 2 * borda, w + 2 * borda, c), dtype=d.dtype)
    for a in range(k):
        for b in range(k):
            volta[:, a:a + passo * ho:passo, b:b + passo * wo:passo, :] += dcols[..., a, b]
    return dW, volta[:, borda:borda + h, borda:borda + w, :]


def max_pool_3x3(x):
    """O max pooling do começo da ResNet: janela 3 x 3, passo 2, borda 1 (a borda nunca ganha)."""
    x = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), constant_values=-np.inf)
    n, h, w, c = x.shape
    ho, wo = (h - 3) // 2 + 1, (w - 3) // 2 + 1
    return sliding_window_view(x, (3, 3), axis=(1, 2))[:, ::2, ::2][:, :ho, :wo].max(axis=(-2, -1))


# ---------------------------------------------------------------- a rede

def bn_como_escala(p, nome):
    """Batch normalization em modo de avaliação: cada canal vira uma escala e um deslocamento fixos."""
    escala = p[f"{nome}.weight"] / np.sqrt(p[f"{nome}.running_var"] + EPS)
    return escala.astype(np.float32), (p[f"{nome}.bias"] - p[f"{nome}.running_mean"] * escala).astype(np.float32)


def dobrar(p, conv_nome, bn_nome):
    """Junta conv e batch normalization numa conv com viés: a mesma conta, feita uma vez só."""
    escala, desloca = bn_como_escala(p, bn_nome)
    return (p[f"{conv_nome}.weight"] * escala[:, None, None, None]).astype(np.float32), desloca


class Corpo:
    """A ida pela entrada e pelas etapas 1 a 4 da ResNet-18, tudo parado (congelado)."""

    def __init__(self, p):
        self.p = p
        self.entrada = dobrar(p, "conv1", "bn1")
        self.blocos = {}
        for etapa in range(1, 5):
            for bloco in range(2):
                nome = f"layer{etapa}.{bloco}"
                d = {"c1": dobrar(p, f"{nome}.conv1", f"{nome}.bn1"), "c2": dobrar(p, f"{nome}.conv2", f"{nome}.bn2"),
                     "passo": 2 if (etapa > 1 and bloco == 0) else 1}
                if f"{nome}.downsample.0.weight" in p:
                    d["atalho"] = dobrar(p, f"{nome}.downsample.0", f"{nome}.downsample.1")
                self.blocos[nome] = d

    def bloco(self, x, nome):
        d = self.blocos[nome]
        z, _ = conv(x, d["c1"][0], d["passo"], 1, d["c1"][1])
        z, _ = conv(np.maximum(z, 0), d["c2"][0], 1, 1, d["c2"][1])
        atalho = conv(x, d["atalho"][0], d["passo"], 0, d["atalho"][1])[0] if "atalho" in d else x
        return np.maximum(z + atalho, 0)

    def ate(self, x, etapa_final=4):
        """A ida até o fim da etapa indicada. Com 4, retorna o mapa antes da média."""
        z, _ = conv(x, self.entrada[0], 2, 3, self.entrada[1])
        x = max_pool_3x3(np.maximum(z, 0))
        for etapa in range(1, etapa_final + 1):
            for bloco in range(2):
                x = self.bloco(x, f"layer{etapa}.{bloco}")
        return x

    def em_lotes(self, fotos_uint8, lado, etapa_final=4, lote=200, media=True, normalizar=True):
        """Prepara e passa as fotos em lotes. Com media=True e etapa 4, retorna os 512 números de cada foto."""
        saidas = []
        for i in range(0, len(fotos_uint8), lote):
            m = self.ate(preparar(fotos_uint8[i:i + lote], lado, normalizar), etapa_final)
            saidas.append(m.mean(axis=(1, 2)) if media else m)
        return np.concatenate(saidas).astype(np.float32)

    def tamanhos(self, lado):
        """O lado do mapa depois da entrada, do pooling e de cada etapa."""
        lados = [(lado + 6 - 7) // 2 + 1]
        lados.append((lados[-1] + 2 - 3) // 2 + 1)
        for _ in range(3):
            lados.append((lados[-1] + 2 - 3) // 2 + 1)
        return lados        # entrada, pooling (= etapa 1), etapa 2, etapa 3, etapa 4


def contar(p):
    """Parâmetros por parte da rede, sem as médias guardadas do batch normalization."""
    partes = {"entrada": 0, "etapa 1": 0, "etapa 2": 0, "etapa 3": 0, "etapa 4": 0, "densa de 1000": 0}
    for nome, v in p.items():
        if nome.endswith(("running_mean", "running_var", "num_batches_tracked")):
            continue
        if nome.startswith("layer"):
            partes[f"etapa {nome[5]}"] += v.size
        elif nome.startswith("fc"):
            partes["densa de 1000"] += v.size
        else:
            partes["entrada"] += v.size
    return partes


class Cabeca:
    """A densa nova: 512 números da média do mapa -> 10 notas. Começo de He, como na aula 08.

    Antes da densa, cada um dos 512 números é centralizado com a média e o desvio das fotos de treino,
    medidos uma vez na rede pronta, como os pixels na aula 03. Essas medidas não aprendem.
    """

    def __init__(self, f_treino, saidas=10, sorteio=0):
        rng = np.random.default_rng(sorteio)
        entradas = f_treino.shape[1]
        self.media = f_treino.mean(axis=0).astype(np.float32)
        self.desvio = (f_treino.std(axis=0) + 1e-6).astype(np.float32)
        self.W = rng.normal(0, np.sqrt(2 / entradas), size=(saidas, entradas)).astype(np.float32)
        self.b = np.zeros(saidas, dtype=np.float32)

    def copia(self):
        c = Cabeca.__new__(Cabeca)
        c.media, c.desvio, c.W, c.b = self.media, self.desvio, self.W.copy(), self.b.copy()
        return c

    def parametros(self):
        return self.W.size + self.b.size

    def notas(self, f):
        return ((f - self.media) / self.desvio) @ self.W.T + self.b

    def volta(self, f, d):
        """d: erro das notas. Retorna o gradiente de W e b e o erro que volta para os 512 números."""
        z = (f - self.media) / self.desvio
        return d.T @ z, d.sum(axis=0), (d @ self.W) / self.desvio


def log_probabilidades(notas):
    z = notas - notas.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def erro_das_notas(notas, y):
    """Gradiente da log-loss média do lote em relação às notas."""
    d = np.exp(log_probabilidades(notas))
    d[np.arange(len(y)), y] -= 1
    return d / len(y)


class Etapa4:
    """A etapa 4 com ida e volta: dois blocos residuais, cada um com duas convs 3 x 3 e o atalho.

    O batch normalization fica em modo de avaliação: as médias e variâncias guardadas do ImageNet
    não mudam. A escala (weight) e o deslocamento (bias) dele são parâmetros e aprendem junto.
    """
    CONVS = [("layer4.0.conv1", "layer4.0.bn1", 2, 1), ("layer4.0.conv2", "layer4.0.bn2", 1, 1),
             ("layer4.0.downsample.0", "layer4.0.downsample.1", 2, 0),
             ("layer4.1.conv1", "layer4.1.bn1", 1, 1), ("layer4.1.conv2", "layer4.1.bn2", 1, 1)]

    def __init__(self, p):
        self.W, self.g, self.beta, self.media, self.var, self.passo, self.borda = {}, {}, {}, {}, {}, {}, {}
        for c, b, passo, borda in self.CONVS:
            self.W[c] = p[f"{c}.weight"].astype(np.float32).copy()
            self.g[c] = p[f"{b}.weight"].astype(np.float32).copy()
            self.beta[c] = p[f"{b}.bias"].astype(np.float32).copy()
            self.media[c] = p[f"{b}.running_mean"].astype(np.float32)
            self.var[c] = p[f"{b}.running_var"].astype(np.float32)
            self.passo[c], self.borda[c] = passo, borda

    def copia(self):
        e = Etapa4.__new__(Etapa4)
        for k in ("W", "g", "beta"):
            setattr(e, k, {c: v.copy() for c, v in getattr(self, k).items()})
        for k in ("media", "var", "passo", "borda"):
            setattr(e, k, getattr(self, k))
        return e

    def parametros(self):
        return sum(self.W[c].size + self.g[c].size + self.beta[c].size for c in self.W)

    def _cbn(self, x, c, guardar):
        z, cols = conv(x, self.W[c], self.passo[c], self.borda[c])
        inv = 1.0 / np.sqrt(self.var[c] + EPS)
        xhat = (z - self.media[c]) * inv
        if guardar is not None:
            guardar[c] = (cols, x.shape, xhat, inv)
        return xhat * self.g[c] + self.beta[c]

    def ida(self, x, guardar=None):
        """x: o mapa da etapa 3, (n, h, w, 256). Retorna os 512 números de cada foto."""
        a = np.maximum(self._cbn(x, "layer4.0.conv1", guardar), 0)
        s1 = self._cbn(a, "layer4.0.conv2", guardar) + self._cbn(x, "layer4.0.downsample.0", guardar)
        o1 = np.maximum(s1, 0)
        b = np.maximum(self._cbn(o1, "layer4.1.conv1", guardar), 0)
        o2 = np.maximum(self._cbn(b, "layer4.1.conv2", guardar) + o1, 0)
        if guardar is not None:
            guardar.update(a=a, o1=o1, b=b, o2=o2)
        return o2.mean(axis=(1, 2))

    def _cbn_volta(self, d, c, guardado, grads):
        cols, forma, xhat, inv = guardado[c]
        grads[c] = {"g": (d * xhat).sum(axis=(0, 1, 2)), "beta": d.sum(axis=(0, 1, 2))}
        dz = d * (self.g[c] * inv)
        dW, dx = conv_volta(dz, cols, self.W[c], forma, self.passo[c], self.borda[c])
        grads[c]["W"] = dW
        return dx

    def volta(self, df, guardado):
        """df: o erro dos 512 números. Retorna o gradiente de cada peso, escala e deslocamento."""
        o2 = guardado["o2"]
        n, h, w, c = o2.shape
        d = np.broadcast_to(df[:, None, None, :] / (h * w), o2.shape) * (o2 > 0)
        grads = {}
        db = self._cbn_volta(d, "layer4.1.conv2", guardado, grads) * (guardado["b"] > 0)
        do1 = d + self._cbn_volta(db, "layer4.1.conv1", guardado, grads)
        ds1 = do1 * (guardado["o1"] > 0)
        da = self._cbn_volta(ds1, "layer4.0.conv2", guardado, grads) * (guardado["a"] > 0)
        self._cbn_volta(da, "layer4.0.conv1", guardado, grads)
        self._cbn_volta(ds1, "layer4.0.downsample.0", guardado, grads)
        return grads

    def passo_sgd(self, grads, taxa):
        for c, g in grads.items():
            self.W[c] -= taxa * g["W"]
            self.g[c] -= taxa * g["g"]
            self.beta[c] -= taxa * g["beta"]

    def mudou(self, original):
        """Tamanho da diferença dos pesos, dividido pelo tamanho dos pesos prontos (como na aula 09)."""
        dif = sum(float(((self.W[c] - original.W[c]) ** 2).sum()) for c in self.W)
        base = sum(float((original.W[c] ** 2).sum()) for c in self.W)
        return np.sqrt(dif / base)
