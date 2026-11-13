"""A venda, o corte e as previsões de comparação da aula «LSTM em vendas: extensão temporal».

Este arquivo não imprime nada: o comparar_lstm.py e o minha_comparacao.py importam daqui.

As regras são as do curso Séries temporais. A previsão do dia D é feita na noite da origem
O = D − horizonte e só lê a venda até O. O calendário entra inteiro, porque a loja já sabe dos
feriados e das promoções antes do dia. A régua, a semana passada e o calendário + AR(1) são os
mesmos das aulas «Naive e naive sazonal» e «Modelo contra baseline» daquele curso.
"""
from pathlib import Path

import numpy as np
import pandas as pd

VENDAS = Path(__file__).resolve().parent.parent / "dados" / "vendas"
INICIO = pd.Timestamp("2024-01-01")
FIM = pd.Timestamp("2025-12-31")
ANO = 365.25
EVENTOS = ["Black Friday", "fim de semana da Black Friday", "Natal", "feriado com a loja aberta"]
FALTA, SOBRA = 6.0, 2.0          # R$ por unidade que falta ou sobra na escala, a conta da Bia


# ---------- a venda ----------

def carregar(coluna="unidades"):
    """Um dia por linha, de 01/01/2024 a 31/12/2025, com o calendário de cada dia.

    `real` é a venda do dia (vazio se não há registro). `usavel` é a venda que pode alimentar um
    modelo: dia com registro, completo (último registro às 21h30 ou depois) e com a loja aberta.
    """
    venda = pd.read_csv(VENDAS / "venda_diaria.csv", parse_dates=["data"], dtype={"ultimo_registro": str})
    cal = pd.read_csv(VENDAS / "calendario.csv", parse_dates=["data"], keep_default_na=False).set_index("data")
    dias = pd.date_range(INICIO, FIM)
    v = venda.set_index("data").reindex(dias)
    c = cal.reindex(dias)
    d = pd.DataFrame(index=dias)
    d["real"] = v[coluna].astype(float)
    d["tem_linha"] = v[coluna].notna()
    d["incompleto"] = v["ultimo_registro"].notna() & (v["ultimo_registro"] < "21:30")
    d["fechada"] = c["loja_fechada"] == 1
    d["usavel"] = d["real"].where(~(d["real"].isna() | d["incompleto"] | d["fechada"]))
    for nome in EVENTOS[:3]:
        d[nome] = c["promocao"] == nome
    d[EVENTOS[3]] = (c["feriado"] != "") & ~d["fechada"]
    d["evento"] = d[EVENTOS].any(axis=1)
    return d


# ---------- as previsões simples ----------

def semana_passada(d, dia, horizonte):
    """O mesmo dia da semana, na semana mais recente que já existe na origem."""
    k = -(-horizonte // 7)
    for _ in range(8):
        x = d["usavel"].get(dia - pd.Timedelta(days=7 * k))
        if x is not None and not np.isnan(x):
            return float(x)
        k += 1
    return np.nan


def mesmo_dia_4_semanas(d, dia, horizonte):
    """A régua: média do mesmo dia da semana nas 4 semanas que terminam na origem."""
    k0 = -(-horizonte // 7)
    vals = [d["usavel"].get(dia - pd.Timedelta(days=7 * k)) for k in range(k0, k0 + 4)]
    vals = [x for x in vals if x is not None and not np.isnan(x)]
    return float(np.mean(vals)) if vals else np.nan


# ---------- calendário + AR(1), no log ----------

def colunas_calendario(d, datas):
    """Constante, tendência, onda do ano (seno e cosseno), dia da semana e eventos."""
    ang = 2 * np.pi * datas.dayofyear.to_numpy() / ANO
    x = {"constante": np.ones(len(datas)), "tendência": (datas - INICIO).days.to_numpy() / ANO,
         "sen1": np.sin(ang), "cos1": np.cos(ang), "sen2": np.sin(2 * ang), "cos2": np.cos(2 * ang)}
    for k in range(1, 7):
        x[f"dia {k}"] = (datas.dayofweek == k).astype(float)
    for nome in EVENTOS:
        x[nome] = d.loc[datas, nome].to_numpy(dtype=float)
    return pd.DataFrame(x, index=datas)


def calendario_ar1(d, origem, dia):
    """Ajusta na venda até a origem e prevê o dia: exp(calendário + fi^h × último resíduo)."""
    treino = d.loc[:origem]
    treino = treino[treino["usavel"].notna()]
    X = colunas_calendario(d, treino.index)
    X = X.loc[:, X.abs().sum() > 0]
    y = np.log(treino["usavel"].to_numpy())
    with np.errstate(all="ignore"):              # o NumPy do macOS avisa à toa no produto de matrizes
        coef, *_ = np.linalg.lstsq(X.to_numpy(), y, rcond=None)
        res = y - X.to_numpy() @ coef
    s = pd.Series(res, index=treino.index).reindex(pd.date_range(treino.index[0], origem))
    par = pd.concat([s, s.shift(1)], axis=1).dropna()
    fi = float(np.sum(par.iloc[:, 0] * par.iloc[:, 1]) / np.sum(par.iloc[:, 1] ** 2))
    r = s.dropna()
    x = colunas_calendario(d, pd.DatetimeIndex([dia]))[list(X.columns)].to_numpy()[0]
    with np.errstate(all="ignore"):
        log = float(x @ coef) + fi ** (dia - r.index[-1]).days * float(r.iloc[-1])
    return float(np.exp(log))


# ---------- a janela que a LSTM lê ----------

def janela(d, origem, tamanho, media, desvio, com_calendario):
    """Os `tamanho` dias que terminam na noite da origem, um vetor por dia.

    Cada dia tem a venda no log, centrada e dividida (0 quando o dia não serve), e um 1 quando o
    dia não serve. Com calendário, mais as 4 colunas de evento de cada dia. Nada depois da origem.
    """
    dias = pd.date_range(origem - pd.Timedelta(days=tamanho - 1), origem)
    if dias[0] < INICIO:
        raise ValueError(f"a janela de {tamanho} dias da origem {origem:%d/%m/%Y} começa antes da venda")
    u = d.loc[dias, "usavel"].to_numpy()
    falta = np.isnan(u)
    z = np.where(falta, 0.0, (np.log(np.where(falta, 1.0, u)) - media) / desvio)
    partes = [z, falta.astype(float)]
    if com_calendario:
        partes += [d.loc[dias, nome].to_numpy(dtype=float) for nome in EVENTOS]
    return np.stack(partes, axis=1)


def calendario_do_dia(d, dia):
    """O que a loja já sabe do dia previsto: dia da semana, onda do ano e eventos."""
    ang = 2 * np.pi * dia.dayofyear / ANO
    semana = [float(dia.dayofweek == k) for k in range(1, 7)]
    return np.array([np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)] + semana
                    + [float(d.at[dia, nome]) for nome in EVENTOS])


# ---------- as contas do erro ----------

def wmape(real, prev):
    return float(np.sum(np.abs(real - prev)) / np.sum(real) * 100)


def custo(real, prev):
    erro = np.asarray(real) - np.asarray(prev)          # positivo: faltou gente; negativo: sobrou
    return float(FALTA * erro.clip(min=0).sum() + SOBRA * (-erro).clip(min=0).sum())


def br(x, casas=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "vazio"
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


def sinal(x, casas=1):
    return ("+" if x >= 0 else "−") + br(abs(x), casas)
