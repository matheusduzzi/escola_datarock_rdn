"""Tarefa 5 do diagnóstico: o que uma rede de vendas precisa bater.

    python3 tarefa5.py

Lê a venda diária da loja física em ../dados/vendas/ e mede o erro de três previsões simples
nos dois últimos meses (novembro e dezembro de 2025). A previsão de cada dia sai na noite anterior,
depois que a loja fecha, e só usa o que já se sabia nessa hora.
"""
from pathlib import Path

import pandas as pd

PASTA = Path(__file__).resolve().parent.parent / 'dados' / 'vendas'
CORTE = '2025-11-01'

venda = pd.read_csv(PASTA / 'venda_diaria.csv', parse_dates=['data'])
cal = pd.read_csv(PASTA / 'calendario.csv', parse_dates=['data'])
dias = cal[cal.data <= venda.data.max()].merge(venda, on='data', how='left')
print(f"dias no calendário: {len(dias)}  dias com linha de venda: {dias.pedidos.notna().sum()}  "
      f"de {dias.data.min():%d/%m/%Y} a {dias.data.max():%d/%m/%Y}")

serie = dias.set_index('data').pedidos
prev = pd.DataFrame({
    'pedidos': serie,
    'ontem': serie.shift(1),  # o dia anterior já fechou quando a previsão sai
    'mesmo dia da semana passada': serie.shift(7),
    'média antes do corte': serie[serie.index < CORTE].mean(),
})
aberta = dias.set_index('data').loja_fechada.eq(0)
teste = prev[(prev.index >= CORTE) & aberta].dropna()
print(f"dias avaliados (loja aberta, com venda e com a previsão disponível): {len(teste)} a partir de {CORTE[8:]}/{CORTE[5:7]}/{CORTE[:4]}")
print(f"pedidos por dia nesses dias: média {teste.pedidos.mean():.1f}")
for nome in ['ontem', 'mesmo dia da semana passada', 'média antes do corte']:
    erro = (teste[nome] - teste.pedidos).abs().mean()
    print(f"erro médio absoluto, {nome}: {erro:.1f} pedidos por dia")
