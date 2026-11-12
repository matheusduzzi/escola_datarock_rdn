# Venda diária da loja física da Loja Exemplo

Dois anos de venda da loja física e o calendário que a loja conhece antes de prever. São os mesmos arquivos do curso Séries temporais. A aula 12 usa essa venda para comparar uma LSTM com as previsões simples daquele curso, e a tarefa 5 do diagnóstico mede essas previsões.

| Arquivo | Linhas | Colunas | O que tem |
|---|---|---|---|
| `venda_diaria.csv` | 722 dias | `data`, `pedidos`, `unidades`, `ultimo_registro` (hora do último pedido gravado) | a venda de 01/01/2024 a 31/12/2025 |
| `calendario.csv` | 762 dias | `data`, `feriado` (nome, vazio se não é feriado), `loja_fechada` (1 se fechou), `promocao` (nome, vazio se não há) | de 01/01/2024 a 31/01/2026 |

Uma linha da venda é um dia. O arquivo não tem linha para todo dia do calendário, e nem toda linha é um dia completo: a hora do último registro mostra quando o sistema parou de gravar. A loja vende das 10h às 22h.

O calendário vai um mês além da venda de propósito: feriados, dias fechados e promoções planejadas já são conhecidos antes do dia. A rede pode usar o que está no calendário; a venda do próprio dia ela só conhece depois.

## Como os dados nascem

Os números são inventados e saem do `gerar.py` dessa pasta. O começo do arquivo conta o mecanismo: nível da venda, crescimento por ano, peso de cada dia da semana, onda do ano, promoções, feriados, ruído e falhas do sistema. Faça as análises das aulas antes de abrir o gerador. Depois abra e compare.

## Gerar de novo

```bash
python3 gerar.py
```

A saída é `venda_diaria=722 dias_no_calendario=731 sem_registro=9 calendario=762`. O gerador só usa a biblioteca padrão do Python e sai igual toda vez. Não edite os CSVs à mão.
