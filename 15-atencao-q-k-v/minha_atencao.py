"""Exercício da aula «Atenção Q/K/V»: a camada de atenção do Caio, com duas cabeças.

    python3 testar_minha_atencao.py
    python3 minha_atencao.py

Você só escreve a função atencao. O resto do arquivo já está pronto.
"""
import numpy as np

PALAVRAS = ["meu", "pedido", "não", "chegou"]

# Cada palavra vira 5 números (d_model = 5).
X = np.array([[1, 0, 0, 0, 1],    # meu
              [0, 1, 0, 0, 1],    # pedido
              [0, 0, 1, 0, 1],    # não
              [0, 1, 0, 1, 0]],   # chegou
             dtype=float)

CABECAS = 2
# Cabeça 1 nas 3 primeiras colunas de W_Q e W_K e nas 2 primeiras de W_V; cabeça 2 nas seguintes.
W_Q = np.array([[0, 0, 1,   0, 1, 0],
                [1, 0, 0,   0, 0, 0],
                [0, 2, 0,   1, 0, 0],
                [0, 1, 1,   2, 0, 0],
                [0, 0, 0,   0, 0, 1]], dtype=float)
W_K = np.array([[0, 0, 1,   0, 0, 1],
                [1, 0, 0,   2, 0, 0],
                [0, 0, 0,   0, 1, 0],
                [0, 2, 0,   0, 0, 0],
                [0, 0, 0,   0, 0, 1]], dtype=float)
W_V = np.array([[0, 1,   1, 0],
                [1, 0,   0, 1],
                [0, -1,  0, 0],
                [1, 1,   1, 1],
                [0, 0,   0, 0]], dtype=float)
# W_O junta as duas cabeças (2 + 2 = 4 números por palavra) e volta para d_model = 5.
W_O = np.array([[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1]], dtype=float)


def atencao(X, W_Q, W_K, W_V, W_O, cabecas, causal=False):
    """Atenção com várias cabeças. Retorna (pesos, saida).

    X: (n, d_model). W_Q e W_K: (d_model, cabecas * d_k). W_V: (d_model, cabecas * d_v).
    W_O: (cabecas * d_v, d_saida). As colunas de cada cabeça vêm em blocos seguidos: a cabeça 1
    usa as primeiras d_k colunas de W_Q e de W_K e as primeiras d_v colunas de W_V; a cabeça 2,
    as seguintes; e assim por diante.

    pesos: (cabecas, n, n). Na cabeça c, a linha i diz quanto a palavra i olha para cada palavra.
    Cada linha soma 1, mesmo com notas grandes como 1000 (sem nan nem inf).
    saida: (n, d_saida). As saídas das cabeças ficam lado a lado, na ordem das cabeças, e passam por W_O.
    As notas de cada cabeça são divididas pela raiz do d_k dessa cabeça.
    Com causal=True, a palavra i só olha para as palavras 0 a i: os pesos acima da diagonal são 0.

    Para com ValueError se X não for uma matriz, se as linhas de W_Q, W_K ou W_V não forem d_model,
    se W_Q e W_K tiverem formatos diferentes, se as colunas de W_Q ou de W_V não dividirem por
    cabecas, ou se as linhas de W_O não forem as colunas de W_V.
    """
    raise NotImplementedError("escreva atencao")


# ---------------------------------------------------------------------------------------------
# Daqui para baixo já está pronto.

def forma(m):
    return "(" + " x ".join(str(d) for d in np.shape(m)) + ")"


def mostrar_pesos(titulo, pesos):
    print(titulo)
    print("    " + " " * 7 + "".join(f"{p:>9}" for p in PALAVRAS) + "     soma")
    for i, linha in enumerate(pesos):
        print(f"    {PALAVRAS[i]:<7}" + "".join(f"{v:9.3f}" for v in linha) + f"    {linha.sum():.3f}")


if __name__ == "__main__":
    print("A camada do Caio: " + " ".join(PALAVRAS))
    for nome, m in (("X", X), ("W_Q", W_Q), ("W_K", W_K), ("W_V", W_V), ("W_O", W_O)):
        print(f"  {nome:<4}{forma(m)}")
    pesos, saida = atencao(X, W_Q, W_K, W_V, W_O, CABECAS)
    print(f"  pesos {forma(pesos)}   saída {forma(saida)}")
    for c in range(CABECAS):
        mostrar_pesos(f"\nCabeça {c + 1}, sem máscara", pesos[c])
    print("\nSaída da camada, sem máscara")
    for i, linha in enumerate(saida):
        print(f"    {PALAVRAS[i]:<7}" + "".join(f"{v:9.3f}" for v in linha))
    pesos_c, _ = atencao(X, W_Q, W_K, W_V, W_O, CABECAS, causal=True)
    for c in range(CABECAS):
        mostrar_pesos(f"\nCabeça {c + 1}, com máscara causal", pesos_c[c])
