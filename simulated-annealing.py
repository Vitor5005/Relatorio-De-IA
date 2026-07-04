import numpy as np
import random
import math
import time

# =========================
# PARÂMETROS
# =========================
NOME_ARQUIVO = "test.in"

TMAX = 1000.0
# Temperatura inicial.
# Valores maiores permitem aceitar mais soluções piores no começo,
# aumentando a exploração do espaço de busca.

TMIN = 1.0 # Padrão: 1.0
# Temperatura final mínima.
# Quando a temperatura chega abaixo desse valor, o algoritmo para.
# Valores menores fazem o algoritmo rodar por mais tempo.

ALPHA = 0.995
# Taxa de resfriamento.
# A cada ciclo, a temperatura é multiplicada por esse fator.
# Quanto mais próximo de 1, mais lento é o resfriamento.

ITER_POR_TEMPERATURA = 100
# Número de vizinhos testados em cada nível de temperatura.
# Valores maiores aumentam o tempo de execução, mas podem melhorar a busca.

PENALIDADE = 1000
# Penalidade aplicada quando a solução ultrapassa a capacidade da mochila.
# Quanto maior esse valor, mais o algoritmo evita soluções inviáveis.

SEMENTE = 42
# Semente aleatória.
# Garante reprodutibilidade, ou seja, execuções iguais com os mesmos parâmetros.

# =========================
# LEITURA DA INSTÂNCIA
# =========================
def ler_instancia(nome_arquivo):
    with open(nome_arquivo, "r") as arquivo:
        linhas = [linha.strip() for linha in arquivo if linha.strip()]

    quantidade_itens = int(linhas[0])

    valores = []
    pesos = []

    for i in range(1, quantidade_itens + 1):
        partes = linhas[i].split()
        valor = int(partes[1])
        peso = int(partes[2])
        valores.append(valor)
        pesos.append(peso)

    capacidade = int(linhas[quantidade_itens + 1])
    return np.array(valores), np.array(pesos), capacidade



# =========================
# FUNÇÕES AUXILIARES
# =========================
def avaliar(solucao):
    valor_total = int(np.dot(solucao, valores))
    peso_total = int(np.dot(solucao, pesos))
    return valor_total, peso_total

def energia(solucao, penalidade=PENALIDADE):
    valor, peso = avaliar(solucao)
    if peso <= capacidade:
        return -valor
    return -valor + penalidade * (peso - capacidade)

def solucao_inicial_gulosa():
    razao = valores / pesos
    ordem = np.argsort(-razao)
    sol = np.zeros(n, dtype=np.int8)
    peso_atual = 0

    for i in ordem:
        if peso_atual + pesos[i] <= capacidade:
            sol[i] = 1
            peso_atual += pesos[i]

    return sol

def solucao_inicial_aleatoria():
    sol = np.zeros(n, dtype=np.int8)
    ordem = np.random.permutation(n)
    peso_atual = 0

    for i in ordem:
        if np.random.rand() < 0.5 and peso_atual + pesos[i] <= capacidade:
            sol[i] = 1
            peso_atual += pesos[i]

    return sol

def gerar_vizinho(solucao):
    vizinho = solucao.copy()
    i = np.random.randint(n)
    vizinho[i] = 1 - vizinho[i]

    if np.dot(vizinho, pesos) <= capacidade:
        return vizinho

    indices_1 = np.where(vizinho == 1)[0]
    if len(indices_1) > 0:
        j = np.random.choice(indices_1)
        vizinho[j] = 0

    return vizinho

# =========================
# SIMULATED ANNEALING
# =========================
def simulated_annealing(
    Tmax=TMAX,
    Tmin=TMIN,
    alpha=ALPHA,
    iter_por_temperatura=ITER_POR_TEMPERATURA,
    penalidade=PENALIDADE,
    usar_gulosa=True,
    semente=SEMENTE
):
    if semente is not None:
        np.random.seed(semente)
        random.seed(semente)

    if usar_gulosa:
        atual = solucao_inicial_gulosa()
    else:
        atual = solucao_inicial_aleatoria()

    melhor = atual.copy()
    e_atual = energia(atual, penalidade)
    e_melhor = e_atual

    T = Tmax
    historico = []
    inicio = time.time()

    while T > Tmin:
        for _ in range(iter_por_temperatura):
            vizinho = gerar_vizinho(atual)
            e_vizinho = energia(vizinho, penalidade)
            delta = e_vizinho - e_atual

            if delta < 0 or np.random.rand() < math.exp(-delta / T):
                atual = vizinho
                e_atual = e_vizinho

                if e_atual < e_melhor:
                    melhor = atual.copy()
                    e_melhor = e_atual

        valor_melhor, peso_melhor = avaliar(melhor)
        historico.append((T, valor_melhor, peso_melhor))
        T *= alpha

    tempo_total = time.time() - inicio
    valor_final, peso_final = avaliar(melhor)

    return {
        "solucao": melhor,
        "valor": valor_final,
        "peso": peso_final,
        "viavel": peso_final <= capacidade,
        "tempo": tempo_total,
        "historico": historico
    }

# =========================
# PROGRAMA PRINCIPAL
# =========================
valores, pesos, capacidade = ler_instancia(NOME_ARQUIVO)
n = len(valores)

resultado = simulated_annealing()

print("Valor:", resultado["valor"])
print("Peso:", resultado["peso"])
print("Viável:", resultado["viavel"])
print("Tempo:", resultado["tempo"])