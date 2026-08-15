import random
import math
import numpy as np
from pathlib import Path


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

NOME_ARQUIVO = ROOT_DIR / "test_3.in"
ARQUIVO_SAIDA = "saida1_sa.csv"

MAX_AVALIACOES = 10000
NUM_EXECUCOES = 10

# SA ORIGINAL
T_MAX = 100.0
T_MIN = 0.01
ALPHA = 0.90
MAX_ITERATION = 100
M_FLIP = 2


# ==========================================================
# LEITURA DA INSTÂNCIA
# ==========================================================

def ler_instancia(nome_arquivo):
    with open(nome_arquivo, "r") as arquivo:
        linhas = [
            linha.strip()
            for linha in arquivo.readlines()
            if linha.strip()
        ]

    quantidade_itens = int(linhas[0])

    valores = []
    pesos = []

    for i in range(1, quantidade_itens + 1):
        partes = linhas[i].split()

        # Formato:
        # id valor peso
        valor = int(partes[1])
        peso = int(partes[2])

        valores.append(valor)
        pesos.append(peso)

    capacidade = int(linhas[quantidade_itens + 1])

    return (
        np.array(valores),
        np.array(pesos),
        capacidade
    )


valores, pesos, capacidade = ler_instancia(NOME_ARQUIVO)
quantidade_itens = len(valores)


# ==========================================================
# FUNÇÃO DE APTIDÃO
# ==========================================================

def fitness(solucao):
    valor_total = int(np.dot(solucao, valores))
    peso_total = int(np.dot(solucao, pesos))

    if peso_total > capacidade:
        return 0

    return valor_total


# ==========================================================
# SOLUÇÃO INICIAL
# ==========================================================

def solucao_aleatoria(rng):
    return [
        rng.randint(0, 1)
        for _ in range(quantidade_itens)
    ]


# ==========================================================
# VIZINHANÇA
# ==========================================================

def vizinho(solucao, rng):
    nova = solucao.copy()

    posicoes = rng.sample(
        range(quantidade_itens),
        k=min(M_FLIP, quantidade_itens)
    )

    for p in posicoes:
        nova[p] = 1 - nova[p]

    return nova


# ==========================================================
# EXECUÇÃO DE UMA RODADA DO SA
# ==========================================================

def executar_sa(seed):
    rng = random.Random(seed)

    s = solucao_aleatoria(rng)
    f_s = fitness(s)

    melhor_solucao = list(s)
    melhor_valor = f_s

    historico = []

    # Primeira avaliação
    historico.append(melhor_valor)
    avaliacao_id = 1

    T = T_MAX

    while avaliacao_id < MAX_AVALIACOES and T >= T_MIN:

        for _ in range(MAX_ITERATION):

            if avaliacao_id >= MAX_AVALIACOES:
                break

            s_prime = vizinho(s, rng)
            f_prime = fitness(s_prime)

            # Positivo quando a candidata é pior
            delta = f_s - f_prime

            if f_prime >= f_s:
                s = s_prime
                f_s = f_prime

            else:
                prob = math.exp(-delta / T)

                if rng.random() < prob:
                    s = s_prime
                    f_s = f_prime

            if f_s > melhor_valor:
                melhor_valor = f_s
                melhor_solucao = list(s)

            historico.append(melhor_valor)
            avaliacao_id += 1

        T *= ALPHA

    avaliacoes_efetivas = avaliacao_id

    # Completa a série até 10000 posições
    while len(historico) < MAX_AVALIACOES:
        historico.append(melhor_valor)

    melhor_solucao_arr = np.array(melhor_solucao)

    peso_final = int(
        np.dot(melhor_solucao_arr, pesos)
    )

    quantidade_itens_escolhidos = int(
        np.sum(melhor_solucao_arr)
    )

    return {
        "historico": historico,
        "melhor_valor": melhor_valor,
        "peso_final": peso_final,
        "quantidade_itens": quantidade_itens_escolhidos,
        "avaliacoes_efetivas": avaliacoes_efetivas
    }


# ==========================================================
# 10 EXECUÇÕES
# ==========================================================

resultados = []

for seed in range(1, NUM_EXECUCOES + 1):

    print("\n" + "=" * 60)
    print(f"EXECUÇÃO {seed}/{NUM_EXECUCOES}")
    print(f"SEED = {seed}")
    print("=" * 60)

    resultado = executar_sa(seed)
    resultados.append(resultado)

    print(f"Melhor valor: {resultado['melhor_valor']}")
    print(
        f"Peso: {resultado['peso_final']}/{capacidade}"
    )
    print(
        f"Itens escolhidos: "
        f"{resultado['quantidade_itens']}"
    )
    print(
        f"Avaliações efetivas: "
        f"{resultado['avaliacoes_efetivas']}"
    )


# ==========================================================
# SALVAR CSV
# ==========================================================

with open(
    ARQUIVO_SAIDA,
    "w",
    newline="",
    encoding="utf-8"
) as arquivo:

    cabecalho = ["avaliacao"]

    for execucao in range(1, NUM_EXECUCOES + 1):
        cabecalho.append(f"sa{execucao}")

    arquivo.write(",".join(cabecalho) + "\n")

    for avaliacao in range(MAX_AVALIACOES):

        linha = [str(avaliacao)]

        for resultado in resultados:
            linha.append(
                str(resultado["historico"][avaliacao])
            )

        arquivo.write(",".join(linha) + "\n")


# ==========================================================
# ESTATÍSTICAS FINAIS
# ==========================================================

valores_finais = [
    resultado["melhor_valor"]
    for resultado in resultados
]

media_final = np.mean(valores_finais)
desvio_padrao = np.std(
    valores_finais,
    ddof=1
)

print("\n")
print("=" * 60)
print("RESULTADO FINAL - SA ORIGINAL")
print("=" * 60)

for i, valor in enumerate(valores_finais, start=1):
    print(f"Execução {i}: {valor}")

print("-" * 60)

print(f"Média final: {media_final:.2f}")
print(f"Desvio-padrão: {desvio_padrao:.2f}")
print(f"Melhor execução: {max(valores_finais)}")
print(f"Pior execução: {min(valores_finais)}")

print("=" * 60)

print(
    f"\nResultados salvos em: {ARQUIVO_SAIDA}"
)