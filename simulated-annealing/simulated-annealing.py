import random
import math
import numpy as np
from pathlib import Path



# CONFIGURAÇÕES
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

NOME_ARQUIVO  =  ROOT_DIR / "test_3.in"
ARQUIVO_SAIDA = "saida2_sa.csv"

MAX_AVALIACOES = 10000

# Parâmetros do SA
T_MAX = 100.0
T_MIN = 0.01
ALPHA = 0.95
MAX_ITERATION = 100 
M_FLIP = 2           
SEED = 1


# LEITURA DA INSTÂNCIA (idêntica à da GA)
def ler_instancia(nome_arquivo):
    with open(nome_arquivo, "r") as arquivo:
        linhas = [linha.strip() for linha in arquivo.readlines() if linha.strip()]

    quantidade_itens = int(linhas[0])

    valores = []
    pesos = []

    for i in range(1, quantidade_itens + 1):
        partes = linhas[i].split()
        # Formato esperado: id valor peso
        valor = int(partes[1])
        peso = int(partes[2])
        valores.append(valor)
        pesos.append(peso)

    capacidade = int(linhas[quantidade_itens + 1])

    return np.array(valores), np.array(pesos), capacidade


valores, pesos, capacidade = ler_instancia(NOME_ARQUIVO)
quantidade_itens = len(valores)


# LEITURA/ESCRITA DO CSV (mesmo esquema da GA)
try:
    with open(ARQUIVO_SAIDA, "r", newline="", encoding="utf-8") as arquivo:
        linhas_csv = arquivo.read().splitlines()
        linhas_csv = [linha.split(",") for linha in linhas_csv]
except FileNotFoundError:
    linhas_csv = [["avaliacao"]]

coluna_execucao = len(linhas_csv[0])
linhas_csv[0].append("sa" + str(coluna_execucao))


def salvar_avaliacao_no_csv(avaliacao_id, valor_para_salvar):
    linha_csv = avaliacao_id + 1

    if linha_csv < len(linhas_csv):
        while len(linhas_csv[linha_csv]) < coluna_execucao:
            linhas_csv[linha_csv].append("")
        linhas_csv[linha_csv].append(str(valor_para_salvar))
    else:
        nova_linha = [str(avaliacao_id)]
        while len(nova_linha) < coluna_execucao:
            nova_linha.append("")
        nova_linha.append(str(valor_para_salvar))
        linhas_csv.append(nova_linha)


# FUNÇÃO DE APTIDÃO (idêntica à penalização usada na GA)
def fitness(solucao):
    valor_total = int(np.dot(solucao, valores))
    peso_total = int(np.dot(solucao, pesos))

    if peso_total > capacidade:
        return 0  # mesma penalização simples da GA - sem reparo

    return valor_total


# SOLUÇÃO INICIAL (aleatória)
def solucao_aleatoria(rng):
    return [rng.randint(0, 1) for _ in range(quantidade_itens)]

# VIZINHANÇA: m-flip simples (sem reparo/Zhan_RI)
def vizinho(solucao, rng, m=M_FLIP):
    nova = solucao.copy()
    posicoes = rng.sample(range(quantidade_itens), k=min(m, quantidade_itens))
    for p in posicoes:
        nova[p] = 1 - nova[p]
    return nova

# EXECUÇÃO DO SA
# Usa a contagem atual de colunas do CSV como semente dinâmica (ex: 1, 2, 3...)
SEED_DINAMICA = coluna_execucao 
rng = random.Random(SEED_DINAMICA)

s = solucao_aleatoria(rng)
f_s = fitness(s)

melhor_solucao = list(s)
melhor_valor_ate_agora = f_s

avaliacao_id = 0
salvar_avaliacao_no_csv(avaliacao_id, melhor_valor_ate_agora)
avaliacao_id += 1

T = T_MAX
while avaliacao_id < MAX_AVALIACOES and T >= T_MIN:
    for _ in range(MAX_ITERATION):
        if avaliacao_id >= MAX_AVALIACOES:
            break

        s_prime = vizinho(s, rng)
        f_prime = fitness(s_prime)

        delta = f_s - f_prime  # positivo = candidata é pior

        if f_prime >= f_s:
            s, f_s = s_prime, f_prime
        else:
            prob = math.exp(-delta / T) if T > 0 else 0.0
            if rng.random() < prob:
                s, f_s = s_prime, f_prime

        if f_s > melhor_valor_ate_agora:
            melhor_valor_ate_agora = f_s
            melhor_solucao = list(s)

        salvar_avaliacao_no_csv(avaliacao_id, melhor_valor_ate_agora)
        avaliacao_id += 1

        print(
            f"Avaliando solução {avaliacao_id}: "
            f"fitness={f_prime}, melhor até agora={melhor_valor_ate_agora}"
        )

    T *= ALPHA



while avaliacao_id < MAX_AVALIACOES:
    salvar_avaliacao_no_csv(avaliacao_id, melhor_valor_ate_agora)
    avaliacao_id += 1

# RESULTADO FINAL
melhor_solucao_arr = np.array(melhor_solucao).astype(int)
valor_final = int(np.dot(melhor_solucao_arr, valores))
peso_final = int(np.dot(melhor_solucao_arr, pesos))
quantidade_itens_escolhidos = int(np.sum(melhor_solucao_arr))

print("\n===== MELHOR SOLUÇÃO ENCONTRADA - SA (simples) =====")
print(f"Valor total: {valor_final}")
print(f"Peso total: {peso_final}/{capacidade}")
print(f"Quantidade de itens escolhidos: {quantidade_itens_escolhidos}")
print(f"Total de avaliações registradas: {avaliacao_id}")


with open(ARQUIVO_SAIDA, "w+", newline="", encoding="utf-8") as arquivo:
    for linha in linhas_csv:
        arquivo.write(",".join(linha) + "\n")