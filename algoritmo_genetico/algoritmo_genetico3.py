import pygad
import numpy as np
from pathlib import Path


# =========================
# CONFIGURAÇÕES
# =========================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

NOME_ARQUIVO = ROOT_DIR / "test_3.in"
ARQUIVO_SAIDA = BASE_DIR / "saida3.csv"

MAX_AVALIACOES = 10000

SOL_PER_POP = 100

# O PyGAD avalia a população inicial.
# Então: 100 avaliações iniciais + 99 gerações * 100 = 10000
NUM_GENERATIONS = (MAX_AVALIACOES // SOL_PER_POP) - 1

NUM_PARENTS_MATING = 15

PARENT_SELECTION_TYPE = "rank"
CROSSOVER_TYPE = "uniform"
MUTATION_TYPE = "random"
MUTATION_PERCENT_GENES = 4

avaliacao_id = 0
first = True
total_avaliacoes = MAX_AVALIACOES

# Guarda o melhor valor encontrado até o momento
melhor_valor_ate_agora = 0

# Guarda a posição da coluna dessa execução no CSV
coluna_execucao = None


# =========================
# LENDO ARQUIVO DE SAIDA
# =========================

try:
    with open(ARQUIVO_SAIDA, "r", newline="", encoding="utf-8") as arquivo:
        csv = arquivo.read()
        csv = csv.splitlines()

        for i in range(len(csv)):
            csv[i] = csv[i].split(",")

except FileNotFoundError:
    csv = [["avaliacao"]]


# =========================
# LEITURA DA INSTÂNCIA
# =========================

def ler_instancia(nome_arquivo):
    with open(nome_arquivo, "r") as arquivo:
        linhas = [linha.strip() for linha in arquivo.readlines() if linha.strip()]

    quantidade_itens = int(linhas[0])

    valores = []
    pesos = []

    for i in range(1, quantidade_itens + 1):
        partes = linhas[i].split()

        # Formato esperado:
        # id valor peso
        valor = int(partes[1])
        peso = int(partes[2])

        valores.append(valor)
        pesos.append(peso)

    capacidade = int(linhas[quantidade_itens + 1])

    return np.array(valores), np.array(pesos), capacidade


# =========================
# CARREGANDO OS DADOS
# =========================

valores, pesos, capacidade = ler_instancia(NOME_ARQUIVO)

quantidade_itens = len(valores)


# =========================
# FUNÇÃO AUXILIAR PARA SALVAR NO CSV
# =========================

def salvar_avaliacao_no_csv(valor_para_salvar):
    global csv, avaliacao_id, coluna_execucao

    linha_csv = avaliacao_id + 1

    if linha_csv < len(csv):
        # Garante alinhamento da coluna correta
        while len(csv[linha_csv]) < coluna_execucao:
            csv[linha_csv].append("")

        csv[linha_csv].append(str(valor_para_salvar))

    else:
        nova_linha = [str(avaliacao_id)]

        # Preenche colunas anteriores, se existirem
        while len(nova_linha) < coluna_execucao:
            nova_linha.append("")

        nova_linha.append(str(valor_para_salvar))

        csv.append(nova_linha)


# =========================
# FUNÇÃO DE APTIDÃO
# =========================

def fitness_func(ga_instance, solution, solution_idx):
    global first, avaliacao_id, total_avaliacoes, csv
    global melhor_valor_ate_agora, coluna_execucao

    valor_total = int(np.sum(solution * valores))
    peso_total = int(np.sum(solution * pesos))

    # Penalização simples:
    # se passar da capacidade, a solução recebe fitness 0.
    if peso_total > capacidade:
        fitness = 0
    else:
        fitness = valor_total

    # Se já registrou 10000 avaliações, não salva mais no CSV
    if avaliacao_id >= MAX_AVALIACOES:
        return fitness

    # Atualiza o melhor valor encontrado até agora
    if fitness > melhor_valor_ate_agora:
        melhor_valor_ate_agora = fitness

    # Cria uma nova coluna para esta execução do AG3
    if first:
        coluna_execucao = len(csv[0])
        csv[0].append("ex" + str(coluna_execucao))
        first = False

    # Salva o melhor valor encontrado até esta avaliação
    salvar_avaliacao_no_csv(melhor_valor_ate_agora)

    print(
        f"Avaliando solução {avaliacao_id}: "
        f"valor={valor_total}, fitness={fitness}, melhor até agora={melhor_valor_ate_agora}"
    )

    total_avaliacoes -= 1
    avaliacao_id += 1

    return fitness


# =========================
# CRIAÇÃO DO ALGORITMO GENÉTICO
# =========================

ga_instance = pygad.GA(
    num_generations=NUM_GENERATIONS,
    sol_per_pop=SOL_PER_POP,
    num_parents_mating=NUM_PARENTS_MATING,
    num_genes=quantidade_itens,
    gene_space=[0, 1],
    gene_type=int,
    fitness_func=fitness_func,
    parent_selection_type=PARENT_SELECTION_TYPE,
    crossover_type=CROSSOVER_TYPE,
    mutation_type=MUTATION_TYPE,
    mutation_percent_genes=MUTATION_PERCENT_GENES,

    # Importante para reduzir reaproveitamento de indivíduos
    keep_parents=0,
    keep_elitism=1
)


# =========================
# EXECUÇÃO
# =========================

ga_instance.run()


# =========================
# GARANTINDO 10000 REGISTROS
# =========================

# Caso o PyGAD faça menos chamadas reais da fitness_func,
# completa o CSV com o último melhor valor encontrado.
if first:
    coluna_execucao = len(csv[0])
    csv[0].append("ex" + str(coluna_execucao))
    first = False

while avaliacao_id < MAX_AVALIACOES:
    salvar_avaliacao_no_csv(melhor_valor_ate_agora)
    avaliacao_id += 1


# =========================
# RESULTADO FINAL
# =========================

solution, solution_fitness, solution_idx = ga_instance.best_solution()

solution = np.array(solution).astype(int)

valor_final = int(np.sum(solution * valores))
peso_final = int(np.sum(solution * pesos))
quantidade_itens_escolhidos = int(np.sum(solution))

print("\n===== MELHOR SOLUÇÃO ENCONTRADA - AG3 =====")
print(f"Valor total: {valor_final}")
print(f"Peso total: {peso_final}/{capacidade}")
print(f"Quantidade de itens escolhidos: {quantidade_itens_escolhidos}")
print(f"Fitness da solução: {solution_fitness}")
print(f"Total de avaliações registradas: {avaliacao_id}")
print(f"Número de gerações usado: {NUM_GENERATIONS}")


with open(ARQUIVO_SAIDA, "w+", newline="", encoding="utf-8") as arquivo:
    for linha in csv:
        arquivo.write(",".join(linha) + "\n")