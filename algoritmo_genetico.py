import pygad
import numpy as np
from datetime import datetime
import time


# =========================
# CONFIGURAÇÕES DO ALGORITMO
# =========================

NOME_ARQUIVO = "test.in"

NUM_GENERATIONS = 10000
SOL_PER_POP = 100
NUM_PARENTS_MATING = 20
MUTATION_PERCENT_GENES = 2

PARENT_SELECTION_TYPE = "sss"
CROSSOVER_TYPE = "single_point"
MUTATION_TYPE = "random"

INTERVALO_EXIBICAO = 100


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

        # Formato da linha:
        # id valor peso
        valor = int(partes[1])
        peso = int(partes[2])

        valores.append(valor)
        pesos.append(peso)

    capacidade = int(linhas[quantidade_itens + 1])

    return np.array(valores), np.array(pesos), capacidade


# =========================
# REPARO DA SOLUÇÃO
# =========================

def reparar_solucao(solution, valores, pesos, capacidade):
    solution = np.array(solution).copy().astype(int)

    peso_total = np.sum(solution * pesos)

    if peso_total <= capacidade:
        return solution

    itens_escolhidos = np.where(solution == 1)[0]

    # Ordena os itens escolhidos pela pior relação valor/peso
    ordem_remocao = itens_escolhidos[
        np.argsort(valores[itens_escolhidos] / pesos[itens_escolhidos])
    ]

    for item in ordem_remocao:
        if peso_total <= capacidade:
            break

        solution[item] = 0
        peso_total -= pesos[item]

    return solution


# =========================
# CRIA POPULAÇÃO INICIAL VÁLIDA
# =========================

def criar_populacao_inicial(tamanho_populacao, quantidade_itens, valores, pesos, capacidade):
    populacao = []

    for _ in range(tamanho_populacao):
        individuo = np.random.randint(0, 2, quantidade_itens)

        individuo = reparar_solucao(
            individuo,
            valores,
            pesos,
            capacidade
        )

        populacao.append(individuo)

    return np.array(populacao)


# =========================
# PROGRAMA PRINCIPAL
# =========================

valores, pesos, capacidade = ler_instancia(NOME_ARQUIVO)

quantidade_itens = len(valores)

populacao_inicial = criar_populacao_inicial(
    SOL_PER_POP,
    quantidade_itens,
    valores,
    pesos,
    capacidade
)


# =========================
# FUNÇÃO DE AVALIAÇÃO
# =========================

def fitness_func(ga_instance, solution, solution_idx):
    solution_corrigida = reparar_solucao(
        solution,
        valores,
        pesos,
        capacidade
    )

    peso_total = np.sum(solution_corrigida * pesos)
    valor_total = np.sum(solution_corrigida * valores)

    if peso_total > capacidade:
        return 0

    return valor_total


# =========================
# EXIBIÇÃO POR GERAÇÃO
# =========================

def mostrar_geracao(ga_instance):
    if ga_instance.generations_completed % INTERVALO_EXIBICAO == 0:
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        solution = reparar_solucao(
            solution,
            valores,
            pesos,
            capacidade
        )

        peso_total = np.sum(solution * pesos)
        valor_total = np.sum(solution * valores)
        quantidade_itens_escolhidos = np.sum(solution == 1)

        print(
            f"Geração {ga_instance.generations_completed} "
            f"| Melhor valor: {int(valor_total)} "
            f"| Peso: {int(peso_total)}/{capacidade} "
            f"| Itens selecionados: {int(quantidade_itens_escolhidos)}"
        )


# =========================
# CONFIGURAÇÃO DO PYGAD
# =========================

ga_instance = pygad.GA(
    num_generations=NUM_GENERATIONS,
    num_parents_mating=NUM_PARENTS_MATING,
    fitness_func=fitness_func,
    initial_population=populacao_inicial,
    num_genes=quantidade_itens,
    gene_space=[0, 1],
    gene_type=int,
    parent_selection_type=PARENT_SELECTION_TYPE,
    crossover_type=CROSSOVER_TYPE,
    mutation_type=MUTATION_TYPE,
    mutation_percent_genes=MUTATION_PERCENT_GENES,
    on_generation=mostrar_geracao
)


# =========================
# MEDIÇÃO DO TEMPO
# =========================

inicio_data_hora = datetime.now()
inicio_tempo = time.perf_counter()

print("\n===== INÍCIO DA EXECUÇÃO =====")
print("Começou em:", inicio_data_hora.strftime("%d/%m/%Y %H:%M:%S"))


# =========================
# EXECUTA O ALGORITMO
# =========================

ga_instance.run()


fim_data_hora = datetime.now()
fim_tempo = time.perf_counter()

tempo_total = fim_tempo - inicio_tempo


# =========================
# RESULTADO FINAL
# =========================

solution, solution_fitness, solution_idx = ga_instance.best_solution()

solution = reparar_solucao(
    solution,
    valores,
    pesos,
    capacidade
)

peso_total = np.sum(solution * pesos)
valor_total = np.sum(solution * valores)

itens_escolhidos = []

for i in range(len(solution)):
    if solution[i] == 1:
        itens_escolhidos.append(i + 1)


print("\n===== MELHOR SOLUÇÃO ENCONTRADA =====")
print("Quantidade de itens disponíveis:", quantidade_itens)
print("Quantidade de itens escolhidos:", len(itens_escolhidos))

if quantidade_itens <= 100:
    print("Itens escolhidos:", itens_escolhidos)
else:
    print("Itens escolhidos: lista ocultada porque há muitos itens")

print("Valor total:", int(valor_total))
print("Peso total:", int(peso_total))
print("Capacidade:", capacidade)

if peso_total <= capacidade:
    print("Solução válida: o peso está dentro da capacidade.")
else:
    print("Solução inválida: passou da capacidade.")


# =========================
# TEMPO DE EXECUÇÃO
# =========================

print("\n===== TEMPO DE EXECUÇÃO =====")
print("Começou em:", inicio_data_hora.strftime("%d/%m/%Y %H:%M:%S"))
print("Terminou em:", fim_data_hora.strftime("%d/%m/%Y %H:%M:%S"))
print(f"Tempo total: {tempo_total:.2f} segundos")


# =========================
# CONFIGURAÇÕES UTILIZADAS
# =========================

print("\n===== CONFIGURAÇÕES UTILIZADAS =====")
print("Arquivo de entrada:", NOME_ARQUIVO)
print("Quantidade de itens disponíveis:", quantidade_itens)
print("Capacidade da mochila:", capacidade)
print("Número de gerações:", NUM_GENERATIONS)
print("Tamanho da população:", SOL_PER_POP)
print("Quantidade de pais para cruzamento:", NUM_PARENTS_MATING)
print("Taxa de mutação (% genes):", MUTATION_PERCENT_GENES)
print("Tipo de seleção:", PARENT_SELECTION_TYPE)
print("Tipo de cruzamento:", CROSSOVER_TYPE)
print("Tipo de mutação:", MUTATION_TYPE)
print("Intervalo de exibição das gerações:", INTERVALO_EXIBICAO)