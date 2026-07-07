import pygad
import numpy as np
from datetime import datetime
import time
import csv
from pathlib import Path


# =========================
# CONFIGURAÇÕES DO ALGORITMO
# =========================

NOME_ARQUIVO = "test.in"

NUM_GENERATIONS = 1000
SOL_PER_POP = 100
NUM_PARENTS_MATING = 20
MUTATION_PERCENT_GENES = 2

PARENT_SELECTION_TYPE = "sss"
CROSSOVER_TYPE = "single_point"
MUTATION_TYPE = "random"

INTERVALO_EXIBICAO = 100

RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)


# =========================
# PASTAS DE SAÍDA
# =========================

PASTA_SAIDA = Path("resultados_algoritmo_genetico")
PASTA_CSV = PASTA_SAIDA / "csv"
PASTA_TXT = PASTA_SAIDA / "txt"
PASTA_GRAFICOS = PASTA_SAIDA / "graficos"
PASTA_GRAFICOS_COMPLETOS = PASTA_GRAFICOS / "completo"
PASTA_GRAFICOS_ESCALA_100 = PASTA_GRAFICOS / "escala_100"
PASTA_GRAFICOS_ESCALA_1000 = PASTA_GRAFICOS / "escala_1000"
PASTA_GRAFICOS_BLOCOS = PASTA_GRAFICOS / "resumo_blocos"

for pasta in [
    PASTA_SAIDA,
    PASTA_CSV,
    PASTA_TXT,
    PASTA_GRAFICOS,
    PASTA_GRAFICOS_COMPLETOS,
    PASTA_GRAFICOS_ESCALA_100,
    PASTA_GRAFICOS_ESCALA_1000,
    PASTA_GRAFICOS_BLOCOS
]:
    pasta.mkdir(exist_ok=True)


# =========================
# ARQUIVOS GERADOS
# =========================

ARQUIVO_ANALISE_GERACOES = PASTA_CSV / "analise_geracoes_ag.csv"
ARQUIVO_ANALISE_ESCALA_100 = PASTA_CSV / "analise_geracoes_ag_escala_100.csv"
ARQUIVO_ANALISE_ESCALA_1000 = PASTA_CSV / "analise_geracoes_ag_escala_1000.csv"
ARQUIVO_RESUMO_BLOCOS_100 = PASTA_CSV / "resumo_blocos_100_geracoes.csv"
ARQUIVO_RESUMO_BLOCOS_1000 = PASTA_CSV / "resumo_blocos_1000_geracoes.csv"

ARQUIVO_SAIDA_TXT = PASTA_TXT / "saida_algoritmo_genetico.txt"
ARQUIVO_ITENS_ESCOLHIDOS = PASTA_TXT / "itens_escolhidos_ag.txt"


# =========================
# VARIÁVEIS DE CONTROLE
# =========================

analise_geracoes = []

tempo_inicio_execucao = None
tempo_ultima_geracao = None

melhor_global_solucao = None
melhor_global_valor = -1
melhor_global_peso = 0
melhor_global_qtd_itens = 0
melhor_global_geracao = 0


# =========================
# FUNÇÃO PARA PRINTAR E SALVAR
# =========================

def registrar_saida(texto=""):
    print(texto)

    with open(ARQUIVO_SAIDA_TXT, "a", encoding="utf-8") as arquivo:
        arquivo.write(str(texto) + "\n")


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

        valor = int(partes[1])
        peso = int(partes[2])

        valores.append(valor)
        pesos.append(peso)

    capacidade = int(linhas[quantidade_itens + 1])

    return np.array(valores), np.array(pesos), capacidade


# =========================
# CÁLCULOS DA SOLUÇÃO
# =========================

def calcular_valor_total(solution, valores):
    return int(np.sum(solution * valores))


def calcular_peso_total(solution, pesos):
    return int(np.sum(solution * pesos))


def calcular_quantidade_itens(solution):
    return int(np.sum(solution == 1))


def obter_itens_escolhidos(solution):
    itens = []

    for i in range(len(solution)):
        if solution[i] == 1:
            itens.append(i + 1)

    return itens


# =========================
# REPARO DA SOLUÇÃO
# =========================

def reparar_solucao(solution, valores, pesos, capacidade):
    solution = np.array(solution).copy().astype(int)

    peso_total = np.sum(solution * pesos)

    if peso_total <= capacidade:
        return solution

    itens_escolhidos = np.where(solution == 1)[0]

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
# FUNÇÃO DE AVALIAÇÃO
# =========================

def fitness_func(ga_instance, solution, solution_idx):
    solution_corrigida = reparar_solucao(
        solution,
        valores,
        pesos,
        capacidade
    )

    peso_total = calcular_peso_total(solution_corrigida, pesos)
    valor_total = calcular_valor_total(solution_corrigida, valores)

    if peso_total > capacidade:
        return 0

    return valor_total


# =========================
# ANÁLISE DE CADA GERAÇÃO
# =========================

def registrar_geracao(ga_instance):
    global tempo_ultima_geracao
    global melhor_global_solucao
    global melhor_global_valor
    global melhor_global_peso
    global melhor_global_qtd_itens
    global melhor_global_geracao

    agora = time.perf_counter()

    geracao_atual = ga_instance.generations_completed

    tempo_geracao = agora - tempo_ultima_geracao
    tempo_acumulado = agora - tempo_inicio_execucao
    tempo_ultima_geracao = agora

    fitness_populacao = np.array(ga_instance.last_generation_fitness, dtype=float)

    fitness_medio = float(np.mean(fitness_populacao))
    fitness_pior = float(np.min(fitness_populacao))

    indice_melhor = int(np.argmax(fitness_populacao))

    melhor_solucao_geracao = ga_instance.population[indice_melhor]

    melhor_solucao_geracao = reparar_solucao(
        melhor_solucao_geracao,
        valores,
        pesos,
        capacidade
    )

    valor_acumulado_geracao = calcular_valor_total(melhor_solucao_geracao, valores)
    peso_acumulado_geracao = calcular_peso_total(melhor_solucao_geracao, pesos)
    qtd_itens_geracao = calcular_quantidade_itens(melhor_solucao_geracao)

    houve_melhoria_global = False

    if valor_acumulado_geracao > melhor_global_valor:
        melhor_global_solucao = melhor_solucao_geracao.copy()
        melhor_global_valor = valor_acumulado_geracao
        melhor_global_peso = peso_acumulado_geracao
        melhor_global_qtd_itens = qtd_itens_geracao
        melhor_global_geracao = geracao_atual
        houve_melhoria_global = True

    linha = {
        "geracao": geracao_atual,
        "tempo_geracao_s": round(tempo_geracao, 6),
        "tempo_acumulado_s": round(tempo_acumulado, 6),

        "valor_acumulado_melhor_geracao": valor_acumulado_geracao,
        "peso_acumulado_melhor_geracao": peso_acumulado_geracao,
        "quantidade_itens_melhor_geracao": qtd_itens_geracao,

        "fitness_medio_populacao": round(fitness_medio, 4),
        "fitness_pior_populacao": round(fitness_pior, 4),

        "melhor_valor_global_ate_agora": melhor_global_valor,
        "peso_melhor_global_ate_agora": melhor_global_peso,
        "quantidade_itens_melhor_global_ate_agora": melhor_global_qtd_itens,
        "geracao_melhor_global": melhor_global_geracao,

        "houve_melhoria_global": houve_melhoria_global
    }

    analise_geracoes.append(linha)

    if geracao_atual % INTERVALO_EXIBICAO == 0:
        registrar_saida(
            f"Geração {geracao_atual} "
            f"| Tempo da geração: {tempo_geracao:.4f}s "
            f"| Valor acumulado: {valor_acumulado_geracao} "
            f"| Peso acumulado: {peso_acumulado_geracao}/{capacidade} "
            f"| Itens selecionados: {qtd_itens_geracao} "
            f"| Melhor global: {melhor_global_valor}"
        )


# =========================
# SALVAR CSV
# =========================

def salvar_csv(caminho, dados):
    if not dados:
        registrar_saida(f"Nenhum dado para salvar em {caminho}.")
        return

    colunas = list(dados[0].keys())

    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas, delimiter=";")
        escritor.writeheader()
        escritor.writerows(dados)

    registrar_saida(f"CSV salvo em: {caminho}")


def filtrar_por_escala(dados, escala):
    filtrados = []

    for linha in dados:
        geracao = linha["geracao"]

        if geracao == 1 or geracao % escala == 0 or geracao == NUM_GENERATIONS:
            filtrados.append(linha)

    return filtrados


def gerar_resumo_por_blocos(dados, tamanho_bloco):
    blocos = []

    for inicio in range(1, NUM_GENERATIONS + 1, tamanho_bloco):
        fim = min(inicio + tamanho_bloco - 1, NUM_GENERATIONS)

        linhas_bloco = [
            linha for linha in dados
            if inicio <= linha["geracao"] <= fim
        ]

        if not linhas_bloco:
            continue

        melhor_linha_bloco = max(
            linhas_bloco,
            key=lambda linha: linha["valor_acumulado_melhor_geracao"]
        )

        ultima_linha_bloco = linhas_bloco[-1]

        tempo_total_bloco = sum(
            linha["tempo_geracao_s"]
            for linha in linhas_bloco
        )

        tempo_medio_bloco = np.mean([
            linha["tempo_geracao_s"]
            for linha in linhas_bloco
        ])

        media_valor_melhor_geracao = np.mean([
            linha["valor_acumulado_melhor_geracao"]
            for linha in linhas_bloco
        ])

        media_fitness_populacao = np.mean([
            linha["fitness_medio_populacao"]
            for linha in linhas_bloco
        ])

        media_itens = np.mean([
            linha["quantidade_itens_melhor_geracao"]
            for linha in linhas_bloco
        ])

        media_peso = np.mean([
            linha["peso_acumulado_melhor_geracao"]
            for linha in linhas_bloco
        ])

        quantidade_melhorias = sum(
            1 for linha in linhas_bloco
            if linha["houve_melhoria_global"]
        )

        blocos.append({
            "bloco_inicio": inicio,
            "bloco_fim": fim,
            "geracao_representativa": fim,

            "melhor_valor_no_bloco": melhor_linha_bloco["valor_acumulado_melhor_geracao"],
            "geracao_do_melhor_valor_no_bloco": melhor_linha_bloco["geracao"],
            "peso_do_melhor_valor_no_bloco": melhor_linha_bloco["peso_acumulado_melhor_geracao"],
            "itens_do_melhor_valor_no_bloco": melhor_linha_bloco["quantidade_itens_melhor_geracao"],

            "melhor_global_ao_final_do_bloco": ultima_linha_bloco["melhor_valor_global_ate_agora"],
            "peso_melhor_global_ao_final_do_bloco": ultima_linha_bloco["peso_melhor_global_ate_agora"],
            "itens_melhor_global_ao_final_do_bloco": ultima_linha_bloco["quantidade_itens_melhor_global_ate_agora"],

            "media_valor_melhor_geracao_no_bloco": round(float(media_valor_melhor_geracao), 4),
            "media_fitness_populacao_no_bloco": round(float(media_fitness_populacao), 4),
            "media_itens_melhor_geracao_no_bloco": round(float(media_itens), 4),
            "media_peso_melhor_geracao_no_bloco": round(float(media_peso), 4),

            "tempo_total_bloco_s": round(float(tempo_total_bloco), 6),
            "tempo_medio_geracao_bloco_s": round(float(tempo_medio_bloco), 6),

            "quantidade_melhorias_globais_no_bloco": quantidade_melhorias
        })

    return blocos


# =========================
# SALVAR ITENS ESCOLHIDOS
# =========================

def salvar_itens_escolhidos():
    if melhor_global_solucao is None:
        registrar_saida("Nenhuma melhor solução global foi encontrada.")
        return

    itens_escolhidos = obter_itens_escolhidos(melhor_global_solucao)

    with open(ARQUIVO_ITENS_ESCOLHIDOS, "w", encoding="utf-8") as arquivo:
        arquivo.write("===== ITENS ESCOLHIDOS NA MELHOR SOLUÇÃO GLOBAL =====\n\n")

        arquivo.write(f"Quantidade de itens disponíveis: {quantidade_itens}\n")
        arquivo.write(f"Quantidade de itens escolhidos: {len(itens_escolhidos)}\n")
        arquivo.write(f"Valor acumulado dos itens escolhidos: {melhor_global_valor}\n")
        arquivo.write(f"Peso acumulado dos itens escolhidos: {melhor_global_peso}/{capacidade}\n")
        arquivo.write(f"Geração em que a melhor solução apareceu: {melhor_global_geracao}\n\n")

        arquivo.write("Itens escolhidos:\n")
        arquivo.write(str(itens_escolhidos))
        arquivo.write("\n\n")

        arquivo.write("Detalhamento dos itens escolhidos:\n")
        arquivo.write("id_item;valor;peso\n")

        for item_id in itens_escolhidos:
            indice = item_id - 1
            arquivo.write(f"{item_id};{valores[indice]};{pesos[indice]}\n")

    registrar_saida(f"Arquivo com itens escolhidos salvo em: {ARQUIVO_ITENS_ESCOLHIDOS}")


# =========================
# GRÁFICOS
# =========================

def preparar_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        registrar_saida("")
        registrar_saida("Matplotlib não está instalado. Os gráficos não foram gerados.")
        registrar_saida("Para gerar gráficos, instale com: pip install matplotlib")
        return None


def gerar_graficos_linhas(dados, pasta, sufixo_titulo, sufixo_arquivo):
    plt = preparar_matplotlib()

    if plt is None:
        return

    if not dados:
        return

    geracoes = [linha["geracao"] for linha in dados]

    valor_melhor_geracao = [
        linha["valor_acumulado_melhor_geracao"]
        for linha in dados
    ]

    melhor_global = [
        linha["melhor_valor_global_ate_agora"]
        for linha in dados
    ]

    tempo_geracao = [
        linha["tempo_geracao_s"]
        for linha in dados
    ]

    qtd_itens = [
        linha["quantidade_itens_melhor_geracao"]
        for linha in dados
    ]

    fitness_medio = [
        linha["fitness_medio_populacao"]
        for linha in dados
    ]

    peso_melhor_geracao = [
        linha["peso_acumulado_melhor_geracao"]
        for linha in dados
    ]

    # 1. Valor acumulado
    plt.figure(figsize=(12, 6))
    plt.plot(geracoes, valor_melhor_geracao, label="Melhor valor da geração")
    plt.plot(geracoes, melhor_global, label="Melhor valor global até agora")
    plt.xlabel("Geração")
    plt.ylabel("Valor acumulado")
    plt.title(f"Evolução do valor acumulado {sufixo_titulo}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        pasta / f"grafico_valor_acumulado_{sufixo_arquivo}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 2. Tempo por geração
    plt.figure(figsize=(12, 6))
    plt.plot(geracoes, tempo_geracao)
    plt.xlabel("Geração")
    plt.ylabel("Tempo da geração (s)")
    plt.title(f"Tempo de execução por geração {sufixo_titulo}")
    plt.grid(True)
    plt.savefig(
        pasta / f"grafico_tempo_por_geracao_{sufixo_arquivo}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 3. Quantidade de itens
    plt.figure(figsize=(12, 6))
    plt.plot(geracoes, qtd_itens)
    plt.xlabel("Geração")
    plt.ylabel("Quantidade de itens")
    plt.title(f"Itens selecionados pela melhor solução {sufixo_titulo}")
    plt.grid(True)
    plt.savefig(
        pasta / f"grafico_itens_por_geracao_{sufixo_arquivo}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 4. Peso acumulado
    plt.figure(figsize=(12, 6))
    plt.plot(geracoes, peso_melhor_geracao, label="Peso acumulado")
    plt.axhline(y=capacidade, linestyle="--", label="Capacidade da mochila")
    plt.xlabel("Geração")
    plt.ylabel("Peso acumulado")
    plt.title(f"Peso acumulado da melhor solução {sufixo_titulo}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        pasta / f"grafico_peso_acumulado_{sufixo_arquivo}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 5. Melhor global vs média da população
    plt.figure(figsize=(12, 6))
    plt.plot(geracoes, melhor_global, label="Melhor valor global")
    plt.plot(geracoes, fitness_medio, label="Fitness médio da população")
    plt.xlabel("Geração")
    plt.ylabel("Valor")
    plt.title(f"Melhor global vs. média da população {sufixo_titulo}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        pasta / f"grafico_melhor_global_vs_media_{sufixo_arquivo}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def gerar_graficos_blocos(blocos, pasta, tamanho_bloco):
    plt = preparar_matplotlib()

    if plt is None:
        return

    if not blocos:
        return

    geracoes = [
        linha["geracao_representativa"]
        for linha in blocos
    ]

    melhor_valor_bloco = [
        linha["melhor_valor_no_bloco"]
        for linha in blocos
    ]

    melhor_global_final_bloco = [
        linha["melhor_global_ao_final_do_bloco"]
        for linha in blocos
    ]

    media_fitness_bloco = [
        linha["media_fitness_populacao_no_bloco"]
        for linha in blocos
    ]

    tempo_total_bloco = [
        linha["tempo_total_bloco_s"]
        for linha in blocos
    ]

    tempo_medio_bloco = [
        linha["tempo_medio_geracao_bloco_s"]
        for linha in blocos
    ]

    melhorias_bloco = [
        linha["quantidade_melhorias_globais_no_bloco"]
        for linha in blocos
    ]

    media_itens_bloco = [
        linha["media_itens_melhor_geracao_no_bloco"]
        for linha in blocos
    ]

    media_peso_bloco = [
        linha["media_peso_melhor_geracao_no_bloco"]
        for linha in blocos
    ]

    # 1. Melhor valor por bloco
    plt.figure(figsize=(12, 6))
    plt.plot(geracoes, melhor_valor_bloco, label=f"Melhor valor no bloco de {tamanho_bloco}")
    plt.plot(geracoes, melhor_global_final_bloco, label="Melhor global ao final do bloco")
    plt.xlabel("Geração final do bloco")
    plt.ylabel("Valor")
    plt.title(f"Evolução por blocos de {tamanho_bloco} gerações")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        pasta / f"grafico_valor_por_blocos_{tamanho_bloco}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 2. Tempo total por bloco
    plt.figure(figsize=(12, 6))
    plt.plot(geracoes, tempo_total_bloco)
    plt.xlabel("Geração final do bloco")
    plt.ylabel("Tempo total do bloco (s)")
    plt.title(f"Tempo total por bloco de {tamanho_bloco} gerações")
    plt.grid(True)
    plt.savefig(
        pasta / f"grafico_tempo_total_blocos_{tamanho_bloco}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 3. Tempo médio por geração em cada bloco
    plt.figure(figsize=(12, 6))
    plt.plot(geracoes, tempo_medio_bloco)
    plt.xlabel("Geração final do bloco")
    plt.ylabel("Tempo médio por geração (s)")
    plt.title(f"Tempo médio por geração em blocos de {tamanho_bloco}")
    plt.grid(True)
    plt.savefig(
        pasta / f"grafico_tempo_medio_blocos_{tamanho_bloco}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 4. Melhorias globais por bloco
    plt.figure(figsize=(12, 6))
    plt.bar(geracoes, melhorias_bloco)
    plt.xlabel("Geração final do bloco")
    plt.ylabel("Quantidade de melhorias globais")
    plt.title(f"Quantidade de melhorias globais por bloco de {tamanho_bloco} gerações")
    plt.grid(True)
    plt.savefig(
        pasta / f"grafico_melhorias_blocos_{tamanho_bloco}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 5. Média de itens por bloco
    plt.figure(figsize=(12, 6))
    plt.plot(geracoes, media_itens_bloco)
    plt.xlabel("Geração final do bloco")
    plt.ylabel("Média de itens selecionados")
    plt.title(f"Média de itens selecionados por bloco de {tamanho_bloco} gerações")
    plt.grid(True)
    plt.savefig(
        pasta / f"grafico_media_itens_blocos_{tamanho_bloco}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 6. Média de peso por bloco
    plt.figure(figsize=(12, 6))
    plt.plot(geracoes, media_peso_bloco, label="Peso médio")
    plt.axhline(y=capacidade, linestyle="--", label="Capacidade da mochila")
    plt.xlabel("Geração final do bloco")
    plt.ylabel("Peso médio")
    plt.title(f"Peso médio da melhor solução por bloco de {tamanho_bloco} gerações")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        pasta / f"grafico_media_peso_blocos_{tamanho_bloco}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 7. Melhor global vs fitness médio por bloco
    plt.figure(figsize=(12, 6))
    plt.plot(geracoes, melhor_global_final_bloco, label="Melhor global ao final do bloco")
    plt.plot(geracoes, media_fitness_bloco, label="Fitness médio da população no bloco")
    plt.xlabel("Geração final do bloco")
    plt.ylabel("Valor")
    plt.title(f"Melhor global vs. fitness médio por blocos de {tamanho_bloco}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        pasta / f"grafico_global_vs_media_blocos_{tamanho_bloco}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def gerar_todos_os_graficos():
    if not analise_geracoes:
        registrar_saida("Nenhum dado registrado para gerar gráficos.")
        return

    dados_escala_100 = filtrar_por_escala(analise_geracoes, 100)
    dados_escala_1000 = filtrar_por_escala(analise_geracoes, 1000)

    resumo_blocos_100 = gerar_resumo_por_blocos(analise_geracoes, 100)
    resumo_blocos_1000 = gerar_resumo_por_blocos(analise_geracoes, 1000)

    salvar_csv(ARQUIVO_ANALISE_ESCALA_100, dados_escala_100)
    salvar_csv(ARQUIVO_ANALISE_ESCALA_1000, dados_escala_1000)
    salvar_csv(ARQUIVO_RESUMO_BLOCOS_100, resumo_blocos_100)
    salvar_csv(ARQUIVO_RESUMO_BLOCOS_1000, resumo_blocos_1000)

    gerar_graficos_linhas(
        dados=analise_geracoes,
        pasta=PASTA_GRAFICOS_COMPLETOS,
        sufixo_titulo="-- todas as gerações",
        sufixo_arquivo="completo"
    )

    gerar_graficos_linhas(
        dados=dados_escala_100,
        pasta=PASTA_GRAFICOS_ESCALA_100,
        sufixo_titulo="-- visualização de 100 em 100 gerações",
        sufixo_arquivo="escala_100"
    )

    gerar_graficos_linhas(
        dados=dados_escala_1000,
        pasta=PASTA_GRAFICOS_ESCALA_1000,
        sufixo_titulo="-- visualização de 1000 em 1000 gerações",
        sufixo_arquivo="escala_1000"
    )

    gerar_graficos_blocos(
        blocos=resumo_blocos_100,
        pasta=PASTA_GRAFICOS_BLOCOS,
        tamanho_bloco=100
    )

    gerar_graficos_blocos(
        blocos=resumo_blocos_1000,
        pasta=PASTA_GRAFICOS_BLOCOS,
        tamanho_bloco=1000
    )

    registrar_saida("")
    registrar_saida("===== GRÁFICOS GERADOS =====")
    registrar_saida(f"Gráficos completos: {PASTA_GRAFICOS_COMPLETOS}")
    registrar_saida(f"Gráficos em escala de 100 em 100: {PASTA_GRAFICOS_ESCALA_100}")
    registrar_saida(f"Gráficos em escala de 1000 em 1000: {PASTA_GRAFICOS_ESCALA_1000}")
    registrar_saida(f"Gráficos de resumo por blocos: {PASTA_GRAFICOS_BLOCOS}")


# =========================
# RESUMO FINAL
# =========================

def salvar_resumo_execucao(inicio_data_hora, fim_data_hora, tempo_total):
    tempos = [linha["tempo_geracao_s"] for linha in analise_geracoes]

    tempo_medio_geracao = np.mean(tempos)
    tempo_menor_geracao = np.min(tempos)
    tempo_maior_geracao = np.max(tempos)

    total_melhorias_globais = sum(
        1 for linha in analise_geracoes
        if linha["houve_melhoria_global"]
    )

    registrar_saida("")
    registrar_saida("===== RESUMO FINAL DA EXECUÇÃO =====")
    registrar_saida("Arquivo de entrada: " + NOME_ARQUIVO)
    registrar_saida(f"Quantidade de itens disponíveis: {quantidade_itens}")
    registrar_saida(f"Capacidade da mochila: {capacidade}")

    registrar_saida("")
    registrar_saida("===== MELHOR SOLUÇÃO GLOBAL ENCONTRADA =====")
    registrar_saida(f"Valor acumulado dos itens escolhidos: {melhor_global_valor}")
    registrar_saida(f"Peso acumulado dos itens escolhidos: {melhor_global_peso}/{capacidade}")
    registrar_saida(f"Quantidade de itens escolhidos: {melhor_global_qtd_itens}")
    registrar_saida(f"Geração em que apareceu: {melhor_global_geracao}")
    registrar_saida(f"Quantidade total de melhorias globais: {total_melhorias_globais}")

    if melhor_global_peso <= capacidade:
        registrar_saida("Solução válida: o peso está dentro da capacidade.")
    else:
        registrar_saida("Solução inválida: passou da capacidade.")

    registrar_saida("")
    registrar_saida("===== TEMPO DE EXECUÇÃO =====")
    registrar_saida("Começou em: " + inicio_data_hora.strftime("%d/%m/%Y %H:%M:%S"))
    registrar_saida("Terminou em: " + fim_data_hora.strftime("%d/%m/%Y %H:%M:%S"))
    registrar_saida(f"Tempo total: {tempo_total:.2f} segundos")
    registrar_saida(f"Tempo médio por geração: {tempo_medio_geracao:.6f} segundos")
    registrar_saida(f"Menor tempo de geração: {tempo_menor_geracao:.6f} segundos")
    registrar_saida(f"Maior tempo de geração: {tempo_maior_geracao:.6f} segundos")

    registrar_saida("")
    registrar_saida("===== CONFIGURAÇÕES UTILIZADAS =====")
    registrar_saida(f"Número de gerações: {NUM_GENERATIONS}")
    registrar_saida(f"Tamanho da população: {SOL_PER_POP}")
    registrar_saida(f"Quantidade de pais para cruzamento: {NUM_PARENTS_MATING}")
    registrar_saida(f"Taxa de mutação (% genes): {MUTATION_PERCENT_GENES}")
    registrar_saida(f"Tipo de seleção: {PARENT_SELECTION_TYPE}")
    registrar_saida(f"Tipo de cruzamento: {CROSSOVER_TYPE}")
    registrar_saida(f"Tipo de mutação: {MUTATION_TYPE}")
    registrar_saida(f"Semente aleatória: {RANDOM_SEED}")


# =========================
# PROGRAMA PRINCIPAL
# =========================

with open(ARQUIVO_SAIDA_TXT, "w", encoding="utf-8") as arquivo:
    arquivo.write("")

valores, pesos, capacidade = ler_instancia(NOME_ARQUIVO)

quantidade_itens = len(valores)

populacao_inicial = criar_populacao_inicial(
    SOL_PER_POP,
    quantidade_itens,
    valores,
    pesos,
    capacidade
)

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
    on_generation=registrar_geracao,
    random_seed=RANDOM_SEED
)


# =========================
# MEDIÇÃO DO TEMPO
# =========================

inicio_data_hora = datetime.now()
tempo_inicio_execucao = time.perf_counter()
tempo_ultima_geracao = tempo_inicio_execucao

registrar_saida("===== INÍCIO DA EXECUÇÃO =====")
registrar_saida("Começou em: " + inicio_data_hora.strftime("%d/%m/%Y %H:%M:%S"))
registrar_saida(f"Arquivo de entrada: {NOME_ARQUIVO}")
registrar_saida(f"Quantidade de itens disponíveis: {quantidade_itens}")
registrar_saida(f"Capacidade da mochila: {capacidade}")
registrar_saida(f"Pasta de saída: {PASTA_SAIDA}")
registrar_saida("")


# =========================
# EXECUTA O ALGORITMO
# =========================

ga_instance.run()


fim_data_hora = datetime.now()
fim_tempo = time.perf_counter()

tempo_total = fim_tempo - tempo_inicio_execucao


# =========================
# SALVA OS RESULTADOS
# =========================

salvar_csv(ARQUIVO_ANALISE_GERACOES, analise_geracoes)
salvar_itens_escolhidos()
gerar_todos_os_graficos()
salvar_resumo_execucao(inicio_data_hora, fim_data_hora, tempo_total)


registrar_saida("")
registrar_saida("===== ARQUIVOS E PASTAS GERADOS =====")
registrar_saida(f"- Saída principal: {ARQUIVO_SAIDA_TXT}")
registrar_saida(f"- Itens escolhidos: {ARQUIVO_ITENS_ESCOLHIDOS}")
registrar_saida(f"- CSV completo: {ARQUIVO_ANALISE_GERACOES}")
registrar_saida(f"- CSV escala 100: {ARQUIVO_ANALISE_ESCALA_100}")
registrar_saida(f"- CSV escala 1000: {ARQUIVO_ANALISE_ESCALA_1000}")
registrar_saida(f"- CSV resumo blocos 100: {ARQUIVO_RESUMO_BLOCOS_100}")
registrar_saida(f"- CSV resumo blocos 1000: {ARQUIVO_RESUMO_BLOCOS_1000}")
registrar_saida(f"- Gráficos completos: {PASTA_GRAFICOS_COMPLETOS}")
registrar_saida(f"- Gráficos escala 100: {PASTA_GRAFICOS_ESCALA_100}")
registrar_saida(f"- Gráficos escala 1000: {PASTA_GRAFICOS_ESCALA_1000}")
registrar_saida(f"- Gráficos resumo por blocos: {PASTA_GRAFICOS_BLOCOS}")