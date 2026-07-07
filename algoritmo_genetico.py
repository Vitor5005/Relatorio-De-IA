import pygad
import numpy as np
from datetime import datetime
import time
import csv
from pathlib import Path


# ==========================================================
# CONFIGURAÇÕES DO ALGORITMO
# ==========================================================

NOME_ARQUIVO = "test_3.in"

NUM_GENERATIONS = 1000
SOL_PER_POP = 100
NUM_PARENTS_MATING = 20
MUTATION_PERCENT_GENES = 2

PARENT_SELECTION_TYPE = "sss"
CROSSOVER_TYPE = "single_point"
MUTATION_TYPE = "random"

# Mostra resultado no terminal de 100 em 100 gerações
INTERVALO_EXIBICAO = 100

RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)


# ==========================================================
# CONFIGURAÇÕES DE VISUALIZAÇÃO
# ==========================================================

ESCALA_100 = 100
ESCALA_1000 = 1000

MINIMO_PONTOS_GRAFICO = 3

# Para 1000 gerações, gráficos de 1000 em 1000 geram poucos pontos.
# Por isso, só serão gerados se houver pelo menos 3000 gerações.
GERAR_ESCALA_1000 = NUM_GENERATIONS >= 3000
GERAR_BLOCOS_1000 = NUM_GENERATIONS >= 3000


# ==========================================================
# PASTAS DE SAÍDA
# ==========================================================

PASTA_SAIDA = Path("resultados_algoritmo_genetico")
PASTA_CSV = PASTA_SAIDA / "csv"
PASTA_TXT = PASTA_SAIDA / "txt"
PASTA_GRAFICOS = PASTA_SAIDA / "graficos_corrigidos"

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
    pasta.mkdir(parents=True, exist_ok=True)


# ==========================================================
# ARQUIVOS GERADOS
# ==========================================================

ARQUIVO_ANALISE_GERACOES = PASTA_CSV / "analise_geracoes_ag.csv"
ARQUIVO_ANALISE_ESCALA_100 = PASTA_CSV / "analise_geracoes_ag_escala_100.csv"
ARQUIVO_ANALISE_ESCALA_1000 = PASTA_CSV / "analise_geracoes_ag_escala_1000.csv"

ARQUIVO_RESUMO_BLOCOS_100 = PASTA_CSV / "resumo_blocos_100_geracoes.csv"
ARQUIVO_RESUMO_BLOCOS_1000 = PASTA_CSV / "resumo_blocos_1000_geracoes.csv"

ARQUIVO_SAIDA_TXT = PASTA_TXT / "saida_algoritmo_genetico.txt"
ARQUIVO_ITENS_ESCOLHIDOS = PASTA_TXT / "itens_escolhidos_ag.txt"

# Arquivo específico para consultar o tempo de cada geração.
ARQUIVO_TEMPOS_GERACOES_CSV = PASTA_CSV / "tempos_por_geracao_ag.csv"
ARQUIVO_TEMPOS_GERACOES_TXT = PASTA_TXT / "tempos_por_geracao_ag.txt"


# ==========================================================
# VARIÁVEIS DE CONTROLE
# ==========================================================

analise_geracoes = []

# Medições de tempo
tempo_inicio_run = None
tempo_fim_callback_anterior = None
tempo_total_callback = 0.0

# Melhor solução global
melhor_global_solucao = None
melhor_global_valor = -1
melhor_global_peso = 0
melhor_global_qtd_itens = 0
melhor_global_geracao = 0


# ==========================================================
# FUNÇÃO PARA PRINTAR E SALVAR
# ==========================================================

def registrar_saida(texto=""):
    print(texto)

    with open(ARQUIVO_SAIDA_TXT, "a", encoding="utf-8") as arquivo:
        arquivo.write(str(texto) + "\n")


# ==========================================================
# LEITURA DA INSTÂNCIA
# ==========================================================

def ler_instancia(nome_arquivo):
    with open(nome_arquivo, "r") as arquivo:
        linhas = [linha.strip() for linha in arquivo.readlines() if linha.strip()]

    quantidade_itens = int(linhas[0])

    valores = []
    pesos = []

    for i in range(1, quantidade_itens + 1):
        partes = linhas[i].split()

        # Formato da linha: id valor peso
        valor = int(partes[1])
        peso = int(partes[2])

        valores.append(valor)
        pesos.append(peso)

    capacidade = int(linhas[quantidade_itens + 1])

    return np.array(valores), np.array(pesos), capacidade


# ==========================================================
# CÁLCULOS DA SOLUÇÃO
# ==========================================================

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


# ==========================================================
# REPARO DA SOLUÇÃO
# ==========================================================

def reparar_solucao(solution, valores, pesos, capacidade):
    solution = np.array(solution).copy().astype(int)

    peso_total = np.sum(solution * pesos)

    if peso_total <= capacidade:
        return solution

    itens_escolhidos = np.where(solution == 1)[0]

    # Remove primeiro os itens com pior relação valor/peso.
    ordem_remocao = itens_escolhidos[
        np.argsort(valores[itens_escolhidos] / pesos[itens_escolhidos])
    ]

    for item in ordem_remocao:
        if peso_total <= capacidade:
            break

        solution[item] = 0
        peso_total -= pesos[item]

    return solution


# ==========================================================
# POPULAÇÃO INICIAL
# ==========================================================

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


# ==========================================================
# FUNÇÃO DE APTIDÃO
# ==========================================================

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


# ==========================================================
# REGISTRO DE CADA GERAÇÃO COM TEMPO MAIS PRECISO
# ==========================================================

def registrar_geracao(ga_instance):
    global tempo_fim_callback_anterior
    global tempo_total_callback

    global melhor_global_solucao
    global melhor_global_valor
    global melhor_global_peso
    global melhor_global_qtd_itens
    global melhor_global_geracao

    # Momento em que o PyGAD terminou a geração e entrou no callback.
    inicio_callback = time.perf_counter()

    geracao_atual = ga_instance.generations_completed

    # Tempo da geração:
    # mede o intervalo entre o fim do callback anterior e o início deste callback.
    # Assim, o tempo gasto imprimindo, salvando e analisando dentro do callback
    # não entra no tempo da próxima geração.
    if tempo_fim_callback_anterior is None:
        tempo_geracao = 0.0
    else:
        tempo_geracao = inicio_callback - tempo_fim_callback_anterior

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

    # Saída limpa de 100 em 100 gerações, mantendo o tempo da geração.
    if geracao_atual % INTERVALO_EXIBICAO == 0:
        registrar_saida(
            f"Geração {geracao_atual:04d}/{NUM_GENERATIONS} | "
            f"Tempo geração: {tempo_geracao:.6f}s | "
            f"Melhor da geração: valor={valor_acumulado_geracao}, "
            f"peso={peso_acumulado_geracao}/{capacidade}, "
            f"itens={qtd_itens_geracao} | "
            f"Melhor global: valor={melhor_global_valor}, "
            f"geração={melhor_global_geracao}"
        )

    # Mede quanto tempo o próprio callback gastou:
    # inclui cálculo das métricas, print e escrita no arquivo.
    fim_callback = time.perf_counter()
    tempo_callback = fim_callback - inicio_callback
    tempo_total_callback += tempo_callback

    linha = {
        "geracao": geracao_atual,

        # Tempo mais limpo da geração, sem o tempo do callback anterior.
        "tempo_geracao_s": round(tempo_geracao, 6),

        # Tempo gasto dentro do callback desta geração.
        "tempo_callback_s": round(tempo_callback, 6),

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

    # A próxima geração começa a contar depois que todo o callback terminou.
    tempo_fim_callback_anterior = fim_callback


# ==========================================================
# CSV
# ==========================================================

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


def salvar_tempos_por_geracao():
    """
    Salva o tempo de TODAS as gerações em arquivos separados.

    O terminal continua limpo, mostrando apenas de 100 em 100 gerações,
    mas estes arquivos guardam o tempo individual de cada geração.
    """

    if not analise_geracoes:
        registrar_saida("Nenhum tempo de geração para salvar.")
        return

    dados_tempos = []

    for linha in analise_geracoes:
        dados_tempos.append({
            "geracao": linha["geracao"],
            "tempo_geracao_s": linha["tempo_geracao_s"],
            "tempo_callback_s": linha["tempo_callback_s"]
        })

    # CSV para análise no Excel/LibreOffice.
    with open(ARQUIVO_TEMPOS_GERACOES_CSV, "w", newline="", encoding="utf-8") as arquivo:
        colunas = ["geracao", "tempo_geracao_s", "tempo_callback_s"]
        escritor = csv.DictWriter(arquivo, fieldnames=colunas, delimiter=";")
        escritor.writeheader()
        escritor.writerows(dados_tempos)

    # TXT para leitura rápida.
    with open(ARQUIVO_TEMPOS_GERACOES_TXT, "w", encoding="utf-8") as arquivo:
        arquivo.write("===== TEMPO DE CADA GERAÇÃO =====\n\n")
        arquivo.write("geracao;tempo_geracao_s;tempo_callback_s\n")

        for linha in dados_tempos:
            arquivo.write(
                f"{linha['geracao']};"
                f"{linha['tempo_geracao_s']};"
                f"{linha['tempo_callback_s']}\n"
            )

    registrar_saida(f"CSV com tempo de cada geração salvo em: {ARQUIVO_TEMPOS_GERACOES_CSV}")
    registrar_saida(f"TXT com tempo de cada geração salvo em: {ARQUIVO_TEMPOS_GERACOES_TXT}")


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

        tempo_callback_bloco = sum(
            linha["tempo_callback_s"]
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
            "tempo_callback_bloco_s": round(float(tempo_callback_bloco), 6),
            "tempo_medio_geracao_bloco_s": round(float(tempo_medio_bloco), 6),

            "quantidade_melhorias_globais_no_bloco": quantidade_melhorias
        })

    return blocos


# ==========================================================
# SALVAR ITENS ESCOLHIDOS
# ==========================================================

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


# ==========================================================
# FUNÇÕES AUXILIARES DOS GRÁFICOS
# ==========================================================

def preparar_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        registrar_saida("")
        registrar_saida("Matplotlib não está instalado. Os gráficos não foram gerados.")
        registrar_saida("Para gerar gráficos, instale com: pip install matplotlib")
        return None


def series_iguais(serie_a, serie_b):
    if len(serie_a) != len(serie_b):
        return False

    return all(a == b for a, b in zip(serie_a, serie_b))


def salvar_grafico(plt, caminho):
    plt.tight_layout()
    plt.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close()


def ajustar_eixo_y(plt, valores, margem_percentual=0.05):
    valores = [v for v in valores if v is not None]

    if not valores:
        return

    minimo = min(valores)
    maximo = max(valores)

    if minimo == maximo:
        margem = max(1, abs(maximo) * margem_percentual)
        plt.ylim(minimo - margem, maximo + margem)
    else:
        margem = (maximo - minimo) * margem_percentual
        plt.ylim(minimo - margem, maximo + margem)


def tem_pontos_suficientes(dados, nome):
    if len(dados) < MINIMO_PONTOS_GRAFICO:
        registrar_saida(
            f"Gráfico '{nome}' não gerado: poucos pontos ({len(dados)})."
        )
        return False

    return True


# ==========================================================
# GRÁFICOS POR GERAÇÃO
# ==========================================================

def gerar_graficos_linhas(dados, pasta, sufixo_titulo, sufixo_arquivo, usar_marcador=False):
    plt = preparar_matplotlib()

    if plt is None:
        return

    if not tem_pontos_suficientes(dados, sufixo_arquivo):
        return

    geracoes = [linha["geracao"] for linha in dados]

    melhor_geracao = [
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

    itens_geracao = [
        linha["quantidade_itens_melhor_geracao"]
        for linha in dados
    ]

    itens_global = [
        linha["quantidade_itens_melhor_global_ate_agora"]
        for linha in dados
    ]

    fitness_medio = [
        linha["fitness_medio_populacao"]
        for linha in dados
    ]

    peso_geracao = [
        linha["peso_acumulado_melhor_geracao"]
        for linha in dados
    ]

    peso_global = [
        linha["peso_melhor_global_ate_agora"]
        for linha in dados
    ]

    marcador = "o" if usar_marcador else None
    tamanho_marcador = 4 if usar_marcador else 0

    # 1. Valor acumulado
    plt.figure(figsize=(12, 6))

    if series_iguais(melhor_geracao, melhor_global):
        plt.step(
            geracoes,
            melhor_global,
            where="post",
            label="Melhor valor global até agora",
            marker=marcador,
            markersize=tamanho_marcador
        )
        titulo_extra = " -- séries sobrepostas"
    else:
        plt.plot(
            geracoes,
            melhor_geracao,
            label="Melhor valor da geração",
            marker=marcador,
            markersize=tamanho_marcador,
            alpha=0.65
        )

        plt.step(
            geracoes,
            melhor_global,
            where="post",
            label="Melhor valor global até agora",
            marker=marcador,
            markersize=tamanho_marcador
        )
        titulo_extra = ""

    plt.xlabel("Geração")
    plt.ylabel("Valor acumulado")
    plt.title(f"Evolução do valor acumulado {sufixo_titulo}{titulo_extra}")
    plt.legend(loc="best")
    plt.grid(True)
    ajustar_eixo_y(plt, melhor_geracao + melhor_global)
    salvar_grafico(
        plt,
        pasta / f"grafico_valor_acumulado_{sufixo_arquivo}.png"
    )

    # 2. Tempo da geração
    plt.figure(figsize=(12, 6))
    plt.plot(
        geracoes,
        tempo_geracao,
        marker=marcador,
        markersize=tamanho_marcador
    )
    plt.xlabel("Geração")
    plt.ylabel("Tempo da geração (s)")
    plt.title(f"Tempo estimado da geração {sufixo_titulo}")
    plt.grid(True)
    ajustar_eixo_y(plt, tempo_geracao, margem_percentual=0.10)
    salvar_grafico(
        plt,
        pasta / f"grafico_tempo_por_geracao_{sufixo_arquivo}.png"
    )

    # 3. Quantidade de itens
    plt.figure(figsize=(12, 6))

    if series_iguais(itens_geracao, itens_global):
        plt.step(
            geracoes,
            itens_global,
            where="post",
            label="Itens da melhor solução global",
            marker=marcador,
            markersize=tamanho_marcador
        )
    else:
        plt.plot(
            geracoes,
            itens_geracao,
            label="Itens da melhor solução da geração",
            marker=marcador,
            markersize=tamanho_marcador,
            alpha=0.65
        )

        plt.step(
            geracoes,
            itens_global,
            where="post",
            label="Itens da melhor solução global",
            marker=marcador,
            markersize=tamanho_marcador
        )

    plt.xlabel("Geração")
    plt.ylabel("Quantidade de itens")
    plt.title(f"Itens selecionados {sufixo_titulo}")
    plt.legend(loc="best")
    plt.grid(True)
    ajustar_eixo_y(plt, itens_geracao + itens_global)
    salvar_grafico(
        plt,
        pasta / f"grafico_itens_por_geracao_{sufixo_arquivo}.png"
    )

    # 4. Peso acumulado
    plt.figure(figsize=(12, 6))

    plt.plot(
        geracoes,
        peso_geracao,
        label="Peso da melhor solução da geração",
        marker=marcador,
        markersize=tamanho_marcador,
        alpha=0.65
    )

    plt.step(
        geracoes,
        peso_global,
        where="post",
        label="Peso da melhor solução global",
        marker=marcador,
        markersize=tamanho_marcador
    )

    plt.axhline(
        y=capacidade,
        linestyle="--",
        label="Capacidade da mochila"
    )

    plt.xlabel("Geração")
    plt.ylabel("Peso acumulado")
    plt.title(f"Peso acumulado {sufixo_titulo}")
    plt.legend(loc="best")
    plt.grid(True)
    ajustar_eixo_y(plt, peso_geracao + peso_global + [capacidade])
    salvar_grafico(
        plt,
        pasta / f"grafico_peso_acumulado_{sufixo_arquivo}.png"
    )

    # 5. Melhor global vs média da população
    plt.figure(figsize=(12, 6))

    plt.step(
        geracoes,
        melhor_global,
        where="post",
        label="Melhor valor global",
        marker=marcador,
        markersize=tamanho_marcador
    )

    plt.plot(
        geracoes,
        fitness_medio,
        label="Fitness médio da população",
        marker=marcador,
        markersize=tamanho_marcador,
        alpha=0.75
    )

    plt.xlabel("Geração")
    plt.ylabel("Valor")
    plt.title(f"Melhor global vs. média da população {sufixo_titulo}")
    plt.legend(loc="best")
    plt.grid(True)
    ajustar_eixo_y(plt, melhor_global + fitness_medio)
    salvar_grafico(
        plt,
        pasta / f"grafico_melhor_global_vs_media_{sufixo_arquivo}.png"
    )


# ==========================================================
# GRÁFICOS POR BLOCOS
# ==========================================================

def gerar_graficos_blocos(blocos, pasta, tamanho_bloco):
    plt = preparar_matplotlib()

    if plt is None:
        return

    if len(blocos) < 2:
        registrar_saida(
            f"Gráficos por blocos de {tamanho_bloco} não gerados: apenas {len(blocos)} bloco útil."
        )
        return

    geracoes = [
        linha["geracao_representativa"]
        for linha in blocos
    ]

    melhor_valor_bloco = [
        linha["melhor_valor_no_bloco"]
        for linha in blocos
    ]

    melhor_global_bloco = [
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

    largura_barra = tamanho_bloco * 0.70

    # 1. Valor por blocos
    plt.figure(figsize=(12, 6))

    if series_iguais(melhor_valor_bloco, melhor_global_bloco):
        plt.step(
            geracoes,
            melhor_global_bloco,
            where="post",
            label="Melhor global ao final do bloco",
            marker="o"
        )
    else:
        plt.plot(
            geracoes,
            melhor_valor_bloco,
            label=f"Melhor valor no bloco de {tamanho_bloco}",
            marker="o",
            alpha=0.65
        )

        plt.step(
            geracoes,
            melhor_global_bloco,
            where="post",
            label="Melhor global ao final do bloco",
            marker="o"
        )

    plt.xlabel("Geração final do bloco")
    plt.ylabel("Valor")
    plt.title(f"Evolução por blocos de {tamanho_bloco} gerações")
    plt.legend(loc="best")
    plt.grid(True)
    ajustar_eixo_y(plt, melhor_valor_bloco + melhor_global_bloco)
    salvar_grafico(
        plt,
        pasta / f"grafico_valor_por_blocos_{tamanho_bloco}.png"
    )

    # 2. Melhorias por bloco
    plt.figure(figsize=(12, 6))
    plt.bar(
        geracoes,
        melhorias_bloco,
        width=largura_barra,
        align="center"
    )
    plt.xlabel("Geração final do bloco")
    plt.ylabel("Quantidade de melhorias globais")
    plt.title(f"Melhorias globais por bloco de {tamanho_bloco} gerações")
    plt.grid(True, axis="y")
    ajustar_eixo_y(plt, melhorias_bloco, margem_percentual=0.15)
    salvar_grafico(
        plt,
        pasta / f"grafico_melhorias_blocos_{tamanho_bloco}.png"
    )

    # 3. Tempo total estimado das gerações por bloco
    plt.figure(figsize=(12, 6))
    plt.plot(geracoes, tempo_total_bloco, marker="o")
    plt.xlabel("Geração final do bloco")
    plt.ylabel("Tempo das gerações no bloco (s)")
    plt.title(f"Tempo estimado das gerações por bloco de {tamanho_bloco}")
    plt.grid(True)
    ajustar_eixo_y(plt, tempo_total_bloco, margem_percentual=0.08)
    salvar_grafico(
        plt,
        pasta / f"grafico_tempo_total_blocos_{tamanho_bloco}.png"
    )

    # 4. Tempo médio por geração em cada bloco
    plt.figure(figsize=(12, 6))
    plt.plot(geracoes, tempo_medio_bloco, marker="o")
    plt.xlabel("Geração final do bloco")
    plt.ylabel("Tempo médio por geração (s)")
    plt.title(f"Tempo médio estimado por geração em blocos de {tamanho_bloco}")
    plt.grid(True)
    ajustar_eixo_y(plt, tempo_medio_bloco, margem_percentual=0.08)
    salvar_grafico(
        plt,
        pasta / f"grafico_tempo_medio_blocos_{tamanho_bloco}.png"
    )

    # 5. Média de itens por bloco
    plt.figure(figsize=(12, 6))
    plt.plot(geracoes, media_itens_bloco, marker="o")
    plt.xlabel("Geração final do bloco")
    plt.ylabel("Média de itens selecionados")
    plt.title(f"Média de itens selecionados por bloco de {tamanho_bloco} gerações")
    plt.grid(True)
    ajustar_eixo_y(plt, media_itens_bloco)
    salvar_grafico(
        plt,
        pasta / f"grafico_media_itens_blocos_{tamanho_bloco}.png"
    )

    # 6. Peso médio por bloco
    plt.figure(figsize=(12, 6))
    plt.plot(geracoes, media_peso_bloco, label="Peso médio", marker="o")
    plt.axhline(y=capacidade, linestyle="--", label="Capacidade da mochila")
    plt.xlabel("Geração final do bloco")
    plt.ylabel("Peso médio")
    plt.title(f"Peso médio da melhor solução por bloco de {tamanho_bloco} gerações")
    plt.legend(loc="best")
    plt.grid(True)
    ajustar_eixo_y(plt, media_peso_bloco + [capacidade])
    salvar_grafico(
        plt,
        pasta / f"grafico_media_peso_blocos_{tamanho_bloco}.png"
    )

    # 7. Melhor global vs fitness médio
    plt.figure(figsize=(12, 6))

    plt.step(
        geracoes,
        melhor_global_bloco,
        where="post",
        label="Melhor global ao final do bloco",
        marker="o"
    )

    plt.plot(
        geracoes,
        media_fitness_bloco,
        label="Fitness médio da população no bloco",
        marker="o",
        alpha=0.75
    )

    plt.xlabel("Geração final do bloco")
    plt.ylabel("Valor")
    plt.title(f"Melhor global vs. fitness médio por blocos de {tamanho_bloco}")
    plt.legend(loc="best")
    plt.grid(True)
    ajustar_eixo_y(plt, melhor_global_bloco + media_fitness_bloco)
    salvar_grafico(
        plt,
        pasta / f"grafico_global_vs_media_blocos_{tamanho_bloco}.png"
    )


# ==========================================================
# GERAR TODOS OS GRÁFICOS
# ==========================================================

def gerar_todos_os_graficos():
    if not analise_geracoes:
        registrar_saida("Nenhum dado registrado para gerar gráficos.")
        return

    dados_escala_100 = filtrar_por_escala(analise_geracoes, ESCALA_100)
    resumo_blocos_100 = gerar_resumo_por_blocos(analise_geracoes, 100)

    salvar_csv(ARQUIVO_ANALISE_ESCALA_100, dados_escala_100)
    salvar_csv(ARQUIVO_RESUMO_BLOCOS_100, resumo_blocos_100)

    gerar_graficos_linhas(
        dados=analise_geracoes,
        pasta=PASTA_GRAFICOS_COMPLETOS,
        sufixo_titulo="-- todas as gerações",
        sufixo_arquivo="completo",
        usar_marcador=False
    )

    gerar_graficos_linhas(
        dados=dados_escala_100,
        pasta=PASTA_GRAFICOS_ESCALA_100,
        sufixo_titulo="-- visualização de 100 em 100 gerações",
        sufixo_arquivo="escala_100",
        usar_marcador=True
    )

    gerar_graficos_blocos(
        blocos=resumo_blocos_100,
        pasta=PASTA_GRAFICOS_BLOCOS,
        tamanho_bloco=100
    )

    if GERAR_ESCALA_1000:
        dados_escala_1000 = filtrar_por_escala(analise_geracoes, ESCALA_1000)
        salvar_csv(ARQUIVO_ANALISE_ESCALA_1000, dados_escala_1000)

        gerar_graficos_linhas(
            dados=dados_escala_1000,
            pasta=PASTA_GRAFICOS_ESCALA_1000,
            sufixo_titulo="-- visualização de 1000 em 1000 gerações",
            sufixo_arquivo="escala_1000",
            usar_marcador=True
        )
    else:
        registrar_saida(
            "Gráficos de 1000 em 1000 não foram gerados porque há poucas gerações para essa escala."
        )

    if GERAR_BLOCOS_1000:
        resumo_blocos_1000 = gerar_resumo_por_blocos(analise_geracoes, 1000)
        salvar_csv(ARQUIVO_RESUMO_BLOCOS_1000, resumo_blocos_1000)

        gerar_graficos_blocos(
            blocos=resumo_blocos_1000,
            pasta=PASTA_GRAFICOS_BLOCOS,
            tamanho_bloco=1000
        )
    else:
        registrar_saida(
            "Gráficos por blocos de 1000 não foram gerados porque haveria apenas um bloco útil."
        )

    registrar_saida("")
    registrar_saida("===== GRÁFICOS GERADOS =====")
    registrar_saida(f"Gráficos completos: {PASTA_GRAFICOS_COMPLETOS}")
    registrar_saida(f"Gráficos de 100 em 100: {PASTA_GRAFICOS_ESCALA_100}")
    registrar_saida(f"Gráficos por blocos de 100: {PASTA_GRAFICOS_BLOCOS}")


# ==========================================================
# RESUMO FINAL
# ==========================================================

def salvar_resumo_execucao(inicio_data_hora, fim_data_hora, tempo_total_run):
    tempos_geracao = [linha["tempo_geracao_s"] for linha in analise_geracoes]
    tempos_callback = [linha["tempo_callback_s"] for linha in analise_geracoes]

    tempo_total_geracoes = sum(tempos_geracao)
    tempo_total_callbacks = sum(tempos_callback)

    tempo_medio_geracao = np.mean(tempos_geracao)
    tempo_menor_geracao = np.min(tempos_geracao)
    tempo_maior_geracao = np.max(tempos_geracao)

    total_melhorias_globais = sum(
        1 for linha in analise_geracoes
        if linha["houve_melhoria_global"]
    )

    registrar_saida("")
    registrar_saida("===== RESUMO FINAL DA EXECUÇÃO =====")
    registrar_saida(f"Arquivo de entrada: {NOME_ARQUIVO}")
    registrar_saida(f"Itens disponíveis: {quantidade_itens}")
    registrar_saida(f"Capacidade da mochila: {capacidade}")

    registrar_saida("")
    registrar_saida("===== MELHOR SOLUÇÃO GLOBAL =====")
    registrar_saida(f"Valor acumulado: {melhor_global_valor}")
    registrar_saida(f"Peso acumulado: {melhor_global_peso}/{capacidade}")
    registrar_saida(f"Itens selecionados: {melhor_global_qtd_itens}")
    registrar_saida(f"Geração em que apareceu: {melhor_global_geracao}")
    registrar_saida(f"Melhorias globais encontradas: {total_melhorias_globais}")

    if melhor_global_peso <= capacidade:
        registrar_saida("Status: solução válida")
    else:
        registrar_saida("Status: solução inválida")

    registrar_saida("")
    registrar_saida("===== TEMPO =====")
    registrar_saida(f"Tempo total do algoritmo: {tempo_total_run:.6f} segundos")
    registrar_saida(f"Tempo médio por geração: {tempo_medio_geracao:.6f} segundos")
    registrar_saida(f"Menor tempo de geração: {tempo_menor_geracao:.6f} segundos")
    registrar_saida(f"Maior tempo de geração: {tempo_maior_geracao:.6f} segundos")
    registrar_saida(f"Tempo total estimado das gerações: {tempo_total_geracoes:.6f} segundos")
    registrar_saida(f"Tempo gasto em callbacks/logs: {tempo_total_callbacks:.6f} segundos")

    registrar_saida("")
    registrar_saida("===== CONFIGURAÇÕES UTILIZADAS =====")
    registrar_saida(f"Número de gerações: {NUM_GENERATIONS}")
    registrar_saida(f"Tamanho da população: {SOL_PER_POP}")
    registrar_saida(f"Pais para cruzamento: {NUM_PARENTS_MATING}")
    registrar_saida(f"Mutação: {MUTATION_PERCENT_GENES}% dos genes")
    registrar_saida(f"Seleção: {PARENT_SELECTION_TYPE}")
    registrar_saida(f"Cruzamento: {CROSSOVER_TYPE}")
    registrar_saida(f"Tipo de mutação: {MUTATION_TYPE}")
    registrar_saida(f"Semente aleatória: {RANDOM_SEED}")


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================

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


# ==========================================================
# MEDIÇÃO DO TEMPO E EXECUÇÃO
# ==========================================================

inicio_data_hora = datetime.now()

tempo_inicio_run = time.perf_counter()
tempo_fim_callback_anterior = tempo_inicio_run

registrar_saida("===== INÍCIO DA EXECUÇÃO =====")
registrar_saida("Começou em: " + inicio_data_hora.strftime("%d/%m/%Y %H:%M:%S"))
registrar_saida(f"Arquivo: {NOME_ARQUIVO}")
registrar_saida(f"Itens disponíveis: {quantidade_itens}")
registrar_saida(f"Capacidade: {capacidade}")
registrar_saida(f"População: {SOL_PER_POP}")
registrar_saida(f"Gerações: {NUM_GENERATIONS}")
registrar_saida("")
registrar_saida("===== ACOMPANHAMENTO A CADA 100 GERAÇÕES =====")


# Executa o Algoritmo Genético.
ga_instance.run()


fim_tempo_run = time.perf_counter()
fim_data_hora = datetime.now()

tempo_total_run = fim_tempo_run - tempo_inicio_run


# ==========================================================
# SALVA OS RESULTADOS
# ==========================================================

salvar_csv(ARQUIVO_ANALISE_GERACOES, analise_geracoes)
salvar_tempos_por_geracao()
salvar_itens_escolhidos()
gerar_todos_os_graficos()
salvar_resumo_execucao(inicio_data_hora, fim_data_hora, tempo_total_run)


registrar_saida("")
registrar_saida("===== ARQUIVOS GERADOS =====")
registrar_saida(f"Saída principal: {ARQUIVO_SAIDA_TXT}")
registrar_saida(f"Itens escolhidos: {ARQUIVO_ITENS_ESCOLHIDOS}")
registrar_saida(f"CSV completo: {ARQUIVO_ANALISE_GERACOES}")
registrar_saida(f"CSV tempos por geração: {ARQUIVO_TEMPOS_GERACOES_CSV}")
registrar_saida(f"TXT tempos por geração: {ARQUIVO_TEMPOS_GERACOES_TXT}")
registrar_saida(f"CSV escala 100: {ARQUIVO_ANALISE_ESCALA_100}")
registrar_saida(f"CSV resumo blocos 100: {ARQUIVO_RESUMO_BLOCOS_100}")
registrar_saida(f"Gráficos completos: {PASTA_GRAFICOS_COMPLETOS}")
registrar_saida(f"Gráficos escala 100: {PASTA_GRAFICOS_ESCALA_100}")
registrar_saida(f"Gráficos por blocos: {PASTA_GRAFICOS_BLOCOS}")

if GERAR_ESCALA_1000:
    registrar_saida(f"CSV escala 1000: {ARQUIVO_ANALISE_ESCALA_1000}")
    registrar_saida(f"Gráficos escala 1000: {PASTA_GRAFICOS_ESCALA_1000}")

if GERAR_BLOCOS_1000:
    registrar_saida(f"CSV resumo blocos 1000: {ARQUIVO_RESUMO_BLOCOS_1000}")
