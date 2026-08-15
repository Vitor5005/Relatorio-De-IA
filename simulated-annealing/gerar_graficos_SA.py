import csv
import os
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# CAMINHOS
# ==========================================================

# Pasta onde este arquivo gerar_graficos_SA.py está localizado
PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Arquivos CSV das três configurações
ARQUIVOS = {
    "SA Original": os.path.join(PASTA_SCRIPT, "saida1_sa.csv"),
    "SA Variação 1": os.path.join(PASTA_SCRIPT, "saida2_sa.csv"),
    "SA Variação 2": os.path.join(PASTA_SCRIPT, "saida3_sa.csv")
}

# Pasta onde os gráficos serão salvos
PASTA_SAIDA = os.path.join(PASTA_SCRIPT, "graficos")

os.makedirs(PASTA_SAIDA, exist_ok=True)


# ==========================================================
# CORES
# ==========================================================

CORES = {
    "SA Original": "tab:blue",
    "SA Variação 1": "tab:orange",
    "SA Variação 2": "tab:green"
}


# ==========================================================
# NOMES DOS GRÁFICOS
# Mantidos exatamente como estavam anteriormente
# ==========================================================

NOMES_ARQUIVOS = {
    "SA Original": "Figure_2.png",
    "SA Variação 1": "Figure_3.png",
    "SA Variação 2": "Figure_4.png"
}

NOME_GRAFICO_COMPARATIVO = "Figure_1.png"


# ==========================================================
# CONFIGURAÇÃO DO EXPERIMENTO
# ==========================================================

NUM_EXECUCOES_ESPERADAS = 10
NUM_POSICOES_ESPERADAS = 10000


# ==========================================================
# DETECTAR DELIMITADOR DO CSV
# ==========================================================

def detectar_delimitador(caminho_arquivo):

    with open(
        caminho_arquivo,
        "r",
        encoding="utf-8"
    ) as arquivo:

        amostra = arquivo.read(2048)

    try:

        dialeto = csv.Sniffer().sniff(
            amostra,
            delimiters=",;"
        )

        return dialeto.delimiter

    except csv.Error:

        return ","


# ==========================================================
# CARREGAR CSV E CALCULAR MÉDIA
# ==========================================================

def carregar_csv_resultados(caminho_arquivo):

    delimitador = detectar_delimitador(caminho_arquivo)

    df = pd.read_csv(
        caminho_arquivo,
        delimiter=delimitador,
        encoding="utf-8"
    )

    # ------------------------------------------------------
    # Verifica se o arquivo está vazio
    # ------------------------------------------------------

    if df.empty:

        raise ValueError(
            f"O arquivo {caminho_arquivo} está vazio."
        )

    # ------------------------------------------------------
    # Primeira coluna deve representar a avaliação
    # ------------------------------------------------------

    primeira_coluna = df.columns[0]

    if primeira_coluna != "avaliacao":

        df = df.rename(
            columns={
                primeira_coluna: "avaliacao"
            }
        )

    # Converte avaliação para número
    df["avaliacao"] = pd.to_numeric(
        df["avaliacao"],
        errors="coerce"
    )

    # Remove linhas sem avaliação válida
    df = df.dropna(
        subset=["avaliacao"]
    )

    # ------------------------------------------------------
    # Identifica colunas das execuções
    # ------------------------------------------------------

    colunas_execucoes = [
        coluna
        for coluna in df.columns
        if coluna != "avaliacao"
    ]

    if len(colunas_execucoes) == 0:

        raise ValueError(
            f"O arquivo {caminho_arquivo} "
            f"não possui colunas de execução."
        )

    # ------------------------------------------------------
    # Verifica se existem exatamente 10 execuções
    # ------------------------------------------------------

    if len(colunas_execucoes) != NUM_EXECUCOES_ESPERADAS:

        raise ValueError(
            f"O arquivo {os.path.basename(caminho_arquivo)} possui "
            f"{len(colunas_execucoes)} execuções, mas eram esperadas "
            f"{NUM_EXECUCOES_ESPERADAS}.\n"
            f"Colunas encontradas: {colunas_execucoes}"
        )

    # ------------------------------------------------------
    # Converte valores das execuções para números
    # ------------------------------------------------------

    for coluna in colunas_execucoes:

        df[coluna] = pd.to_numeric(
            df[coluna],
            errors="coerce"
        )

    # ------------------------------------------------------
    # Verifica valores ausentes
    # ------------------------------------------------------

    valores_ausentes = df[colunas_execucoes].isna().sum().sum()

    if valores_ausentes > 0:

        raise ValueError(
            f"O arquivo {os.path.basename(caminho_arquivo)} possui "
            f"{valores_ausentes} valores ausentes ou inválidos."
        )

    # ------------------------------------------------------
    # Ordena pela avaliação
    # ------------------------------------------------------

    df = df.sort_values(
        "avaliacao"
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # Verifica quantidade de posições
    # ------------------------------------------------------

    if len(df) != NUM_POSICOES_ESPERADAS:

        print(
            f"AVISO: {os.path.basename(caminho_arquivo)} possui "
            f"{len(df)} posições. "
            f"Eram esperadas {NUM_POSICOES_ESPERADAS}."
        )

    # ------------------------------------------------------
    # Calcula média das 10 execuções em cada posição
    # ------------------------------------------------------

    df["media"] = df[
        colunas_execucoes
    ].mean(
        axis=1
    )

    # ------------------------------------------------------
    # Valores finais de cada uma das 10 execuções
    # ------------------------------------------------------

    valores_finais = df[
        colunas_execucoes
    ].iloc[-1]

    media_final = valores_finais.mean()

    # Desvio-padrão amostral
    desvio_padrao_final = valores_finais.std(ddof=1)

    melhor_execucao = valores_finais.max()
    pior_execucao = valores_finais.min()

    return {
        "dados": df,
        "colunas_execucoes": colunas_execucoes,
        "valores_finais": valores_finais,
        "media_final": media_final,
        "desvio_padrao_final": desvio_padrao_final,
        "melhor_execucao": melhor_execucao,
        "pior_execucao": pior_execucao
    }


# ==========================================================
# CARREGAR RESULTADOS DOS 3 SAs
# ==========================================================

resultados = {}


for nome_algoritmo, caminho_arquivo in ARQUIVOS.items():

    if not os.path.exists(caminho_arquivo):

        print(
            f"AVISO: arquivo não encontrado:"
        )

        print(caminho_arquivo)

        print()

        continue

    try:

        resultado = carregar_csv_resultados(
            caminho_arquivo
        )

        resultados[
            nome_algoritmo
        ] = resultado

        df = resultado["dados"]

        print("=" * 60)
        print(nome_algoritmo)
        print("=" * 60)

        print(
            f"Posições carregadas: "
            f"{len(df)}"
        )

        print(
            f"Execuções detectadas: "
            f"{len(resultado['colunas_execucoes'])}"
        )

        print()

        print("Valores finais das execuções:")

        for nome_execucao, valor in resultado[
            "valores_finais"
        ].items():

            print(
                f"  {nome_execucao}: "
                f"{valor:.0f}"
            )

        print()

        print(
            f"Média final: "
            f"{resultado['media_final']:.2f}"
        )

        print(
            f"Desvio-padrão final: "
            f"{resultado['desvio_padrao_final']:.2f}"
        )

        print(
            f"Melhor execução: "
            f"{resultado['melhor_execucao']:.0f}"
        )

        print(
            f"Pior execução: "
            f"{resultado['pior_execucao']:.0f}"
        )

        print()

    except Exception as erro:

        print(
            f"ERRO ao carregar {nome_algoritmo}:"
        )

        print(erro)

        print()


# ==========================================================
# VERIFICA SE OS TRÊS RESULTADOS FORAM CARREGADOS
# ==========================================================

if len(resultados) == 0:

    print(
        "Nenhum resultado foi carregado."
    )

    print(
        "Verifique se os arquivos:"
    )

    print(
        "saida1_sa.csv"
    )

    print(
        "saida2_sa.csv"
    )

    print(
        "saida3_sa.csv"
    )

    print(
        "estão na mesma pasta do gerar_graficos_SA.py."
    )

    raise SystemExit


if len(resultados) != 3:

    print(
        "AVISO: nem todas as três configurações "
        "foram carregadas."
    )

    print()


# ==========================================================
# GRÁFICO COMPARATIVO
# Figure_1.png
# ==========================================================

plt.figure(
    figsize=(12, 6)
)


for nome_algoritmo, resultado in resultados.items():

    df = resultado["dados"]

    plt.plot(
        df["avaliacao"],
        df["media"],
        label=nome_algoritmo,
        color=CORES[nome_algoritmo]
    )


plt.title(
    "Função objetiva média por posição de acompanhamento"
)

plt.xlabel(
    "Posição de acompanhamento"
)

plt.ylabel(
    "Média dos melhores valores acumulados"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


caminho_comparativo = os.path.join(
    PASTA_SAIDA,
    NOME_GRAFICO_COMPARATIVO
)


plt.savefig(
    caminho_comparativo,
    dpi=300
)

plt.close()


print(
    "Gráfico comparativo salvo em:"
)

print(
    caminho_comparativo
)

print()


# ==========================================================
# GRÁFICOS INDIVIDUAIS
# Figure_2.png
# Figure_3.png
# Figure_4.png
# ==========================================================

for nome_algoritmo, resultado in resultados.items():

    df = resultado["dados"]

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        df["avaliacao"],
        df["media"],
        label=nome_algoritmo,
        color=CORES[nome_algoritmo]
    )

    plt.title(
        f"Função objetiva média por posição de acompanhamento - "
        f"{nome_algoritmo}"
    )

    plt.xlabel(
        "Posição de acompanhamento"
    )

    plt.ylabel(
        "Média dos melhores valores acumulados"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()


    caminho_individual = os.path.join(
        PASTA_SAIDA,
        NOMES_ARQUIVOS[nome_algoritmo]
    )


    plt.savefig(
        caminho_individual,
        dpi=300
    )

    plt.close()


    print(
        f"Gráfico individual de "
        f"{nome_algoritmo} salvo em:"
    )

    print(
        caminho_individual
    )

    print()


# ==========================================================
# CSV COMPARATIVO DAS MÉDIAS
# ==========================================================

comparativo = pd.DataFrame()


for nome_algoritmo, resultado in resultados.items():

    df = resultado["dados"]

    serie = df.set_index(
        "avaliacao"
    )["media"]

    comparativo[
        nome_algoritmo
    ] = serie


comparativo = comparativo.reset_index()


caminho_csv_media = os.path.join(
    PASTA_SAIDA,
    "medias_comparativo.csv"
)


comparativo.to_csv(
    caminho_csv_media,
    index=False,
    encoding="utf-8"
)


print(
    "CSV com as médias salvo em:"
)

print(
    caminho_csv_media
)

print()


# ==========================================================
# RESUMO ESTATÍSTICO FINAL
# ==========================================================

print()
print("=" * 70)
print("RESUMO ESTATÍSTICO FINAL DO SIMULATED ANNEALING")
print("=" * 70)

for nome_algoritmo, resultado in resultados.items():

    print()
    print(nome_algoritmo)

    print(
        f"  Média final: "
        f"{resultado['media_final']:.2f}"
    )

    print(
        f"  Desvio-padrão: "
        f"{resultado['desvio_padrao_final']:.2f}"
    )

    print(
        f"  Melhor execução: "
        f"{resultado['melhor_execucao']:.0f}"
    )

    print(
        f"  Pior execução: "
        f"{resultado['pior_execucao']:.0f}"
    )


print()
print("=" * 70)


# ==========================================================
# FINAL
# ==========================================================

print()
print(
    "Processamento concluído!"
)

print()

print(
    "Arquivos gerados:"
)

print(
    "- Figure_1.png"
)

print(
    "- Figure_2.png"
)

print(
    "- Figure_3.png"
)

print(
    "- Figure_4.png"
)

print(
    "- medias_comparativo.csv"
)