import csv
import os
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# CAMINHOS
# ==========================================================

# Pasta onde este arquivo .py está localizado
PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Arquivos CSV
ARQUIVOS = {
    "AG Original": os.path.join(PASTA_SCRIPT, "saida.csv"),
    "AG Variação 1": os.path.join(PASTA_SCRIPT, "saida2.csv"),
    "AG Variação 2": os.path.join(PASTA_SCRIPT, "saida3.csv")
}

# Pasta onde os gráficos serão salvos
PASTA_SAIDA = os.path.join(PASTA_SCRIPT, "graficos")

os.makedirs(PASTA_SAIDA, exist_ok=True)


# ==========================================================
# CORES
# ==========================================================

CORES = {
    "AG Original": "tab:blue",
    "AG Variação 1": "tab:orange",
    "AG Variação 2": "tab:green"
}


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

    with open(
        caminho_arquivo,
        "r",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        leitor = csv.reader(
            arquivo,
            delimiter=delimitador
        )

        linhas = list(leitor)

    # ------------------------------------------------------
    # Verifica se o arquivo está vazio
    # ------------------------------------------------------

    if len(linhas) == 0:

        raise ValueError(
            f"O arquivo {caminho_arquivo} está vazio."
        )

    # ------------------------------------------------------
    # Cabeçalho
    # ------------------------------------------------------

    cabecalho = linhas[0]

    maior_tamanho = max(
        len(linha)
        for linha in linhas
    )

    # Caso existam mais colunas do que nomes no cabeçalho
    while len(cabecalho) < maior_tamanho:

        cabecalho.append(
            f"ex{len(cabecalho)}"
        )

    # ------------------------------------------------------
    # Organiza os dados
    # ------------------------------------------------------

    dados = []

    for linha in linhas[1:]:

        while len(linha) < maior_tamanho:
            linha.append("")

        dados.append(linha)

    df = pd.DataFrame(
        dados,
        columns=cabecalho
    )

    # ------------------------------------------------------
    # Primeira coluna = posição de acompanhamento
    # ------------------------------------------------------

    primeira_coluna = df.columns[0]

    df = df.rename(
        columns={
            primeira_coluna: "avaliacao"
        }
    )

    # Converte posição para número
    df["avaliacao"] = pd.to_numeric(
        df["avaliacao"],
        errors="coerce"
    )

    # Remove linhas sem posição válida
    df = df.dropna(
        subset=["avaliacao"]
    )

    # ------------------------------------------------------
    # Identifica as colunas das execuções
    # ------------------------------------------------------

    colunas_execucoes = [
        coluna
        for coluna in df.columns
        if coluna != "avaliacao"
    ]

    if len(colunas_execucoes) == 0:

        raise ValueError(
            f"O arquivo {caminho_arquivo} não possui "
            f"colunas de execução."
        )

    # ------------------------------------------------------
    # Converte resultados para números
    # ------------------------------------------------------

    for coluna in colunas_execucoes:

        df[coluna] = pd.to_numeric(
            df[coluna],
            errors="coerce"
        )

    # ------------------------------------------------------
    # Calcula a média das execuções
    # ------------------------------------------------------

    df["media"] = df[
        colunas_execucoes
    ].mean(
        axis=1,
        skipna=True
    )

    # Quantidade de execuções válidas naquela posição
    df["qtd_execucoes"] = df[
        colunas_execucoes
    ].notna().sum(axis=1)

    # Remove linhas sem média
    df = df.dropna(
        subset=["media"]
    )

    # Ordena pela posição de acompanhamento
    df = df.sort_values(
        "avaliacao"
    )

    return df[
        [
            "avaliacao",
            "media",
            "qtd_execucoes"
        ]
    ]


# ==========================================================
# CARREGAR RESULTADOS DOS 3 ALGORITMOS
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

        df = carregar_csv_resultados(
            caminho_arquivo
        )

        resultados[
            nome_algoritmo
        ] = df

        print(
            f"{nome_algoritmo}:"
        )

        print(
            f"  Posições carregadas: {len(df)}"
        )

        print(
            f"  Execuções detectadas: "
            f"{int(df['qtd_execucoes'].iloc[-1])}"
        )

        print(
            f"  Média final: "
            f"{df['media'].iloc[-1]:.2f}"
        )

        print()

    except Exception as erro:

        print(
            f"Erro ao carregar {nome_algoritmo}:"
        )

        print(erro)

        print()


# ==========================================================
# VERIFICA SE ALGUM RESULTADO FOI CARREGADO
# ==========================================================

if len(resultados) == 0:

    print(
        "Nenhum resultado foi carregado."
    )

    print(
        "Verifique se os arquivos:"
    )

    print(
        "saida.csv"
    )

    print(
        "saida2.csv"
    )

    print(
        "saida3.csv"
    )

    print(
        "estão na mesma pasta do gerar_graficos_AG.py."
    )

    raise SystemExit


# ==========================================================
# GRÁFICO COMPARATIVO
# ==========================================================

plt.figure(
    figsize=(12, 6)
)


for nome_algoritmo, df in resultados.items():

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
    "Média dos resultados da função objetiva"
)

plt.ylim(top=46000)

plt.legend()

plt.grid(True)

plt.tight_layout()


caminho_comparativo = os.path.join(
    PASTA_SAIDA,
    "grafico_comparativo_media.png"
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
# ==========================================================

NOMES_ARQUIVOS = {

    "AG Original":
        "grafico_ag_original.png",

    "AG Variação 1":
        "grafico_ag_variacao_1.png",

    "AG Variação 2":
        "grafico_ag_variacao_2.png"
}


for nome_algoritmo, df in resultados.items():

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
        "Média dos resultados da função objetiva"
    )

    plt.ylim(top=46000)

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


for nome_algoritmo, df in resultados.items():

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
    "- grafico_comparativo_media.png"
)

print(
    "- grafico_ag_original.png"
)

print(
    "- grafico_ag_variacao_1.png"
)

print(
    "- grafico_ag_variacao_2.png"
)

print(
    "- medias_comparativo.csv"
)