import csv
import os
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# CAMINHOS
# ==========================================================

# Pasta onde este arquivo está localizado.
# Como o arquivo ficará na pasta IA, esta será a pasta raiz.
PASTA_RAIZ = os.path.dirname(
    os.path.abspath(__file__)
)

# Pasta do Algoritmo Genético
PASTA_AG = os.path.join(
    PASTA_RAIZ,
    "algoritmo_genetico"
)

# Pasta do Simulated Annealing
PASTA_SA = os.path.join(
    PASTA_RAIZ,
    "simulated-annealing"
)


# ==========================================================
# ARQUIVOS DO ALGORITMO GENÉTICO
# ==========================================================

ARQUIVOS_AG = {
    "AG Original": os.path.join(
        PASTA_AG,
        "saida.csv"
    ),

    "AG Variação 1": os.path.join(
        PASTA_AG,
        "saida2.csv"
    ),

    "AG Variação 2": os.path.join(
        PASTA_AG,
        "saida3.csv"
    )
}


# ==========================================================
# ARQUIVOS DO SIMULATED ANNEALING
# ==========================================================

ARQUIVOS_SA = {
    "SA Original": os.path.join(
        PASTA_SA,
        "saida1_sa.csv"
    ),

    "SA Variação 1": os.path.join(
        PASTA_SA,
        "saida2_sa.csv"
    ),

    "SA Variação 2": os.path.join(
        PASTA_SA,
        "saida3_sa.csv"
    )
}


# ==========================================================
# ARQUIVO DE SAÍDA
# ==========================================================

# O gráfico será salvo na própria pasta raiz IA.
CAMINHO_GRAFICO = os.path.join(
    PASTA_RAIZ,
    "grafico_melhor_ag_vs_melhor_sa.png"
)


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

    delimitador = detectar_delimitador(
        caminho_arquivo
    )

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


    # Se alguma linha tiver mais colunas do que o cabeçalho,
    # cria nomes adicionais para as execuções.
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
    # Primeira coluna = avaliação
    # ------------------------------------------------------

    primeira_coluna = df.columns[0]

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
    # Identifica as colunas das execuções
    # ------------------------------------------------------

    colunas_execucoes = [
        coluna
        for coluna in df.columns
        if coluna != "avaliacao"
    ]


    if len(colunas_execucoes) == 0:

        raise ValueError(
            f"O arquivo {caminho_arquivo} "
            f"não possui execuções."
        )


    # ------------------------------------------------------
    # Converte as execuções para números
    # ------------------------------------------------------

    for coluna in colunas_execucoes:

        df[coluna] = pd.to_numeric(
            df[coluna],
            errors="coerce"
        )


    # ------------------------------------------------------
    # Média dos melhores resultados das execuções
    # em cada avaliação
    # ------------------------------------------------------

    df["media"] = df[
        colunas_execucoes
    ].mean(
        axis=1,
        skipna=True
    )


    # Quantidade de execuções válidas naquela avaliação
    df["qtd_execucoes"] = df[
        colunas_execucoes
    ].notna().sum(
        axis=1
    )


    # Remove linhas que não possuem média
    df = df.dropna(
        subset=["media"]
    )


    # Ordena por avaliação
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
# CARREGAR UM GRUPO DE ALGORITMOS
# ==========================================================

def carregar_algoritmos(arquivos):

    resultados = {}

    for nome_algoritmo, caminho in arquivos.items():

        print(
            f"Carregando {nome_algoritmo}..."
        )

        if not os.path.exists(caminho):

            print(
                f"ERRO: arquivo não encontrado:"
            )

            print(caminho)

            print()

            continue


        try:

            df = carregar_csv_resultados(
                caminho
            )

            resultados[
                nome_algoritmo
            ] = df


            print(
                f"  Avaliações: {len(df)}"
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
                f"Erro ao carregar "
                f"{nome_algoritmo}:"
            )

            print(erro)

            print()


    return resultados


# ==========================================================
# CARREGANDO OS RESULTADOS DO AG
# ==========================================================

print()
print(
    "=========================================="
)
print(
    "ALGORITMOS GENÉTICOS"
)
print(
    "=========================================="
)
print()

resultados_ag = carregar_algoritmos(
    ARQUIVOS_AG
)


# ==========================================================
# CARREGANDO OS RESULTADOS DO SA
# ==========================================================

print()
print(
    "=========================================="
)
print(
    "SIMULATED ANNEALING"
)
print(
    "=========================================="
)
print()

resultados_sa = carregar_algoritmos(
    ARQUIVOS_SA
)


# ==========================================================
# VERIFICAÇÃO
# ==========================================================

if len(resultados_ag) == 0:

    print(
        "Nenhum resultado de Algoritmo Genético "
        "foi carregado."
    )

    raise SystemExit


if len(resultados_sa) == 0:

    print(
        "Nenhum resultado de Simulated Annealing "
        "foi carregado."
    )

    raise SystemExit


# ==========================================================
# ENCONTRAR O MELHOR ALGORITMO GENÉTICO
# ==========================================================

# O melhor é definido pela maior média
# na última avaliação.

melhor_ag_nome = max(
    resultados_ag,
    key=lambda nome:
        resultados_ag[nome]["media"].iloc[-1]
)

melhor_ag_df = resultados_ag[
    melhor_ag_nome
]

melhor_ag_media = melhor_ag_df[
    "media"
].iloc[-1]


# ==========================================================
# ENCONTRAR O MELHOR SIMULATED ANNEALING
# ==========================================================

melhor_sa_nome = max(
    resultados_sa,
    key=lambda nome:
        resultados_sa[nome]["media"].iloc[-1]
)

melhor_sa_df = resultados_sa[
    melhor_sa_nome
]

melhor_sa_media = melhor_sa_df[
    "media"
].iloc[-1]


# ==========================================================
# EXIBIR RESULTADOS
# ==========================================================

print()
print(
    "=========================================="
)
print(
    "MELHORES CONFIGURAÇÕES"
)
print(
    "=========================================="
)
print()


print(
    f"Melhor AG: {melhor_ag_nome}"
)

print(
    f"Média final: {melhor_ag_media:.2f}"
)

print()


print(
    f"Melhor SA: {melhor_sa_nome}"
)

print(
    f"Média final: {melhor_sa_media:.2f}"
)

print()


# ==========================================================
# DIFERENÇA ENTRE OS ALGORITMOS
# ==========================================================

diferenca = (
    melhor_sa_media
    - melhor_ag_media
)


if melhor_ag_media != 0:

    diferenca_percentual = (
        diferenca
        / melhor_ag_media
    ) * 100

else:

    diferenca_percentual = 0


print(
    f"Diferença absoluta: "
    f"{diferenca:.2f}"
)

print(
    f"Diferença percentual: "
    f"{diferenca_percentual:.2f}%"
)

print()


# ==========================================================
# GERAR GRÁFICO
# ==========================================================

plt.figure(
    figsize=(12, 6)
)


# ----------------------------------------------------------
# Melhor Algoritmo Genético
# ----------------------------------------------------------

plt.plot(
    melhor_ag_df["avaliacao"],
    melhor_ag_df["media"],
    label=melhor_ag_nome,
    color="tab:orange"
)


# ----------------------------------------------------------
# Melhor Simulated Annealing
# ----------------------------------------------------------

plt.plot(
    melhor_sa_df["avaliacao"],
    melhor_sa_df["media"],
    label=melhor_sa_nome,
    color="tab:green"
)


# ==========================================================
# CONFIGURAÇÕES DO GRÁFICO
# ==========================================================

plt.title(
    "Comparação entre o melhor AG e o melhor SA"
)

plt.xlabel(
    "Número de Avaliações"
)

plt.ylabel(
    "Média dos resultados da função objetiva"
)


# Limite superior do eixo Y
plt.ylim(
    top=50000
)


plt.legend()

plt.grid(True)

plt.tight_layout()


# ==========================================================
# SALVAR GRÁFICO
# ==========================================================

plt.savefig(
    CAMINHO_GRAFICO,
    dpi=300
)

plt.close()


# ==========================================================
# FINAL
# ==========================================================

print(
    "=========================================="
)

print(
    "GRÁFICO GERADO COM SUCESSO"
)

print(
    "=========================================="
)

print()

print(
    f"Arquivo salvo em:"
)

print(
    CAMINHO_GRAFICO
)