import csv
import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIGURAÇÕES
# =========================

ARQUIVOS = {
    "AG Original": "saida.csv",
    "AG Variação 1": "saida2.csv",
    "AG Variação 2": "saida3.csv"
}

PASTA_SAIDA = "graficos"

os.makedirs(PASTA_SAIDA, exist_ok=True)


# =========================
# LEITURA GENÉRICA DO CSV
# =========================

def detectar_delimitador(caminho_arquivo):
    with open(caminho_arquivo, "r", encoding="utf-8") as arquivo:
        amostra = arquivo.read(2048)

    try:
        dialeto = csv.Sniffer().sniff(amostra, delimiters=",;")
        return dialeto.delimiter
    except csv.Error:
        return ","


def carregar_csv_resultados(caminho_arquivo):
    delimitador = detectar_delimitador(caminho_arquivo)

    with open(caminho_arquivo, "r", newline="", encoding="utf-8") as arquivo:
        leitor = csv.reader(arquivo, delimiter=delimitador)
        linhas = list(leitor)

    if len(linhas) == 0:
        raise ValueError(f"O arquivo {caminho_arquivo} está vazio.")

    cabecalho = linhas[0]

    # Garante que todas as linhas tenham o mesmo tamanho
    maior_tamanho = max(len(linha) for linha in linhas)

    while len(cabecalho) < maior_tamanho:
        cabecalho.append("ex" + str(len(cabecalho)))

    dados = []

    for linha in linhas[1:]:
        while len(linha) < maior_tamanho:
            linha.append("")

        dados.append(linha)

    df = pd.DataFrame(dados, columns=cabecalho)

    # A primeira coluna deve ser a avaliação
    primeira_coluna = df.columns[0]
    df = df.rename(columns={primeira_coluna: "avaliacao"})

    # Converte avaliação para número
    df["avaliacao"] = pd.to_numeric(df["avaliacao"], errors="coerce")

    # Remove linhas sem avaliação válida
    df = df.dropna(subset=["avaliacao"])

    # Colunas das execuções: ex1, ex2, ex3...
    colunas_execucoes = [coluna for coluna in df.columns if coluna != "avaliacao"]

    # Converte os resultados para número
    for coluna in colunas_execucoes:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    # Calcula a média das execuções em cada avaliação
    df["media"] = df[colunas_execucoes].mean(axis=1, skipna=True)

    # Conta quantas execuções foram usadas naquela linha
    df["qtd_execucoes"] = df[colunas_execucoes].notna().sum(axis=1)

    # Remove linhas sem média
    df = df.dropna(subset=["media"])

    # Ordena por avaliação
    df = df.sort_values("avaliacao")

    return df[["avaliacao", "media", "qtd_execucoes"]]


# =========================
# CARREGANDO OS RESULTADOS
# =========================

resultados = {}

for nome_algoritmo, caminho_arquivo in ARQUIVOS.items():
    if not os.path.exists(caminho_arquivo):
        print(f"Aviso: arquivo não encontrado: {caminho_arquivo}")
        continue

    df = carregar_csv_resultados(caminho_arquivo)
    resultados[nome_algoritmo] = df

    print(f"{nome_algoritmo}: {len(df)} avaliações carregadas.")
    print(f"Execuções detectadas na última linha: {int(df['qtd_execucoes'].iloc[-1])}")
    print(f"Melhor média final: {df['media'].iloc[-1]:.2f}")
    print()


# =========================
# GERANDO GRÁFICO COMPARATIVO
# =========================

plt.figure(figsize=(12, 6))

for nome_algoritmo, df in resultados.items():
    plt.plot(
        df["avaliacao"],
        df["media"],
        label=nome_algoritmo
    )

plt.title("Função objetiva média por avaliação")
plt.xlabel("Avaliação")
plt.ylabel("Média dos resultados da função objetiva")
plt.legend()
plt.grid(True)
plt.tight_layout()

caminho_grafico = os.path.join(PASTA_SAIDA, "grafico_comparativo_media.png")
plt.savefig(caminho_grafico, dpi=300)
plt.show()

print(f"Gráfico comparativo salvo em: {caminho_grafico}")


# =========================
# GERANDO GRÁFICOS INDIVIDUAIS
# =========================

for nome_algoritmo, df in resultados.items():
    plt.figure(figsize=(12, 6))

    plt.plot(
        df["avaliacao"],
        df["media"],
        label=nome_algoritmo
    )

    plt.title(f"Função objetiva média por avaliação - {nome_algoritmo}")
    plt.xlabel("Avaliação")
    plt.ylabel("Média dos resultados da função objetiva")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    nome_arquivo = nome_algoritmo.lower()
    nome_arquivo = nome_arquivo.replace(" ", "_")
    nome_arquivo = nome_arquivo.replace("ç", "c")
    nome_arquivo = nome_arquivo.replace("ã", "a")

    caminho_individual = os.path.join(PASTA_SAIDA, f"grafico_{nome_arquivo}.png")

    plt.savefig(caminho_individual, dpi=300)
    plt.show()

    print(f"Gráfico individual salvo em: {caminho_individual}")


# =========================
# SALVANDO CSV COM AS MÉDIAS
# =========================

comparativo = pd.DataFrame()

for nome_algoritmo, df in resultados.items():
    serie = df.set_index("avaliacao")["media"]
    comparativo[nome_algoritmo] = serie

comparativo = comparativo.reset_index()

caminho_csv_media = os.path.join(PASTA_SAIDA, "medias_comparativo.csv")
comparativo.to_csv(caminho_csv_media, index=False, encoding="utf-8")

print(f"CSV com as médias salvo em: {caminho_csv_media}")