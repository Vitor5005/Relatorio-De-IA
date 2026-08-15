import os
import numpy as np
import pandas as pd

# ============================================================
# DIRETÓRIO BASE DO PROJETO
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# CONFIGURAÇÃO DOS ARQUIVOS
# ============================================================

ARQUIVOS = {
    "AG Original": {
        "arquivo": os.path.join(
            BASE_DIR,
            "algoritmo_genetico",
            "saida.csv"
        ),
        "algoritmo": "AG"
    },

    "AG Variação 1": {
        "arquivo": os.path.join(
            BASE_DIR,
            "algoritmo_genetico",
            "saida2.csv"
        ),
        "algoritmo": "AG"
    },

    "AG Variação 2": {
        "arquivo": os.path.join(
            BASE_DIR,
            "algoritmo_genetico",
            "saida3.csv"
        ),
        "algoritmo": "AG"
    },

    "SA Original": {
        "arquivo": os.path.join(
            BASE_DIR,
            "simulated-annealing",
            "saida1_sa.csv"
        ),
        "algoritmo": "SA"
    },

    "SA Variação 1": {
        "arquivo": os.path.join(
            BASE_DIR,
            "simulated-annealing",
            "saida2_sa.csv"
        ),
        "algoritmo": "SA"
    },

    "SA Variação 2": {
        "arquivo": os.path.join(
            BASE_DIR,
            "simulated-annealing",
            "saida3_sa.csv"
        ),
        "algoritmo": "SA"
    }
}

NUM_EXECUCOES_ESPERADAS = 10

ARQUIVO_SAIDA = os.path.join(
    BASE_DIR,
    "estatisticas_finais.csv"
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def carregar_csv(caminho):
    """
    Carrega o CSV.

    Primeiro tenta o formato padrão separado por vírgula.
    Caso seja identificado apenas um campo, tenta detectar
    automaticamente o separador.
    """

    df = pd.read_csv(caminho)

    if len(df.columns) == 1:
        df = pd.read_csv(
            caminho,
            sep=None,
            engine="python"
        )

    return df


def obter_resultados_finais(caminho):
    """
    Obtém o resultado final de cada uma das 10 execuções.

    A primeira coluna do CSV representa a posição de
    acompanhamento.

    As demais colunas representam as execuções independentes.

    Como cada série armazena o melhor valor acumulado
    (best-so-far), o último valor de cada coluna representa
    o resultado final daquela execução.
    """

    df = carregar_csv(caminho)

    if df.empty:
        raise ValueError(
            f"O arquivo '{caminho}' está vazio."
        )

    if len(df.columns) < 2:
        raise ValueError(
            f"O arquivo '{caminho}' não possui colunas suficientes."
        )

    # Primeira coluna = posição de acompanhamento
    # Demais colunas = execuções
    colunas_execucoes = list(df.columns[1:])

    quantidade_execucoes = len(colunas_execucoes)

    if quantidade_execucoes != NUM_EXECUCOES_ESPERADAS:
        raise ValueError(
            f"O arquivo '{caminho}' deveria possuir "
            f"{NUM_EXECUCOES_ESPERADAS} execuções, "
            f"mas possui {quantidade_execucoes}."
        )

    resultados = []

    for coluna in colunas_execucoes:

        serie = pd.to_numeric(
            df[coluna],
            errors="coerce"
        ).dropna()

        if serie.empty:
            raise ValueError(
                f"A coluna '{coluna}' do arquivo "
                f"'{caminho}' não possui valores numéricos."
            )

        # Último resultado disponível da execução
        resultado_final = float(serie.iloc[-1])

        resultados.append(resultado_final)

    return np.array(resultados, dtype=float)


def calcular_intervalo_confianca(media, desvio_padrao, n):
    """
    Calcula o intervalo de confiança de 95% da média
    usando a distribuição t de Student.

    Para n = 10:
        graus de liberdade = 9

    Caso scipy não esteja instalada, retorna None.
    """

    try:
        from scipy.stats import t

        erro_padrao = desvio_padrao / np.sqrt(n)

        valor_t = t.ppf(
            0.975,
            df=n - 1
        )

        margem = valor_t * erro_padrao

        limite_inferior = media - margem
        limite_superior = media + margem

        return (
            limite_inferior,
            limite_superior,
            margem
        )

    except ImportError:
        return None


def calcular_estatisticas(valores):
    """
    Calcula as estatísticas descritivas dos resultados
    finais das execuções.
    """

    n = len(valores)

    media = np.mean(valores)

    # ddof=1 -> desvio-padrão AMOSTRAL
    desvio_padrao = np.std(
        valores,
        ddof=1
    )

    mediana = np.median(valores)

    melhor = np.max(valores)

    pior = np.min(valores)

    amplitude = melhor - pior

    erro_padrao = desvio_padrao / np.sqrt(n)

    intervalo = calcular_intervalo_confianca(
        media,
        desvio_padrao,
        n
    )

    if intervalo is not None:
        ic_inferior, ic_superior, margem = intervalo
    else:
        ic_inferior = np.nan
        ic_superior = np.nan
        margem = np.nan

    return {
        "n": n,
        "media": media,
        "desvio_padrao": desvio_padrao,
        "mediana": mediana,
        "melhor": melhor,
        "pior": pior,
        "amplitude": amplitude,
        "erro_padrao": erro_padrao,
        "ic95_inferior": ic_inferior,
        "ic95_superior": ic_superior,
        "margem_erro_95": margem
    }


def formatar_numero(valor, casas=2):
    """
    Formata número no padrão brasileiro apenas
    para exibição no terminal.
    """

    if pd.isna(valor):
        return "-"

    texto = f"{valor:,.{casas}f}"

    texto = (
        texto
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    return texto


# ============================================================
# PROCESSAMENTO
# ============================================================

def main():

    print()
    print("=" * 78)
    print("ANÁLISE ESTATÍSTICA DAS EXECUÇÕES")
    print("ALGORITMO GENÉTICO E SIMULATED ANNEALING")
    print("=" * 78)

    resultados_resumo = []

    resultados_individuais = {}

    for configuracao, dados in ARQUIVOS.items():

        arquivo = dados["arquivo"]
        algoritmo = dados["algoritmo"]

        print()
        print("=" * 78)
        print(configuracao)
        print("=" * 78)

        if not os.path.exists(arquivo):
            print(
                f"ERRO: arquivo '{arquivo}' não encontrado."
            )
            continue

        try:
            valores = obter_resultados_finais(
                arquivo
            )

        except Exception as erro:
            print(
                f"ERRO ao processar '{arquivo}': {erro}"
            )
            continue

        resultados_individuais[configuracao] = valores

        # ----------------------------------------------------
        # Mostrar valores finais das 10 execuções
        # ----------------------------------------------------

        print("\nResultados finais das execuções:\n")

        for indice, valor in enumerate(
            valores,
            start=1
        ):
            print(
                f"Execução {indice:02d}: "
                f"{formatar_numero(valor, 0)}"
            )

        # ----------------------------------------------------
        # Estatísticas
        # ----------------------------------------------------

        estatisticas = calcular_estatisticas(
            valores
        )

        print("\nEstatísticas:")
        print("-" * 78)

        print(
            f"Número de execuções: "
            f"{estatisticas['n']}"
        )

        print(
            f"Média final:          "
            f"{formatar_numero(estatisticas['media'], 2)}"
        )

        print(
            f"Desvio-padrão:        "
            f"{formatar_numero(estatisticas['desvio_padrao'], 2)}"
        )

        print(
            f"Mediana:              "
            f"{formatar_numero(estatisticas['mediana'], 2)}"
        )

        print(
            f"Melhor execução:      "
            f"{formatar_numero(estatisticas['melhor'], 0)}"
        )

        print(
            f"Pior execução:        "
            f"{formatar_numero(estatisticas['pior'], 0)}"
        )

        print(
            f"Amplitude:             "
            f"{formatar_numero(estatisticas['amplitude'], 0)}"
        )

        print(
            f"Erro-padrão da média: "
            f"{formatar_numero(estatisticas['erro_padrao'], 2)}"
        )

        if not np.isnan(
            estatisticas["ic95_inferior"]
        ):

            print(
                "IC 95% da média:       "
                f"[{formatar_numero(estatisticas['ic95_inferior'], 2)}; "
                f"{formatar_numero(estatisticas['ic95_superior'], 2)}]"
            )

        else:

            print(
                "IC 95% da média:       "
                "não calculado (scipy não instalada)"
            )

        # ----------------------------------------------------
        # Armazenar resumo
        # ----------------------------------------------------

        resultados_resumo.append({
            "Algoritmo": algoritmo,
            "Configuração": configuracao,
            "Execuções": estatisticas["n"],
            "Média final": estatisticas["media"],
            "Desvio-padrão": estatisticas["desvio_padrao"],
            "Mediana": estatisticas["mediana"],
            "Melhor": estatisticas["melhor"],
            "Pior": estatisticas["pior"],
            "Amplitude": estatisticas["amplitude"],
            "Erro-padrão": estatisticas["erro_padrao"],
            "IC95 inferior": estatisticas["ic95_inferior"],
            "IC95 superior": estatisticas["ic95_superior"]
        })

    # ========================================================
    # SALVAR RESUMO EM CSV
    # ========================================================

    if resultados_resumo:

        df_resumo = pd.DataFrame(
            resultados_resumo
        )

        df_resumo.to_csv(
            ARQUIVO_SAIDA,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print("=" * 78)
        print("RESUMO GERAL")
        print("=" * 78)

        print()

        for resultado in resultados_resumo:

            print(
                f"{resultado['Configuração']:<18} | "
                f"Média: "
                f"{formatar_numero(resultado['Média final'], 1):>10} | "
                f"DP: "
                f"{formatar_numero(resultado['Desvio-padrão'], 2):>9}"
            )

        print()
        print(
            f"Arquivo de resumo salvo em: "
            f"{ARQUIVO_SAIDA}"
        )

    # ========================================================
    # CONFERÊNCIA DA MELHOR CONFIGURAÇÃO DE CADA MÉTODO
    # ========================================================

    if resultados_resumo:

        df_resultados = pd.DataFrame(
            resultados_resumo
        )

        for algoritmo in ["AG", "SA"]:

            grupo = df_resultados[
                df_resultados["Algoritmo"]
                == algoritmo
            ]

            if grupo.empty:
                continue

            indice_melhor = (
                grupo["Média final"].idxmax()
            )

            melhor = df_resultados.loc[
                indice_melhor
            ]

            print()
            print(
                f"Melhor configuração do {algoritmo}: "
                f"{melhor['Configuração']}"
            )

            print(
                f"Média final: "
                f"{formatar_numero(melhor['Média final'], 1)}"
            )

            print(
                f"Desvio-padrão: "
                f"{formatar_numero(melhor['Desvio-padrão'], 2)}"
            )

    print()
    print("=" * 78)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()