"""
Simulated Annealing (SA) para o Problema da Mochila 0-1 (KP01)
================================================================

Ponto de partida baseado em:

- Estrutura geral de projeto da instância (pesos, valores, capacidade):
  inspirado no repositório "Knapsack-Simulated-Annealing" (JVictor011, GitHub).

- Lógica do algoritmo SA em si (Algoritmo 1, geração de solução inicial,
  operador de vizinhança, reparo de soluções infactíveis e critério de
  aceitação de Boltzmann):
  MORADI, N.; KAYVANFAR, V.; RAFIEE, M. "An efficient population-based
  simulated annealing algorithm for 0-1 knapsack problem". Engineering
  with Computers, 2021. (Algoritmos 1 a 7 do artigo)

- Discussão de paralelismo / múltiplas cadeias de Markov como extensão futura:
  LEMOS, D. V. X.; LONGO, H. J. "Uso de GPUs na resolução do Problema da
  Mochila Multidimensional" (revisão sobre SA paralelo para MKP/KP).

O código foi escrito para deixar EXPLÍCITOS os pontos que você deve variar
e justificar no relatório:
    - método de geração da solução inicial (RISP vs GISP)
    - operador de vizinhança (quantidade de bits "flipados", m)
    - método de reparo/melhoria de soluções infactíveis (Zhan_RI)
    - parâmetros do cronograma de temperatura (T_max, T_min, alpha, max_iter)
"""

from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional


# ---------------------------------------------------------------------------
# 1. Representação da instância do problema
# ---------------------------------------------------------------------------

@dataclass
class KnapsackInstance:
    """Representa uma instância do KP01.

    weights[i] e profits[i] são o peso e o lucro do item i.
    capacity é a capacidade máxima da mochila (C no artigo, Eq. 1).
    """
    weights: List[float]
    profits: List[float]
    capacity: float

    def __post_init__(self):
        assert len(self.weights) == len(self.profits)
        self.n = len(self.weights)
        # v_i = p_i / w_i -> "density metric" (métrica de densidade),
        # usada no GISP e nos operadores de reparo (Seção 2, do artigo).
        self.density = [self.profits[i] / self.weights[i] for i in range(self.n)]
        # índices ordenados do MENOR para o MAIOR v_i (usado nos Algoritmos 3, 5 e 6)
        self.order_by_density = sorted(range(self.n), key=lambda i: self.density[i])

    @staticmethod
    def random_instance(
        n: int,
        capacity_ratio: float = 0.75,
        correlation: str = "uncorrelated",
        seed: Optional[int] = None,
    ) -> "KnapsackInstance":
        """Gera uma instância aleatória seguindo a Tabela 7 do artigo do Moradi et al.

        correlation:
            "uncorrelated"      -> p_i ~ U(10,100), w_i ~ U(10,100)
            "weakly_correlated" -> p_i ~ U(w_i-10, w_i+10), w_i ~ U(10,100)
            "strongly_correlated" -> p_i = w_i + 10, w_i ~ U(10,100)

        capacity_ratio: C = capacity_ratio * soma dos pesos (0.75 no artigo)
        """
        rng = random.Random(seed)
        weights = [rng.uniform(10, 100) for _ in range(n)]

        if correlation == "uncorrelated":
            profits = [rng.uniform(10, 100) for _ in range(n)]
        elif correlation == "weakly_correlated":
            profits = [rng.uniform(w - 10, w + 10) for w in weights]
        elif correlation == "strongly_correlated":
            profits = [w + 10 for w in weights]
        else:
            raise ValueError(f"correlation desconhecida: {correlation}")

        capacity = capacity_ratio * sum(weights)
        return KnapsackInstance(weights, profits, capacity)


def load_instance_pisinger_format(path):
    """Carrega uma instância no formato usado pelos geradores de instâncias
    "difíceis" de Pisinger (comum em benchmarks de KP01), como o arquivo
    test.in:

        n
        id_1  profit_1  weight_1
        id_2  profit_2  weight_2
        ...
        id_n  profit_n  weight_n
        capacity

    Os ids (1..n) são apenas identificadores e não são usados; a ordem das
    linhas define a ordem dos itens no vetor de solução.
    """
    with open(path, "r") as f:
        tokens = f.read().split()

    idx = 0
    n = int(tokens[idx]); idx += 1

    weights = []
    profits = []
    for _ in range(n):
        idx += 1  # pula o id do item (não usado)
        profit = float(tokens[idx]); idx += 1
        weight = float(tokens[idx]); idx += 1
        profits.append(profit)
        weights.append(weight)

    capacity = float(tokens[idx]); idx += 1

    return KnapsackInstance(weights=weights, profits=profits, capacity=capacity)


Solution = List[int]  # vetor binário X_i em {0,1}


def total_weight(sol: Solution, inst: KnapsackInstance) -> float:
    return sum(inst.weights[i] for i, x in enumerate(sol) if x == 1)


def fitness(sol: Solution, inst: KnapsackInstance) -> float:
    """Função objetivo: soma dos lucros dos itens selecionados (Eq. 1)."""
    return sum(inst.profits[i] for i, x in enumerate(sol) if x == 1)


def is_feasible(sol: Solution, inst: KnapsackInstance) -> bool:
    return total_weight(sol, inst) <= inst.capacity


# ---------------------------------------------------------------------------
# 2. Geração da solução inicial: RISP e GISP (Algoritmos 2 e 3 do artigo)
# ---------------------------------------------------------------------------

def risp_initial_solution(inst: KnapsackInstance, rng: random.Random) -> Solution:
    """Random Initial Solution Phase: cada bit é sorteado aleatoriamente.
    É rápido, mas normalmente gera uma solução infactível ou de baixa
    qualidade -> precisa de reparo em seguida."""
    return [rng.randint(0, 1) for _ in range(inst.n)]


def gisp_initial_solution(inst: KnapsackInstance) -> Solution:
    """Greedy Initial Solution Phase (Algoritmo 3): ordena os itens por
    densidade v_i = p_i/w_i (decrescente) e vai inserindo enquanto não
    ultrapassar a capacidade. Gera soluções de melhor qualidade que o RISP,
    ao custo de menos diversidade."""
    sol = [0] * inst.n
    order_desc = list(reversed(inst.order_by_density))  # do maior p/w pro menor
    current_weight = 0.0
    for idx in order_desc:
        if current_weight + inst.weights[idx] <= inst.capacity:
            sol[idx] = 1
            current_weight += inst.weights[idx]
    return sol


def generate_initial_solution(
    inst: KnapsackInstance, method: str, rng: random.Random
) -> Solution:
    if method == "RISP":
        sol = risp_initial_solution(inst, rng)
        sol = repair_and_improve(sol, inst)  # RISP quase sempre precisa reparo
        return sol
    elif method == "GISP":
        return gisp_initial_solution(inst)
    else:
        raise ValueError("method deve ser 'RISP' ou 'GISP'")


# ---------------------------------------------------------------------------
# 3. Reparo e melhoria de soluções infactíveis: Zhan_RI (Algoritmos 5 e 6)
# ---------------------------------------------------------------------------

def repair(sol: Solution, inst: KnapsackInstance) -> Solution:
    """Algoritmo 5: remove itens com MENOR p/w enquanto a mochila estiver
    acima da capacidade."""
    sol = sol.copy()
    weight = total_weight(sol, inst)
    for idx in inst.order_by_density:  # do menor p/w pro maior
        if weight <= inst.capacity:
            break
        if sol[idx] == 1:
            sol[idx] = 0
            weight -= inst.weights[idx]
    return sol


def improve(sol: Solution, inst: KnapsackInstance) -> Solution:
    """Algoritmo 6: tenta adicionar itens com MAIOR p/w enquanto não
    ultrapassar a capacidade (melhora soluções factíveis "frouxas")."""
    sol = sol.copy()
    weight = total_weight(sol, inst)
    for idx in reversed(inst.order_by_density):  # do maior p/w pro menor
        if sol[idx] == 0 and weight + inst.weights[idx] <= inst.capacity:
            sol[idx] = 1
            weight += inst.weights[idx]
    return sol


def repair_and_improve(sol: Solution, inst: KnapsackInstance) -> Solution:
    """Zhan_RI: repara (se infactível) e depois melhora (Figs. 5 e 6)."""
    if not is_feasible(sol, inst):
        sol = repair(sol, inst)
    sol = improve(sol, inst)
    return sol


# ---------------------------------------------------------------------------
# 4. Operador de vizinhança: m-flipping (Algoritmo 7)
# ---------------------------------------------------------------------------

def m_flip_neighbor(sol: Solution, m: int, rng: random.Random) -> Solution:
    """Sorteia m posições e inverte o bit (0->1 ou 1->0).
    m é um parâmetro a ser calibrado: valores pequenos (m=1,2,3) fazem
    busca local mais "fina"; valores maiores tornam a busca mais "global"
    (mais próxima de uma nova solução aleatória), aumentando diversificação
    mas podendo prejudicar a convergência."""
    sol = sol.copy()
    n = len(sol)
    positions = rng.sample(range(n), k=min(m, n))
    for j in positions:
        sol[j] = 1 - sol[j]
    return sol


# ---------------------------------------------------------------------------
# 5. Simulated Annealing (Algoritmo 1)
# ---------------------------------------------------------------------------

@dataclass
class SAResult:
    best_solution: Solution
    best_fitness: float
    history: List[Tuple[float, float]] = field(default_factory=list)
    # history: lista de (temperatura, melhor_fitness_até_o_momento)
    # útil para reproduzir gráficos de convergência como as Figs. 12-15 do artigo


def simulated_annealing(
    inst: KnapsackInstance,
    T_max: float = 1000.0,
    T_min: float = 0.0001,
    alpha: float = 0.98,
    max_iteration: int = 100,
    m: int = 2,
    initial_method: str = "GISP",
    seed: Optional[int] = None,
) -> SAResult:
    """Implementação do Algoritmo 1 (SSA) do artigo do Moradi et al.

    Parâmetros a variar/justificar no relatório:
        T_max, T_min, alpha  -> cronograma de temperatura (cooling schedule)
        max_iteration        -> nº de iterações por temperatura fixa
        m                     -> "força" do operador de vizinhança (m-flip)
        initial_method        -> "RISP" (aleatório) ou "GISP" (guloso)
    """
    rng = random.Random(seed)

    s = generate_initial_solution(inst, initial_method, rng)
    f_s = fitness(s, inst)

    best_sol, best_fit = s.copy(), f_s
    history: List[Tuple[float, float]] = [(T_max, best_fit)]

    T = T_max
    while T >= T_min:
        for _ in range(max_iteration):
            s_prime = m_flip_neighbor(s, m, rng)
            s_prime = repair_and_improve(s_prime, inst)
            f_s_prime = fitness(s_prime, inst)

            delta = f_s - f_s_prime  # delta_E = f(s') - f(s), mas maximizamos
            # (invertemos o sinal em relação ao artigo, que minimiza por
            # convenção do pseudocódigo genérico de SA)
            if f_s_prime >= f_s:
                s, f_s = s_prime, f_s_prime  # aceita solução melhor/igual
            else:
                # aceita solução pior com probabilidade e^{-delta/T}
                # (critério de Boltzmann, linha 11 do Algoritmo 1)
                prob = math.exp(-delta / T) if T > 0 else 0.0
                if rng.random() < prob:
                    s, f_s = s_prime, f_s_prime

            if f_s > best_fit:
                best_sol, best_fit = s.copy(), f_s

        history.append((T, best_fit))
        T *= alpha  # cronograma de resfriamento geométrico T = alpha * T

    return SAResult(best_solution=best_sol, best_fitness=best_fit, history=history)


# ---------------------------------------------------------------------------
# 6. Simulated Annealing com LOG detalhado (todas as iterações/passos)
# ---------------------------------------------------------------------------

def simulated_annealing_with_log(
    inst: KnapsackInstance,
    T_max: float = 1000.0,
    T_min: float = 0.0001,
    alpha: float = 0.98,
    max_iteration: int = 100,
    m: int = 2,
    initial_method: str = "GISP",
    seed: Optional[int] = None,
    log_path: Optional[str] = None,
):
    """Versão do SA (idêntica em lógica à `simulated_annealing`) que registra,
    para CADA iteração/passo executado, os dados necessários para auditar
    o comportamento do algoritmo:

        step                 -> número sequencial do passo (1, 2, 3, ...)
        temperature          -> temperatura T no momento do passo
        iteration_in_temp    -> qual iteração dentro do nível de temperatura atual
        current_fitness      -> fitness da solução corrente (s) ANTES do passo
        candidate_fitness    -> fitness da solução vizinha gerada (s')
        delta                -> f(s) - f(s') (positivo = candidata é pior)
        acceptance_prob      -> probabilidade de aceitação usada (1.0 se candidata
                                 é melhor/igual; e^{-delta/T} caso contrário)
        accepted             -> se a solução candidata foi aceita neste passo
        best_fitness_so_far  -> melhor fitness encontrado até este passo (inclusive)

    Se `log_path` for informado, o log completo é salvo em um arquivo CSV
    nesse caminho. Retorna (SAResult, log), onde `log` é uma lista de dicts
    (um por passo), útil para gerar gráficos de convergência ou depurar o
    comportamento do algoritmo (ex.: taxa de aceitação por temperatura).
    """
    import csv

    rng = random.Random(seed)

    s = generate_initial_solution(inst, initial_method, rng)
    f_s = fitness(s, inst)

    best_sol, best_fit = s.copy(), f_s
    log: List[dict] = []
    step = 0

    T = T_max
    while T >= T_min:
        for it in range(max_iteration):
            step += 1
            s_prime = m_flip_neighbor(s, m, rng)
            s_prime = repair_and_improve(s_prime, inst)
            f_s_prime = fitness(s_prime, inst)

            delta = f_s - f_s_prime

            if f_s_prime >= f_s:
                accepted = True
                acceptance_prob = 1.0
                s, f_s = s_prime, f_s_prime
            else:
                acceptance_prob = math.exp(-delta / T) if T > 0 else 0.0
                accepted = rng.random() < acceptance_prob
                if accepted:
                    s, f_s = s_prime, f_s_prime

            if f_s > best_fit:
                best_sol, best_fit = s.copy(), f_s

            log.append({
                "step": step,
                "temperature": T,
                "iteration_in_temp": it + 1,
                "current_fitness": f_s,
                "candidate_fitness": f_s_prime,
                "delta": delta,
                "acceptance_prob": acceptance_prob,
                "accepted": accepted,
                "best_fitness_so_far": best_fit,
            })

        T *= alpha

    result = SAResult(best_solution=best_sol, best_fitness=best_fit)

    if log_path is not None:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log[0].keys()))
            writer.writeheader()
            writer.writerows(log)

    return result, log


# ---------------------------------------------------------------------------
# 7. Simulated Annealing "verbose" (log no formato pedido pelo professor)
# ---------------------------------------------------------------------------

def simulated_annealing_verbose(
    inst: KnapsackInstance,
    T_max: float = 1000.0,
    T_min: float = 0.0001,
    alpha: float = 0.98,
    max_iteration: int = 100,
    m: int = 2,
    initial_method: str = "GISP",
    seed: Optional[int] = None,
    print_every: int = 100,
):
    """Versão do SA que imprime o progresso no formato:

        SA (Orçamento: <N> avaliações)...
        Ciclo 0000 | Temp: 200.00 | Melhor Custo: 18844.0 | Tempo Decorrido: 0.00s
        Ciclo 0100 | Temp: 1.18   | Melhor Custo: 18844.0 | Tempo Decorrido: 1.20s
        ...

    Cada "Ciclo" corresponde a um nível de temperatura (isto é, ao laço
    externo `while T >= T_min`), da mesma forma que uma "Geração" no GA
    corresponde a uma população inteira avaliada. Dentro de cada ciclo,
    `max_iteration` vizinhos são avaliados antes de resfriar (T *= alpha).

    IMPORTANTE: o KP01 é um problema de MAXIMIZAÇÃO (queremos o MAIOR lucro
    possível), então "Melhor Custo" aqui tende a CRESCER ao longo dos ciclos
    -- ao contrário de problemas de minimização (ex.: os logs de exemplo do
    GA/TSP do professor), onde o custo tende a CAIR. Se o professor exigir
    que o rótulo/tendência seja literalmente "Custo" decrescente, basta
    reportar o negativo do fitness (custo = -melhor_fitness) na impressão.
    """
    import time

    rng = random.Random(seed)

    s = generate_initial_solution(inst, initial_method, rng)
    f_s = fitness(s, inst)
    best_sol, best_fit = s.copy(), f_s

    # "Orçamento" análogo ao do GA: total de avaliações de vizinhos que serão
    # feitas ao longo de toda a execução (ciclos de resfriamento x iterações).
    n_cycles_estimate = max(1, int(math.log(T_min / T_max) / math.log(alpha)))
    total_evaluations = n_cycles_estimate * max_iteration
    print(f"Iniciando SA (Orçamento: {total_evaluations} avaliações)...")

    t_start = time.time()
    log: List[dict] = []
    cycle = 0

    T = T_max
    while T >= T_min:
        for _ in range(max_iteration):
            s_prime = m_flip_neighbor(s, m, rng)
            s_prime = repair_and_improve(s_prime, inst)
            f_s_prime = fitness(s_prime, inst)

            delta = f_s - f_s_prime
            if f_s_prime >= f_s:
                s, f_s = s_prime, f_s_prime
            else:
                prob = math.exp(-delta / T) if T > 0 else 0.0
                if rng.random() < prob:
                    s, f_s = s_prime, f_s_prime

            if f_s > best_fit:
                best_sol, best_fit = s.copy(), f_s

        elapsed = time.time() - t_start
        log.append({
            "cycle": cycle,
            "temperature": T,
            "best_cost": best_fit,
            "elapsed_time": elapsed,
        })

        if cycle % print_every == 0:
            print(
                f"Ciclo {cycle:04d} | Temp: {T:10.2f} | "
                f"Melhor Custo: {best_fit:.1f} | Tempo Decorrido: {elapsed:.2f}s"
            )

        cycle += 1
        T *= alpha

    # garante que o último ciclo também apareça no log/print, mesmo que não
    # seja múltiplo de print_every
    elapsed = time.time() - t_start
    print(
        f"Ciclo {cycle - 1:04d} | Temp: {T / alpha:10.2f} | "
        f"Melhor Custo: {best_fit:.1f} | Tempo Decorrido: {elapsed:.2f}s  (final)"
    )

    result = SAResult(best_solution=best_sol, best_fitness=best_fit)
    return result, log


# ---------------------------------------------------------------------------
# 8. Exemplo de uso / teste rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Instância pequena "de bolso" para checar rapidamente que está funcionando
    weights = [2, 3, 4, 5, 9]
    profits = [3, 4, 5, 8, 10]
    capacity = 10
    inst = KnapsackInstance(weights, profits, capacity)

    result = simulated_annealing(
        inst,
        T_max=1000,
        T_min=0.001,
        alpha=0.95,
        max_iteration=50,
        m=1,
        initial_method="GISP",
        seed=42,
    )

    print("Melhor solução:", result.best_solution)
    print("Melhor valor (fitness):", result.best_fitness)
    print("Peso total:", total_weight(result.best_solution, inst), "/", capacity)

    # Instância maior e aleatória, para testes de calibração de parâmetros
    big_inst = KnapsackInstance.random_instance(
        n=200, capacity_ratio=0.75, correlation="uncorrelated", seed=1
    )
    result_big = simulated_annealing(big_inst, seed=1)
    print("\nInstância aleatória (n=200):")
    print("Melhor valor encontrado:", result_big.best_fitness)

    # ---------------------------------------------------------------------
    # Carregando a instância real do arquivo test.in (formato Pisinger)
    # ---------------------------------------------------------------------
    import time

    # test_inst = KnapsackInstance.random_instance(200)
    test_inst = load_instance_pisinger_format("./test.in")
    print(f"\nInstância test.in: n={test_inst.n}, capacidade={test_inst.capacity}")

    t0 = time.time()
    result_test = simulated_annealing(
        test_inst,
        T_max=200,
        T_min=0.001,
        alpha=0.95,
        max_iteration=60,
        m=4,
        initial_method="GISP",
        seed=1,
    )
    elapsed = time.time() - t0

    print("\nMelhor valor encontrado:", result_test.best_fitness)
    print(
        "Peso total usado:",
        total_weight(result_test.best_solution, test_inst),
        "/",
        test_inst.capacity,
    )
    print(f"Tempo de execução: {elapsed:.2f}s")

    # ---------------------------------------------------------------------
    # Gerando o LOG completo de execução (todas as iterações), salvo em CSV
    # ---------------------------------------------------------------------
    result_logged, log = simulated_annealing_with_log(
        test_inst,
        T_max=200,
        T_min=0.001,
        alpha=0.95,
        max_iteration=60,
        m=4,
        initial_method="GISP",
        seed=1,
        log_path="sa_log.csv",
    )

    print(f"\nLog gerado com {len(log)} passos (salvo em sa_log.csv)")
    print("Melhor valor (via versão com log):", result_logged.best_fitness)
    print("\nPrimeiros 3 passos do log:")
    for row in log[:3]:
        print(row)
    print("\nÚltimos 3 passos do log:")
    for row in log[-3:]:
        print(row)

    # ---------------------------------------------------------------------
    # Versão "verbose" (formato Ciclo/Temp/Melhor Custo/Tempo Decorrido)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    result_verbose, cycles_log = simulated_annealing_verbose(
        test_inst,
        T_max=200,
        T_min=0.001,
        alpha=0.95,
        max_iteration=60,
        m=4,
        initial_method="GISP",
        seed=1,
        print_every=10,
    )