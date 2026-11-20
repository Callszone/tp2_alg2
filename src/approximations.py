"""Algoritmos aproximados para o TSP Métrico: Twice-around-the-tree e Christofides."""

from typing import Dict, List
import time
import numpy as np
import networkx as nx

from src.mst import compute_mst_prim
from src.profiling import measure_subroutine, track_peak_memory
from src.types import TourResult


def _preorder_dfs(adjacency: List[List[int]], root: int) -> List[int]:
    """Percurso em pré-ordem iterativo (pilha explícita) sobre a MST em O(N).

    Equivale ao passeio euleriano na MST duplicada com atalhos (shortcutting)
    sobre vértices já visitados. Empilha os vizinhos em ordem reversa para
    preservar a ordem canônica determinística de pré-ordem sem risco de
    estouro da pilha de recursão em árvores degeneradas.
    """
    n = len(adjacency)
    start = int(root)
    visited = [False] * n
    visited[start] = True
    stack: List[int] = [start]
    tour: List[int] = []

    while stack:
        u = stack.pop()
        tour.append(u)
        for v in reversed(adjacency[u]):
            if not visited[v]:
                visited[v] = True
                stack.append(v)

    return tour


def _compute_tour_cost(tour: List[int], dist_matrix: np.ndarray) -> float:
    """Calcula o custo total do ciclo hamiltoniano fechado em O(N)."""
    if len(tour) <= 1:
        return 0.0

    cost = 0.0
    prev = tour[-1]
    for curr in tour:
        cost += float(dist_matrix[prev, curr])
        prev = curr
    return cost


def twice_around_the_tree(
    dist_matrix: np.ndarray,
    root: int = 0,
    track_memory: bool = False,
) -> TourResult:
    """Algoritmo aproximado 2-aproximado baseado em MST + percurso em pré-ordem.

    Etapas instrumentadas em `subroutine_times`:
        - ``mst``: construção da MST via Prim denso em O(N^2) tempo / O(N) espaço.
        - ``preorder_dfs``: percurso em pré-ordem iterativo na MST em O(N).
        - ``tour_cost``: avaliação do custo do ciclo hamiltoniano fechado em O(N).

    Args:
        dist_matrix: Matriz simétrica de distâncias de formato (N, N).
        root: Vértice raiz da MST e ponto inicial do percurso (padrão: 0).
        track_memory: Se True, mede `peak_memory_bytes` via `tracemalloc`.
            Mantido False por padrão para não contaminar medições de tempo.

    Returns:
        TourResult com rota válida (0..N-1 sem repetir o início), custo total,
        tempo total, tempos por subrotina e pico de memória.
    """
    matrix = np.asarray(dist_matrix)
    timings: Dict[str, float] = {}

    with track_peak_memory(enabled=track_memory) as mem:
        total_start = time.perf_counter()

        with measure_subroutine(timings, "mst"):
            mst = compute_mst_prim(matrix, root=root)

        with measure_subroutine(timings, "preorder_dfs"):
            tour = _preorder_dfs(mst.adjacency, root=root)

        with measure_subroutine(timings, "tour_cost"):
            cost = _compute_tour_cost(tour, matrix)

        elapsed_time = time.perf_counter() - total_start

    return TourResult(
        tour=tour,
        cost=cost,
        elapsed_time=elapsed_time,
        subroutine_times=timings,
        peak_memory_bytes=mem.peak_bytes,
    )


def christofides(
    dist_matrix: np.ndarray,
    track_memory: bool = False,
) -> TourResult:
    """Algoritmo aproximado 1.5-aproximado (Christofides-Serdyukov).

    Garante que o custo do ciclo hamiltoniano seja no maximo 1.5 * OPT
    em espacos metricos (que satisfazem a desigualdade triangular).

    Etapas instrumentadas em `subroutine_times`:
        - ``mst``: arvore geradora minima via Prim denso em O(N^2).
        - ``odd_vertices``: identificacao dos vertices com grau impar na MST em O(N).
        - ``min_weight_matching``: emparelhamento perfeito de peso minimo via Blossom.
        - ``eulerian_circuit``: montagem do multigrafo euleriano e extracao do circuito.
        - ``shortcutting``: atalhos pulando vertices visitados para obter o ciclo.
        - ``tour_cost``: calculo do custo total do ciclo hamiltoniano fechado em O(N).

    Args:
        dist_matrix: Matriz simétrica de distâncias de formato (N, N).
        track_memory: Se True, mede `peak_memory_bytes` via `tracemalloc`.
            Mantido False por padrão para não contaminar medições de tempo.

    Returns:
        TourResult com rota válida (0..N-1 sem repetir o início), custo total,
        tempo total, tempos por subrotina e pico de memória.
    """
    matrix = np.asarray(dist_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"dist_matrix deve ser uma matriz quadrada (N, N); recebido shape={matrix.shape}."
        )

    n = matrix.shape[0]
    if n == 0:
        raise ValueError("dist_matrix não pode estar vazia.")

    if n == 1:
        return TourResult(
            tour=[0],
            cost=0.0,
            elapsed_time=0.0,
            subroutine_times={},
            peak_memory_bytes=0,
        )

    timings: Dict[str, float] = {}

    with track_peak_memory(enabled=track_memory) as mem:
        total_start = time.perf_counter()

        with measure_subroutine(timings, "mst"):
            mst = compute_mst_prim(matrix, root=0)

        with measure_subroutine(timings, "odd_vertices"):
            odd_vertices = [v for v, deg in enumerate(mst.degrees()) if deg % 2 != 0]

        with measure_subroutine(timings, "min_weight_matching"):
            subgraph = nx.Graph()
            for i, u in enumerate(odd_vertices):
                for w in odd_vertices[i + 1 :]:
                    subgraph.add_edge(u, w, weight=float(matrix[u, w]))
            matching = nx.min_weight_matching(subgraph, weight="weight")

        with measure_subroutine(timings, "eulerian_circuit"):
            multi_graph = nx.MultiGraph()
            for u, v in mst.edges():
                multi_graph.add_edge(u, v)
            for u, v in matching:
                multi_graph.add_edge(u, v)
            eulerian_edges = list(nx.eulerian_circuit(multi_graph, source=0))

        with measure_subroutine(timings, "shortcutting"):
            visited = [False] * n
            tour: List[int] = []
            for u, _ in eulerian_edges:
                if not visited[u]:
                    visited[u] = True
                    tour.append(u)

            if len(tour) < n:
                for v in range(n):
                    if not visited[v]:
                        tour.append(v)

            if tour and tour[0] != 0 and 0 in tour:
                idx0 = tour.index(0)
                tour = tour[idx0:] + tour[:idx0]

        with measure_subroutine(timings, "tour_cost"):
            cost = _compute_tour_cost(tour, matrix)

        elapsed_time = time.perf_counter() - total_start

    return TourResult(
        tour=tour,
        cost=cost,
        elapsed_time=elapsed_time,
        subroutine_times=timings,
        peak_memory_bytes=mem.peak_bytes,
    )
