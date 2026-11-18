"""Algoritmo exato Branch-and-Bound para o Problema do Caixeiro Viajante."""

import math
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.profiling import measure_subroutine, track_peak_memory
from src.types import BnBResult

_EPS = 1e-9
_TIMEOUT_CHECK_MASK = 1023  # Checa relógio a cada 1024 nós expandidos


def _compute_tour_cost(tour: List[int], dist_matrix: np.ndarray) -> float:
    n = len(tour)
    return float(sum(dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n)))


def _precompute_vertex_bounds(
    dist_matrix: np.ndarray,
) -> Tuple[List[float], List[float], List[List[int]]]:
    """Pré-computa m1[v] (menor aresta), m1[v]+m2[v] e vizinhos ordenados por custo."""
    n = dist_matrix.shape[0]
    min1: List[float] = []
    two_min_sum: List[float] = []
    ordered_neighbors: List[List[int]] = []

    for i in range(n):
        row = [(float(dist_matrix[i, j]), j) for j in range(n) if j != i]
        row.sort(key=lambda x: x[0])
        m1 = row[0][0]
        m2 = row[1][0] if len(row) > 1 else row[0][0]
        min1.append(m1)
        two_min_sum.append(m1 + m2)
        ordered_neighbors.append([j for _, j in row])

    return min1, two_min_sum, ordered_neighbors


def _mst_cost_subset(dist_list: List[List[float]], nodes: List[int]) -> float:
    """Calcula o custo da MST sobre um subconjunto de vértices via Prim denso O(k^2) sem cópia de matriz."""
    k = len(nodes)
    if k <= 1:
        return 0.0
    if k == 2:
        return dist_list[nodes[0]][nodes[1]]

    in_mst = [False] * k
    in_mst[0] = True
    row0 = dist_list[nodes[0]]
    min_edge = [row0[nodes[i]] for i in range(k)]
    total = 0.0

    for _ in range(k - 1):
        best_idx = -1
        best_w = math.inf
        for i in range(1, k):
            if not in_mst[i] and min_edge[i] < best_w:
                best_w = min_edge[i]
                best_idx = i

        if best_idx == -1:
            break

        in_mst[best_idx] = True
        total += best_w
        row_u = dist_list[nodes[best_idx]]
        for i in range(1, k):
            if not in_mst[i]:
                w = row_u[nodes[i]]
                if w < min_edge[i]:
                    min_edge[i] = w

    return total


def solve_branch_and_bound(
    dist_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    initial_bound: Optional[float] = None,
    strategy_name: str = "cold_start",
    timeout_sec: float = 1800.0,
    track_memory: bool = False,
    lower_bound_mode: str = "two_min",
) -> BnBResult:
    """Resolve o TSP via Branch-and-Bound (DFS + bitmask).

    Modos de lower bound admissível (`lower_bound_mode`):
        - ``"two_min"``: O(1) incremental via soma das duas menores arestas.
        - ``"mst"``: O(k^2) via MST nos vértices não visitados + 2 arestas de conexão.
    """
    if lower_bound_mode not in ("two_min", "mst"):
        raise ValueError(f"lower_bound_mode inválido: {lower_bound_mode!r}. Use 'two_min' ou 'mst'.")
    n = dist_matrix.shape[0]
    timings: Dict[str, float] = {}

    if n < 2:
        return BnBResult(
            best_tour=list(range(n)),
            best_cost=0.0,
            elapsed_time=0.0,
            timed_out=False,
            nodes_expanded=1,
            nodes_pruned=0,
            max_depth=0,
            initial_bound=0.0,
            strategy=strategy_name,
            subroutine_times=timings,
            peak_memory_bytes=0,
        )

    best_tour: List[int] = list(initial_tour) if initial_tour is not None else []
    if initial_bound is not None:
        best_cost = float(initial_bound)
    elif initial_tour is not None:
        best_cost = _compute_tour_cost(best_tour, dist_matrix)
    else:
        best_cost = math.inf

    starting_bound = best_cost

    # Caso base N == 2
    if n == 2:
        cost_01 = float(dist_matrix[0, 1] + dist_matrix[1, 0])
        if cost_01 < best_cost - _EPS:
            best_cost = cost_01
            best_tour = [0, 1]
        return BnBResult(
            best_tour=best_tour,
            best_cost=best_cost,
            elapsed_time=0.0,
            timed_out=False,
            nodes_expanded=1,
            nodes_pruned=0,
            max_depth=1,
            initial_bound=starting_bound,
            strategy=strategy_name,
            subroutine_times=timings,
            peak_memory_bytes=0,
        )

    with track_peak_memory(enabled=track_memory) as mem:
        start_time = time.perf_counter()

        with measure_subroutine(timings, "precompute_bounds"):
            min1, two_min_sum, ordered_neighbors = _precompute_vertex_bounds(dist_matrix)
            # Lista de listas Python puro para acesso O(1) sem overhead de indexação NumPy
            dist_list: List[List[float]] = dist_matrix.tolist()

        nodes_expanded = 0
        nodes_pruned = 0
        max_depth = 0
        timed_out = False

        path: List[int] = [0]
        initial_rem_sum = sum(two_min_sum[1:])

        def dfs(curr: int, depth: int, visited_mask: int, g_cost: float, rem_two_min: float) -> None:
            nonlocal best_cost, best_tour, nodes_expanded, nodes_pruned, max_depth, timed_out

            if timed_out:
                return

            if depth > max_depth:
                max_depth = depth

            if depth == n - 1:
                total_cost = g_cost + dist_list[curr][0]
                if total_cost < best_cost - _EPS:
                    best_cost = total_cost
                    best_tour = path.copy()
                return

            nodes_expanded += 1
            if (nodes_expanded & _TIMEOUT_CHECK_MASK) == 0:
                if time.perf_counter() - start_time >= timeout_sec:
                    timed_out = True
                    return

            row_curr = dist_list[curr]
            if lower_bound_mode == "two_min":
                for nxt in ordered_neighbors[curr]:
                    if visited_mask & (1 << nxt):
                        continue

                    new_g = g_cost + row_curr[nxt]
                    new_rem = rem_two_min - two_min_sum[nxt]
                    lb = new_g + 0.5 * (min1[0] + min1[nxt] + new_rem)

                    if lb >= best_cost - _EPS:
                        nodes_pruned += 1
                        continue

                    path.append(nxt)
                    dfs(nxt, depth + 1, visited_mask | (1 << nxt), new_g, new_rem)
                    path.pop()

                    if timed_out:
                        return
            else:
                row0 = dist_list[0]
                for nxt in ordered_neighbors[curr]:
                    nxt_bit = 1 << nxt
                    if visited_mask & nxt_bit:
                        continue

                    new_g = g_cost + row_curr[nxt]
                    if new_g >= best_cost - _EPS:
                        nodes_pruned += 1
                        continue

                    next_mask = visited_mask | nxt_bit
                    unvisited = [v for v in range(1, n) if not (next_mask & (1 << v))]

                    if not unvisited:
                        lb = new_g + dist_list[nxt][0]
                    else:
                        row_nxt = dist_list[nxt]
                        min_from_nxt = min(row_nxt[u] for u in unvisited)
                        min_to_zero = min(row0[u] for u in unvisited)
                        lb = new_g + min_from_nxt + _mst_cost_subset(dist_list, unvisited) + min_to_zero

                    if lb >= best_cost - _EPS:
                        nodes_pruned += 1
                        continue

                    path.append(nxt)
                    dfs(nxt, depth + 1, next_mask, new_g, 0.0)
                    path.pop()

                    if timed_out:
                        return

        with measure_subroutine(timings, "search_loop"):
            dfs(curr=0, depth=0, visited_mask=1, g_cost=0.0, rem_two_min=initial_rem_sum)

        total_elapsed = time.perf_counter() - start_time

    return BnBResult(
        best_tour=best_tour,
        best_cost=best_cost,
        elapsed_time=None if timed_out else total_elapsed,
        timed_out=timed_out,
        nodes_expanded=nodes_expanded,
        nodes_pruned=nodes_pruned,
        max_depth=max_depth,
        initial_bound=starting_bound,
        strategy=strategy_name,
        subroutine_times=timings,
        peak_memory_bytes=mem.peak_bytes,
    )
