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


def solve_branch_and_bound(
    dist_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    initial_bound: Optional[float] = None,
    strategy_name: str = "cold_start",
    timeout_sec: float = 1800.0,
    track_memory: bool = False,
) -> BnBResult:
    """Resolve o TSP via Branch-and-Bound (DFS + bitmask + lower bound O(1)).

    O lower bound admissível é dado por:
        LB = g + 0.5 * (min1[0] + min1[curr] + sum_{u não visitado} (min1[u] + min2[u]))
    mantido incrementalmente em O(1) por nó, sem cópia de matrizes.
    """
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
