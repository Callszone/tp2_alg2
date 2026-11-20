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
    lower_bound_mode: str = "two_min",
) -> BnBResult:
    """Resolve o TSP via Branch-and-Bound (DFS + bitmask + quebra de simetria O(1)).

    Modos de lower bound admissível (`lower_bound_mode`):
        - ``"two_min"``: O(1) incremental via soma das duas menores arestas.
        - ``"mst"``: filtro rápido O(1) + O(k^2) via MST nos vértices não visitados
          com buffers pré-alocados (zero alocação de listas por nó).
    """
    if lower_bound_mode not in ("two_min", "mst"):
        raise ValueError(f"lower_bound_mode inválido: {lower_bound_mode!r}. Use 'two_min' ou 'mst'.")

    n = dist_matrix.shape[0]
    timings: Dict[str, float] = {}
    memory_peaks: Dict[str, int] = {}

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
            subroutine_memory_bytes=memory_peaks,
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
            subroutine_memory_bytes=memory_peaks,
            peak_memory_bytes=0,
        )

    with track_peak_memory(enabled=track_memory) as mem:
        start_time = time.perf_counter()

        with measure_subroutine(timings, "precompute_bounds", memory_peaks):
            min1, two_min_sum, ordered_neighbors = _precompute_vertex_bounds(dist_matrix)
            dist_list: List[List[float]] = dist_matrix.tolist()
            is_symmetric = bool(np.allclose(dist_matrix, dist_matrix.T, atol=1e-9))
            all_visited_mask = (1 << n) - 1
            # Para cada v1 in 0..N-1, máscara com todos os vértices > v1
            greater_masks = [all_visited_mask & ~((1 << (v + 1)) - 1) for v in range(n)]

        nodes_expanded = 0
        nodes_pruned = 0
        max_depth = 0
        timed_out = False

        path: List[int] = [0]
        initial_rem_sum = sum(two_min_sum[1:])

        # Buffers reutilizáveis para Prim em lower_bound_mode == "mst" (evita alocação por nó)
        unvisited_buf = [0] * n
        in_mst_buf = [False] * n
        min_edge_buf = [0.0] * n

        def mst_lb_preallocated(next_mask: int, nxt: int) -> float:
            k = 0
            for v in range(1, n):
                if not (next_mask & (1 << v)):
                    unvisited_buf[k] = v
                    k += 1

            if k == 0:
                return dist_list[nxt][0]

            row_nxt = dist_list[nxt]
            row0 = dist_list[0]
            if k == 1:
                u0 = unvisited_buf[0]
                return row_nxt[u0] + row0[u0]

            min_from_nxt = math.inf
            min_to_zero = math.inf
            first_u = unvisited_buf[0]
            row_first = dist_list[first_u]

            for i in range(k):
                u = unvisited_buf[i]
                d_nxt = row_nxt[u]
                if d_nxt < min_from_nxt:
                    min_from_nxt = d_nxt
                d_zero = row0[u]
                if d_zero < min_to_zero:
                    min_to_zero = d_zero
                in_mst_buf[i] = False
                min_edge_buf[i] = row_first[u]

            in_mst_buf[0] = True
            mst_total = 0.0

            for _ in range(k - 1):
                best_idx = -1
                best_w = math.inf
                for i in range(1, k):
                    if not in_mst_buf[i] and min_edge_buf[i] < best_w:
                        best_w = min_edge_buf[i]
                        best_idx = i
                if best_idx == -1:
                    break
                in_mst_buf[best_idx] = True
                mst_total += best_w
                row_u = dist_list[unvisited_buf[best_idx]]
                for i in range(1, k):
                    if not in_mst_buf[i]:
                        w = row_u[unvisited_buf[i]]
                        if w < min_edge_buf[i]:
                            min_edge_buf[i] = w

            return min_from_nxt + mst_total + min_to_zero

        use_mst = lower_bound_mode == "mst"

        def dfs(
            curr: int,
            depth: int,
            visited_mask: int,
            g_cost: float,
            rem_two_min: float,
            gt_v1_mask: int,
        ) -> None:
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
                nxt_bit = 1 << nxt
                if visited_mask & nxt_bit:
                    continue

                next_mask = visited_mask | nxt_bit
                next_gt_mask = greater_masks[nxt] if depth == 0 else gt_v1_mask

                # Quebra de simetria O(1): exige primeiro vértice (v1) < último vértice (v_{N-1}).
                # Se ainda não estamos escolhendo o último vértice (next_mask != all_visited_mask)
                # e todos os vértices > v1 já foram visitados, nenhum fechamento terá v_{N-1} > v1.
                if (
                    is_symmetric
                    and next_mask != all_visited_mask
                    and (next_mask & next_gt_mask) == next_gt_mask
                ):
                    nodes_pruned += 1
                    continue

                new_g = g_cost + row_curr[nxt]
                new_rem = rem_two_min - two_min_sum[nxt]
                lb_fast = new_g + 0.5 * (min1[0] + min1[nxt] + new_rem)

                if lb_fast >= best_cost - _EPS:
                    nodes_pruned += 1
                    continue

                if use_mst:
                    lb_mst = new_g + mst_lb_preallocated(next_mask, nxt)
                    if lb_mst >= best_cost - _EPS:
                        nodes_pruned += 1
                        continue

                path.append(nxt)
                dfs(nxt, depth + 1, next_mask, new_g, new_rem, next_gt_mask)
                path.pop()

                if timed_out:
                    return

        with measure_subroutine(timings, "search_loop", memory_peaks):
            dfs(
                curr=0,
                depth=0,
                visited_mask=1,
                g_cost=0.0,
                rem_two_min=initial_rem_sum,
                gt_v1_mask=all_visited_mask,
            )

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
        subroutine_memory_bytes=memory_peaks,
        peak_memory_bytes=mem.peak_bytes,
    )
