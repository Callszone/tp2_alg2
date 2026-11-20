"""Heurísticas geradoras de solução inicial (warm-start) para o Branch-and-Bound."""

import math
import time
from typing import Dict, List, Tuple
import numpy as np

from src.profiling import measure_subroutine, track_peak_memory
from src.types import TourResult


def _validate_dist_matrix(dist_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(dist_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Esperada matriz quadrada (N, N), recebido {matrix.shape}.")
    return matrix


def _compute_tour_cost(tour: List[int], matrix: np.ndarray) -> float:
    if len(tour) <= 1:
        return 0.0
    cost = 0.0
    prev = tour[-1]
    for curr in tour:
        cost += float(matrix[prev, curr])
        prev = curr
    return cost


def cold_start(dist_matrix: np.ndarray, track_memory: bool = False) -> TourResult:
    """Solução trivial: 0 -> 1 -> 2 -> ... -> N-1 -> 0."""
    matrix = _validate_dist_matrix(dist_matrix)
    n = matrix.shape[0]
    timings: Dict[str, float] = {}
    memory_peaks: Dict[str, int] = {}

    with track_peak_memory(enabled=track_memory) as mem:
        start_time = time.perf_counter()

        with measure_subroutine(timings, "tour_construction", memory_peaks):
            tour = list(range(n))

        with measure_subroutine(timings, "tour_cost", memory_peaks):
            cost = _compute_tour_cost(tour, matrix)

        elapsed = time.perf_counter() - start_time

    return TourResult(
        tour=tour,
        cost=cost,
        elapsed_time=elapsed,
        subroutine_times=timings,
        subroutine_memory_bytes=memory_peaks,
        peak_memory_bytes=mem.peak_bytes,
    )


def nearest_neighbor(
    dist_matrix: np.ndarray,
    start: int = 0,
    track_memory: bool = False,
) -> TourResult:
    """Heurística gulosa do Vizinho Mais Próximo (Nearest Neighbor)."""
    matrix = _validate_dist_matrix(dist_matrix)
    n = matrix.shape[0]
    if n == 0:
        return TourResult(tour=[], cost=0.0, elapsed_time=0.0)
    if not (0 <= start < n):
        raise ValueError(f"Vértice inicial inválido: {start} (esperado 0 <= start < {n}).")

    timings: Dict[str, float] = {}
    memory_peaks: Dict[str, int] = {}
    with track_peak_memory(enabled=track_memory) as mem:
        start_time = time.perf_counter()

        with measure_subroutine(timings, "greedy_selection", memory_peaks):
            visited = [False] * n
            curr = int(start)
            visited[curr] = True
            tour: List[int] = [curr]

            for _ in range(n - 1):
                row = matrix[curr]
                best_next = -1
                best_dist = math.inf
                for v in range(n):
                    if not visited[v] and float(row[v]) < best_dist:
                        best_dist = float(row[v])
                        best_next = v
                visited[best_next] = True
                tour.append(best_next)
                curr = best_next

        with measure_subroutine(timings, "tour_cost", memory_peaks):
            cost = _compute_tour_cost(tour, matrix)

        elapsed = time.perf_counter() - start_time

    return TourResult(
        tour=tour,
        cost=cost,
        elapsed_time=elapsed,
        subroutine_times=timings,
        subroutine_memory_bytes=memory_peaks,
        peak_memory_bytes=mem.peak_bytes,
    )


def compute_convex_hull_indices(coords: np.ndarray) -> List[int]:
    """Calcula os índices do fecho convexo 2D via Andrew's Monotone Chain em O(N log N)."""
    n = coords.shape[0]
    if n <= 2:
        return list(range(n))

    pts: List[Tuple[float, float, int]] = [
        (float(coords[i, 0]), float(coords[i, 1]), i) for i in range(n)
    ]
    pts.sort(key=lambda p: (p[0], p[1], p[2]))

    def cross(
        o: Tuple[float, float, int],
        a: Tuple[float, float, int],
        b: Tuple[float, float, int],
    ) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float, int]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float, int]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = [p[2] for p in lower[:-1] + upper[:-1]]
    return hull if len(hull) >= 2 else [pts[0][2], pts[-1][2]]


# Alias para compatibilidade interna
_monotone_chain_hull = compute_convex_hull_indices


def convex_hull_insertion(
    coords: np.ndarray,
    dist_matrix: np.ndarray,
    track_memory: bool = False,
) -> TourResult:
    """Heurística de inserção gulosa (Cheapest Insertion) em O(N^2) a partir do fecho convexo 2D.

    Refatoração O(N^3) -> O(N^2): mantém para cada vértice restante k a melhor aresta
    de inserção (best_u[k], best_w[k], best_delta[k]). Quando k* é inserido na aresta
    (u*, w*), apenas vértices cuja melhor aresta era (u*, w*) reavaliam o ciclo; os
    demais testam apenas as 2 novas arestas (u*, k*) e (k*, w*) em O(1).
    """
    matrix = _validate_dist_matrix(dist_matrix)
    coords_arr = np.asarray(coords, dtype=np.float64)
    n = matrix.shape[0]
    if coords_arr.shape != (n, 2):
        raise ValueError(f"Esperado coords com shape ({n}, 2), recebido {coords_arr.shape}.")

    if n == 0:
        return TourResult(tour=[], cost=0.0, elapsed_time=0.0)
    if n <= 2:
        return cold_start(matrix, track_memory=track_memory)

    timings: Dict[str, float] = {}
    memory_peaks: Dict[str, int] = {}
    with track_peak_memory(enabled=track_memory) as mem:
        start_time = time.perf_counter()

        with measure_subroutine(timings, "convex_hull", memory_peaks):
            hull = compute_convex_hull_indices(coords_arr)

        with measure_subroutine(timings, "greedy_insertion", memory_peaks):
            dist_list = matrix.tolist()
            # Representa o ciclo como lista encadeada via array next_node[u] = w
            next_node = [-1] * n
            m = len(hull)
            for idx in range(m):
                next_node[hull[idx]] = hull[(idx + 1) % m]
            tour_vertices = list(hull)

            remaining = [v for v in range(n) if next_node[v] == -1]
            best_delta = [math.inf] * n
            best_u = [-1] * n
            best_w = [-1] * n

            def recompute_best_for(k: int) -> None:
                row_k = dist_list[k]
                b_delta = math.inf
                b_u = -1
                b_w = -1
                for u in tour_vertices:
                    w = next_node[u]
                    delta = dist_list[u][k] + row_k[w] - dist_list[u][w]
                    if delta < b_delta:
                        b_delta = delta
                        b_u = u
                        b_w = w
                best_delta[k] = b_delta
                best_u[k] = b_u
                best_w[k] = b_w

            for k in remaining:
                recompute_best_for(k)

            while remaining:
                chosen_idx = 0
                chosen_k = remaining[0]
                min_d = best_delta[chosen_k]
                for r_idx in range(1, len(remaining)):
                    cand = remaining[r_idx]
                    if best_delta[cand] < min_d:
                        min_d = best_delta[cand]
                        chosen_k = cand
                        chosen_idx = r_idx

                u_star = best_u[chosen_k]
                w_star = best_w[chosen_k]
                remaining.pop(chosen_idx)

                # Insere chosen_k entre u_star e w_star
                next_node[u_star] = chosen_k
                next_node[chosen_k] = w_star
                tour_vertices.append(chosen_k)

                d_u_k = dist_list[u_star][chosen_k]
                d_k_w = dist_list[chosen_k][w_star]
                row_u = dist_list[u_star]
                row_chosen = dist_list[chosen_k]

                for k in remaining:
                    if best_u[k] == u_star and best_w[k] == w_star:
                        recompute_best_for(k)
                    else:
                        row_cand = dist_list[k]
                        d1 = row_u[k] + row_cand[chosen_k] - d_u_k
                        if d1 < best_delta[k]:
                            best_delta[k] = d1
                            best_u[k] = u_star
                            best_w[k] = chosen_k
                        d2 = row_chosen[k] + row_cand[w_star] - d_k_w
                        if d2 < best_delta[k]:
                            best_delta[k] = d2
                            best_u[k] = chosen_k
                            best_w[k] = w_star

            # Reconstrói o ciclo a partir do vértice 0
            tour: List[int] = []
            curr = 0
            for _ in range(n):
                tour.append(curr)
                curr = next_node[curr]

        with measure_subroutine(timings, "tour_cost", memory_peaks):
            cost = _compute_tour_cost(tour, matrix)

        elapsed = time.perf_counter() - start_time

    return TourResult(
        tour=tour,
        cost=cost,
        elapsed_time=elapsed,
        subroutine_times=timings,
        subroutine_memory_bytes=memory_peaks,
        peak_memory_bytes=mem.peak_bytes,
    )
