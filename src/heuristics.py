"""Heurísticas geradoras de solução inicial (warm-start) para o Branch-and-Bound."""

import math
import time
from typing import Dict, List
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

    with track_peak_memory(enabled=track_memory) as mem:
        start_time = time.perf_counter()

        with measure_subroutine(timings, "tour_construction"):
            tour = list(range(n))

        with measure_subroutine(timings, "tour_cost"):
            cost = _compute_tour_cost(tour, matrix)

        elapsed = time.perf_counter() - start_time

    return TourResult(
        tour=tour,
        cost=cost,
        elapsed_time=elapsed,
        subroutine_times=timings,
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
    with track_peak_memory(enabled=track_memory) as mem:
        start_time = time.perf_counter()

        with measure_subroutine(timings, "greedy_selection"):
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

        with measure_subroutine(timings, "tour_cost"):
            cost = _compute_tour_cost(tour, matrix)

        elapsed = time.perf_counter() - start_time

    return TourResult(
        tour=tour,
        cost=cost,
        elapsed_time=elapsed,
        subroutine_times=timings,
        peak_memory_bytes=mem.peak_bytes,
    )


def convex_hull_insertion(coords: np.ndarray, dist_matrix: np.ndarray) -> TourResult:
    """Heurística de inserção gulosa a partir do fecho convexo 2D."""
    raise NotImplementedError("Implementar convex_hull_insertion.")
