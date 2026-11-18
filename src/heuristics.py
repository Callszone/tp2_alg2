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


def _monotone_chain_hull(coords: np.ndarray) -> List[int]:
    """Calcula os índices do fecho convexo 2D via Andrew's Monotone Chain em O(N log N)."""
    n = coords.shape[0]
    if n <= 2:
        return list(range(n))

    pts: List[Tuple[float, float, int]] = [
        (float(coords[i, 0]), float(coords[i, 1]), i) for i in range(n)
    ]
    pts.sort(key=lambda p: (p[0], p[1], p[2]))

    def cross(o: Tuple[float, float, int], a: Tuple[float, float, int], b: Tuple[float, float, int]) -> float:
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


def convex_hull_insertion(
    coords: np.ndarray,
    dist_matrix: np.ndarray,
    track_memory: bool = False,
) -> TourResult:
    """Heurística de inserção gulosa (Cheapest Insertion) a partir do fecho convexo 2D."""
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
    with track_peak_memory(enabled=track_memory) as mem:
        start_time = time.perf_counter()

        with measure_subroutine(timings, "convex_hull"):
            tour = _monotone_chain_hull(coords_arr)

        with measure_subroutine(timings, "greedy_insertion"):
            in_tour = [False] * n
            for v in tour:
                in_tour[v] = True
            remaining = [v for v in range(n) if not in_tour[v]]

            while remaining:
                best_delta = math.inf
                best_v_idx = 0
                best_pos = 1
                m = len(tour)

                for r_idx, k in enumerate(remaining):
                    row_k = matrix[k]
                    for pos in range(m):
                        u = tour[pos]
                        w = tour[(pos + 1) % m]
                        delta = float(matrix[u, k] + row_k[w] - matrix[u, w])
                        if delta < best_delta:
                            best_delta = delta
                            best_v_idx = r_idx
                            best_pos = pos + 1

                chosen = remaining.pop(best_v_idx)
                tour.insert(best_pos, chosen)

            # Rotaciona para começar canonicamente no vértice 0
            zero_idx = tour.index(0)
            tour = tour[zero_idx:] + tour[:zero_idx]

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
