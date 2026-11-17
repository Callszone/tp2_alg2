"""Heurísticas geradoras de solução inicial (warm-start) para o Branch-and-Bound."""

import numpy as np
from src.types import TourResult


def cold_start(dist_matrix: np.ndarray) -> TourResult:
    """Solução trivial: 0 -> 1 -> 2 -> ... -> N-1 -> 0."""
    raise NotImplementedError("Implementar cold_start.")


def nearest_neighbor(dist_matrix: np.ndarray, start: int = 0) -> TourResult:
    """Heurística gulosa do Vizinho Mais Próximo."""
    raise NotImplementedError("Implementar nearest_neighbor.")


def convex_hull_insertion(coords: np.ndarray, dist_matrix: np.ndarray) -> TourResult:
    """Heurística de inserção gulosa a partir do fecho convexo 2D."""
    raise NotImplementedError("Implementar convex_hull_insertion.")
