"""Algoritmos aproximados para o TSP Métrico: Twice-around-the-tree e Christofides."""

import numpy as np
from src.types import TourResult


def twice_around_the_tree(dist_matrix: np.ndarray) -> TourResult:
    """Algoritmo aproximado 2-aproximado baseado em MST + percurso em pré-ordem."""
    raise NotImplementedError("Implementar twice_around_the_tree.")


def christofides(dist_matrix: np.ndarray) -> TourResult:
    """Algoritmo aproximado 1.5-aproximado (MST + Matching Perfeito Mínimo + Euler)."""
    raise NotImplementedError("Implementar christofides.")
