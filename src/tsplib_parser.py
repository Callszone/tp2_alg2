"""Parser de instâncias TSPLIB (tipo EUC_2D) e cálculo de matriz de distâncias."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class TSPInstance:
    """Representa uma instância TSP geométrica 2D."""

    name: str
    dimension: int
    edge_weight_type: str
    coords: np.ndarray  # shape (N, 2)
    dist_matrix: np.ndarray  # shape (N, N)
    optimal_cost: Optional[float] = None


def load_tsplib_euc2d(filepath: Path, optimal_cost: Optional[float] = None) -> TSPInstance:
    """Lê um arquivo .tsp da TSPLIB validando EDGE_WEIGHT_TYPE == EUC_2D."""
    raise NotImplementedError("Implementar parser TSPLIB EUC_2D.")
