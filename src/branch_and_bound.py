"""Algoritmo exato Branch-and-Bound para o Problema do Caixeiro Viajante."""

from typing import List, Optional
import numpy as np
from src.types import BnBResult


def solve_branch_and_bound(
    dist_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    initial_bound: Optional[float] = None,
    strategy_name: str = "cold_start",
    timeout_sec: float = 1800.0,
) -> BnBResult:
    """Resolve o TSP via Branch-and-Bound com suporte a warm-start e timeout.

    Args:
        dist_matrix: Matriz simétrica de distâncias (N, N).
        initial_tour: Rota inicial para inicializar o upper bound (warm-start).
        initial_bound: Custo da rota inicial. Se None, calcula a partir de initial_tour
            ou usa infinito.
        strategy_name: Nome da estratégia de warm-start para registro nas métricas.
        timeout_sec: Tempo máximo de execução em segundos (padrão: 1800s / 30min).

    Returns:
        BnBResult com melhor rota, custo, contadores de nós (expandidos/podados),
        profundidade máxima e tempos por subrotina.
    """
    raise NotImplementedError("Implementar Branch-and-Bound.")
