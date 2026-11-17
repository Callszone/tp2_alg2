"""Helpers de validação de rotas e contratos para testes unitários."""

from typing import List
import numpy as np


def compute_tour_cost(tour: List[int], dist_matrix: np.ndarray) -> float:
    """Calcula o custo total de um ciclo hamiltoniano fechado."""
    n = len(tour)
    assert n == dist_matrix.shape[0], "Tamanho da rota difere do número de vértices."
    assert sorted(tour) == list(range(n)), "Rota não é uma permutação válida de 0..N-1."
    return float(sum(dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n)))


def test_compute_tour_cost_triangle() -> None:
    """Valida cálculo de custo em instância mínima de 3 vértices."""
    dist = np.array(
        [
            [0.0, 3.0, 4.0],
            [3.0, 0.0, 5.0],
            [4.0, 5.0, 0.0],
        ]
    )
    assert compute_tour_cost([0, 1, 2], dist) == 12.0
