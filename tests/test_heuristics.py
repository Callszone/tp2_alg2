"""Testes unitários e de integração para as heurísticas de warm-start."""

from pathlib import Path
import numpy as np

from src.branch_and_bound import solve_branch_and_bound
from src.heuristics import cold_start, convex_hull_insertion, nearest_neighbor
from src.tsplib_parser import compute_euc2d_matrix, load_tsplib_euc2d
from tests.test_contracts import compute_tour_cost


def test_cold_start_trivial_order_and_cost() -> None:
    """Cold start deve gerar 0 -> 1 -> ... -> N-1 -> 0 com custo exato."""
    dist = np.array(
        [
            [0.0, 2.0, 9.0, 10.0],
            [2.0, 0.0, 6.0, 4.0],
            [9.0, 6.0, 0.0, 8.0],
            [10.0, 4.0, 8.0, 0.0],
        ]
    )
    res = cold_start(dist)
    assert res.tour == [0, 1, 2, 3]
    assert res.cost == 26.0
    assert compute_tour_cost(res.tour, dist) == 26.0
    assert res.peak_memory_bytes == 0
    assert "tour_construction" in res.subroutine_times
    assert "tour_cost" in res.subroutine_times


def test_nearest_neighbor_picks_greedy_path() -> None:
    """Nearest Neighbor deve escolher sempre a menor aresta para vértice não visitado."""
    dist = np.array(
        [
            [0.0, 1.0, 10.0, 10.0],
            [1.0, 0.0, 2.0, 10.0],
            [10.0, 2.0, 0.0, 3.0],
            [10.0, 10.0, 3.0, 0.0],
        ]
    )
    res = nearest_neighbor(dist, start=0, track_memory=True)
    assert res.tour == [0, 1, 2, 3]
    assert res.cost == 16.0
    assert res.peak_memory_bytes > 0
    assert "greedy_selection" in res.subroutine_times


def test_heuristics_warm_start_bnb_on_eil51_n12() -> None:
    """Valida cold_start e nearest_neighbor como warm-start no B&B para eil51_n12."""
    inst = load_tsplib_euc2d(Path("data/tsplib/eil51_n12.tsp"))
    cs = cold_start(inst.dist_matrix)
    nn = nearest_neighbor(inst.dist_matrix)

    assert nn.cost <= cs.cost

    res_cs = solve_branch_and_bound(
        inst.dist_matrix,
        initial_tour=cs.tour,
        initial_bound=cs.cost,
        strategy_name="cold_start",
        lower_bound_mode="mst",
    )
    res_nn = solve_branch_and_bound(
        inst.dist_matrix,
        initial_tour=nn.tour,
        initial_bound=nn.cost,
        strategy_name="nearest_neighbor",
        lower_bound_mode="mst",
    )

    assert res_cs.best_cost == res_nn.best_cost
    assert res_nn.nodes_expanded <= res_cs.nodes_expanded


def test_convex_hull_insertion_square_with_interior_point() -> None:
    """Quadrado com ponto interno deve produzir ciclo válido e começar em 0."""
    coords = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [5.0, 1.0],
        ]
    )
    dist = compute_euc2d_matrix(coords)
    res = convex_hull_insertion(coords, dist, track_memory=True)

    assert res.tour[0] == 0
    assert sorted(res.tour) == [0, 1, 2, 3, 4]
    assert abs(compute_tour_cost(res.tour, dist) - res.cost) < 1e-9
    assert res.peak_memory_bytes > 0
    assert "convex_hull" in res.subroutine_times
    assert "greedy_insertion" in res.subroutine_times


def test_convex_hull_insertion_handles_collinear_points() -> None:
    """Pontos colineares não devem falhar e devem gerar ciclo válido."""
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0], [9.0, 0.0]])
    dist = compute_euc2d_matrix(coords)
    res = convex_hull_insertion(coords, dist)

    assert sorted(res.tour) == [0, 1, 2, 3]
    assert res.cost == 18.0


def test_convex_hull_warm_start_on_eil51_n12() -> None:
    """Valida convex_hull_insertion como warm-start no B&B para eil51_n12."""
    inst = load_tsplib_euc2d(Path("data/tsplib/eil51_n12.tsp"))
    ch = convex_hull_insertion(inst.coords, inst.dist_matrix)
    res = solve_branch_and_bound(
        inst.dist_matrix,
        initial_tour=ch.tour,
        initial_bound=ch.cost,
        strategy_name="convex_hull",
        lower_bound_mode="mst",
    )
    assert not res.timed_out
    assert res.best_cost <= ch.cost + 1e-9
