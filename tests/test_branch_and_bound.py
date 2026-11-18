"""Testes unitários e diferenciais para o Branch-and-Bound."""

import itertools
import numpy as np

from src.approximations import twice_around_the_tree
from src.branch_and_bound import solve_branch_and_bound
from tests.test_contracts import compute_tour_cost


def _brute_force_tsp(dist_matrix: np.ndarray) -> float:
    n = dist_matrix.shape[0]
    best = float("inf")
    for perm in itertools.permutations(range(1, n)):
        tour = [0, *perm]
        cost = compute_tour_cost(tour, dist_matrix)
        if cost < best:
            best = cost
    return best


def _random_euc2d_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 100.0, size=(n, 2))
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.hypot(diff[:, :, 0], diff[:, :, 1])


def test_bnb_known_square_4_vertices() -> None:
    """Quadrado unitário: ciclo ótimo tem custo 4.0."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])

    res = solve_branch_and_bound(dist)
    assert not res.timed_out
    assert res.elapsed_time is not None
    assert abs(res.best_cost - 4.0) < 1e-9
    assert abs(compute_tour_cost(res.best_tour, dist) - 4.0) < 1e-9


def test_bnb_matches_brute_force_random_instances() -> None:
    """Compara o B&B contra força bruta em instâncias aleatórias N=5..8."""
    for n, seed in [(5, 42), (6, 7), (7, 99), (8, 123)]:
        dist = _random_euc2d_matrix(n, seed)
        exact_opt = _brute_force_tsp(dist)

        res = solve_branch_and_bound(dist)
        assert not res.timed_out
        assert abs(res.best_cost - exact_opt) < 1e-7
        assert abs(compute_tour_cost(res.best_tour, dist) - exact_opt) < 1e-7
        assert res.nodes_expanded > 0
        assert res.max_depth == n - 1


def test_bnb_warm_start_reduces_or_matches_nodes() -> None:
    """Warm-start com upper bound ótimo deve expandir menos ou igual nós que cold start."""
    dist = _random_euc2d_matrix(9, seed=2026)
    cold = solve_branch_and_bound(dist, strategy_name="cold_start")

    warm = solve_branch_and_bound(
        dist,
        initial_tour=cold.best_tour,
        initial_bound=cold.best_cost,
        strategy_name="optimal_warm",
    )
    assert abs(warm.best_cost - cold.best_cost) < 1e-9
    assert warm.nodes_expanded <= cold.nodes_expanded


def test_bnb_timeout_returns_na_and_partial_stats() -> None:
    """Ao estourar o timeout, elapsed_time deve ser None mas contadores preservados."""
    dist = _random_euc2d_matrix(22, seed=55)
    res = solve_branch_and_bound(dist, timeout_sec=0.05)

    assert res.timed_out is True
    assert res.elapsed_time is None
    assert res.nodes_expanded > 0
    assert res.max_depth > 0


def test_bnb_warm_start_with_twice_around_the_tree_and_memory_tracking() -> None:
    """Integra twice_around_the_tree como warm-start e valida track_memory."""
    dist = _random_euc2d_matrix(9, seed=77)
    approx = twice_around_the_tree(dist)

    res_no_mem = solve_branch_and_bound(
        dist,
        initial_tour=approx.tour,
        initial_bound=approx.cost,
        strategy_name="twice_around_the_tree",
        track_memory=False,
    )
    res_with_mem = solve_branch_and_bound(
        dist,
        initial_tour=approx.tour,
        initial_bound=approx.cost,
        strategy_name="twice_around_the_tree",
        track_memory=True,
    )

    assert res_no_mem.peak_memory_bytes == 0
    assert res_with_mem.peak_memory_bytes > 0
    assert abs(res_no_mem.best_cost - res_with_mem.best_cost) < 1e-9
    assert res_with_mem.best_cost <= approx.cost + 1e-9


def test_bnb_mst_lower_bound_matches_brute_force_and_prunes_more() -> None:
    """Valida que lower_bound_mode='mst' acha o ótimo exato e expande <= nós que 'two_min'."""
    for n, seed in [(5, 42), (6, 7), (7, 99), (8, 123)]:
        dist = _random_euc2d_matrix(n, seed)
        exact_opt = _brute_force_tsp(dist)

        res_two_min = solve_branch_and_bound(dist, lower_bound_mode="two_min")
        res_mst = solve_branch_and_bound(dist, lower_bound_mode="mst")

        assert abs(res_mst.best_cost - exact_opt) < 1e-7
        assert abs(compute_tour_cost(res_mst.best_tour, dist) - exact_opt) < 1e-7
        assert res_mst.nodes_expanded <= res_two_min.nodes_expanded


def test_bnb_invalid_lower_bound_mode_raises() -> None:
    """Modo de lower bound desconhecido deve falhar explicitamente."""
    dist = _random_euc2d_matrix(4, seed=1)
    try:
        solve_branch_and_bound(dist, lower_bound_mode="invalido")
        assert False, "Esperava ValueError para lower_bound_mode inválido"
    except ValueError:
        pass
