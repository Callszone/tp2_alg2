"""Testes unitários para Twice-around-the-tree, Christofides e instrumentação de profiling."""

from itertools import permutations
import tracemalloc
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from src.approximations import christofides, twice_around_the_tree
from src.mst import compute_mst_prim
from src.profiling import track_peak_memory
from src.types import TourResult
from tests.test_contracts import compute_tour_cost


def _brute_force_tsp_opt(dist_matrix: np.ndarray) -> float:
    """Resolve o TSP exatamente por força bruta fixando o vértice 0 (para N pequeno)."""
    n = dist_matrix.shape[0]
    best = float("inf")
    for perm in permutations(range(1, n)):
        tour = [0, *perm]
        cost = compute_tour_cost(tour, dist_matrix)
        if cost < best:
            best = cost
    return best


def _assert_valid_tour_result(
    result: TourResult,
    dist_matrix: np.ndarray,
    expected_root: int = 0,
    expect_memory: bool = False,
) -> None:
    """Valida todos os contratos de TourResult para twice_around_the_tree."""
    n = dist_matrix.shape[0]
    assert isinstance(result, TourResult)
    assert isinstance(result.tour, list)
    assert len(result.tour) == n
    assert all(type(v) is int for v in result.tour)
    assert result.tour[0] == expected_root
    assert sorted(result.tour) == list(range(n))

    assert isinstance(result.cost, float)
    if n == 1:
        assert result.cost == 0.0
    else:
        assert result.cost == pytest.approx(compute_tour_cost(result.tour, dist_matrix))

    assert set(result.subroutine_times.keys()) == {"mst", "preorder_dfs", "tour_cost"}
    for t in result.subroutine_times.values():
        assert isinstance(t, float) and t >= 0.0
    assert result.elapsed_time >= sum(result.subroutine_times.values())

    if expect_memory:
        assert result.peak_memory_bytes > 0
    else:
        assert result.peak_memory_bytes == 0


def test_tatt_unit_square_deterministic() -> None:
    """No quadrado unitário, TATT produz ciclo ótimo de custo 4.0."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    dist = cdist(coords, coords)

    result = twice_around_the_tree(dist)
    _assert_valid_tour_result(result, dist, expected_root=0, expect_memory=False)
    assert result.cost == pytest.approx(4.0)


def test_tatt_collinear_points() -> None:
    """Em pontos colineares 0, 1, 3, 7, o tour fechado ótimo tem custo 14.0."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [7.0, 0.0]])
    dist = cdist(coords, coords)

    result = twice_around_the_tree(dist, root=0)
    _assert_valid_tour_result(result, dist, expected_root=0)
    assert result.cost == pytest.approx(14.0)


@pytest.mark.parametrize("n", [4, 6, 8])
def test_tatt_2_approximation_guarantee_vs_exact(n: int) -> None:
    """Verifica MST <= OPT <= TATT <= 2 * MST <= 2 * OPT em instâncias EUC_2D."""
    rng = np.random.default_rng(seed=100 + n)
    for _ in range(5):
        coords = rng.uniform(0.0, 100.0, size=(n, 2))
        dist = cdist(coords, coords)

        mst = compute_mst_prim(dist, root=0)
        opt = _brute_force_tsp_opt(dist)
        result = twice_around_the_tree(dist, root=0)

        _assert_valid_tour_result(result, dist, expected_root=0)
        assert mst.total_weight <= opt + 1e-9
        assert opt <= result.cost + 1e-9
        assert result.cost <= 2.0 * mst.total_weight + 1e-9
        assert result.cost <= 2.0 * opt + 1e-9


@pytest.mark.parametrize("n", [40, 75])
def test_tatt_tp2_target_sizes_and_immutability(n: int) -> None:
    """Valida instâncias na faixa principal e limite do TP2 sem mutar dist_matrix."""
    rng = np.random.default_rng(seed=2026 + n)
    coords = rng.uniform(0.0, 1000.0, size=(n, 2))
    dist = cdist(coords, coords)
    dist_copy = dist.copy()

    mst = compute_mst_prim(dist, root=0)
    result = twice_around_the_tree(dist, root=0)

    _assert_valid_tour_result(result, dist, expected_root=0, expect_memory=False)
    assert mst.total_weight <= result.cost <= 2.0 * mst.total_weight + 1e-9
    np.testing.assert_array_equal(dist, dist_copy)


def test_tatt_memory_tracking_flag_and_nested_tracemalloc() -> None:
    """Valida track_memory=False (padrão) e track_memory=True sem vazar estado."""
    rng = np.random.default_rng(seed=7)
    coords = rng.uniform(0.0, 100.0, size=(30, 2))
    dist = cdist(coords, coords)

    assert not tracemalloc.is_tracing()

    # Padrão: track_memory=False não liga tracemalloc e mantém peak_memory_bytes=0
    res_no_mem = twice_around_the_tree(dist)
    assert not tracemalloc.is_tracing()
    _assert_valid_tour_result(res_no_mem, dist, expect_memory=False)

    # track_memory=True mede memória e encerra tracemalloc ao finalizar
    res_mem = twice_around_the_tree(dist, track_memory=True)
    assert not tracemalloc.is_tracing()
    _assert_valid_tour_result(res_mem, dist, expect_memory=True)

    # Se tracemalloc já estava ativo externamente, não deve interrompê-lo
    tracemalloc.start()
    try:
        with track_peak_memory(enabled=True) as mem:
            _ = [i for i in range(1000)]
        assert tracemalloc.is_tracing()
        assert mem.peak_bytes > 0
    finally:
        tracemalloc.stop()


def test_tatt_edge_cases_and_different_root() -> None:
    """Valida N=1, N=2, N=3, pontos coincidentes, dtype inteiro e root != 0."""
    # N = 1
    res1 = twice_around_the_tree(np.array([[0.0]]))
    _assert_valid_tour_result(res1, np.array([[0.0]]), expected_root=0)
    assert res1.tour == [0]
    assert res1.cost == 0.0

    # N = 2 com dtype int64
    dist2 = np.array([[0, 5], [5, 0]], dtype=np.int64)
    res2 = twice_around_the_tree(dist2, root=np.int64(1))
    _assert_valid_tour_result(res2, dist2, expected_root=1)
    assert res2.tour == [1, 0]
    assert res2.cost == pytest.approx(10.0)

    # N = 3 triângulo 3-4-5
    dist3 = np.array([[0.0, 3.0, 4.0], [3.0, 0.0, 5.0], [4.0, 5.0, 0.0]])
    res3 = twice_around_the_tree(dist3, root=2)
    _assert_valid_tour_result(res3, dist3, expected_root=2)
    assert res3.cost == pytest.approx(12.0)

    # Pontos coincidentes (distância zero)
    coords = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])
    dist_zero = cdist(coords, coords)
    res_zero = twice_around_the_tree(dist_zero)
    _assert_valid_tour_result(res_zero, dist_zero)
    assert res_zero.cost == pytest.approx(10.0)


def test_tatt_degenerate_long_path_no_recursion_error() -> None:
    """Garante que o DFS iterativo não sofre RecursionError em árvore caminho N=1500."""
    n = 1500
    x = np.arange(n, dtype=np.float64)[:, None]
    dist = np.abs(x - x.T)

    result = twice_around_the_tree(dist, root=0)
    _assert_valid_tour_result(result, dist, expected_root=0)
    # Ida de 0 a n-1 (custo n-1) + volta direta de n-1 a 0 (custo n-1)
    assert result.cost == pytest.approx(2.0 * (n - 1))


def test_tatt_invalid_inputs() -> None:
    """Valida propagação de ValueError para entradas inválidas."""
    with pytest.raises(ValueError):
        twice_around_the_tree(np.empty((0, 0)))

    with pytest.raises(ValueError):
        twice_around_the_tree(np.zeros((2, 3)))

    with pytest.raises(ValueError):
        twice_around_the_tree(np.zeros((3, 3)), root=3)

    disconnected = np.array(
        [
            [0.0, 1.0, np.inf],
            [1.0, 0.0, np.inf],
            [np.inf, np.inf, 0.0],
        ]
    )
    with pytest.raises(ValueError):
        twice_around_the_tree(disconnected)


def _assert_valid_christofides_result(
    result: TourResult,
    dist_matrix: np.ndarray,
    expect_memory: bool = False,
) -> None:
    """Valida todos os contratos de TourResult para christofides."""
    n = dist_matrix.shape[0]
    assert isinstance(result, TourResult)
    assert isinstance(result.tour, list)
    assert len(result.tour) == n
    assert all(type(v) is int for v in result.tour)
    assert result.tour[0] == 0
    assert sorted(result.tour) == list(range(n))

    assert isinstance(result.cost, float)
    if n == 1:
        assert result.cost == 0.0
    else:
        assert result.cost == pytest.approx(compute_tour_cost(result.tour, dist_matrix))

    expected_subroutines = {
        "mst",
        "odd_vertices",
        "min_weight_matching",
        "eulerian_circuit",
        "shortcutting",
        "tour_cost",
    }
    if n > 1:
        assert set(result.subroutine_times.keys()) == expected_subroutines
        for t in result.subroutine_times.values():
            assert isinstance(t, float) and t >= 0.0
        assert result.elapsed_time >= sum(result.subroutine_times.values())

    if expect_memory:
        assert result.peak_memory_bytes > 0
    else:
        assert result.peak_memory_bytes == 0


def test_christofides_unit_square_deterministic() -> None:
    """No quadrado unitário, Christofides produz o ciclo ótimo de custo 4.0."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    dist = cdist(coords, coords)

    result = christofides(dist)
    _assert_valid_christofides_result(result, dist, expect_memory=False)
    assert result.cost == pytest.approx(4.0)


def test_christofides_collinear_points() -> None:
    """Em pontos colineares 0, 1, 3, 7, o tour fechado ótimo tem custo 14.0."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [7.0, 0.0]])
    dist = cdist(coords, coords)

    result = christofides(dist)
    _assert_valid_christofides_result(result, dist)
    assert result.cost == pytest.approx(14.0)


@pytest.mark.parametrize("n", [4, 6, 8])
def test_christofides_1_5_approximation_guarantee_vs_exact(n: int) -> None:
    """Verifica OPT <= Christofides <= 1.5 * OPT em instâncias EUC_2D aleatórias."""
    rng = np.random.default_rng(seed=200 + n)
    for _ in range(5):
        coords = rng.uniform(0.0, 100.0, size=(n, 2))
        dist = cdist(coords, coords)

        opt = _brute_force_tsp_opt(dist)
        result = christofides(dist)

        _assert_valid_christofides_result(result, dist)
        assert opt <= result.cost + 1e-9
        assert result.cost <= 1.5 * opt + 1e-9


@pytest.mark.parametrize("n", [20, 35])
def test_christofides_immutability_and_scaling(n: int) -> None:
    """Valida que Christofides não modifica a matriz de entrada e respeita cota da MST."""
    rng = np.random.default_rng(seed=3000 + n)
    coords = rng.uniform(0.0, 1000.0, size=(n, 2))
    dist = cdist(coords, coords)
    dist_copy = dist.copy()

    mst = compute_mst_prim(dist, root=0)
    result = christofides(dist)

    _assert_valid_christofides_result(result, dist, expect_memory=False)
    assert mst.total_weight <= result.cost
    np.testing.assert_array_equal(dist, dist_copy)


def test_christofides_memory_tracking() -> None:
    """Valida medição isolada de memória para Christofides."""
    rng = np.random.default_rng(seed=42)
    coords = rng.uniform(0.0, 100.0, size=(25, 2))
    dist = cdist(coords, coords)

    assert not tracemalloc.is_tracing()

    res_no_mem = christofides(dist, track_memory=False)
    assert not tracemalloc.is_tracing()
    _assert_valid_christofides_result(res_no_mem, dist, expect_memory=False)

    res_mem = christofides(dist, track_memory=True)
    assert not tracemalloc.is_tracing()
    _assert_valid_christofides_result(res_mem, dist, expect_memory=True)


def test_christofides_edge_cases() -> None:
    """Valida N=1, N=2, N=3 e pontos coincidentes."""
    # N = 1
    res1 = christofides(np.array([[0.0]]))
    _assert_valid_christofides_result(res1, np.array([[0.0]]))
    assert res1.tour == [0]
    assert res1.cost == 0.0

    # N = 2
    dist2 = np.array([[0.0, 7.0], [7.0, 0.0]])
    res2 = christofides(dist2)
    _assert_valid_christofides_result(res2, dist2)
    assert res2.tour == [0, 1]
    assert res2.cost == pytest.approx(14.0)

    # N = 3
    dist3 = np.array([[0.0, 3.0, 4.0], [3.0, 0.0, 5.0], [4.0, 5.0, 0.0]])
    res3 = christofides(dist3)
    _assert_valid_christofides_result(res3, dist3)
    assert res3.cost == pytest.approx(12.0)

    # Pontos coincidentes
    coords = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])
    dist_zero = cdist(coords, coords)
    res_zero = christofides(dist_zero)
    _assert_valid_christofides_result(res_zero, dist_zero)
    assert res_zero.cost == pytest.approx(10.0)


def test_christofides_invalid_inputs() -> None:
    """Valida propagação de exceções para entradas incorretas."""
    with pytest.raises(ValueError):
        christofides(np.empty((0, 0)))

    with pytest.raises(ValueError):
        christofides(np.zeros((2, 3)))

    disconnected = np.array(
        [
            [0.0, 1.0, np.inf],
            [1.0, 0.0, np.inf],
            [np.inf, np.inf, 0.0],
        ]
    )
    with pytest.raises(ValueError):
        christofides(disconnected)
