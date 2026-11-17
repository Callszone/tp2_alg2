"""Testes unitários para o algoritmo de MST (Prim denso com arrays simples)."""

import networkx as nx
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from src.mst import MSTResult, compute_mst_prim


def _assert_valid_spanning_tree(mst: MSTResult, dist_matrix: np.ndarray) -> None:
    """Valida invariantes estruturais de uma árvore geradora sobre dist_matrix."""
    n = dist_matrix.shape[0]
    edges = mst.edges()
    assert len(edges) == n - 1
    assert int(mst.parent[mst.root]) == -1

    # Lista de adjacência não-direcionada consistente com edges e graus.
    total_half_edges = sum(len(nbrs) for nbrs in mst.adjacency)
    assert total_half_edges == 2 * (n - 1)
    assert mst.degrees() == [len(nbrs) for nbrs in mst.adjacency]

    # Conectividade a partir da raiz (como há N-1 arestas, conexo => árvore acíclica).
    visited = set()
    stack = [mst.root]
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        for v in mst.adjacency[u]:
            assert u in mst.adjacency[v]
            if v not in visited:
                stack.append(v)
    assert len(visited) == n

    # Peso total consistente com a soma das arestas na matriz original.
    recomputed_weight = sum(float(dist_matrix[u, v]) for u, v in edges)
    assert mst.total_weight == pytest.approx(recomputed_weight)


def test_mst_known_square_graph() -> None:
    """Valida MST em quadrado unitário (peso ótimo = 3.0)."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    dist = cdist(coords, coords)
    mst = compute_mst_prim(dist, root=0)

    _assert_valid_spanning_tree(mst, dist)
    assert mst.total_weight == pytest.approx(3.0)


def test_mst_known_weighted_graph() -> None:
    """Valida peso, arestas e graus em grafo ponderado conhecido (CLRS-like)."""
    dist = np.array(
        [
            [0.0, 2.0, 10.0, 6.0, 10.0],
            [2.0, 0.0, 3.0, 8.0, 5.0],
            [10.0, 3.0, 0.0, 10.0, 7.0],
            [6.0, 8.0, 10.0, 0.0, 9.0],
            [10.0, 5.0, 7.0, 9.0, 0.0],
        ]
    )
    # Arestas ótimas únicas: (0,1)=2, (1,2)=3, (1,4)=5, (0,3)=6 => peso total = 16.0
    mst = compute_mst_prim(dist, root=0)

    _assert_valid_spanning_tree(mst, dist)
    assert mst.total_weight == pytest.approx(16.0)
    expected_edges = {(0, 1), (1, 2), (1, 4), (0, 3)}
    actual_edges = {tuple(sorted(e)) for e in mst.edges()}
    assert actual_edges == expected_edges
    # Vértice 1 é o hub com grau 3; vértice 0 tem grau 2; folhas têm grau 1.
    assert mst.degrees() == [2, 3, 1, 1, 1]


@pytest.mark.parametrize("n", [5, 16, 40, 75])
def test_mst_matches_networkx_random_euc2d(n: int) -> None:
    """Valida peso ótimo da MST contra NetworkX em instâncias EUC_2D aleatórias."""
    rng = np.random.default_rng(seed=42 + n)
    coords = rng.uniform(0.0, 1000.0, size=(n, 2))
    dist = cdist(coords, coords)
    dist_copy = dist.copy()

    mst = compute_mst_prim(dist, root=0)
    _assert_valid_spanning_tree(mst, dist)

    # Verifica que dist_matrix não foi mutada.
    np.testing.assert_array_equal(dist, dist_copy)

    # Compara peso ótimo com referência externa (NetworkX).
    g = nx.complete_graph(n)
    for i in range(n):
        for j in range(i + 1, n):
            g[i][j]["weight"] = float(dist[i, j])
    nx_mst = nx.minimum_spanning_tree(g, algorithm="kruskal")
    nx_weight = sum(data["weight"] for _, _, data in nx_mst.edges(data=True))

    assert mst.total_weight == pytest.approx(nx_weight, rel=1e-9, abs=1e-9)


def test_mst_invariance_to_root() -> None:
    """O peso total da MST não depende da escolha do vértice raiz."""
    rng = np.random.default_rng(seed=123)
    coords = rng.uniform(0.0, 100.0, size=(20, 2))
    dist = cdist(coords, coords)

    weights = [compute_mst_prim(dist, root=r).total_weight for r in [0, 5, 13, 19]]
    for w in weights[1:]:
        assert w == pytest.approx(weights[0], rel=1e-12)


def test_mst_coincident_points_zero_weight_edges() -> None:
    """Trata corretamente arestas de peso 0.0 (pontos coincidentes)."""
    coords = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])
    dist = cdist(coords, coords)

    mst = compute_mst_prim(dist)
    _assert_valid_spanning_tree(mst, dist)
    assert mst.total_weight == pytest.approx(5.0)


def test_mst_edge_cases_small_and_int_dtype() -> None:
    """Valida N=1, N=2 e matrizes com dtype inteiro."""
    # N = 1
    mst1 = compute_mst_prim(np.array([[0.0]]))
    assert mst1.total_weight == 0.0
    assert mst1.adjacency == [[]]
    assert mst1.edges() == []
    assert mst1.degrees() == [0]

    # N = 2 com dtype int64
    dist2 = np.array([[0, 7], [7, 0]], dtype=np.int64)
    mst2 = compute_mst_prim(dist2)
    _assert_valid_spanning_tree(mst2, dist2)
    assert mst2.total_weight == pytest.approx(7.0)


def test_mst_invalid_inputs() -> None:
    """Valida exceções para entradas inválidas ou grafo desconexo."""
    with pytest.raises(ValueError):
        compute_mst_prim(np.empty((0, 0)))

    with pytest.raises(ValueError):
        compute_mst_prim(np.zeros((3, 4)))

    with pytest.raises(ValueError):
        compute_mst_prim(np.zeros((3, 3)), root=3)

    with pytest.raises(ValueError):
        compute_mst_prim(np.zeros((3, 3)), root=-1)

    disconnected = np.array(
        [
            [0.0, 1.0, np.inf],
            [1.0, 0.0, np.inf],
            [np.inf, np.inf, 0.0],
        ]
    )
    with pytest.raises(ValueError):
        compute_mst_prim(disconnected)
