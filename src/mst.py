"""Árvore Geradora Mínima (MST) via algoritmo de Prim denso com arrays simples.

Para instâncias do TSP geométrico, o grafo de entrada é completo e denso
(|V| = N, |E| = N(N-1)/2) e já está representado como matriz de adjacência.
Nesse cenário, o algoritmo de Prim com busca linear em arrays 1D simples atinge
complexidade temporal O(N^2) e espaço auxiliar O(N), evitando o overhead de
O(N^2 log N) de uma fila de prioridade binária e evitando materializar todas
as O(N^2) arestas ou copiar a matriz de distâncias.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class MSTResult:
    """Representação compacta da Árvore Geradora Mínima."""

    parent: np.ndarray  # shape (N,), parent[root] == -1
    adjacency: List[List[int]]  # lista de adjacência não-direcionada (N-1 arestas)
    total_weight: float
    root: int = 0

    def edges(self) -> List[Tuple[int, int]]:
        """Retorna a lista das N-1 arestas (pai, filho) da MST."""
        return [
            (int(self.parent[v]), v)
            for v in range(len(self.parent))
            if self.parent[v] != -1
        ]

    def degrees(self) -> List[int]:
        """Retorna o grau de cada vértice na MST."""
        return [len(neighbors) for neighbors in self.adjacency]


def build_mst_adjacency(parent: np.ndarray) -> List[List[int]]:
    """Constrói a lista de adjacência não-direcionada da MST em O(N) a partir de parent."""
    n = len(parent)
    adjacency: List[List[int]] = [[] for _ in range(n)]
    for v in range(n):
        p = int(parent[v])
        if p != -1:
            adjacency[p].append(v)
            adjacency[v].append(p)
    return adjacency


def compute_mst_prim(dist_matrix: np.ndarray, root: int = 0) -> MSTResult:
    """Calcula a MST via Prim denso com busca linear em arrays 1D.

    Opera diretamente sobre views das linhas de `dist_matrix`, sem copiar a matriz
    nem utilizar fila de prioridade.

    Complexidade:
        - Tempo: O(N^2)
        - Espaço auxiliar: O(N) (arrays 1D + lista de adjacência com 2*(N-1) entradas)

    Args:
        dist_matrix: Matriz simétrica de distâncias de formato (N, N).
        root: Índice do vértice raiz da árvore (padrão: 0).

    Returns:
        MSTResult contendo vetor `parent`, lista de adjacência não-direcionada
        `adjacency`, peso total `total_weight` e `root`.

    Raises:
        ValueError: Se a matriz não for 2D quadrada e não vazia, se `root` estiver
            fora dos limites ou se o grafo for desconexo / contiver pesos não finitos.
    """
    matrix = np.asarray(dist_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"dist_matrix deve ser uma matriz quadrada (N, N); recebido shape={matrix.shape}."
        )

    n = matrix.shape[0]
    if n == 0:
        raise ValueError("dist_matrix não pode estar vazia.")
    if not (0 <= root < n):
        raise ValueError(f"Raiz inválida: {root}. Deve estar no intervalo [0, {n - 1}].")

    if n == 1:
        return MSTResult(
            parent=np.array([-1], dtype=np.intp),
            adjacency=[[]],
            total_weight=0.0,
            root=root,
        )

    # Arrays 1D simples: O(N) em memória, pré-alocados para evitar cópias no laço.
    min_edge_cost = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.intp)
    unvisited = np.ones(n, dtype=bool)
    improved = np.empty(n, dtype=bool)

    min_edge_cost[root] = 0.0
    total_weight = 0.0

    for step in range(n):
        # Busca linear O(N) pelo vértice fora da MST com menor aresta de corte.
        # Vértices já adicionados têm min_edge_cost == +inf e não são reescolhidos.
        u = int(np.argmin(min_edge_cost))
        cost_u = float(min_edge_cost[u])
        if not np.isfinite(cost_u):
            raise ValueError(
                "Não foi possível conectar todos os vértices na MST "
                "(grafo desconexo ou pesos não finitos)."
            )

        unvisited[u] = False
        min_edge_cost[u] = np.inf
        total_weight += cost_u

        if step == n - 1:
            break

        # Relaxamento O(N) sobre a linha u (view sem cópia de dist_matrix).
        row = matrix[u]
        np.less(row, min_edge_cost, out=improved)
        improved &= unvisited
        np.copyto(min_edge_cost, row, where=improved, casting="unsafe")
        np.copyto(parent, u, where=improved)

    adjacency = build_mst_adjacency(parent)
    return MSTResult(
        parent=parent,
        adjacency=adjacency,
        total_weight=total_weight,
        root=root,
    )
