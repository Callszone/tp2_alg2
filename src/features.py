"""Extração de métricas geométricas e topológicas para análise de sensibilidade."""

import math
from typing import Dict
import numpy as np

from src.heuristics import compute_convex_hull_indices
from src.mst import compute_mst_prim

_ENTROPY_GRID_SIZE = 4  # Grade fixa 4x4 (16 células) para evitar saltos de discretização em sqrt(N)


def extract_instance_features(coords: np.ndarray, dist_matrix: np.ndarray) -> Dict[str, float]:
    """Extrai métricas geométricas e topológicas de uma instância TSP 2D.

    Features retornadas:
        - ``aspect_ratio``: razão max(dx, dy) / min(dx, dy) do bounding box (sempre finita >= 1.0).
        - ``convex_hull_ratio``: fração de pontos pertencentes ao fecho convexo.
        - ``spatial_entropy``: entropia de Shannon normalizada em grade fixa 4x4.
        - ``mst_max_degree``: grau máximo de um vértice na MST (presença de hubs).
        - ``mst_leaf_ratio``: fração de folhas (grau 1) na MST.
        - ``mst_total_cost``: custo total da MST.
        - ``nn_dist_cv``: coeficiente de variação (std / mean) das distâncias ao vizinho mais próximo.
    """
    pts = np.asarray(coords, dtype=np.float64)
    matrix = np.asarray(dist_matrix, dtype=np.float64)
    n = pts.shape[0]
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Esperado coords (N, 2), recebido {pts.shape}.")
    if matrix.shape != (n, n):
        raise ValueError(f"Esperado dist_matrix ({n}, {n}), recebido {matrix.shape}.")
    if n == 0:
        raise ValueError("Instância vazia não possui features.")

    # 1. Razão de aspecto do bounding box (sempre finita para evitar perda em regressões)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span_x = float(max_xy[0] - min_xy[0])
    span_y = float(max_xy[1] - min_xy[1])
    hi_span = max(span_x, span_y)
    lo_span = min(span_x, span_y)
    if lo_span > 1e-9:
        aspect_ratio = hi_span / lo_span
    else:
        aspect_ratio = max(1.0, hi_span)

    # 2. Proporção no fecho convexo
    hull = compute_convex_hull_indices(pts)
    convex_hull_ratio = float(len(hull)) / float(n)

    # 3. Entropia da distribuição espacial em grade fixa 4x4
    grid_size = _ENTROPY_GRID_SIZE
    if span_x <= 1e-12:
        gx = np.zeros(n, dtype=int)
    else:
        gx = np.clip(((pts[:, 0] - min_xy[0]) / span_x * grid_size).astype(int), 0, grid_size - 1)
    if span_y <= 1e-12:
        gy = np.zeros(n, dtype=int)
    else:
        gy = np.clip(((pts[:, 1] - min_xy[1]) / span_y * grid_size).astype(int), 0, grid_size - 1)

    cell_ids = gx * grid_size + gy
    _, counts = np.unique(cell_ids, return_counts=True)
    probs = counts.astype(np.float64) / float(n)
    entropy = float(-np.sum(probs * np.log(probs)))
    max_entropy = math.log(grid_size * grid_size)
    spatial_entropy = entropy / max_entropy if max_entropy > 0.0 else 0.0

    # 4. Estatísticas da Árvore Geradora Mínima
    mst = compute_mst_prim(matrix, root=0)
    degrees = mst.degrees()
    mst_max_degree = float(max(degrees)) if degrees else 0.0
    mst_leaf_ratio = float(sum(1 for d in degrees if d == 1)) / float(n)

    # 5. Coeficiente de variação das distâncias ao vizinho mais próximo
    if n <= 1:
        nn_dist_cv = 0.0
    else:
        masked = matrix.copy()
        np.fill_diagonal(masked, np.inf)
        nn_dists = masked.min(axis=1)
        nn_mean = float(np.mean(nn_dists))
        nn_std = float(np.std(nn_dists))
        nn_dist_cv = nn_std / nn_mean if nn_mean > 1e-12 else 0.0

    return {
        "aspect_ratio": aspect_ratio,
        "convex_hull_ratio": convex_hull_ratio,
        "spatial_entropy": spatial_entropy,
        "mst_max_degree": mst_max_degree,
        "mst_leaf_ratio": mst_leaf_ratio,
        "mst_total_cost": float(mst.total_weight),
        "nn_dist_cv": nn_dist_cv,
    }
