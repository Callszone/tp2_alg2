"""Extração de métricas geométricas e topológicas para análise de sensibilidade."""

from typing import Dict
import numpy as np


def extract_instance_features(coords: np.ndarray, dist_matrix: np.ndarray) -> Dict[str, float]:
    """Extrai features como aspect ratio, proporção no fecho convexo, grau máximo na MST e CV do NN."""
    raise NotImplementedError("Implementar extrator de features.")
