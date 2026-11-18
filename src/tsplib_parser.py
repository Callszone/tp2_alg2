"""Parser de instâncias TSPLIB (tipo EUC_2D) e cálculo de matriz de distâncias."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Custos ótimos oficiais publicados na TSPLIB95 (EUC_2D com arredondamento nint)
KNOWN_OPTIMA: Dict[str, float] = {
    "eil51": 426.0,
    "berlin52": 7542.0,
    "st70": 675.0,
}


@dataclass
class TSPInstance:
    """Representa uma instância TSP geométrica 2D."""

    name: str
    dimension: int
    edge_weight_type: str
    coords: np.ndarray  # shape (N, 2)
    dist_matrix: np.ndarray  # shape (N, N)
    optimal_cost: Optional[float] = None


def compute_euc2d_matrix(coords: np.ndarray) -> np.ndarray:
    """Calcula a matriz de distâncias EUC_2D conforme padrão TSPLIB95 (nint = floor(d + 0.5))."""
    coords_f = np.asarray(coords, dtype=np.float64)
    if coords_f.ndim != 2 or coords_f.shape[1] != 2:
        raise ValueError(f"Esperado array (N, 2) de coordenadas, recebido {coords_f.shape}.")

    diff = coords_f[:, np.newaxis, :] - coords_f[np.newaxis, :, :]
    raw_dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    dist_matrix = np.floor(raw_dist + 0.5)
    np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix


def load_tsplib_euc2d(filepath: Path, optimal_cost: Optional[float] = None) -> TSPInstance:
    """Lê um arquivo .tsp da TSPLIB validando EDGE_WEIGHT_TYPE == EUC_2D."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo TSPLIB não encontrado: {path}")

    metadata: Dict[str, str] = {}
    coords_list: List[List[float]] = []
    in_coords = False

    with path.open("r", encoding="latin-1") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line == "EOF":
                break
            if line == "NODE_COORD_SECTION":
                in_coords = True
                continue
            if in_coords:
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Linha inválida em NODE_COORD_SECTION ({path}): {line!r}")
                coords_list.append([float(parts[1]), float(parts[2])])
            elif ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip().upper()] = val.strip()

    ewt = metadata.get("EDGE_WEIGHT_TYPE", "").upper()
    if ewt != "EUC_2D":
        raise ValueError(f"Instância {path.name} tem EDGE_WEIGHT_TYPE={ewt!r}, esperado 'EUC_2D'.")

    name = metadata.get("NAME", path.stem)
    dimension = int(metadata.get("DIMENSION", len(coords_list)))
    if len(coords_list) != dimension:
        raise ValueError(
            f"Instância {name}: DIMENSION={dimension}, mas encontradas {len(coords_list)} coordenadas."
        )

    coords = np.asarray(coords_list, dtype=np.float64)
    dist_matrix = compute_euc2d_matrix(coords)

    resolved_opt = optimal_cost if optimal_cost is not None else KNOWN_OPTIMA.get(name)

    return TSPInstance(
        name=name,
        dimension=dimension,
        edge_weight_type=ewt,
        coords=coords,
        dist_matrix=dist_matrix,
        optimal_cost=resolved_opt,
    )


def list_tsplib_instances(directory: Path = Path("data/tsplib")) -> List[TSPInstance]:
    """Carrega todas as instâncias .tsp de um diretório ordenadas por dimensão (N)."""
    paths = sorted(Path(directory).glob("*.tsp"))
    instances = [load_tsplib_euc2d(p) for p in paths]
    instances.sort(key=lambda inst: (inst.dimension, inst.name))
    return instances
