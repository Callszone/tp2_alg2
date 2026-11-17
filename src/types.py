"""Contratos de dados compartilhados entre os módulos do projeto."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TourResult:
    """Resultado padrão para heurísticas e algoritmos aproximados."""

    tour: List[int]
    cost: float
    elapsed_time: float
    subroutine_times: Dict[str, float] = field(default_factory=dict)
    peak_memory_bytes: int = 0


@dataclass
class BnBResult:
    """Resultado padrão para o algoritmo exato Branch-and-Bound."""

    best_tour: List[int]
    best_cost: float
    elapsed_time: Optional[float]  # None quando timed_out=True (reportar como NA)
    timed_out: bool
    nodes_expanded: int
    nodes_pruned: int
    max_depth: int
    initial_bound: float
    strategy: str
    subroutine_times: Dict[str, float] = field(default_factory=dict)
    peak_memory_bytes: int = 0
