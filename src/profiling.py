"""Utilitários de profiling de tempo e memória em nível de subrotina."""

from contextlib import contextmanager
from typing import Dict, Iterator
import time


@contextmanager
def measure_subroutine(timings: Dict[str, float], name: str) -> Iterator[None]:
    """Acumula o tempo de execução (perf_counter) de uma subrotina no dicionário timings."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - start)
