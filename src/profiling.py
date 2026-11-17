"""Utilitários de profiling de tempo e memória em nível de subrotina."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator
import time
import tracemalloc


@dataclass
class MemoryMeasurement:
    """Armazena o pico de memória (em bytes) capturado por `track_peak_memory`."""

    peak_bytes: int = 0


@contextmanager
def measure_subroutine(timings: Dict[str, float], name: str) -> Iterator[None]:
    """Acumula o tempo de execução (perf_counter) de uma subrotina no dicionário timings."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - start)


@contextmanager
def track_peak_memory(enabled: bool = False) -> Iterator[MemoryMeasurement]:
    """Mede o pico de memória alocada durante o bloco via `tracemalloc`.

    Quando `enabled=False` (padrão), não ativa `tracemalloc` para evitar overhead
    temporal que distorça medições de complexidade empírica, mantendo
    `peak_bytes = 0`.

    Se `tracemalloc` já estiver ativo antes da entrada no bloco, mede o pico
    incremental sem interromper o rastreamento externo.
    """
    measurement = MemoryMeasurement(peak_bytes=0)
    if not enabled:
        yield measurement
        return

    already_tracing = tracemalloc.is_tracing()
    if already_tracing:
        baseline_current, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
    else:
        baseline_current = 0
        tracemalloc.start()

    try:
        yield measurement
    finally:
        _, peak = tracemalloc.get_traced_memory()
        if already_tracing:
            measurement.peak_bytes = max(0, peak - baseline_current)
        else:
            measurement.peak_bytes = peak
            tracemalloc.stop()
