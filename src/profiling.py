"""Utilitários de profiling de tempo e memória em nível de subrotina."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Optional
import time
import tracemalloc

_high_water_mark: int = 0


@dataclass
class MemoryMeasurement:
    """Armazena o pico de memória (em bytes) capturado por `track_peak_memory`."""

    peak_bytes: int = 0


@contextmanager
def measure_subroutine(
    timings: Dict[str, float],
    name: str,
    memory_peaks: Optional[Dict[str, int]] = None,
) -> Iterator[None]:
    """Acumula o tempo de execução e (se tracemalloc ativo) o pico de memória da subrotina."""
    global _high_water_mark
    tracing = memory_peaks is not None and tracemalloc.is_tracing()
    if tracing:
        sub_base_curr, prev_peak = tracemalloc.get_traced_memory()
        if prev_peak > _high_water_mark:
            _high_water_mark = prev_peak
        tracemalloc.reset_peak()
    else:
        sub_base_curr = 0

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[name] = timings.get(name, 0.0) + elapsed
        if tracing and memory_peaks is not None:
            _, sub_peak = tracemalloc.get_traced_memory()
            if sub_peak > _high_water_mark:
                _high_water_mark = sub_peak
            inc_peak = max(0, sub_peak - sub_base_curr)
            memory_peaks[name] = max(memory_peaks.get(name, 0), inc_peak)


@contextmanager
def track_peak_memory(enabled: bool = False) -> Iterator[MemoryMeasurement]:
    """Mede o pico de memória alocada durante o bloco via `tracemalloc`.

    Quando `enabled=False` (padrão), não ativa `tracemalloc` para evitar overhead
    temporal que distorça medições de complexidade empírica, mantendo
    `peak_bytes = 0`.

    Se `tracemalloc` já estiver ativo antes da entrada no bloco, mede o pico
    incremental sem interromper o rastreamento externo.
    """
    global _high_water_mark
    measurement = MemoryMeasurement(peak_bytes=0)
    if not enabled:
        yield measurement
        return

    already_tracing = tracemalloc.is_tracing()
    if already_tracing:
        baseline_current, prev_peak = tracemalloc.get_traced_memory()
        _high_water_mark = max(_high_water_mark, prev_peak)
        tracemalloc.reset_peak()
    else:
        baseline_current = 0
        _high_water_mark = 0
        tracemalloc.start()

    local_hwm_start = baseline_current
    try:
        yield measurement
    finally:
        _, peak = tracemalloc.get_traced_memory()
        true_peak = max(peak, _high_water_mark, local_hwm_start)
        if already_tracing:
            measurement.peak_bytes = max(0, true_peak - baseline_current)
        else:
            measurement.peak_bytes = true_peak
            tracemalloc.stop()
            _high_water_mark = 0
