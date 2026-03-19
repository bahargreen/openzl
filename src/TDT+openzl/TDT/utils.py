from __future__ import annotations

import statistics
import time
from typing import Callable, Any


def median_runtime(fn: Callable[[], Any], n_runs: int = 5, n_warmup: int = 1) -> tuple[float, Any]:
    """
    Run fn multiple times and return median runtime in seconds plus last result.
    """
    last_result = None

    for _ in range(n_warmup):
        last_result = fn()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        last_result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return statistics.median(times), last_result


def mb_per_sec(num_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return (num_bytes / (1024 * 1024)) / seconds