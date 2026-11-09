"""Run time per stage of a patient, and the machine it ran on (for the timing report)."""
from __future__ import annotations

import os
import platform
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager

# in processing order; the stages of the optional checks are 0 when they are not used
STAGES = ("inventory", "read", "segment", "anatomy_check", "mask", "targets", "fill", "write", "verify", "qc",
          "face_check", "rt_copy")


class Timer:
    """Adds up the seconds of each named stage; a patient with several CT series adds up over them."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start


def machine_info(device: str) -> str:
    """Short description of the hardware, to report next to the timings."""
    parts = [f"{platform.system()} {platform.machine()}", f"{os.cpu_count()} CPUs"]
    cpu = platform.processor()
    if cpu and cpu != platform.machine():
        parts.insert(1, cpu)
    if device.startswith("gpu"):
        try:
            import torch
            if torch.cuda.is_available():
                parts.append(torch.cuda.get_device_name(0))
        except Exception:                       # the timing report must never stop a run
            pass
    return ", ".join(parts)


def _spread(values: list[float]) -> str:
    """Median and interquartile range."""
    if len(values) < 2:
        return f"{values[0]:.1f}"
    q1, med, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return f"{med:.1f} [{q1:.1f}-{q3:.1f}]"


def summary_lines(rows: list[tuple[str, float, dict[str, float]]]) -> list[str]:
    """Report lines for (device, total seconds, seconds per stage) of the finished patients: the
    median [interquartile range] of the total and of each stage, for each device used."""
    lines = []
    for device in sorted({d for d, _, _ in rows}):
        used = [r for r in rows if r[0] == device]
        lines.append(f"Timing on {device} ({len(used)} patient(s)), seconds, median [IQR]: "
                     f"total {_spread([t for _, t, _ in used])}")
        for stage in STAGES:
            values = [s.get(stage, 0.0) for _, _, s in used]
            if any(values):
                lines.append(f"    {stage:<14} {_spread(values)}")
    return lines
