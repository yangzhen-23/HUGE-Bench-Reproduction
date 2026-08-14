"""Non-blocking execution smoke test for the local official metric module.
看看官方 `metric.py` 在你这个 Python 环境里能不能正常 import、调用、返回有限数值。

"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable
from uuid import uuid4

import numpy as np


SMOKE_LABEL = "Synthetic metric implementation smoke test \u2014 not a paper result."
_REQUIRED_FUNCTIONS = (
    "compute_avg_tcr",
    "compute_ndtw",
    "compute_nsp",
    "compute_success",
    "path_length_xyz",
)
_LIMITATIONS = (
    "synthetic identical trajectories only; no predicted model trajectory, collision mesh, "
    "3DGS simulator, or closed-loop evaluation; the result cannot reproduce or support "
    "the paper's model-performance table; softdtw_gamma=0.1 is used because the current "
    "official implementation with gamma 0 exhibits division-by-zero / non-finite behavior "
    "in the local probe."
)


@dataclass(frozen=True)
class MetricSmokeResult:
    status: str
    label: str
    avg_tcr: float | None
    ndtw: float | None
    nsp: float | None
    success: float | None
    path_length: float | None
    limitations: str


def _skip(reason: str) -> MetricSmokeResult:
    return MetricSmokeResult(
        status="SKIP",
        label=SMOKE_LABEL,
        avg_tcr=None,
        ndtw=None,
        nsp=None,
        success=None,
        path_length=None,
        limitations=f"Metric smoke skipped: {reason}. {_LIMITATIONS}",
    )


def _load_isolated_module(metric_path: Path) -> ModuleType:
    module_name = f"_huge_lightweight_metric_smoke_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, metric_path)
    if spec is None or spec.loader is None:
        raise ImportError("could not create a module specification")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    sys.modules.pop(module_name, None)
    return module


def _required_callables(module: ModuleType) -> dict[str, Callable[..., object]]:
    functions: dict[str, Callable[..., object]] = {}
    for name in _REQUIRED_FUNCTIONS:
        value = getattr(module, name, None)
        if not callable(value):
            raise AttributeError(f"required callable {name!r} is absent")
        functions[name] = value
    return functions


def _finite_float(name: str, value: object) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} returned an unconvertible result ({type(exc).__name__}: {exc})") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} returned a non-finite result")
    return converted


def run_metric_smoke(metric_path: Path) -> MetricSmokeResult:
    """Run released pure metrics on an identical synthetic trajectory, or return ``SKIP``."""
    path = Path(metric_path)
    if not path.exists():
        return _skip(f"metric path does not exist: {path}")
    if not path.is_file():
        return _skip(f"metric path is not a regular file: {path}")

    try:
        module = _load_isolated_module(path)
        functions = _required_callables(module)
        trajectory = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
            ]
        )
        gt = trajectory
        pred = trajectory
        avg_tcr = _finite_float("compute_avg_tcr", functions["compute_avg_tcr"](gt, pred, [0.5, 1.0, 2.0]))
        ndtw = _finite_float(
            "compute_ndtw",
            functions["compute_ndtw"](gt, pred, eta=1.0, yaw_weight=1.0, softdtw_gamma=0.1),
        )
        nsp = _finite_float("compute_nsp", functions["compute_nsp"](gt[:, :3], pred[:, :3]))
        success = _finite_float(
            "compute_success", functions["compute_success"](gt[:, :3], pred[:, :3], success_thresh_m=1.0)
        )
        path_length = _finite_float("path_length_xyz", functions["path_length_xyz"](pred[:, :3]))
    # This optional smoke boundary must not terminate the core annotation
    # pipeline when a supplied metric module raises a BaseException subclass.
    except BaseException as exc:
        return _skip(f"{type(exc).__name__}: {exc}")

    return MetricSmokeResult(
        status="PASS",
        label=SMOKE_LABEL,
        avg_tcr=avg_tcr,
        ndtw=ndtw,
        nsp=nsp,
        success=success,
        path_length=path_length,
        limitations=_LIMITATIONS,
    )
