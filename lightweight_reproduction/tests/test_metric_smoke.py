from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from huge_lightweight.metric_smoke import MetricSmokeResult, run_metric_smoke


OFFICIAL_METRIC = Path(__file__).resolve().parents[2] / "HUGE-Bench" / "metric.py"
EXPECTED_LABEL = "Synthetic metric implementation smoke test \u2014 not a paper result."
NUMERIC_FIELDS = ("avg_tcr", "ndtw", "nsp", "success", "path_length")


def _assert_skip(result: MetricSmokeResult) -> None:
    assert result.status == "SKIP"
    assert result.label == EXPECTED_LABEL
    assert all(getattr(result, field) is None for field in NUMERIC_FIELDS)
    assert result.limitations


def test_current_official_metric_passes_with_expected_plain_float_values() -> None:
    result = run_metric_smoke(OFFICIAL_METRIC)

    assert result.status == "PASS"
    assert result.label == EXPECTED_LABEL
    assert result.avg_tcr == pytest.approx(1.0)
    assert result.ndtw == pytest.approx(1.0)
    assert result.nsp == pytest.approx(1.0)
    assert result.success == pytest.approx(1.0)
    assert result.path_length == pytest.approx(2.0)
    assert all(type(getattr(result, field)) is float for field in NUMERIC_FIELDS)
    assert "synthetic identical trajectories only" in result.limitations
    assert "softdtw_gamma=0.1" in result.limitations


def test_missing_path_skips_without_numeric_results(tmp_path: Path) -> None:
    result = run_metric_smoke(tmp_path / "missing_metric.py")

    _assert_skip(result)
    assert "does not exist" in result.limitations


def test_syntax_error_during_dynamic_import_skips(tmp_path: Path) -> None:
    metric_path = tmp_path / "syntax_error_metric.py"
    metric_path.write_text("def broken(:\n", encoding="utf-8")

    result = run_metric_smoke(metric_path)

    _assert_skip(result)
    assert "SyntaxError" in result.limitations


def test_import_time_system_exit_skips_and_removes_dynamic_module(tmp_path: Path) -> None:
    metric_path = tmp_path / "system_exit_import_metric.py"
    metric_path.write_text("raise SystemExit('import stopped')\n", encoding="utf-8")

    result = run_metric_smoke(metric_path)

    _assert_skip(result)
    assert "SystemExit: import stopped" in result.limitations
    assert not any(name.startswith("_huge_lightweight_metric_smoke_") for name in sys.modules)


def test_missing_required_function_skips_and_names_function(tmp_path: Path) -> None:
    metric_path = tmp_path / "incomplete_metric.py"
    metric_path.write_text(
        "\n".join(
            (
                "def compute_avg_tcr(gt, pred, thresholds): return 1.0",
                "def compute_ndtw(gt, pred, eta, yaw_weight, softdtw_gamma): return 1.0",
                "def compute_nsp(gt, pred): return 1.0",
                "def compute_success(gt, pred, success_thresh_m): return 1.0",
            )
        ),
        encoding="utf-8",
    )

    result = run_metric_smoke(metric_path)

    _assert_skip(result)
    assert "path_length_xyz" in result.limitations


@pytest.mark.parametrize(
    ("body", "expected_reason"),
    (
        ("raise RuntimeError('metric exploded')", "RuntimeError: metric exploded"),
        ("return float('nan')", "non-finite"),
    ),
)
def test_raising_or_nan_metric_result_skips_with_concrete_reason(
    tmp_path: Path, body: str, expected_reason: str
) -> None:
    metric_path = tmp_path / "bad_metric.py"
    metric_path.write_text(
        "\n".join(
            (
                "def compute_avg_tcr(gt, pred, thresholds): " + body,
                "def compute_ndtw(gt, pred, eta, yaw_weight, softdtw_gamma): return 1.0",
                "def compute_nsp(gt, pred): return 1.0",
                "def compute_success(gt, pred, success_thresh_m): return 1.0",
                "def path_length_xyz(pred): return 2.0",
            )
        ),
        encoding="utf-8",
    )

    result = run_metric_smoke(metric_path)

    _assert_skip(result)
    assert expected_reason in result.limitations


def test_metric_call_system_exit_skips_and_leaves_no_dynamic_module(tmp_path: Path) -> None:
    metric_path = tmp_path / "system_exit_call_metric.py"
    metric_path.write_text(
        "\n".join(
            (
                "def compute_avg_tcr(gt, pred, thresholds): raise SystemExit('call stopped')",
                "def compute_ndtw(gt, pred, eta, yaw_weight, softdtw_gamma): return 1.0",
                "def compute_nsp(gt, pred): return 1.0",
                "def compute_success(gt, pred, success_thresh_m): return 1.0",
                "def path_length_xyz(pred): return 2.0",
            )
        ),
        encoding="utf-8",
    )

    result = run_metric_smoke(metric_path)

    _assert_skip(result)
    assert "SystemExit: call stopped" in result.limitations
    assert not any(name.startswith("_huge_lightweight_metric_smoke_") for name in sys.modules)


def test_repeated_calls_use_isolated_dynamic_module_names_without_repo_sys_path(tmp_path: Path) -> None:
    names_path = tmp_path / "module_names.txt"
    metric_path = tmp_path / "recording_metric.py"
    metric_path.write_text(
        "\n".join(
            (
                "from pathlib import Path",
                f"NAMES_PATH = Path({str(names_path)!r})",
                "def compute_avg_tcr(gt, pred, thresholds):",
                "    with NAMES_PATH.open('a', encoding='utf-8') as handle:",
                "        handle.write(__name__ + '\\n')",
                "    return 1.0",
                "def compute_ndtw(gt, pred, eta, yaw_weight, softdtw_gamma): return 1.0",
                "def compute_nsp(gt, pred): return 1.0",
                "def compute_success(gt, pred, success_thresh_m): return 1.0",
                "def path_length_xyz(pred): return 2.0",
            )
        ),
        encoding="utf-8",
    )
    before_paths = tuple(sys.path)

    first = run_metric_smoke(metric_path)
    second = run_metric_smoke(metric_path)

    assert first.status == second.status == "PASS"
    names = names_path.read_text(encoding="utf-8").splitlines()
    assert len(names) == 2
    assert names[0] != names[1]
    assert tuple(sys.path) == before_paths
    assert str(OFFICIAL_METRIC.parent) not in sys.path
