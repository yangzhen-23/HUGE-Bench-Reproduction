from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import pytest


REPORT_MD = "HUGE_Bench_轻量复现报告.md"
REPORT_HTML = "HUGE_Bench_轻量复现报告.html"
TABLES = {
    "dataset_overview.csv",
    "task_statistics.csv",
    "task_scene_matrix.csv",
    "stage_statistics.csv",
    "validation_report.csv",
}


def _cli_module():
    try:
        from huge_lightweight import cli
    except ImportError:
        pytest.fail("huge_lightweight.cli is not implemented")
    return cli


def _minimal_metric(repo: Path) -> None:
    repo.mkdir()
    (repo / "metric.py").write_text(
        """import numpy as np
def compute_avg_tcr(gt, pred, thresholds): return 1.0
def compute_ndtw(gt, pred, eta, yaw_weight, softdtw_gamma): return 1.0
def compute_nsp(gt, pred): return 1.0
def compute_success(gt, pred, success_thresh_m): return 1.0
def path_length_xyz(path): return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
""",
        encoding="utf-8",
    )


@pytest.fixture
def successful_run(tmp_path: Path, valid_sidecar: Path, capsys):
    cli = _cli_module()
    repo = tmp_path / "repo"
    _minimal_metric(repo)
    output = tmp_path / "output"
    command = ["huge-lightweight", "--token", "value with spaces"]
    code = cli.run_pipeline(valid_sidecar, repo, output, command=command)
    captured = capsys.readouterr()
    return output, repo, command, code, captured


def test_run_pipeline_creates_complete_success_inventory(successful_run, capsys):
    output, _, _, code, captured = successful_run

    assert code == 0
    assert captured.out == "LIGHTWEIGHT_REPRODUCTION_OK episodes=1 segments=2\n"
    assert captured.err == ""
    assert {path.name for path in (output / "tables").iterdir()} == TABLES
    assert len(tuple((output / "figures").glob("*.png"))) == 8
    assert (output / REPORT_MD).is_file()
    assert (output / REPORT_HTML).is_file()
    assert (output / "summary.json").is_file()
    assert (output / "run_manifest.json").is_file()


def test_success_summary_uses_measured_resources_and_no_placeholder(successful_run):
    output, _, _, _, _ = successful_run
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["resources"]["measured"] is True
    for name in (REPORT_MD, REPORT_HTML):
        text = (output / name).read_text(encoding="utf-8")
        assert "Task 6 中的 0 bytes 和 0.000 秒是占位值" not in text


def test_manifest_contract_hashes_every_non_manifest_output(successful_run):
    output, repo, command, _, _ = successful_run
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))

    assert list(manifest) == [
        "status", "command", "started_utc", "ended_utc", "elapsed_seconds",
        "runtime", "platform", "inputs", "resources", "metric_smoke", "outputs",
    ]
    assert manifest["status"] == "SUCCESS"
    assert manifest["command"] == command
    assert set(manifest["runtime"]) == {"python", "numpy", "matplotlib", "pillow"}
    assert all(isinstance(value, str) and value for value in manifest["runtime"].values())
    assert set(manifest["platform"]) == {"platform", "system", "release", "machine", "processor"}
    assert manifest["inputs"]["repo_root"] == str(repo.resolve())
    assert manifest["resources"]["measured"] is True
    assert isinstance(manifest["resources"]["peak_python_memory_bytes"], int)
    assert manifest["resources"]["peak_python_memory_bytes"] >= 0
    assert manifest["resources"]["elapsed_seconds"] >= 0.0
    assert set(manifest["metric_smoke"]) == {
        "status", "label", "avg_tcr", "ndtw", "nsp", "success", "path_length", "limitations",
    }
    assert manifest["metric_smoke"]["status"] == "PASS"

    paths = list(manifest["outputs"])
    assert paths == sorted(paths)
    assert len(paths) == 16
    assert "run_manifest.json" not in paths
    for relative, metadata in manifest["outputs"].items():
        digest = hashlib.sha256()
        size = 0
        with (output / relative).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
                size += len(chunk)
        assert metadata == {
            "sha256": digest.hexdigest(),
            "size_bytes": size,
        }


def test_manifest_timestamps_are_ordered_aware_utc(successful_run):
    output, _, command, _, _ = successful_run
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    started = datetime.fromisoformat(manifest["started_utc"].replace("Z", "+00:00"))
    ended = datetime.fromisoformat(manifest["ended_utc"].replace("Z", "+00:00"))
    assert started.tzinfo is not None and started.utcoffset() == timezone.utc.utcoffset(started)
    assert ended >= started
    assert manifest["command"] == command


def test_repository_commit_helper_returns_real_temp_commit(monkeypatch, tmp_path: Path):
    cli = _cli_module()
    completed = subprocess.CompletedProcess(["git"], 0, stdout="a" * 40 + "\n", stderr="")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return completed

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    value, diagnostic = cli._repository_commit(tmp_path)
    assert value == "a" * 40 and diagnostic is None
    assert calls == [(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], {
        "shell": False, "capture_output": True, "text": True, "check": False,
    })]


def test_validation_failure_writes_only_diagnostics(valid_sidecar: Path, tmp_path: Path, capsys):
    cli = _cli_module()
    manifest_path = valid_sidecar / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["total_episodes"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "failed"

    assert cli.run_pipeline(valid_sidecar, tmp_path / "repo", output) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("LIGHTWEIGHT_REPRODUCTION_VALIDATION_FAILED ")
    known = sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file())
    assert known == ["summary.json", "tables/validation_report.csv"]
    with (output / "tables" / "validation_report.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and list(rows[0]) == ["check_id", "scope", "status", "expected", "actual", "details"]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "VALIDATION_FAILED"
    assert summary["validation"]["fail_count"] > 0
    assert summary["validation"]["failure_ids"]
    assert summary["validation"]["failure_scopes"]
    assert summary["validation"]["failure_details"]
    assert summary["resources"]["measured"] is True


def test_missing_metric_is_nonblocking_skip(valid_sidecar: Path, tmp_path: Path, capsys):
    cli = _cli_module()
    output = tmp_path / "skip-output"
    code = cli.run_pipeline(valid_sidecar, tmp_path / "repo-without-metric", output)
    capsys.readouterr()
    assert code == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert summary["metric_smoke"]["status"] == "SKIP"
    assert manifest["metric_smoke"]["status"] == "SKIP"


def test_main_parses_relative_paths_and_exception_boundary(monkeypatch, tmp_path: Path, capsys):
    cli = _cli_module()
    received = []
    monkeypatch.chdir(tmp_path)

    def fake_pipeline(annotations_root, repo_root, output, *, command=None):
        received.append((annotations_root, repo_root, output, command))
        return 2

    monkeypatch.setattr(cli, "run_pipeline", fake_pipeline)
    argv = ["--annotations-root", "ann", "--repo-root", "repo", "--output", "out"]
    assert cli.main(argv) == 2
    assert received == [(Path("ann"), Path("repo"), Path("out"), ["huge-lightweight", *argv])]

    monkeypatch.setattr(cli, "run_pipeline", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk unavailable")))
    assert cli.main(argv) == 1
    assert capsys.readouterr().err == "LIGHTWEIGHT_REPRODUCTION_ERROR OSError: disk unavailable\n"

    monkeypatch.setattr(cli, "run_pipeline", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        cli.main(argv)


def test_module_entry_records_exact_python_m_command(valid_sidecar: Path, tmp_path: Path):
    output = tmp_path / "module-output"
    repo = tmp_path / "repo"
    argv = [
        "--annotations-root", str(valid_sidecar),
        "--repo-root", str(repo),
        "--output", str(output),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

    completed = subprocess.run(
        [sys.executable, "-m", "huge_lightweight.cli", *argv],
        cwd=tmp_path,
        env=environment,
        shell=False,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"] == [sys.executable, "-m", "huge_lightweight.cli", *argv]


def test_unrelated_output_file_survives_success(valid_sidecar: Path, tmp_path: Path, capsys):
    cli = _cli_module()
    repo = tmp_path / "repo"
    _minimal_metric(repo)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep-me.txt"
    sentinel.write_text("user data", encoding="utf-8")
    assert cli.run_pipeline(valid_sidecar, repo, output) == 0
    capsys.readouterr()
    assert sentinel.read_text(encoding="utf-8") == "user data"


@pytest.mark.parametrize("initially_active", [False, True])
def test_pipeline_preserves_caller_tracemalloc_state(initially_active, valid_sidecar: Path, tmp_path: Path, capsys):
    cli = _cli_module()
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    if initially_active:
        tracemalloc.start()
    try:
        output = tmp_path / ("active" if initially_active else "stopped")
        assert cli.run_pipeline(valid_sidecar, tmp_path / "repo", output) == 0
        capsys.readouterr()
        assert tracemalloc.is_tracing() is initially_active
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()


def _caller_peak() -> int:
    allocation = bytearray(120_000_000)
    peak = tracemalloc.get_traced_memory()[1]
    del allocation
    gc.collect()
    return peak


def test_active_caller_peak_survives_success(valid_sidecar: Path, tmp_path: Path, capsys):
    cli = _cli_module()
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    tracemalloc.start()
    before_peak = _caller_peak()
    try:
        assert cli.run_pipeline(valid_sidecar, tmp_path / "repo", tmp_path / "output") == 0
        capsys.readouterr()
        assert tracemalloc.is_tracing()
        assert tracemalloc.get_traced_memory()[1] >= before_peak
    finally:
        tracemalloc.stop()


def test_active_caller_peak_survives_exception(monkeypatch, tmp_path: Path):
    cli = _cli_module()
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    tracemalloc.start()
    before_peak = _caller_peak()
    monkeypatch.setattr(cli, "load_annotations", lambda path: (_ for _ in ()).throw(OSError("forced load failure")))
    try:
        with pytest.raises(OSError, match="forced load failure"):
            cli.run_pipeline(tmp_path / "annotations", tmp_path / "repo", tmp_path / "output")
        assert tracemalloc.is_tracing()
        assert tracemalloc.get_traced_memory()[1] >= before_peak
    finally:
        tracemalloc.stop()
