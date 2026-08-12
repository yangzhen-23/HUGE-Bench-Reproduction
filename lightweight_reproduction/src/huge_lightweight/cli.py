"""One-command orchestration for the offline HUGE-Bench reproduction."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy
import PIL

from .analysis import analyze, write_csv_tables
from .loader import load_annotations
from .metric_smoke import run_metric_smoke
from .plots import create_all_figures
from .report import (
    FIGURE_PATHS,
    TABLE_PATHS,
    build_summary,
    write_html_report,
    write_markdown_report,
    write_summary_json,
)
from .validation import ValidationResult, validate_annotations


REPORT_MD = "HUGE_Bench_轻量复现报告.md"
REPORT_HTML = "HUGE_Bench_轻量复现报告.html"
_VALIDATION_FIELDS = ("check_id", "scope", "status", "expected", "actual", "details")


def run_pipeline(
    annotations_root: Path,
    repo_root: Path,
    output: Path,
    *,
    command: Sequence[str] | None = None,
) -> int:
    """Run the accepted annotation pipeline and return its process-style code."""
    annotations = Path(annotations_root).resolve()
    repository = Path(repo_root).resolve()
    destination = Path(output).resolve()
    started = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    tracing_was_active = tracemalloc.is_tracing()
    if not tracing_was_active:
        tracemalloc.start()

    try:
        data = load_annotations(annotations)
        validation = validate_annotations(annotations, data)
        manifest_sha256 = _sha256_file(annotations / "manifest.json")[0]
        repository_commit, commit_diagnostic = _repository_commit(repository)

        if not validation.ok:
            destination.mkdir(parents=True, exist_ok=True)
            _write_validation_csv(validation, destination / TABLE_PATHS[-1])
            peak = tracemalloc.get_traced_memory()[1]
            elapsed = _elapsed(started_counter)
            diagnostic = _validation_summary(
                validation,
                annotations=annotations,
                repository=repository,
                manifest_sha256=manifest_sha256,
                repository_commit=repository_commit,
                commit_diagnostic=commit_diagnostic,
                peak=peak,
                elapsed=elapsed,
            )
            write_summary_json(diagnostic, destination / "summary.json")
            failure_ids = ",".join(dict.fromkeys(check.check_id for check in validation.failures))
            print(
                f"LIGHTWEIGHT_REPRODUCTION_VALIDATION_FAILED failures={len(validation.failures)} ids={failure_ids}",
                file=sys.stderr,
            )
            return 2

        result = analyze(data)
        write_csv_tables(result, validation, destination / "tables")
        metric = run_metric_smoke(repository / "metric.py")
        create_all_figures(result, destination / "figures")

        # First pass makes report rendering part of the measured high-water mark.
        peak = tracemalloc.get_traced_memory()[1]
        elapsed = _elapsed(started_counter)
        summary = _build_success_summary(
            result, validation, metric, data, annotations, repository_commit,
            manifest_sha256, peak, elapsed,
        )
        _write_reports(destination, summary)

        # Rewrite with a fresh resource sample before stable output hashing.
        peak = tracemalloc.get_traced_memory()[1]
        elapsed = _elapsed(started_counter)
        summary = _build_success_summary(
            result, validation, metric, data, annotations, repository_commit,
            manifest_sha256, peak, elapsed,
        )
        _write_reports(destination, summary)

        outputs = _output_hashes(destination)
        final_peak = tracemalloc.get_traced_memory()[1]
        final_elapsed = _elapsed(started_counter)
        ended = datetime.now(timezone.utc)
        manifest = _run_manifest(
            command=list(command) if command is not None else _canonical_command(annotations, repository, destination),
            started=started,
            ended=ended,
            elapsed=final_elapsed,
            annotations=annotations,
            repository=repository,
            manifest_sha256=manifest_sha256,
            repository_commit=repository_commit,
            commit_diagnostic=commit_diagnostic,
            peak=final_peak,
            metric=metric,
            outputs=outputs,
        )
        _write_json(manifest, destination / "run_manifest.json")
        print(
            "LIGHTWEIGHT_REPRODUCTION_OK "
            f"episodes={len(result.episode_rows)} segments={len(result.stage_rows)}"
        )
        return 0
    finally:
        if not tracing_was_active and tracemalloc.is_tracing():
            tracemalloc.stop()


def main(
    argv: Sequence[str] | None = None,
    *,
    command: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Run the offline HUGE-Bench lightweight reproduction")
    parser.add_argument("--annotations-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    supplied = list(argv) if argv is not None else sys.argv[1:]
    recorded_command = list(command) if command is not None else (
        [sys.argv[0], *supplied] if argv is None else ["huge-lightweight", *supplied]
    )
    try:
        return run_pipeline(
            arguments.annotations_root,
            arguments.repo_root,
            arguments.output,
            command=recorded_command,
        )
    except Exception as exc:
        print(f"LIGHTWEIGHT_REPRODUCTION_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _build_success_summary(
    result: object,
    validation: ValidationResult,
    metric: object,
    data: object,
    annotations: Path,
    repository_commit: str,
    manifest_sha256: str,
    peak: int,
    elapsed: float,
) -> dict[str, object]:
    manifest = data.manifest  # type: ignore[attr-defined]
    return build_summary(
        result,  # type: ignore[arg-type]
        validation,
        metric,  # type: ignore[arg-type]
        source_dataset=manifest.source_dataset,
        raw_subtask_files=manifest.raw_subtask_files,
        annotations_root=str(annotations),
        repository_commit=repository_commit,
        manifest_sha256=manifest_sha256,
        peak_python_memory_bytes=peak,
        elapsed_seconds=elapsed,
        resources_measured=True,
    )


def _write_reports(destination: Path, summary: Mapping[str, object]) -> None:
    write_summary_json(summary, destination / "summary.json")
    write_markdown_report(summary, FIGURE_PATHS, TABLE_PATHS, destination / REPORT_MD)
    write_html_report(summary, FIGURE_PATHS, TABLE_PATHS, destination / REPORT_HTML)


def _write_validation_csv(validation: ValidationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_VALIDATION_FIELDS, lineterminator="\n")
        writer.writeheader()
        for check in validation.checks:
            writer.writerow({field: getattr(check, field) for field in _VALIDATION_FIELDS})


def _validation_summary(
    validation: ValidationResult,
    *,
    annotations: Path,
    repository: Path,
    manifest_sha256: str,
    repository_commit: str,
    commit_diagnostic: str | None,
    peak: int,
    elapsed: float,
) -> dict[str, object]:
    failures = validation.failures
    inputs: dict[str, object] = {
        "annotations_root": str(annotations),
        "repo_root": str(repository),
        "annotation_manifest_sha256": manifest_sha256,
        "repository_commit": repository_commit,
    }
    if commit_diagnostic is not None:
        inputs["repository_commit_diagnostic"] = commit_diagnostic
    return {
        "status": "VALIDATION_FAILED",
        "inputs": inputs,
        "validation": {
            "total_checks": len(validation.checks),
            "pass_count": len(validation.checks) - len(failures),
            "fail_count": len(failures),
            "failure_ids": [check.check_id for check in failures],
            "failure_scopes": [check.scope for check in failures],
            "failure_details": [check.details for check in failures],
        },
        "resources": {
            "peak_python_memory_bytes": int(peak),
            "elapsed_seconds": elapsed,
            "measured": True,
        },
    }


def _repository_commit(repo_root: Path) -> tuple[str, str | None]:
    argv = ["git", "-C", str(repo_root), "rev-parse", "HEAD"]
    try:
        completed = subprocess.run(
            argv, shell=False, capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        return "unavailable", f"git invocation failed: {type(exc).__name__}: {exc}"
    value = completed.stdout.strip()
    if completed.returncode == 0 and value:
        return value, None
    detail = completed.stderr.strip() or f"git exited with code {completed.returncode}"
    return "unavailable", detail.splitlines()[0]


def _output_hashes(destination: Path) -> dict[str, dict[str, object]]:
    relative_paths = sorted(
        (*TABLE_PATHS, *FIGURE_PATHS, REPORT_MD, REPORT_HTML, "summary.json")
    )
    outputs: dict[str, dict[str, object]] = {}
    for relative in relative_paths:
        digest, size = _sha256_file(destination / Path(relative))
        outputs[relative] = {"sha256": digest, "size_bytes": size}
    return outputs


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _run_manifest(
    *,
    command: list[str],
    started: datetime,
    ended: datetime,
    elapsed: float,
    annotations: Path,
    repository: Path,
    manifest_sha256: str,
    repository_commit: str,
    commit_diagnostic: str | None,
    peak: int,
    metric: object,
    outputs: Mapping[str, object],
) -> dict[str, object]:
    inputs: dict[str, object] = {
        "annotations_root": str(annotations),
        "repo_root": str(repository),
        "annotation_manifest_sha256": manifest_sha256,
        "repository_commit": repository_commit,
    }
    if commit_diagnostic is not None:
        inputs["repository_commit_diagnostic"] = commit_diagnostic
    return {
        "status": "SUCCESS",
        "command": command,
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "elapsed_seconds": elapsed,
        "runtime": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": PIL.__version__,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "inputs": inputs,
        "resources": {
            "peak_python_memory_bytes": int(peak),
            "elapsed_seconds": elapsed,
            "measured": True,
        },
        "metric_smoke": dataclasses.asdict(metric),
        "outputs": dict(outputs),
    }


def _canonical_command(annotations: Path, repository: Path, destination: Path) -> list[str]:
    return [
        "huge-lightweight",
        "--annotations-root", str(annotations),
        "--repo-root", str(repository),
        "--output", str(destination),
    ]


def _write_json(value: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_text(content, encoding="utf-8", newline="")


def _elapsed(started_counter: float) -> float:
    value = time.perf_counter() - started_counter
    if not math.isfinite(value) or value < 0:
        raise RuntimeError("elapsed time measurement is invalid")
    return value


if __name__ == "__main__":
    module_argv = sys.argv[1:]
    raise SystemExit(main(
        module_argv,
        command=[sys.executable, "-m", "huge_lightweight.cli", *module_argv],
    ))
