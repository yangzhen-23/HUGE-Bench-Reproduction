from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from huge_lightweight.analysis import analyze, write_csv_tables
from huge_lightweight.loader import load_annotations
from huge_lightweight.models import AnnotationDataset, EpisodeRecord, StageRecord
from huge_lightweight.validation import ValidationCheck, ValidationResult, validate_annotations


TASK_ORDER = ("0", "hl", "orbit", "building", "road", "farm", "obstacle", "orbit_multi")


def _episode(index: int, task_id: str, env_id: str, length: int, num_stages: int = 1) -> EpisodeRecord:
    return EpisodeRecord(
        episode_index=index,
        task_id=task_id,
        task_episode_index=index,
        env_id=env_id,
        traj_id=index,
        pose_start=0,
        pose_end=length - 1,
        length=length,
        instruction=f"Instruction for {task_id}.",
        num_stages=num_stages,
        annotation_provenance="original_raw",
        subtask_file=None,
    )


def _stage(index: int, task_id: str, env_id: str, subtask_id: int, duration: int, text: str) -> StageRecord:
    return StageRecord(
        episode_index=index,
        task_id=task_id,
        env_id=env_id,
        traj_id=index,
        annotation_provenance="original_raw",
        subtask_id=subtask_id,
        pose_start=0,
        pose_end=duration - 1,
        frame_start=0,
        frame_end=duration - 1,
        subtask_text=text,
    )


def _synthetic_data(valid_sidecar: Path) -> AnnotationDataset:
    base = load_annotations(valid_sidecar)
    episodes = {
        "train": (
            _episode(0, "0", "b_scene", 10),
            _episode(1, "hl", "a_scene", 4),
            _episode(2, "0", "a_scene", 30),
        ),
        "test_seen": (
            _episode(0, "0", "a_scene", 10),
            _episode(1, "orbit", "c_scene", 7),
        ),
        "test_unseen": (
            _episode(0, "0", "a_scene", 30),
        ),
    }
    stages = {
        "train": (
            _stage(0, "0", "b_scene", 0, 2, "alpha"),
            _stage(1, "hl", "a_scene", 0, 4, "beta"),
            _stage(2, "0", "a_scene", 0, 6, "alpha"),
        ),
        "test_seen": (
            _stage(0, "0", "a_scene", 0, 10, "alpha"),
            _stage(1, "orbit", "c_scene", 0, 7, "gamma"),
        ),
        "test_unseen": (_stage(0, "0", "a_scene", 0, 14, "alpha"),),
    }
    return replace(
        base,
        episodes_by_split=MappingProxyType(episodes),
        stages_by_split=MappingProxyType(stages),
    )


def _validation() -> ValidationResult:
    return ValidationResult(
        (
            ValidationCheck("first", "fixture", "PASS", "one", "one", "first detail"),
            ValidationCheck("second", "fixture", "PASS", "two", "two", "second detail"),
        )
    )


def test_fixture_overview_and_all_task_rows(valid_sidecar: Path):
    result = analyze(load_annotations(valid_sidecar))

    train = next(row for row in result.dataset_overview if row["split"] == "train")
    assert train["episodes"] == 1
    assert train["total_frames"] == 5
    assert train["total_hours_at_5fps"] == pytest.approx(5 / 5 / 3600)
    assert train["stage_segments"] == 2
    assert train["mean_stages_per_episode"] == 2
    assert [(row["split"], row["task_id"]) for row in result.task_statistics] == [
        (split, task_id) for split in ("train", "test_seen", "test_unseen", "overall") for task_id in TASK_ORDER
    ]


def test_task_scene_counts_and_stage_percentiles_use_default_linear_method(valid_sidecar: Path):
    result = analyze(_synthetic_data(valid_sidecar))

    assert result.scene_order == ("a_scene", "b_scene", "c_scene")
    matrix = {row["task_id"]: row for row in result.task_scene_matrix}
    assert matrix["0"] == {"task_id": "0", "a_scene": 3, "b_scene": 1, "c_scene": 0}
    assert matrix["hl"] == {"task_id": "hl", "a_scene": 1, "b_scene": 0, "c_scene": 0}
    assert all(row["task_id"] in TASK_ORDER for row in result.task_scene_matrix)
    group = next(row for row in result.stage_statistics if row["task_id"] == "0" and row["subtask_id"] == 0)
    expected = np.percentile([2, 6, 10, 14], [25, 50, 75])
    assert (group["p25_duration_frames"], group["median_duration_frames"], group["p75_duration_frames"]) == pytest.approx(expected)
    assert group["unique_subtask_texts"] == 1


def test_representative_selection_uses_split_then_episode_tie_break(valid_sidecar: Path):
    result = analyze(_synthetic_data(valid_sidecar))

    selected = result.representative_episodes["0"]
    assert selected.episode_index == 0
    assert selected.split == "train"
    assert selected.length == 10


def test_csv_tables_have_exact_schema_types_and_deterministic_order(valid_sidecar: Path, tmp_path: Path):
    data = load_annotations(valid_sidecar)
    result = analyze(data)
    validation = validate_annotations(valid_sidecar, data)
    before = validation.checks
    first = write_csv_tables(result, validation, tmp_path / "first")
    second = write_csv_tables(result, validation, tmp_path / "second")

    assert [path.name for path in first] == [
        "dataset_overview.csv",
        "task_statistics.csv",
        "task_scene_matrix.csv",
        "stage_statistics.csv",
        "validation_report.csv",
    ]
    assert validation.checks == before
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]
    expected_headers = {
        "dataset_overview.csv": ["split", "episodes", "episode_percent", "total_frames", "total_hours_at_5fps", "stage_segments", "mean_episode_frames", "median_episode_frames", "mean_stages_per_episode", "unique_tasks", "unique_scenes"],
        "task_statistics.csv": ["split", "task_id", "episodes", "episode_percent", "total_frames", "total_hours_at_5fps", "mean_episode_frames", "median_episode_frames", "mean_stages_per_episode", "stage_segments", "unique_scenes"],
        "task_scene_matrix.csv": ["task_id", "1_office"],
        "stage_statistics.csv": ["task_id", "subtask_id", "stage_segments", "mean_duration_frames", "median_duration_frames", "p25_duration_frames", "p75_duration_frames", "mean_duration_seconds", "median_duration_seconds", "p25_duration_seconds", "p75_duration_seconds", "unique_subtask_texts"],
        "validation_report.csv": ["check_id", "scope", "status", "expected", "actual", "details"],
    }
    expected_row_counts = {
        "dataset_overview.csv": 4,
        "task_statistics.csv": 32,
        "task_scene_matrix.csv": 8,
        "stage_statistics.csv": 2,
        "validation_report.csv": len(validation.checks),
    }
    for path in first:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == expected_headers[path.name]
        assert len(rows) == expected_row_counts[path.name] + 1
        assert path.read_text(encoding="utf-8").count("\n") == len(rows)
    overview_rows = list(csv.DictReader(first[0].open("r", encoding="utf-8", newline="")))
    assert isinstance(int(overview_rows[0]["episodes"]), int)
    assert isinstance(float(overview_rows[0]["total_hours_at_5fps"]), float)
    validation_rows = list(csv.DictReader(first[-1].open("r", encoding="utf-8", newline="")))
    assert [row["check_id"] for row in validation_rows] == [check.check_id for check in validation.checks]
