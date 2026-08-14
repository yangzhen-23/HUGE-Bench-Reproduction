"""Deterministic summary tables derived from loaded HUGE-Bench annotations.
开始真正做特征统计分析
8个任务ID:("0", "hl", "orbit", "building",
 "road", "farm", "obstacle", "orbit_multi")
计算：
数据集整体统计 dataset_overview.csv
每个任务的统计 task_statistics.csv
Task × Scene 统计 task_scene_matrix.csv
Stage 统计 stage_statistics.csv

"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .loader import SPLIT_ORDER
from .models import AnnotationDataset, EpisodeRecord, StageRecord
from .validation import ValidationResult


TASK_ORDER = ("0", "hl", "orbit", "building", "road", "farm", "obstacle", "orbit_multi")
FRAME_RATE = 5.0

_OVERVIEW_FIELDS = (
    "split", "episodes", "episode_percent", "total_frames", "total_hours_at_5fps",
    "stage_segments", "mean_episode_frames", "median_episode_frames",
    "mean_stages_per_episode", "unique_tasks", "unique_scenes",
)
_TASK_FIELDS = (
    "split", "task_id", "episodes", "episode_percent", "total_frames",
    "total_hours_at_5fps", "mean_episode_frames", "median_episode_frames",
    "mean_stages_per_episode", "stage_segments", "unique_scenes",
)
_STAGE_FIELDS = (
    "task_id", "subtask_id", "stage_segments", "mean_duration_frames",
    "median_duration_frames", "p25_duration_frames", "p75_duration_frames",
    "mean_duration_seconds", "median_duration_seconds", "p25_duration_seconds",
    "p75_duration_seconds", "unique_subtask_texts",
)
_VALIDATION_FIELDS = ("check_id", "scope", "status", "expected", "actual", "details")


@dataclass(frozen=True)
class AnalysisEpisodeRow:
    split: str
    record: EpisodeRecord


@dataclass(frozen=True)
class AnalysisStageRow:
    split: str
    record: StageRecord


@dataclass(frozen=True)
class RepresentativeEpisode:
    split: str
    episode_index: int
    task_id: str
    length: int
    episode: EpisodeRecord


@dataclass(frozen=True)
class AnalysisResult:
    dataset_overview: tuple[Mapping[str, object], ...]
    task_statistics: tuple[Mapping[str, object], ...]
    task_scene_matrix: tuple[Mapping[str, object], ...]
    stage_statistics: tuple[Mapping[str, object], ...]
    episode_rows: tuple[AnalysisEpisodeRow, ...]
    stage_rows: tuple[AnalysisStageRow, ...]
    representative_episodes: Mapping[str, RepresentativeEpisode]
    scene_order: tuple[str, ...]


def analyze(data: AnnotationDataset) -> AnalysisResult:
    """Calculate stable statistics from records already loaded by ``load_annotations``."""
    episode_rows = tuple(
        AnalysisEpisodeRow(split, episode)
        for split in SPLIT_ORDER
        for episode in data.episodes_by_split.get(split, ())
    )
    stage_rows = tuple(
        AnalysisStageRow(split, stage)
        for split in SPLIT_ORDER
        for stage in data.stages_by_split.get(split, ())
    )
    scene_order = tuple(sorted({row.record.env_id for row in episode_rows}))
    all_episodes = tuple(row.record for row in episode_rows)
    all_stages = tuple(row.record for row in stage_rows)

    overview = tuple(
        _frozen_row(_overview_row(split, _episodes_for_split(episode_rows, split), _stages_for_split(stage_rows, split), len(all_episodes)))
        for split in (*SPLIT_ORDER, "overall")
    )
    task_statistics = tuple(
        _frozen_row(_task_row(split, task_id, _episodes_for_split(episode_rows, split), _stages_for_split(stage_rows, split), len(all_episodes)))
        for split in (*SPLIT_ORDER, "overall")
        for task_id in TASK_ORDER
    )
    task_scene_matrix = tuple(
        _frozen_row({"task_id": task_id, **{scene: sum(episode.task_id == task_id and episode.env_id == scene for episode in all_episodes) for scene in scene_order}})
        for task_id in TASK_ORDER
    )
    stage_statistics = tuple(_frozen_row(row) for row in _stage_rows(all_stages))
    representatives = MappingProxyType(_representatives(episode_rows))
    return AnalysisResult(
        dataset_overview=overview,
        task_statistics=task_statistics,
        task_scene_matrix=task_scene_matrix,
        stage_statistics=stage_statistics,
        episode_rows=episode_rows,
        stage_rows=stage_rows,
        representative_episodes=representatives,
        scene_order=scene_order,
    )


def write_csv_tables(result: AnalysisResult, validation: ValidationResult, output_dir: Path) -> tuple[Path, ...]:
    """Write the five deterministic UTF-8 tables and return them in reporting order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "dataset_overview.csv",
        output_dir / "task_statistics.csv",
        output_dir / "task_scene_matrix.csv",
        output_dir / "stage_statistics.csv",
        output_dir / "validation_report.csv",
    )
    _write_rows(paths[0], _OVERVIEW_FIELDS, result.dataset_overview)
    _write_rows(paths[1], _TASK_FIELDS, result.task_statistics)
    _write_rows(paths[2], ("task_id", *result.scene_order), result.task_scene_matrix)
    _write_rows(paths[3], _STAGE_FIELDS, result.stage_statistics)
    _write_rows(paths[4], _VALIDATION_FIELDS, tuple(_validation_row(check) for check in validation.checks))
    return paths


def _episodes_for_split(rows: tuple[AnalysisEpisodeRow, ...], split: str) -> tuple[EpisodeRecord, ...]:
    return tuple(row.record for row in rows) if split == "overall" else tuple(row.record for row in rows if row.split == split)


def _stages_for_split(rows: tuple[AnalysisStageRow, ...], split: str) -> tuple[StageRecord, ...]:
    return tuple(row.record for row in rows) if split == "overall" else tuple(row.record for row in rows if row.split == split)


def _overview_row(split: str, episodes: tuple[EpisodeRecord, ...], stages: tuple[StageRecord, ...], all_episode_count: int) -> dict[str, object]:
    return {
        "split": split,
        "episodes": len(episodes),
        "episode_percent": _percentage(len(episodes), all_episode_count),
        "total_frames": sum(episode.length for episode in episodes),
        "total_hours_at_5fps": sum(episode.length for episode in episodes) / FRAME_RATE / 3600.0,
        "stage_segments": len(stages),
        "mean_episode_frames": _mean([episode.length for episode in episodes]),
        "median_episode_frames": _median([episode.length for episode in episodes]),
        "mean_stages_per_episode": _mean([episode.num_stages for episode in episodes]),
        "unique_tasks": len({episode.task_id for episode in episodes}),
        "unique_scenes": len({episode.env_id for episode in episodes}),
    }


def _task_row(split: str, task_id: str, episodes: tuple[EpisodeRecord, ...], stages: tuple[StageRecord, ...], all_episode_count: int) -> dict[str, object]:
    selected_episodes = tuple(episode for episode in episodes if episode.task_id == task_id)
    selected_stages = tuple(stage for stage in stages if stage.task_id == task_id)
    denominator = all_episode_count if split == "overall" else len(episodes)
    return {
        "split": split,
        "task_id": task_id,
        "episodes": len(selected_episodes),
        "episode_percent": _percentage(len(selected_episodes), denominator),
        "total_frames": sum(episode.length for episode in selected_episodes),
        "total_hours_at_5fps": sum(episode.length for episode in selected_episodes) / FRAME_RATE / 3600.0,
        "mean_episode_frames": _mean([episode.length for episode in selected_episodes]),
        "median_episode_frames": _median([episode.length for episode in selected_episodes]),
        "mean_stages_per_episode": _mean([episode.num_stages for episode in selected_episodes]),
        "stage_segments": len(selected_stages),
        "unique_scenes": len({episode.env_id for episode in selected_episodes}),
    }


def _stage_rows(stages: tuple[StageRecord, ...]) -> tuple[dict[str, object], ...]:
    groups: dict[tuple[str, int], list[StageRecord]] = defaultdict(list)
    for stage in stages:
        groups[(stage.task_id, stage.subtask_id)].append(stage)
    task_rank = {task_id: position for position, task_id in enumerate(TASK_ORDER)}
    rows: list[dict[str, object]] = []
    for (task_id, subtask_id), group in sorted(groups.items(), key=lambda item: (task_rank.get(item[0][0], len(TASK_ORDER)), item[0][0], item[0][1])):
        durations = [stage.duration_frames for stage in group]
        p25, median, p75 = (float(value) for value in np.percentile(durations, [25, 50, 75]))
        rows.append({
            "task_id": task_id,
            "subtask_id": subtask_id,
            "stage_segments": len(group),
            "mean_duration_frames": _mean(durations),
            "median_duration_frames": median,
            "p25_duration_frames": p25,
            "p75_duration_frames": p75,
            "mean_duration_seconds": _mean(durations) / FRAME_RATE,
            "median_duration_seconds": median / FRAME_RATE,
            "p25_duration_seconds": p25 / FRAME_RATE,
            "p75_duration_seconds": p75 / FRAME_RATE,
            "unique_subtask_texts": len({stage.subtask_text for stage in group if stage.subtask_text.strip()}),
        })
    return tuple(rows)


def _representatives(rows: tuple[AnalysisEpisodeRow, ...]) -> dict[str, RepresentativeEpisode]:
    by_task: dict[str, list[AnalysisEpisodeRow]] = defaultdict(list)
    for row in rows:
        by_task[row.record.task_id].append(row)
    split_rank = {split: position for position, split in enumerate(SPLIT_ORDER)}
    selections: dict[str, RepresentativeEpisode] = {}
    for task_id in TASK_ORDER:
        candidates = by_task.get(task_id, [])
        if not candidates:
            continue
        median = float(np.median([row.record.length for row in candidates]))
        selected = min(candidates, key=lambda row: (abs(row.record.length - median), split_rank[row.split], row.record.episode_index))
        selections[task_id] = RepresentativeEpisode(selected.split, selected.record.episode_index, task_id, selected.record.length, selected.record)
    return selections


def _percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _mean(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _median(values: list[int]) -> float:
    return float(np.median(values)) if values else 0.0


def _frozen_row(row: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(row))


def _validation_row(check: object) -> Mapping[str, object]:
    return _frozen_row({field: getattr(check, field) for field in _VALIDATION_FIELDS})


def _write_rows(path: Path, fieldnames: tuple[str, ...], rows: tuple[Mapping[str, object], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row[field]) for field in fieldnames})


def _csv_value(value: object) -> object:
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:.6f}".rstrip("0").rstrip(".") or "0"
    return value
