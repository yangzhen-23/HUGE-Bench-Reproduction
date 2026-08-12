from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ManifestFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SplitManifest:
    episodes: int
    stage_segments: int


@dataclass(frozen=True)
class AnnotationManifest:
    format_version: int
    source_dataset: str
    task_order: tuple[str, ...]
    splits: Mapping[str, SplitManifest]
    total_episodes: int
    total_stage_segments: int
    raw_subtask_files: int
    provenance: Mapping[str, str]
    files: tuple[ManifestFile, ...]


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    task_id: str
    task_episode_index: int
    env_id: str
    traj_id: int | None
    pose_start: int | None
    pose_end: int | None
    length: int
    instruction: str
    num_stages: int
    annotation_provenance: str
    subtask_file: str | None


@dataclass(frozen=True)
class StageRecord:
    episode_index: int
    task_id: str
    env_id: str
    traj_id: int | None
    annotation_provenance: str
    subtask_id: int
    pose_start: int | None
    pose_end: int | None
    frame_start: int
    frame_end: int
    subtask_text: str

    @property
    def duration_frames(self) -> int:
        return self.frame_end - self.frame_start + 1

    @property
    def duration_seconds(self) -> float:
        return self.duration_frames / 5.0


@dataclass(frozen=True)
class AnnotationDataset:
    root: Path
    manifest: AnnotationManifest
    episodes_by_split: Mapping[str, tuple[EpisodeRecord, ...]]
    stages_by_split: Mapping[str, tuple[StageRecord, ...]]
