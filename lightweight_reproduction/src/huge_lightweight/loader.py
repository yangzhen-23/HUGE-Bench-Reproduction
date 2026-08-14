"""把硬盘上的标注读进 Python
`load_annotations()` 的核心逻辑非常直白：
manifest = load_manifest(root)

for split in:
    train
    test_seen
    test_unseen

    读取 episode_mapping
    读取 stage_segments

然后：
JSON
 ↓
dict
 ↓
EpisodeRecord / StageRecord
 ↓
AnnotationDataset

"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar

from .models import (
    AnnotationDataset,
    AnnotationManifest,
    EpisodeRecord,
    ManifestFile,
    SplitManifest,
    StageRecord,
)

SPLIT_ORDER = ("train", "test_seen", "test_unseen")


class JsonlParseError(ValueError):
    """A JSONL syntax, shape, or record-conversion error with source context."""


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    """Yield one-based line numbers and object records without reading the whole file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise JsonlParseError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(value, dict):
                raise JsonlParseError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def load_manifest(root: Path) -> AnnotationManifest:
    manifest_path = root / "manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{manifest_path}: invalid JSON: {error.msg}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{manifest_path}: expected a JSON object")

    try:
        splits_raw = _object(raw, "splits")
        splits = MappingProxyType(
            {
                split: SplitManifest(
                    episodes=_integer(_object(splits_raw, split), "episodes"),
                    stage_segments=_integer(_object(splits_raw, split), "stage_segments"),
                )
                for split in SPLIT_ORDER
            }
        )
        files = tuple(
            ManifestFile(
                path=_string(entry, "path"),
                size_bytes=_integer(entry, "size_bytes"),
                sha256=_string(entry, "sha256"),
            )
            for entry in _object_list(raw, "files")
        )
        return AnnotationManifest(
            format_version=_integer(raw, "format_version"),
            source_dataset=_string(raw, "source_dataset"),
            task_order=tuple(_string_value(value, "task_order item") for value in _list(raw, "task_order")),
            splits=splits,
            total_episodes=_integer(raw, "total_episodes"),
            total_stage_segments=_integer(raw, "total_stage_segments"),
            raw_subtask_files=_integer(raw, "raw_subtask_files"),
            provenance=MappingProxyType(
                {key: _string_value(value, f"provenance.{key}") for key, value in _object(raw, "provenance").items()}
            ),
            files=files,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{manifest_path}: {error}") from error


def load_annotations(root: Path) -> AnnotationDataset:
    manifest = load_manifest(root)
    episodes_by_split: dict[str, tuple[EpisodeRecord, ...]] = {}
    stages_by_split: dict[str, tuple[StageRecord, ...]] = {}
    for split in SPLIT_ORDER:
        episodes_by_split[split] = tuple(
            _convert_jsonl_record(path, line_number, record, _episode_from_record)
            for path, line_number, record in _records(root / "episode_mapping" / f"{split}.jsonl")
        )
        stages_by_split[split] = tuple(
            _convert_jsonl_record(path, line_number, record, _stage_from_record)
            for path, line_number, record in _records(root / "stage_segments" / f"{split}.jsonl")
        )
    return AnnotationDataset(
        root=root,
        manifest=manifest,
        episodes_by_split=MappingProxyType(episodes_by_split),
        stages_by_split=MappingProxyType(stages_by_split),
    )


T = TypeVar("T")


def _records(path: Path) -> Iterator[tuple[Path, int, dict[str, object]]]:
    for line_number, record in iter_jsonl(path):
        yield path, line_number, record


def _convert_jsonl_record(path: Path, line_number: int, record: dict[str, object], converter: object) -> T:
    try:
        return converter(record)  # type: ignore[operator, no-any-return]
    except (KeyError, TypeError, ValueError) as error:
        raise JsonlParseError(f"{path}:{line_number}: invalid typed record: {error}") from error


def _episode_from_record(record: Mapping[str, object]) -> EpisodeRecord:
    return EpisodeRecord(
        episode_index=_integer(record, "episode_index"),
        task_id=_string(record, "task_id"),
        task_episode_index=_integer(record, "task_episode_index"),
        env_id=_string(record, "env_id"),
        traj_id=_nullable_integer(record, "traj_id"),
        pose_start=_nullable_integer(record, "pose_start"),
        pose_end=_nullable_integer(record, "pose_end"),
        length=_integer(record, "length"),
        instruction=_string(record, "instruction"),
        num_stages=_integer(record, "num_stages"),
        annotation_provenance=_string(record, "annotation_provenance"),
        subtask_file=_nullable_string(record, "subtask_file"),
    )


def _stage_from_record(record: Mapping[str, object]) -> StageRecord:
    return StageRecord(
        episode_index=_integer(record, "episode_index"),
        task_id=_string(record, "task_id"),
        env_id=_string(record, "env_id"),
        traj_id=_nullable_integer(record, "traj_id"),
        annotation_provenance=_string(record, "annotation_provenance"),
        subtask_id=_integer(record, "subtask_id"),
        pose_start=_nullable_integer(record, "pose_start"),
        pose_end=_nullable_integer(record, "pose_end"),
        frame_start=_integer(record, "frame_start"),
        frame_end=_integer(record, "frame_end"),
        subtask_text=_string(record, "subtask_text"),
    )


def _object(mapping: Mapping[str, object], key: str) -> dict[str, object]:
    value = mapping[key]
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object")
    return value


def _object_list(mapping: Mapping[str, object], key: str) -> list[dict[str, object]]:
    return [_object({"item": value}, "item") for value in _list(mapping, key)]


def _list(mapping: Mapping[str, object], key: str) -> list[object]:
    value = mapping[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return value


def _integer(mapping: Mapping[str, object], key: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _nullable_integer(mapping: Mapping[str, object], key: str) -> int | None:
    value = mapping[key]
    if value is None:
        return None
    return _integer(mapping, key)


def _string(mapping: Mapping[str, object], key: str) -> str:
    return _string_value(mapping[key], key)


def _nullable_string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping[key]
    return None if value is None else _string_value(value, key)


def _string_value(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value
