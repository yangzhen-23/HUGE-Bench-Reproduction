from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _file_entry(root: Path, relative_path: str) -> dict[str, object]:
    path = root / relative_path
    contents = path.read_bytes()
    return {
        "path": relative_path,
        "size_bytes": len(contents),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }


@pytest.fixture
def valid_sidecar(tmp_path: Path) -> Path:
    root = tmp_path / "sidecar"
    episode_dir = root / "episode_mapping"
    stage_dir = root / "stage_segments"
    episode_dir.mkdir(parents=True)
    stage_dir.mkdir()

    train_episode = {
        "episode_index": 0,
        "task_id": "0",
        "task_episode_index": 0,
        "env_id": "1_office",
        "traj_id": 0,
        "pose_start": 0,
        "pose_end": 4,
        "length": 5,
        "instruction": "Fly to the office.",
        "num_stages": 2,
        "annotation_provenance": "original_raw",
        "subtask_file": "raw_subtasks/task_0/1_office/subtask.txt",
    }
    train_stages = [
        {
            "episode_index": 0,
            "task_id": "0",
            "env_id": "1_office",
            "traj_id": 0,
            "annotation_provenance": "original_raw",
            "subtask_id": 0,
            "pose_start": 0,
            "pose_end": 1,
            "frame_start": 0,
            "frame_end": 1,
            "subtask_text": "Turn toward the office.",
        },
        {
            "episode_index": 0,
            "task_id": "0",
            "env_id": "1_office",
            "traj_id": 0,
            "annotation_provenance": "original_raw",
            "subtask_id": 1,
            "pose_start": 2,
            "pose_end": 4,
            "frame_start": 2,
            "frame_end": 4,
            "subtask_text": "Fly to the office.",
        },
    ]
    (episode_dir / "train.jsonl").write_text(
        json.dumps(train_episode) + "\n", encoding="utf-8"
    )
    (stage_dir / "train.jsonl").write_text(
        "".join(json.dumps(stage) + "\n" for stage in train_stages), encoding="utf-8"
    )
    for split in ("test_seen", "test_unseen"):
        (episode_dir / f"{split}.jsonl").write_text("", encoding="utf-8")
        (stage_dir / f"{split}.jsonl").write_text("", encoding="utf-8")

    file_paths = [
        "episode_mapping/train.jsonl",
        "episode_mapping/test_seen.jsonl",
        "episode_mapping/test_unseen.jsonl",
        "stage_segments/train.jsonl",
        "stage_segments/test_seen.jsonl",
        "stage_segments/test_unseen.jsonl",
    ]
    manifest = {
        "format_version": 1,
        "source_dataset": "fixture-dataset",
        "task_order": ["0"],
        "splits": {
            "train": {"episodes": 1, "stage_segments": 2},
            "test_seen": {"episodes": 0, "stage_segments": 0},
            "test_unseen": {"episodes": 0, "stage_segments": 0},
        },
        "total_episodes": 1,
        "total_stage_segments": 2,
        "raw_subtask_files": 0,
        "provenance": {"original_raw": "fixture"},
        "files": [_file_entry(root, path) for path in file_paths],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root
