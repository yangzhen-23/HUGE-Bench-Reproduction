from __future__ import annotations

import pytest

from huge_lightweight.loader import JsonlParseError, iter_jsonl, load_annotations


def test_loads_typed_episode_and_closed_interval_stage(valid_sidecar):
    data = load_annotations(valid_sidecar)
    episode = data.episodes_by_split["train"][0]
    stages = data.stages_by_split["train"]

    assert episode.length == 5
    assert episode.task_id == "0"
    assert stages[1].duration_frames == 3
    assert stages[1].duration_seconds == 0.6


def test_jsonl_error_reports_file_and_line(valid_sidecar):
    path = valid_sidecar / "episode_mapping" / "train.jsonl"
    path.write_text('{"episode_index": 0}\nnot-json\n', encoding="utf-8")

    with pytest.raises(JsonlParseError, match=r"train\.jsonl:2"):
        list(iter_jsonl(path))


def test_non_object_json_reports_file_and_line(valid_sidecar):
    path = valid_sidecar / "episode_mapping" / "train.jsonl"
    path.write_text('["not", "an", "object"]\n', encoding="utf-8")

    with pytest.raises(JsonlParseError, match=r"train\.jsonl:1"):
        list(iter_jsonl(path))


def test_typed_field_error_reports_file_and_line(valid_sidecar):
    path = valid_sidecar / "episode_mapping" / "train.jsonl"
    path.write_text('{"episode_index": "zero"}\n', encoding="utf-8")

    with pytest.raises(JsonlParseError, match=r"train\.jsonl:1"):
        load_annotations(valid_sidecar)
