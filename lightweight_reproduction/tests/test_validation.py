from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from huge_lightweight import validation
from huge_lightweight.loader import load_annotations
from huge_lightweight.models import AnnotationDataset
from huge_lightweight.validation import validate_annotations


def _with_split_data(
    data: AnnotationDataset,
    *,
    episodes: tuple | None = None,
    stages: tuple | None = None,
) -> AnnotationDataset:
    episode_splits = dict(data.episodes_by_split)
    stage_splits = dict(data.stages_by_split)
    if episodes is not None:
        episode_splits["train"] = episodes
    if stages is not None:
        stage_splits["train"] = stages
    return replace(
        data,
        episodes_by_split=MappingProxyType(episode_splits),
        stages_by_split=MappingProxyType(stage_splits),
    )


def _failed_ids(data: AnnotationDataset) -> set[str]:
    return {check.check_id for check in validate_annotations(data.root, data).failures}


def test_valid_sidecar_returns_stable_passing_checks(valid_sidecar: Path):
    result = validate_annotations(valid_sidecar, load_annotations(valid_sidecar))

    assert result.ok
    assert result.failures == ()
    assert result.checks[0].check_id == "manifest.format_version"
    assert all(
        check.status == "PASS"
        and check.check_id
        and check.scope
        and check.expected
        and check.actual
        and check.details
        for check in result.checks
    )


def test_changed_file_bytes_fail_size_and_hash(valid_sidecar: Path):
    data = load_annotations(valid_sidecar)
    path = valid_sidecar / "stage_segments" / "train.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")

    assert {"manifest.file_size", "manifest.sha256"} <= _failed_ids(data)


def test_missing_manifest_file_accumulates_file_failures(valid_sidecar: Path):
    data = load_annotations(valid_sidecar)
    (valid_sidecar / "stage_segments" / "train.jsonl").unlink()

    assert {"manifest.file_exists", "manifest.file_size", "manifest.sha256"} <= _failed_ids(data)


def test_crlf_working_tree_file_matches_published_lf_bytes(valid_sidecar: Path):
    path = valid_sidecar / "stage_segments" / "train.jsonl"
    published = path.read_bytes()
    crlf_working_tree = published.replace(b"\n", b"\r\n")
    path.write_bytes(crlf_working_tree)
    manifest_path = valid_sidecar / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["path"] == "stage_segments/train.jsonl":
            entry["size_bytes"] = len(published)
            entry["sha256"] = hashlib.sha256(published).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_annotations(valid_sidecar, load_annotations(valid_sidecar))
    checks = [check for check in result.checks if check.scope == "stage_segments/train.jsonl"]

    assert all(check.status == "PASS" for check in checks)
    assert all("raw=" in check.actual and "canonical=" in check.actual for check in checks[1:])
    assert all("working-tree CRLF normalized to published LF bytes" in check.details for check in checks[1:])


def test_crlf_normalization_rejects_other_content_changes(valid_sidecar: Path):
    path = valid_sidecar / "stage_segments" / "train.jsonl"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n") + b" \r\n")

    failed = _failed_ids(load_annotations(valid_sidecar))

    assert {"manifest.file_size", "manifest.sha256"} <= failed


def test_split_names_rejects_extra_or_missing_stage_split(valid_sidecar: Path):
    data = load_annotations(valid_sidecar)
    stages = dict(data.stages_by_split)
    stages.pop("test_unseen")
    stages["unexpected"] = ()
    invalid = replace(data, stages_by_split=MappingProxyType(stages))

    assert "split.names" in _failed_ids(invalid)


@pytest.mark.parametrize(
    ("mutation", "expected_id"),
    [
        (lambda data: _with_split_data(data, episodes=(data.episodes_by_split["train"][0], replace(data.episodes_by_split["train"][0], task_episode_index=1))), "episode.index_contiguous"),
        (lambda data: _with_split_data(data, episodes=(replace(data.episodes_by_split["train"][0], episode_index=1),)), "episode.index_contiguous"),
        (lambda data: _with_split_data(data, stages=(data.stages_by_split["train"][0], replace(data.stages_by_split["train"][1], frame_start=3))), "stage.coverage_contiguous"),
        (lambda data: _with_split_data(data, stages=(data.stages_by_split["train"][0], replace(data.stages_by_split["train"][1], frame_start=1))), "stage.coverage_contiguous"),
        (lambda data: _with_split_data(data, stages=(data.stages_by_split["train"][0], replace(data.stages_by_split["train"][1], frame_end=3))), "stage.coverage_complete"),
    ],
)
def test_structural_invalid_records_are_detected(valid_sidecar: Path, mutation, expected_id: str):
    assert expected_id in _failed_ids(mutation(load_annotations(valid_sidecar)))


@pytest.mark.parametrize(
    "field",
    ["task_id", "env_id", "annotation_provenance"],
)
def test_stage_identity_mismatch_is_detected(valid_sidecar: Path, field: str):
    data = load_annotations(valid_sidecar)
    stage = replace(data.stages_by_split["train"][0], **{field: "mismatch"})

    assert "stage.fields_consistent" in _failed_ids(
        _with_split_data(data, stages=(stage, data.stages_by_split["train"][1]))
    )


@pytest.mark.parametrize("text", ["", " \t "])
def test_blank_subtask_text_is_detected(valid_sidecar: Path, text: str):
    data = load_annotations(valid_sidecar)
    stage = replace(data.stages_by_split["train"][0], subtask_text=text)

    assert "stage.subtask_text_nonempty" in _failed_ids(
        _with_split_data(data, stages=(stage, data.stages_by_split["train"][1]))
    )


@pytest.mark.parametrize(
    "episode_provenance,stage_provenance",
    [("recovered_from_released_actions", "recovered_from_released_actions"), ("invalid", "invalid")],
)
def test_invalid_provenance_is_detected(valid_sidecar: Path, episode_provenance: str, stage_provenance: str):
    data = load_annotations(valid_sidecar)
    episode = replace(data.episodes_by_split["train"][0], annotation_provenance=episode_provenance)
    stages = tuple(replace(stage, annotation_provenance=stage_provenance) for stage in data.stages_by_split["train"])

    assert "provenance.allowed" in _failed_ids(_with_split_data(data, episodes=(episode,), stages=stages))


def test_orphan_stage_is_detected(valid_sidecar: Path):
    data = load_annotations(valid_sidecar)
    orphan = replace(data.stages_by_split["train"][0], episode_index=99)

    assert "stage.orphan_episode" in _failed_ids(
        _with_split_data(data, stages=(orphan, data.stages_by_split["train"][1]))
    )


def test_orphan_stage_accumulates_blank_text_and_invalid_provenance(valid_sidecar: Path):
    data = load_annotations(valid_sidecar)
    orphan = replace(
        data.stages_by_split["train"][0],
        episode_index=99,
        subtask_text=" \t ",
        annotation_provenance="invalid",
    )

    assert {"stage.orphan_episode", "stage.subtask_text_nonempty", "provenance.allowed"} <= _failed_ids(
        _with_split_data(data, stages=(orphan, data.stages_by_split["train"][1]))
    )


def test_streamed_fingerprints_normalize_crlf_across_chunk_boundary(tmp_path: Path, monkeypatch):
    path = tmp_path / "chunk-boundary.txt"
    raw = b"alpha\r\nbeta\r\ngamma\n"
    canonical = b"alpha\nbeta\ngamma\n"
    path.write_bytes(raw)

    def fail_read_bytes(_: Path) -> bytes:
        raise AssertionError("validation must stream rather than call Path.read_bytes")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    fingerprints = validation._file_fingerprints(path, chunk_size=6)

    assert fingerprints.raw_size == len(raw)
    assert fingerprints.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert fingerprints.canonical_size == len(canonical)
    assert fingerprints.canonical_sha256 == hashlib.sha256(canonical).hexdigest()


def test_non_official_fixture_does_not_require_official_totals(valid_sidecar: Path):
    data = load_annotations(valid_sidecar)
    result = validate_annotations(valid_sidecar, data)

    assert result.ok
    assert not any(check.scope == "official-source" for check in result.checks)
