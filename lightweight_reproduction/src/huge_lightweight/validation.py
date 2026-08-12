"""Deterministic integrity validation for loaded HUGE-Bench annotations."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .loader import SPLIT_ORDER
from .models import AnnotationDataset, EpisodeRecord, StageRecord

_OFFICIAL_SOURCE = "yu781986168/HUGE_Dataset_v0"
_OFFICIAL_TASK_IDS = frozenset({"0", "hl", "orbit", "building", "road", "farm", "obstacle", "orbit_multi"})
_OFFICIAL_EPISODES = {"train": 5175, "test_seen": 576, "test_unseen": 417}
_OFFICIAL_STAGES = {"train": 23115, "test_seen": 2573, "test_unseen": 1851}


@dataclass(frozen=True)
class ValidationCheck:
    check_id: str
    scope: str
    status: str
    expected: str
    actual: str
    details: str


@dataclass(frozen=True)
class ValidationResult:
    checks: tuple[ValidationCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.status == "PASS" for check in self.checks)

    @property
    def failures(self) -> tuple[ValidationCheck, ...]:
        return tuple(check for check in self.checks if check.status != "PASS")


def validate_annotations(root: Path, data: AnnotationDataset) -> ValidationResult:
    """Validate sidecar bytes and loaded records without reparsing JSONL files."""
    checks: list[ValidationCheck] = []
    add = _CheckAppender(checks)
    manifest = data.manifest

    add("manifest.format_version", "manifest", "1", str(manifest.format_version), "format version is supported")
    for entry in manifest.files:
        path = root / entry.path
        regular = path.is_file()
        add("manifest.file_exists", entry.path, "existing regular file", "regular file" if regular else "missing or not regular", "manifest-listed path availability", regular)
        if regular:
            fingerprints = _file_fingerprints(path)
            canonical_match = fingerprints.canonical_size == entry.size_bytes and fingerprints.canonical_sha256 == entry.sha256
            raw_match = fingerprints.raw_size == entry.size_bytes and fingerprints.raw_sha256 == entry.sha256
            actual = f"raw=size:{fingerprints.raw_size},sha256:{fingerprints.raw_sha256}; canonical=size:{fingerprints.canonical_size},sha256:{fingerprints.canonical_sha256}"
            details = "raw and canonical results recorded; working-tree CRLF normalized to published LF bytes only when both canonical size and SHA-256 match"
            add("manifest.file_size", entry.path, str(entry.size_bytes), actual, details, raw_match or canonical_match)
            add("manifest.sha256", entry.path, entry.sha256, actual, details, raw_match or canonical_match)
        else:
            add("manifest.file_size", entry.path, str(entry.size_bytes), "unavailable", "file is unavailable for byte count")
            add("manifest.sha256", entry.path, entry.sha256, "unavailable", "file is unavailable for digest")

    actual_dataset_splits = tuple(data.episodes_by_split.keys())
    actual_stage_splits = tuple(data.stages_by_split.keys())
    actual_manifest_splits = tuple(manifest.splits.keys())
    expected_splits = set(SPLIT_ORDER)
    add("split.names", "dataset/manifest", ", ".join(SPLIT_ORDER), f"episodes={actual_dataset_splits}; stages={actual_stage_splits}; manifest={actual_manifest_splits}", "episode, stage, and manifest split collections must contain exactly the required names", set(actual_dataset_splits) == expected_splits and set(actual_stage_splits) == expected_splits and set(actual_manifest_splits) == expected_splits)

    for split in SPLIT_ORDER:
        episodes = data.episodes_by_split.get(split, ())
        stages = data.stages_by_split.get(split, ())
        split_manifest = manifest.splits.get(split)
        expected_episode_count = split_manifest.episodes if split_manifest else "missing split"
        expected_stage_count = split_manifest.stage_segments if split_manifest else "missing split"
        add("split.episode_count", split, str(expected_episode_count), str(len(episodes)), "loaded episode count compared to manifest")
        add("split.stage_count", split, str(expected_stage_count), str(len(stages)), "loaded stage count compared to manifest")

    loaded_episodes = sum(len(data.episodes_by_split.get(split, ())) for split in SPLIT_ORDER)
    loaded_stages = sum(len(data.stages_by_split.get(split, ())) for split in SPLIT_ORDER)
    add("dataset.total_episodes", "dataset", str(manifest.total_episodes), str(loaded_episodes), "loaded total compared to manifest")
    add("dataset.total_stage_segments", "dataset", str(manifest.total_stage_segments), str(loaded_stages), "loaded total compared to manifest")

    if manifest.source_dataset == _OFFICIAL_SOURCE:
        for split in SPLIT_ORDER:
            split_manifest = manifest.splits.get(split)
            add("split.episode_count", f"official-source/{split}", str(_OFFICIAL_EPISODES[split]), str(split_manifest.episodes if split_manifest else "missing split"), "official split episode constant")
            add("split.stage_count", f"official-source/{split}", str(_OFFICIAL_STAGES[split]), str(split_manifest.stage_segments if split_manifest else "missing split"), "official split stage constant")
        add("dataset.total_episodes", "official-source", "6168", str(manifest.total_episodes), "official total episode constant")
        add("dataset.total_stage_segments", "official-source", "27539", str(manifest.total_stage_segments), "official total stage constant")

    for split in SPLIT_ORDER:
        _validate_split(add, split, data.episodes_by_split.get(split, ()), data.stages_by_split.get(split, ()))
    return ValidationResult(tuple(checks))


def _validate_split(add: "_CheckAppender", split: str, episodes: tuple[EpisodeRecord, ...], stages: tuple[StageRecord, ...]) -> None:
    indices = [episode.episode_index for episode in episodes]
    expected_indices = list(range(len(episodes)))
    add("episode.index_contiguous", split, str(expected_indices), str(sorted(indices)), "episode indices must be unique and contiguous", len(indices) == len(set(indices)) and sorted(indices) == expected_indices)
    stages_by_episode: dict[int, list[StageRecord]] = defaultdict(list)
    for stage in stages:
        stages_by_episode[stage.episode_index].append(stage)
    episodes_by_index = {episode.episode_index: episode for episode in episodes}

    for episode in sorted(episodes, key=lambda value: value.episode_index):
        scope = f"{split}/episode/{episode.episode_index}"
        valid_fields = (
            episode.length > 0
            and episode.num_stages > 0
            and episode.task_id in _OFFICIAL_TASK_IDS
            and bool(episode.env_id.strip())
            and bool(episode.instruction.strip())
            and bool(episode.annotation_provenance.strip())
        )
        add("episode.fields_valid", scope, "positive lengths, official task, and nonblank text/provenance", _episode_actual(episode), "episode structural fields", valid_fields)
        add("stage.present", scope, "at least one stage", str(len(stages_by_episode[episode.episode_index])), "stages mapped to episode", bool(stages_by_episode[episode.episode_index]))
        episode_stages = sorted(stages_by_episode[episode.episode_index], key=lambda value: (value.frame_start, value.frame_end, value.subtask_id))
        add("stage.count_matches", scope, str(episode.num_stages), str(len(episode_stages)), "stage count compared to episode num_stages")
        _validate_episode_stages(add, scope, episode, episode_stages)

    for episode_index in sorted(stages_by_episode):
        if episode_index not in episodes_by_index:
            add("stage.orphan_episode", f"{split}/episode/{episode_index}", "episode mapping record", "no episode mapping record", "stage references an absent episode")
            for position, stage in enumerate(sorted(stages_by_episode[episode_index], key=lambda value: (value.frame_start, value.frame_end, value.subtask_id))):
                stage_scope = f"{split}/episode/{episode_index}/stage/{position}"
                add("stage.subtask_text_nonempty", stage_scope, "nonblank subtask text", "nonblank" if stage.subtask_text.strip() else "blank", "stage instruction text", bool(stage.subtask_text.strip()))
                _validate_provenance(add, stage_scope, stage.task_id, stage.annotation_provenance)


def _validate_episode_stages(add: "_CheckAppender", scope: str, episode: EpisodeRecord, stages: list[StageRecord]) -> None:
    for position, stage in enumerate(stages):
        stage_scope = f"{scope}/stage/{position}"
        same_identity = (stage.task_id, stage.env_id, stage.annotation_provenance) == (episode.task_id, episode.env_id, episode.annotation_provenance)
        add("stage.fields_consistent", stage_scope, f"task={episode.task_id}; env={episode.env_id}; provenance={episode.annotation_provenance}", f"task={stage.task_id}; env={stage.env_id}; provenance={stage.annotation_provenance}", "stage identity fields compared to episode", same_identity)
        valid_bounds = 0 <= stage.frame_start <= stage.frame_end < episode.length
        add("stage.bounds_valid", stage_scope, f"0 <= start <= end < {episode.length}", f"start={stage.frame_start}; end={stage.frame_end}", "closed frame interval bounds", valid_bounds)
        add("stage.subtask_text_nonempty", stage_scope, "nonblank subtask text", "nonblank" if stage.subtask_text.strip() else "blank", "stage instruction text", bool(stage.subtask_text.strip()))
        _validate_provenance(add, stage_scope, stage.task_id, stage.annotation_provenance)
    _validate_provenance(add, scope, episode.task_id, episode.annotation_provenance)

    if not stages:
        add("stage.coverage_contiguous", scope, "stages start at frame 0 without gaps or overlaps", "no stages", "coverage cannot be established")
        add("stage.coverage_complete", scope, f"last end is {episode.length - 1}", "no stages", "coverage cannot be complete")
        return
    contiguous = stages[0].frame_start == 0 and all(current.frame_start == previous.frame_end + 1 for previous, current in zip(stages, stages[1:]))
    coverage_actual = _coverage_actual(stages)
    add("stage.coverage_contiguous", scope, "begin at 0 and adjacent boundaries differ by one", coverage_actual, "gap/overlap diagnostics in sorted frame order", contiguous)
    add("stage.coverage_complete", scope, str(episode.length - 1), str(stages[-1].frame_end), "last sorted stage end compared to episode length")


def _validate_provenance(add: "_CheckAppender", scope: str, task_id: str, provenance: str) -> None:
    allowed = {"original_raw", "recovered_from_released_actions"} if task_id == "obstacle" else {"original_raw"}
    add("provenance.allowed", scope, " or ".join(sorted(allowed)), provenance or "blank", "task-specific provenance policy", provenance in allowed)


def _episode_actual(episode: EpisodeRecord) -> str:
    return f"length={episode.length}; num_stages={episode.num_stages}; task={episode.task_id}; env={'nonblank' if episode.env_id.strip() else 'blank'}; instruction={'nonblank' if episode.instruction.strip() else 'blank'}; provenance={'nonblank' if episode.annotation_provenance.strip() else 'blank'}"


def _coverage_actual(stages: list[StageRecord]) -> str:
    if stages[0].frame_start != 0:
        return f"first start={stages[0].frame_start} (expected 0)"
    for previous, current in zip(stages, stages[1:]):
        expected_start = previous.frame_end + 1
        if current.frame_start != expected_start:
            kind = "gap" if current.frame_start > expected_start else "overlap"
            return f"{kind}: previous end={previous.frame_end}; next start={current.frame_start}; expected={expected_start}"
    return "contiguous from frame 0"


@dataclass(frozen=True)
class _FileFingerprints:
    raw_size: int
    raw_sha256: str
    canonical_size: int
    canonical_sha256: str


def _file_fingerprints(path: Path, chunk_size: int = 1024 * 1024) -> _FileFingerprints:
    raw_digest = hashlib.sha256()
    canonical_digest = hashlib.sha256()
    raw_size = 0
    canonical_size = 0
    pending_cr = False
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            raw_digest.update(chunk)
            raw_size += len(chunk)
            canonical_chunk, pending_cr = _canonicalize_chunk(chunk, pending_cr)
            canonical_digest.update(canonical_chunk)
            canonical_size += len(canonical_chunk)
    if pending_cr:
        canonical_digest.update(b"\r")
        canonical_size += 1
    return _FileFingerprints(raw_size, raw_digest.hexdigest(), canonical_size, canonical_digest.hexdigest())


def _canonicalize_chunk(chunk: bytes, pending_cr: bool) -> tuple[bytes, bool]:
    prefix = b"\r" if pending_cr else b""
    content = prefix + chunk
    if content.endswith(b"\r"):
        content = content[:-1]
        pending_cr = True
    else:
        pending_cr = False
    return content.replace(b"\r\n", b"\n"), pending_cr


class _CheckAppender:
    def __init__(self, checks: list[ValidationCheck]) -> None:
        self._checks = checks

    def __call__(self, check_id: str, scope: str, expected: str, actual: str, details: str, passed: bool | None = None) -> None:
        if passed is None:
            passed = expected == actual
        self._checks.append(ValidationCheck(check_id, scope, "PASS" if passed else "FAIL", expected, actual, details))
