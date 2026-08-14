from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image, ImageStat

from huge_lightweight import plots as plots_module
from huge_lightweight.analysis import TASK_ORDER, analyze
from huge_lightweight.loader import SPLIT_ORDER, load_annotations
from huge_lightweight.models import AnnotationDataset, EpisodeRecord, StageRecord
from huge_lightweight.plots import (
    create_all_figures,
    representative_timeline_rows,
    stage_duration_p99,
    stages_per_episode_proportions,
    task_scene_array,
)


EXPECTED_FILENAMES = (
    "01_split_overview.png",
    "02_task_distribution.png",
    "03_task_scene_heatmap.png",
    "04_episode_length_distribution.png",
    "05_stages_per_episode.png",
    "06_stage_duration_by_task.png",
    "07_annotation_provenance.png",
    "08_example_stage_timeline.png",
)


def test_all_figure_objects_omit_explanatory_footers(synthetic_result):
    forbidden = {
        "Source: official HUGE-Bench stage-annotation sidecar",
        "Recovered boundaries apply to released obstacle actions; they are not original human annotations.",
    }
    plotters = (
        plots_module._split_overview,
        plots_module._task_distribution,
        plots_module._task_scene_heatmap,
        plots_module._episode_length_distribution,
        plots_module._stages_per_episode,
        plots_module._stage_duration_by_task,
        plots_module._annotation_provenance,
        plots_module._example_stage_timeline,
    )

    for plotter in plotters:
        figure = plotter(synthetic_result)
        try:
            rendered_text = {text.get_text() for text in figure.texts}
            rendered_text.update(
                text.get_text()
                for axis in figure.axes
                for text in axis.texts
            )
            assert forbidden.isdisjoint(rendered_text), plotter.__name__
        finally:
            plt.close(figure)


def _episode(
    index: int,
    task_id: str,
    scene: str,
    length: int,
    num_stages: int,
    provenance: str,
) -> EpisodeRecord:
    return EpisodeRecord(
        episode_index=index,
        task_id=task_id,
        task_episode_index=index,
        env_id=scene,
        traj_id=index,
        pose_start=0,
        pose_end=length - 1,
        length=length,
        instruction=f"Instruction for {task_id}.",
        num_stages=num_stages,
        annotation_provenance=provenance,
        subtask_file=None,
    )


def _segments(episode: EpisodeRecord) -> tuple[StageRecord, ...]:
    quotient, remainder = divmod(episode.length, episode.num_stages)
    start = 0
    rows = []
    for subtask_id in range(episode.num_stages):
        duration = quotient + (subtask_id < remainder)
        end = start + duration - 1
        rows.append(
            StageRecord(
                episode_index=episode.episode_index,
                task_id=episode.task_id,
                env_id=episode.env_id,
                traj_id=episode.traj_id,
                annotation_provenance=episode.annotation_provenance,
                subtask_id=subtask_id,
                pose_start=start,
                pose_end=end,
                frame_start=start,
                frame_end=end,
                subtask_text=f"Stage {subtask_id} for {episode.task_id}.",
            )
        )
        start = end + 1
    return tuple(rows)


def _synthetic_dataset(valid_sidecar: Path) -> AnnotationDataset:
    base = load_annotations(valid_sidecar)
    episodes: dict[str, list[EpisodeRecord]] = {split: [] for split in SPLIT_ORDER}
    stages: dict[str, list[StageRecord]] = {split: [] for split in SPLIT_ORDER}
    split_indices = {split: 0 for split in SPLIT_ORDER}
    for position, task_id in enumerate(TASK_ORDER):
        split = SPLIT_ORDER[position % len(SPLIT_ORDER)]
        provenance = "recovered_from_released_actions" if task_id == "obstacle" else "original_raw"
        episode = _episode(
            split_indices[split],
            task_id,
            ("alpha_scene", "beta_scene")[position % 2],
            12 + 3 * position,
            1 + position % 4,
            provenance,
        )
        split_indices[split] += 1
        episodes[split].append(episode)
        stages[split].extend(_segments(episode))

    extra = _episode(split_indices["test_seen"], "0", "alpha_scene", 20, 2, "original_raw")
    episodes["test_seen"].append(extra)
    stages["test_seen"].extend(_segments(extra))
    return replace(
        base,
        episodes_by_split=MappingProxyType({key: tuple(value) for key, value in episodes.items()}),
        stages_by_split=MappingProxyType({key: tuple(value) for key, value in stages.items()}),
    )


@pytest.fixture
def synthetic_result(valid_sidecar: Path):
    return analyze(_synthetic_dataset(valid_sidecar))


def test_task_scene_array_has_approved_task_and_lexicographic_scene_order(synthetic_result):
    values, task_order, scene_order = task_scene_array(synthetic_result)

    assert values.shape == (8, 2)
    assert task_order == TASK_ORDER
    assert scene_order == ("alpha_scene", "beta_scene")
    assert values[0].tolist() == [2, 0]
    assert values[1].tolist() == [0, 1]
    assert values[6].tolist() == [1, 0]


def test_stage_proportions_use_deterministic_categories_and_normalize_rows(synthetic_result):
    proportions, categories = stages_per_episode_proportions(synthetic_result)

    assert categories == (1, 2, 3, 4)
    assert proportions.shape == (8, 4)
    assert proportions[0].tolist() == pytest.approx([0.5, 0.5, 0.0, 0.0])
    assert np.sum(proportions, axis=1).tolist() == pytest.approx([1.0] * 8)


def test_stage_proportions_leave_empty_task_rows_at_zero(valid_sidecar: Path):
    result = analyze(load_annotations(valid_sidecar))

    proportions, categories = stages_per_episode_proportions(result)

    assert categories == (2,)
    assert proportions[0].tolist() == [1.0]
    assert np.sum(proportions[1:], axis=1).tolist() == [0.0] * 7


def test_stage_duration_p99_uses_every_closed_interval_duration(synthetic_result):
    expected = np.percentile(
        [row.record.frame_end - row.record.frame_start + 1 for row in synthetic_result.stage_rows],
        99,
    )

    assert stage_duration_p99(synthetic_result) == pytest.approx(expected)


def test_timeline_rows_preserve_representatives_and_closed_intervals(synthetic_result):
    rows = representative_timeline_rows(synthetic_result)

    assert tuple(row.task_id for row in rows) == TASK_ORDER
    for row in rows:
        selected = synthetic_result.representative_episodes[row.task_id]
        assert (row.split, row.episode_index, row.episode_length) == (
            selected.split,
            selected.episode_index,
            selected.length,
        )
        assert tuple(segment.frame_start for segment in row.segments) == tuple(
            sorted(segment.frame_start for segment in row.segments)
        )
        assert all(segment.duration == segment.frame_end - segment.frame_start + 1 for segment in row.segments)
        assert sum(segment.duration for segment in row.segments) == row.episode_length
    obstacle = next(row for row in rows if row.task_id == "obstacle")
    assert obstacle.provenance == "recovered_from_released_actions"


def test_create_all_figures_exports_nonblank_images_and_closes_figures(synthetic_result, tmp_path: Path):
    before = tuple(plt.get_fignums())
    output_dir = tmp_path / "figures"

    first = create_all_figures(synthetic_result, output_dir)
    second = create_all_figures(synthetic_result, output_dir)

    assert tuple(path.name for path in first) == EXPECTED_FILENAMES
    assert tuple(path.name for path in second) == EXPECTED_FILENAMES
    assert tuple(plt.get_fignums()) == before
    for path in first:
        with Image.open(path) as image:
            image.load()
            assert image.mode in {"RGB", "RGBA"}
            assert image.width >= 800
            assert image.height >= 450
            dpi = image.info.get("dpi")
            if dpi is not None:
                assert dpi[0] == pytest.approx(180, abs=0.1)
                assert dpi[1] == pytest.approx(180, abs=0.1)
            rgb = image.convert("RGB")
            assert all(value > 0 for value in ImageStat.Stat(rgb).stddev)
