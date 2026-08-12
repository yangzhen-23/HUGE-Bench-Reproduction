"""Deterministic publication-style figures derived from :class:`AnalysisResult`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .analysis import AnalysisResult, TASK_ORDER
from .loader import SPLIT_ORDER


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


TASK_LABELS = {
    "0": "Landing",
    "hl": "Orbit-H",
    "orbit": "Orbit-R",
    "building": "Inspection-B",
    "road": "Inspection-R",
    "farm": "Mapping",
    "obstacle": "Traversal",
    "orbit_multi": "Spiral Down",
}
TASK_COLORS = {
    task_id: color for task_id, color in zip(
        TASK_ORDER,
        ("#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#76B7B2", "#EDC948", "#9C755F", "#FF9DA7"),
        strict=True,
    )
}
SPLIT_LABELS = {"train": "Train", "test_seen": "Test seen", "test_unseen": "Test unseen"}
SPLIT_COLORS = {"train": "#4C566A", "test_seen": "#8492A6", "test_unseen": "#B8C2CC"}
PROVENANCE_ORDER = ("original_raw", "recovered_from_released_actions")
PROVENANCE_LABELS = {
    "original_raw": "Original raw",
    "recovered_from_released_actions": "Recovered from\nreleased actions",
}
PROVENANCE_COLORS = {"original_raw": "#7A7A7A", "recovered_from_released_actions": "#2A9D8F"}
FOOTER = "Source: official HUGE-Bench stage-annotation sidecar"
FILENAMES = (
    "01_split_overview.png",
    "02_task_distribution.png",
    "03_task_scene_heatmap.png",
    "04_episode_length_distribution.png",
    "05_stages_per_episode.png",
    "06_stage_duration_by_task.png",
    "07_annotation_provenance.png",
    "08_example_stage_timeline.png",
)


@dataclass(frozen=True)
class TimelineSegment:
    frame_start: int
    frame_end: int
    subtask_id: int
    duration: int
    text: str


@dataclass(frozen=True)
class TimelineRow:
    task_id: str
    task_label: str
    split: str
    episode_index: int
    episode_length: int
    provenance: str
    segments: tuple[TimelineSegment, ...]


def task_scene_array(result: AnalysisResult) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Return task-by-scene episode counts in the approved deterministic order."""
    scene_order = tuple(result.scene_order)
    by_task = {str(row["task_id"]): row for row in result.task_scene_matrix}
    values = np.array(
        [[int(by_task[task_id][scene]) for scene in scene_order] for task_id in TASK_ORDER],
        dtype=int,
    )
    return values, TASK_ORDER, scene_order


def stages_per_episode_proportions(result: AnalysisResult) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return task-normalized proportions for every observed stage-count category."""
    categories = tuple(sorted({row.record.num_stages for row in result.episode_rows}))
    values = np.zeros((len(TASK_ORDER), len(categories)), dtype=float)
    category_index = {value: index for index, value in enumerate(categories)}
    for task_index, task_id in enumerate(TASK_ORDER):
        episodes = [row.record for row in result.episode_rows if row.record.task_id == task_id]
        if not episodes:
            continue
        for episode in episodes:
            values[task_index, category_index[episode.num_stages]] += 1.0
        values[task_index] /= len(episodes)
    return values, categories


def stage_duration_p99(result: AnalysisResult) -> float:
    """Return the default NumPy 99th percentile of all closed-interval durations."""
    durations = [row.record.duration_frames for row in result.stage_rows]
    return float(np.percentile(durations, 99)) if durations else 0.0


def representative_timeline_rows(result: AnalysisResult) -> tuple[TimelineRow, ...]:
    """Materialize ordered, immutable segments for each available representative."""
    rows: list[TimelineRow] = []
    for task_id in TASK_ORDER:
        representative = result.representative_episodes.get(task_id)
        if representative is None:
            continue
        stages = sorted(
            (
                row.record
                for row in result.stage_rows
                if row.split == representative.split
                and row.record.task_id == task_id
                and row.record.episode_index == representative.episode_index
            ),
            key=lambda stage: (stage.frame_start, stage.frame_end, stage.subtask_id),
        )
        segments = tuple(
            TimelineSegment(
                frame_start=stage.frame_start,
                frame_end=stage.frame_end,
                subtask_id=stage.subtask_id,
                duration=stage.duration_frames,
                text=stage.subtask_text,
            )
            for stage in stages
        )
        rows.append(
            TimelineRow(
                task_id=task_id,
                task_label=TASK_LABELS[task_id],
                split=representative.split,
                episode_index=representative.episode_index,
                episode_length=representative.length,
                provenance=representative.episode.annotation_provenance,
                segments=segments,
            )
        )
    return tuple(rows)


def create_all_figures(result: AnalysisResult, output_dir: Path) -> tuple[Path, ...]:
    """Create all eight report figures and return their exact reporting order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plotters: tuple[Callable[[AnalysisResult], plt.Figure], ...] = (
        _split_overview,
        _task_distribution,
        _task_scene_heatmap,
        _episode_length_distribution,
        _stages_per_episode,
        _stage_duration_by_task,
        _annotation_provenance,
        _example_stage_timeline,
    )
    paths: list[Path] = []
    for filename, plotter in zip(FILENAMES, plotters, strict=True):
        path = output_dir / filename
        figure = plotter(result)
        _save_figure(figure, path)
        paths.append(path)
    return tuple(paths)


def _new_figure(*, width: float = 7.2, height: float = 4.2, **kwargs: object) -> plt.Figure:
    return plt.figure(figsize=(width, height), constrained_layout=True, **kwargs)


def _footer(figure: plt.Figure) -> None:
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.12, 1.0, 0.85))
    figure.text(0.5, 0.012, FOOTER, ha="center", va="bottom", color="#777777", fontsize=7)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    try:
        figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)


def _split_overview(result: AnalysisResult) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 4.2), constrained_layout=True)
    metrics = (("episodes", "Episodes"), ("total_frames", "Total frames"), ("stage_segments", "Stage segments"))
    overview = {str(row["split"]): row for row in result.dataset_overview}
    x = np.arange(len(SPLIT_ORDER))
    for axis, (field, label) in zip(axes, metrics, strict=True):
        values = [int(overview[split][field]) for split in SPLIT_ORDER]
        bars = axis.bar(x, values, color=[SPLIT_COLORS[split] for split in SPLIT_ORDER], width=0.66)
        axis.set_title(label, fontweight="bold")
        axis.set_xticks(x, [SPLIT_LABELS[split] for split in SPLIT_ORDER], rotation=25, ha="right")
        axis.set_ylim(0, max(values + [1]) * 1.2)
        axis.bar_label(bars, labels=[f"{value:,}" for value in values], padding=2, fontsize=7)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        axis.set_axisbelow(True)
    figure.suptitle("Dataset coverage across official splits", fontsize=11, fontweight="bold")
    _footer(figure)
    return figure


def _task_distribution(result: AnalysisResult) -> plt.Figure:
    figure = _new_figure()
    axis = figure.subplots()
    counts = {str(row["task_id"]): int(row["episodes"]) for row in result.task_statistics if row["split"] == "overall"}
    rank = {task_id: index for index, task_id in enumerate(TASK_ORDER)}
    ordered = sorted(TASK_ORDER, key=lambda task_id: (-counts[task_id], rank[task_id]))
    labels = [TASK_LABELS[task_id] for task_id in ordered]
    values = [counts[task_id] for task_id in ordered]
    y = np.arange(len(ordered))
    bars = axis.barh(y, values, color=[TASK_COLORS[task_id] for task_id in ordered], height=0.7)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Episodes")
    axis.set_title("Episode distribution across eight task types", fontsize=11, fontweight="bold")
    axis.set_xlim(0, max(values + [1]) * 1.13)
    axis.bar_label(bars, labels=[f"{value:,}" for value in values], padding=3, fontsize=7)
    axis.grid(axis="x", color="#E5E5E5", linewidth=0.6)
    axis.set_axisbelow(True)
    _footer(figure)
    return figure


def _task_scene_heatmap(result: AnalysisResult) -> plt.Figure:
    values, task_order, scene_order = task_scene_array(result)
    width = max(7.2, 0.48 * len(scene_order) + 2.6)
    figure = _new_figure(width=width, height=4.8)
    axis = figure.subplots()
    image = axis.imshow(values, cmap="Blues", aspect="auto", interpolation="nearest")
    axis.set_yticks(np.arange(len(task_order)), [TASK_LABELS[task_id] for task_id in task_order])
    axis.set_xticks(np.arange(len(scene_order)), scene_order, rotation=45, ha="right")
    axis.set_xlabel("Scene")
    axis.set_title("Task × scene episode coverage", fontsize=11, fontweight="bold")
    threshold = float(values.max()) * 0.55 if values.size else 0.0
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = int(values[row_index, column_index])
            axis.text(column_index, row_index, str(value), ha="center", va="center", color="white" if value > threshold else "#222222", fontsize=7)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label("Episodes")
    _footer(figure)
    return figure


def _episode_length_distribution(result: AnalysisResult) -> plt.Figure:
    figure = _new_figure()
    axis = figure.subplots()
    by_split = {
        split: np.array([row.record.length for row in result.episode_rows if row.split == split], dtype=float)
        for split in SPLIT_ORDER
    }
    all_lengths = np.concatenate([values for values in by_split.values() if values.size]) if any(values.size for values in by_split.values()) else np.array([0.0, 1.0])
    if float(all_lengths.min()) == float(all_lengths.max()):
        edges = np.linspace(float(all_lengths.min()) - 0.5, float(all_lengths.max()) + 0.5, 11)
    else:
        edges = np.histogram_bin_edges(all_lengths, bins=min(20, max(6, int(np.sqrt(all_lengths.size)))))
    for split in SPLIT_ORDER:
        values = by_split[split]
        if not values.size:
            continue
        weights = np.full(values.size, 1.0 / values.size)
        axis.hist(values, bins=edges, weights=weights, histtype="step", linewidth=1.8, color=SPLIT_COLORS[split], label=SPLIT_LABELS[split])
        median = float(np.median(values))
        axis.axvline(median, color=SPLIT_COLORS[split], linestyle="--", linewidth=1.0)
        label_level = 0.97 - 0.07 * SPLIT_ORDER.index(split)
        axis.text(median, label_level, f"{SPLIT_LABELS[split]} median {median:g}", color=SPLIT_COLORS[split], va="top", ha="left", transform=axis.get_xaxis_transform(), fontsize=6.5)
    axis.set_xlabel("Episode length (frames at 5 FPS)")
    axis.set_ylabel("Proportion within split")
    axis.set_title("Split-normalized episode-length distributions", fontsize=11, fontweight="bold")
    axis.legend(loc="upper right")
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axis.set_axisbelow(True)
    _footer(figure)
    return figure


def _stages_per_episode(result: AnalysisResult) -> plt.Figure:
    proportions, categories = stages_per_episode_proportions(result)
    figure = _new_figure()
    axis = figure.subplots()
    y = np.arange(len(TASK_ORDER))
    left = np.zeros(len(TASK_ORDER), dtype=float)
    colors = plt.get_cmap("Blues")(np.linspace(0.32, 0.88, max(len(categories), 1)))
    for index, category in enumerate(categories):
        values = proportions[:, index]
        axis.barh(y, values, left=left, color=colors[index], height=0.7, label=str(category))
        left += values
    axis.set_yticks(y, [TASK_LABELS[task_id] for task_id in TASK_ORDER])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Proportion of episodes")
    axis.set_title("Stage-count composition by task", fontsize=11, fontweight="bold", pad=52)
    axis.legend(title="Stages per episode", ncols=min(4, max(1, len(categories))), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    axis.grid(axis="x", color="#E5E5E5", linewidth=0.6)
    axis.set_axisbelow(True)
    _footer(figure)
    return figure


def _stage_duration_by_task(result: AnalysisResult) -> plt.Figure:
    figure = _new_figure(height=4.6)
    axis = figure.subplots()
    values = [
        [row.record.duration_frames for row in result.stage_rows if row.record.task_id == task_id]
        for task_id in TASK_ORDER
    ]
    positions = [index + 1 for index, group in enumerate(values) if group]
    present = [group for group in values if group]
    box = axis.boxplot(present, positions=positions, widths=0.58, patch_artist=True, showfliers=False, medianprops={"color": "#222222", "linewidth": 1.2})
    for patch, task_id in zip(box["boxes"], (task_id for task_id, group in zip(TASK_ORDER, values, strict=True) if group), strict=True):
        patch.set_facecolor(TASK_COLORS[task_id])
        patch.set_alpha(0.82)
    p99 = stage_duration_p99(result)
    axis.set_ylim(0, max(1.0, p99))
    axis.set_xticks(np.arange(1, len(TASK_ORDER) + 1), [TASK_LABELS[task_id] for task_id in TASK_ORDER], rotation=25, ha="right")
    axis.set_ylabel("Stage duration (frames)")
    axis.set_title("Closed-interval stage durations by task", fontsize=11, fontweight="bold")
    axis.text(0.99, 0.98, f"Display capped at overall p99 = {p99:.1f} frames; values above p99 are outside the displayed range.", transform=axis.transAxes, ha="right", va="top", fontsize=6.5, color="#555555")
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axis.set_axisbelow(True)
    _footer(figure)
    return figure


def _annotation_provenance(result: AnalysisResult) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), constrained_layout=True)
    episode_counts = {provenance: sum(row.record.annotation_provenance == provenance for row in result.episode_rows) for provenance in PROVENANCE_ORDER}
    stage_counts = {provenance: sum(row.record.annotation_provenance == provenance for row in result.stage_rows) for provenance in PROVENANCE_ORDER}
    for axis, counts, title in zip(axes, (episode_counts, stage_counts), ("Episodes", "Stage segments"), strict=True):
        values = [counts[provenance] for provenance in PROVENANCE_ORDER]
        x = np.arange(len(PROVENANCE_ORDER))
        bars = axis.bar(x, values, color=[PROVENANCE_COLORS[provenance] for provenance in PROVENANCE_ORDER], width=0.65)
        axis.set_xticks(x, [PROVENANCE_LABELS[provenance] for provenance in PROVENANCE_ORDER])
        axis.set_title(title, fontweight="bold")
        axis.set_ylim(0, max(values + [1]) * 1.2)
        axis.bar_label(bars, labels=[f"{value:,}" for value in values], padding=2, fontsize=7)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        axis.set_axisbelow(True)
    figure.suptitle("Annotation provenance coverage", fontsize=11, fontweight="bold")
    figure.text(0.5, 0.075, "Recovered boundaries apply to released obstacle actions; they are not original human annotations.", ha="center", color="#555555", fontsize=7)
    _footer(figure)
    return figure


def _example_stage_timeline(result: AnalysisResult) -> plt.Figure:
    rows = representative_timeline_rows(result)
    figure = _new_figure(height=5.0)
    axis = figure.subplots()
    subtask_ids = sorted({segment.subtask_id for row in rows for segment in row.segments})
    palette = {subtask_id: plt.get_cmap("cividis")(position) for subtask_id, position in zip(subtask_ids, np.linspace(0.15, 0.9, max(len(subtask_ids), 1)), strict=True)}
    for y, row in enumerate(rows):
        for segment in row.segments:
            axis.barh(y, segment.duration, left=segment.frame_start, height=0.64, color=palette[segment.subtask_id], edgecolor="white", linewidth=0.5)
            if segment.duration >= max(8, row.episode_length * 0.08):
                axis.text(segment.frame_start + segment.duration / 2, y, str(segment.duration), ha="center", va="center", fontsize=6.5, color="white" if segment.subtask_id > np.median(subtask_ids) else "#222222")
        suffix = " · recovered" if row.provenance == "recovered_from_released_actions" else ""
        axis.text(row.episode_length + max(1, row.episode_length * 0.02), y, f"{SPLIT_LABELS[row.split]} · ep {row.episode_index}{suffix}", va="center", fontsize=6.5, color="#555555")
    axis.set_yticks(np.arange(len(rows)), [row.task_label for row in rows])
    axis.invert_yaxis()
    maximum = max((row.episode_length for row in rows), default=1)
    axis.set_xlim(0, maximum * 1.34)
    axis.set_xlabel("Episode timeline (frames)")
    axis.set_title("Representative multi-stage episode timelines", fontsize=11, fontweight="bold", pad=48)
    handles = [mpl.patches.Patch(color=palette[subtask_id], label=str(subtask_id)) for subtask_id in subtask_ids]
    axis.legend(handles=handles, title="Subtask ID", ncols=min(6, max(1, len(handles))), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    axis.grid(axis="x", color="#E5E5E5", linewidth=0.6)
    axis.set_axisbelow(True)
    _footer(figure)
    return figure
