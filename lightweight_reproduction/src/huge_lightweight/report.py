"""Deterministic Chinese reports for the annotation-only reproduction."""
"""
把结果整理成“人能看的报告”
"""
from __future__ import annotations

import dataclasses
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import numpy as np

from .analysis import AnalysisResult, FRAME_RATE, TASK_ORDER
from .loader import SPLIT_ORDER
from .metric_smoke import MetricSmokeResult
from .validation import ValidationResult


REPORT_TITLE = "HUGE-Bench 轻量本地复现报告（数据与标注层）"
ALLOWED_CONCLUSION = "复现了 HUGE-Bench 官方公开数据的任务结构、多阶段标注统计与标注完整性检查。"
EXCLUDED_CLAIMS = (
    "未复现 PI0/PI0.5 模型性能。",
    "未复现论文 TCR、nDTW、NSP、CR 或 CSPL 主实验数值。",
    "未完成 3DGS 渲染、闭环飞行或碰撞评估。",
    "recovered_from_released_actions 不等同于原始人工阶段标注。",
)
PROVENANCE_CAVEAT = "recovered obstacle-action boundaries are not original human annotations; recovered_from_released_actions 不等同于原始人工阶段标注。"

FIGURE_PATHS = (
    "figures/01_split_overview.png",
    "figures/02_task_distribution.png",
    "figures/03_task_scene_heatmap.png",
    "figures/04_episode_length_distribution.png",
    "figures/05_stages_per_episode.png",
    "figures/06_stage_duration_by_task.png",
    "figures/07_annotation_provenance.png",
    "figures/08_example_stage_timeline.png",
)
TABLE_PATHS = (
    "tables/dataset_overview.csv",
    "tables/task_statistics.csv",
    "tables/task_scene_matrix.csv",
    "tables/stage_statistics.csv",
    "tables/validation_report.csv",
)
TASK_DISPLAY_NAMES = {
    "0": "基础导航",
    "hl": "高低位导航",
    "orbit": "单目标环绕",
    "building": "建筑导航",
    "road": "道路跟随",
    "farm": "农田导航",
    "obstacle": "避障",
    "orbit_multi": "多目标环绕",
}


@dataclass(frozen=True)
class ReportFigure:
    path: str
    alt: str
    caption: str


@dataclass(frozen=True)
class ReportLink:
    path: str
    label: str


@dataclass(frozen=True)
class ReportSection:
    heading: str
    paragraphs: tuple[str, ...]
    figures: tuple[ReportFigure, ...] = ()
    links: tuple[ReportLink, ...] = ()


def build_summary(
    result: AnalysisResult,
    validation: ValidationResult,
    metric_smoke: MetricSmokeResult,
    *,
    source_dataset: str,
    raw_subtask_files: int,
    annotations_root: str,
    repository_commit: str,
    manifest_sha256: str,
    peak_python_memory_bytes: int,
    elapsed_seconds: float,
    resources_measured: bool = True,
) -> dict[str, object]:
    """Build a JSON-native, deterministically ordered report summary."""
    episodes = tuple(row.record for row in result.episode_rows)
    stages = tuple(row.record for row in result.stage_rows)
    total_episodes = len(episodes)
    total_stages = len(stages)
    total_frames = sum(episode.length for episode in episodes)

    overview_by_split = {str(row["split"]): row for row in result.dataset_overview}
    overall_tasks = {
        str(row["task_id"]): row
        for row in result.task_statistics
        if row["split"] == "overall"
    }
    splits = []
    for split in SPLIT_ORDER:
        row = overview_by_split[split]
        count = int(row["episodes"])
        splits.append({
            "split": split,
            "episodes": count,
            "episode_ratio": _ratio(count, total_episodes),
            "frames": int(row["total_frames"]),
            "stages": int(row["stage_segments"]),
            "hours_at_5fps": _finite_float(row["total_hours_at_5fps"]),
        })

    tasks = []
    for task_id in TASK_ORDER:
        row = overall_tasks[task_id]
        count = int(row["episodes"])
        tasks.append({
            "task_id": task_id,
            "display_name": TASK_DISPLAY_NAMES[task_id],
            "episodes": count,
            "episode_ratio": _ratio(count, total_episodes),
            "frames": int(row["total_frames"]),
            "stages": int(row["stage_segments"]),
            "scenes": int(row["unique_scenes"]),
            "median_episode_frames": _finite_float(row["median_episode_frames"]),
            "mean_stages_per_episode": _finite_float(row["mean_stages_per_episode"]),
        })

    scene_counts = [
        {"scene": scene, "episodes": sum(episode.env_id == scene for episode in episodes)}
        for scene in result.scene_order
    ]
    episode_provenance = _provenance_counts(episode.annotation_provenance for episode in episodes)
    stage_provenance = _provenance_counts(stage.annotation_provenance for stage in stages)
    provenance = {}
    for name in ("original_raw", "recovered_from_released_actions"):
        episode_count = episode_provenance.get(name, 0)
        stage_count = stage_provenance.get(name, 0)
        provenance[name] = {
            "episodes": episode_count,
            "episode_ratio": _ratio(episode_count, total_episodes),
            "stages": stage_count,
            "stage_ratio": _ratio(stage_count, total_stages),
        }
    provenance["caveat"] = PROVENANCE_CAVEAT

    representatives = []
    for task_id in TASK_ORDER:
        representative = result.representative_episodes.get(task_id)
        if representative is not None:
            representatives.append({
                "task_id": task_id,
                "display_name": TASK_DISPLAY_NAMES[task_id],
                "split": representative.split,
                "episode_index": int(representative.episode_index),
                "length": int(representative.length),
                "provenance": representative.episode.annotation_provenance,
                "instruction": representative.episode.instruction,
            })

    failures = validation.failures
    pass_count = sum(check.status == "PASS" for check in validation.checks)
    return {
        "scope": {"allowed_conclusion": ALLOWED_CONCLUSION, "excluded_claims": list(EXCLUDED_CLAIMS)},
        "source": {
            "source_dataset": str(source_dataset),
            "annotations_root": str(annotations_root),
            "repository_commit": str(repository_commit) if repository_commit else "unavailable",
            "manifest_sha256": str(manifest_sha256),
        },
        "counts": {
            "total_episodes": total_episodes,
            "total_stage_segments": total_stages,
            "total_frames": total_frames,
            "total_hours_at_5fps": total_frames / FRAME_RATE / 3600.0,
            "task_count": len({episode.task_id for episode in episodes}),
            "scene_count": len(result.scene_order),
            "raw_subtask_files": int(raw_subtask_files),
            "observed_stages_per_episode": sorted({int(episode.num_stages) for episode in episodes}),
        },
        "splits": splits,
        "tasks": tasks,
        "scenes": scene_counts,
        "episode_length_quantiles_frames": _quantiles([episode.length for episode in episodes]),
        "stage_duration_quantiles_frames": _quantiles([stage.duration_frames for stage in stages]),
        "provenance": provenance,
        "representative_episodes": representatives,
        "validation": {
            "ok": bool(validation.ok),
            "total_checks": len(validation.checks),
            "pass_count": pass_count,
            "fail_count": len(failures),
            "failure_ids": [check.check_id for check in failures],
            "failure_scopes": [check.scope for check in failures],
        },
        "metric_smoke": _json_native(dataclasses.asdict(metric_smoke)),
        "resources": {
            "peak_python_memory_bytes": int(peak_python_memory_bytes),
            "elapsed_seconds": _finite_float(elapsed_seconds),
            "measured": bool(resources_measured),
        },
    }


def write_summary_json(summary: Mapping[str, object], output_path: Path) -> Path:
    """Write deterministic UTF-8 JSON with a trailing newline."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(_json_native(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_text(content, encoding="utf-8", newline="")
    return path


def build_report_sections(
    summary: Mapping[str, object],
    figure_paths: Sequence[str],
    table_paths: Sequence[str],
) -> tuple[ReportSection, ...]:
    """Create the sole shared content model consumed by both renderers."""
    figures, tables = _validated_paths(figure_paths, table_paths)
    counts = _mapping(summary["counts"])
    source = _mapping(summary["source"])
    validation = _mapping(summary["validation"])
    metric = _mapping(summary["metric_smoke"])
    resources = _mapping(summary["resources"])
    scope = _mapping(summary["scope"])
    provenance = _mapping(summary["provenance"])
    splits = [_mapping(item) for item in _sequence(summary["splits"])]
    tasks = [_mapping(item) for item in _sequence(summary["tasks"])]
    representatives = [_mapping(item) for item in _sequence(summary["representative_episodes"])]
    episode_quantiles = _mapping(summary["episode_length_quantiles_frames"])
    stage_quantiles = _mapping(summary["stage_duration_quantiles_frames"])

    nonempty_tasks = [task for task in tasks if int(task["episodes"]) > 0]
    largest = max(nonempty_tasks, key=lambda task: int(task["episodes"])) if nonempty_tasks else None
    smallest = min(nonempty_tasks, key=lambda task: int(task["episodes"])) if nonempty_tasks else None
    split_text = "、".join(f"{item['split']}={int(item['episodes']):,}" for item in splits)
    split_imbalance = max((int(item["episodes"]) for item in splits), default=0) != min(
        (int(item["episodes"]) for item in splits), default=0
    )
    task_extremes = (
        f"任务样本量最大的是 {largest['display_name']}（{int(largest['episodes']):,} 条），"
        f"最小的已观测任务是 {smallest['display_name']}（{int(smallest['episodes']):,} 条）。"
        if largest and smallest else "当前摘要没有已观测任务，无法比较任务样本量。"
    )
    stage_categories = [int(value) for value in _sequence(counts["observed_stages_per_episode"])]
    category_text = "、".join(str(value) for value in stage_categories) if stage_categories else "无"
    representative_text = "；".join(
        f"{item['display_name']}:{item['split']}/{int(item['episode_index'])}"
        for item in representatives
    ) or "无"
    recovered = _mapping(provenance["recovered_from_released_actions"])
    allowed = str(scope["allowed_conclusion"])
    excluded = tuple(str(item) for item in _sequence(scope["excluded_claims"]))

    figure_captions = (
        f"图 1：三种 split 的 episode 数为 {split_text}，分布{'不均衡' if split_imbalance else '均衡'}；这里只描述公开数据构成。",
        f"图 2：{task_extremes}",
        f"图 3：共 {int(counts['scene_count']):,} 个场景；各任务覆盖场景数不同，体现 task-scene 异质性。",
        f"图 4：episode 长度中位数为 {float(episode_quantiles['p50']):,.2f} 帧，按 5 FPS 折合 {float(episode_quantiles['p50']) / FRAME_RATE:,.2f} 秒。",
        f"图 5：逐条 episode 观测到的阶段数类别为 {category_text}；该图呈现多阶段标注结构。",
        f"图 6：阶段时长 p99 为 {float(stage_quantiles['p99']):,.2f} 帧，图中以此作为显示上限，避免极端长尾压缩主体分布。",
        f"图 7：recovered_from_released_actions 包含 {int(recovered['episodes']):,} 个 episode、{int(recovered['stages']):,} 个阶段。{provenance['caveat']}",
        f"图 8：按固定规则为八类任务选择代表时间线（实际列出 {len(representatives)} 条）：{representative_text}。obstacle 的 recovered provenance 必须按发布动作恢复边界解释。",
    )
    alt_texts = (
        "训练与测试划分样本量概览图", "八类任务样本分布图", "任务与场景分布热图",
        "轨迹长度分布图", "单轨迹阶段数量分布图", "各任务阶段时长分布图",
        "标注来源构成图", "八类代表轨迹阶段时间线图",
    )
    report_figures = tuple(ReportFigure(path, alt, caption) for path, alt, caption in zip(figures, alt_texts, figure_captions))
    report_links = tuple(ReportLink(path, Path(path).name) for path in tables) + (ReportLink("summary.json", "summary.json"),)
    status = "PASS" if validation["ok"] else "FAIL"
    resource_status = (
        "以上为调用方提供的实测资源值。"
        if resources["measured"]
        else "Task 6 中的 0 bytes 和 0.000 秒是占位值，并非资源实测；完整流程的峰值内存与耗时将在 Task 7/8 测量。"
    )

    return (
        ReportSection("1. 复现目标与结论边界", (allowed, "本报告仅覆盖数据与标注层，不把数据统计或合成 smoke test 表述为模型或论文结果。")),
        ReportSection("2. 输入、版本与完整性校验", (
            f"源数据集：{source['source_dataset']}；标注根目录：{source['annotations_root']}。",
            f"仓库 commit：{source['repository_commit']}；manifest SHA-256：{source['manifest_sha256']}。",
            f"完整性校验：{status}，共 {int(validation['total_checks']):,} 项（PASS {int(validation['pass_count']):,}，FAIL {int(validation['fail_count']):,}）。",
        )),
        ReportSection("3. 数据规模总览", (
            f"共 {int(counts['total_episodes']):,} 个 episode、{int(counts['total_stage_segments']):,} 个阶段片段、{int(counts['total_frames']):,} 帧，任务数 {int(counts['task_count']):,}，场景数 {int(counts['scene_count']):,}。",
            f"统一按 5 FPS 换算，总时长为 {float(counts['total_hours_at_5fps']):,.4f} 小时；此换算不代表推理耗时。",
        )),
        ReportSection("4. 八张图的逐图解读", ("以下八张图均由同一摘要中的真实统计量解释，不外推到模型性能。",), report_figures),
        ReportSection("5. 多阶段标注与 provenance 说明", (
            f"original_raw：{int(_mapping(provenance['original_raw'])['episodes']):,} 个 episode、{int(_mapping(provenance['original_raw'])['stages']):,} 个阶段；recovered_from_released_actions：{int(recovered['episodes']):,} 个 episode、{int(recovered['stages']):,} 个阶段。",
            str(provenance["caveat"]),
        )),
        ReportSection("6. 官方 metric.py 合成 smoke test", (
            str(metric["label"]),
            f"状态：{metric['status']}；avg_tcr={metric['avg_tcr']}，ndtw={metric['ndtw']}，nsp={metric['nsp']}，success={metric['success']}，path_length={metric['path_length']}。",
            f"限制：{metric['limitations']} 该检查只验证实现可执行性，不是论文指标复现。",
        )),
        ReportSection("7. 本地资源使用", (
            f"峰值 Python 内存：{int(resources['peak_python_memory_bytes']):,} bytes；elapsed：{float(resources['elapsed_seconds']):,.3f} 秒。",
            resource_status,
            "不据此声称使用或未使用 GPU。",
        )),
        ReportSection("8. 可复现文件清单", ("以下相对路径链接到五张 CSV 表和机器可读 summary.json。",), links=report_links),
        ReportSection("9. 不能声称的结果", excluded),
        ReportSection("10. 下一步：AutoDL 阶段 A", ("下一阶段可在独立、可度量的运行中接入模型推理与资源采样；在获得新证据前，本报告的结论边界保持不变。",)),
    )


def write_markdown_report(
    summary: Mapping[str, object],
    figure_paths: Sequence[str],
    table_paths: Sequence[str],
    output_path: Path,
) -> Path:
    sections = build_report_sections(summary, figure_paths, table_paths)
    lines = [f"# {REPORT_TITLE}", ""]
    for section in sections:
        lines.extend((f"## {section.heading}", ""))
        for paragraph in section.paragraphs:
            lines.extend((paragraph, ""))
        for figure in section.figures:
            lines.extend((f"![{figure.alt}]({figure.path})", "", figure.caption, ""))
        for link in section.links:
            lines.append(f"- [{link.label}]({link.path})")
        if section.links:
            lines.append("")
    return _write_text(output_path, "\n".join(lines).rstrip() + "\n")


def write_html_report(
    summary: Mapping[str, object],
    figure_paths: Sequence[str],
    table_paths: Sequence[str],
    output_path: Path,
) -> Path:
    sections = build_report_sections(summary, figure_paths, table_paths)
    parts = [
        '<!doctype html>', '<html lang="zh-CN">', '<head>', '<meta charset="utf-8">',
        f"<title>{_html_text(REPORT_TITLE)}</title>",
        "<style>body{margin:0;background:#f4f6f8;color:#17212b;font-family:system-ui,sans-serif;line-height:1.7}main{max-width:1080px;margin:auto;padding:2rem}section,figure{background:#fff;border-radius:10px;padding:1rem;margin:1rem 0}img{display:block;max-width:100%;height:auto;margin:auto}figcaption{margin-top:.8rem;color:#425466}a{color:#075985;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse}@media(max-width:640px){main{padding:.75rem}}</style>",
        "</head>", '<body>', '<main>', f"<h1>{_html_text(REPORT_TITLE)}</h1>",
    ]
    for section in sections:
        parts.extend(("<section>", f"<h2>{_html_text(section.heading)}</h2>"))
        parts.extend(f"<p>{_html_text(paragraph)}</p>" for paragraph in section.paragraphs)
        for figure in section.figures:
            parts.extend((
                "<figure>",
                f'<img src="{_html_text(figure.path, quote=True)}" alt="{_html_text(figure.alt, quote=True)}">',
                f"<figcaption>{_html_text(figure.caption)}</figcaption>",
                "</figure>",
            ))
        if section.links:
            parts.append("<ul>")
            parts.extend(f'<li><a href="{_html_text(link.path, quote=True)}">{_html_text(link.label)}</a></li>' for link in section.links)
            parts.append("</ul>")
        parts.append("</section>")
    parts.extend(("</main>", "</body>", "</html>"))
    return _write_text(output_path, "\n".join(parts) + "\n")


def _validated_paths(figure_paths: Sequence[str], table_paths: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    figures = tuple(str(path) for path in figure_paths)
    tables = tuple(str(path) for path in table_paths)
    _validate_path_set("figure", figures, FIGURE_PATHS)
    _validate_path_set("table", tables, TABLE_PATHS)
    return figures, tables


def _validate_path_set(kind: str, supplied: tuple[str, ...], expected: tuple[str, ...]) -> None:
    if len(supplied) != len(expected):
        raise ValueError(f"{kind} path count must be exactly {len(expected)}")
    if len(set(supplied)) != len(supplied):
        raise ValueError(f"duplicate {kind} path")
    for value in supplied:
        path = PurePosixPath(value)
        if not value or "\\" in value or path.is_absolute() or ".." in path.parts or "://" in value or value.startswith("//"):
            raise ValueError(f"unsafe {kind} path: {value!r}")
    if supplied != expected:
        raise ValueError(f"{kind} paths must match the approved filenames and order")


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p99": 0.0}
    results = np.percentile(values, [25, 50, 75, 99])
    return {name: _finite_float(value) for name, value in zip(("p25", "p50", "p75", "p99"), results)}


def _provenance_counts(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _finite_float(value: object) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("summary numeric values must be finite")
    return converted


def _json_native(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _json_native(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, float):
        return _finite_float(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"value is not JSON-native: {type(value).__name__}")


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("summary field must be a mapping")
    return value


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("summary field must be a sequence")
    return value


def _write_text(output_path: Path, content: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return path


def _html_text(value: object, *, quote: bool = False) -> str:
    """Escape dynamic text and neutralize external-URI tokens in offline source."""
    escaped = html.escape(str(value), quote=quote)
    return escaped.replace("https://", "https&#58;//").replace("http://", "http&#58;//").replace("data:", "data&#58;")
