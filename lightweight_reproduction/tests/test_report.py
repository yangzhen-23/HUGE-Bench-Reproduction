from __future__ import annotations

import dataclasses
import importlib
import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from huge_lightweight.analysis import analyze
from huge_lightweight.loader import load_annotations
from huge_lightweight.metric_smoke import MetricSmokeResult, SMOKE_LABEL
from huge_lightweight.validation import ValidationCheck, ValidationResult, validate_annotations


FIGURES = (
    "figures/01_split_overview.png",
    "figures/02_task_distribution.png",
    "figures/03_task_scene_heatmap.png",
    "figures/04_episode_length_distribution.png",
    "figures/05_stages_per_episode.png",
    "figures/06_stage_duration_by_task.png",
    "figures/07_annotation_provenance.png",
    "figures/08_example_stage_timeline.png",
)
TABLES = (
    "tables/dataset_overview.csv",
    "tables/task_statistics.csv",
    "tables/task_scene_matrix.csv",
    "tables/stage_statistics.csv",
    "tables/validation_report.csv",
)
ALLOWED = "复现了 HUGE-Bench 官方公开数据的任务结构、多阶段标注统计与标注完整性检查。"
EXCLUDED = (
    "未复现 PI0/PI0.5 模型性能。",
    "未复现论文 TCR、nDTW、NSP、CR 或 CSPL 主实验数值。",
    "未完成 3DGS 渲染、闭环飞行或碰撞评估。",
    "recovered_from_released_actions 不等同于原始人工阶段标注。",
)
PLACEHOLDER_DISCLOSURE = "Task 6 中的 0 bytes 和 0.000 秒是占位值，并非资源实测；完整流程的峰值内存与耗时将在 Task 7/8 测量。"


def _report_module():
    try:
        return importlib.import_module("huge_lightweight.report")
    except ModuleNotFoundError:
        pytest.fail("huge_lightweight.report is not implemented")


def _metric() -> MetricSmokeResult:
    return MetricSmokeResult("PASS", SMOKE_LABEL, 1.0, 1.0, 1.0, 1.0, 2.0, "synthetic only")


def _summary(
    valid_sidecar: Path,
    *,
    annotations_root: str = "fixture-root",
    resources_measured: bool = True,
) -> dict[str, object]:
    report = _report_module()
    data = load_annotations(valid_sidecar)
    result = analyze(data)
    validation = validate_annotations(valid_sidecar, data)
    return report.build_summary(
        result,
        validation,
        _metric(),
        source_dataset="fixture-dataset",
        raw_subtask_files=3,
        annotations_root=annotations_root,
        repository_commit="abc123",
        manifest_sha256="f" * 64,
        peak_python_memory_bytes=1234,
        elapsed_seconds=1.25,
        resources_measured=resources_measured,
    )


def _official_summary(valid_sidecar: Path, *, annotations_root: str = "fixture-root") -> dict[str, object]:
    summary = _summary(valid_sidecar, annotations_root=annotations_root)
    summary["counts"]["total_episodes"] = 6168
    summary["counts"]["total_stage_segments"] = 27539
    summary["splits"] = [
        {"split": "train", "episodes": 5175, "episode_ratio": 5175 / 6168, "frames": 100, "stages": 23115, "hours_at_5fps": 100 / 5 / 3600},
        {"split": "test_seen", "episodes": 576, "episode_ratio": 576 / 6168, "frames": 20, "stages": 2573, "hours_at_5fps": 20 / 5 / 3600},
        {"split": "test_unseen", "episodes": 417, "episode_ratio": 417 / 6168, "frames": 10, "stages": 1851, "hours_at_5fps": 10 / 5 / 3600},
    ]
    summary["provenance"]["recovered_from_released_actions"].update(episodes=42, stages=99)
    return summary


def test_build_summary_contract_is_json_native_and_ordered(valid_sidecar: Path):
    report = _report_module()
    summary = _summary(valid_sidecar)

    assert list(summary) == [
        "scope", "source", "counts", "splits", "tasks", "scenes",
        "episode_length_quantiles_frames", "stage_duration_quantiles_frames",
        "provenance", "representative_episodes", "validation", "metric_smoke", "resources",
    ]
    assert summary["scope"] == {"allowed_conclusion": ALLOWED, "excluded_claims": list(EXCLUDED)}
    assert [item["split"] for item in summary["splits"]] == ["train", "test_seen", "test_unseen"]
    assert [item["task_id"] for item in summary["tasks"]] == ["0", "hl", "orbit", "building", "road", "farm", "obstacle", "orbit_multi"]
    assert summary["counts"] == {
        "total_episodes": 1, "total_stage_segments": 2, "total_frames": 5,
        "total_hours_at_5fps": pytest.approx(5 / 5 / 3600), "task_count": 1,
        "scene_count": 1, "raw_subtask_files": 3, "observed_stages_per_episode": [2],
    }
    assert summary["episode_length_quantiles_frames"] == {"p25": 5.0, "p50": 5.0, "p75": 5.0, "p99": 5.0}
    assert summary["stage_duration_quantiles_frames"] == {"p25": 2.25, "p50": 2.5, "p75": 2.75, "p99": 2.99}
    assert summary["provenance"]["original_raw"] == {"episodes": 1, "episode_ratio": 1.0, "stages": 2, "stage_ratio": 1.0}
    assert summary["validation"]["ok"] is True
    assert summary["validation"]["total_checks"] == len(validate_annotations(valid_sidecar, load_annotations(valid_sidecar)).checks)
    assert summary["metric_smoke"] == dataclasses.asdict(_metric())
    assert summary["resources"] == {"peak_python_memory_bytes": 1234, "elapsed_seconds": 1.25, "measured": True}
    assert summary["representative_episodes"][0]["instruction"] == "Fly to the office."
    assert all(0.0 <= split["episode_ratio"] <= 1.0 for split in summary["splits"])
    json.dumps(summary, allow_nan=False)


def test_failed_validation_serializes_failure_ids_and_scopes(valid_sidecar: Path):
    report = _report_module()
    data = load_annotations(valid_sidecar)
    validation = ValidationResult((ValidationCheck("broken", "train/episode/0", "FAIL", "yes", "no", "fixture"),))
    summary = report.build_summary(
        analyze(data), validation, _metric(), source_dataset="fixture", raw_subtask_files=0,
        annotations_root="root", repository_commit="unavailable", manifest_sha256="a" * 64,
        peak_python_memory_bytes=0, elapsed_seconds=0.0,
    )
    assert summary["validation"] == {
        "ok": False, "total_checks": 1, "pass_count": 0, "fail_count": 1,
        "failure_ids": ["broken"], "failure_scopes": ["train/episode/0"],
    }


class _AuditParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.counts: dict[str, int] = {}
        self.images: list[dict[str, str | None]] = []
        self.links: list[str] = []
        self.attrs: list[tuple[str, list[tuple[str, str | None]]]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.counts[tag] = self.counts.get(tag, 0) + 1
        values = dict(attrs)
        self.attrs.append((tag, attrs))
        if tag == "img":
            self.images.append(values)
        if tag == "a" and values.get("href"):
            self.links.append(values["href"] or "")

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def test_placeholder_resource_status_is_explicit_in_both_renderers(valid_sidecar: Path, tmp_path: Path):
    report = _report_module()
    summary = _summary(valid_sidecar, resources_measured=False)
    summary["resources"].update(peak_python_memory_bytes=0, elapsed_seconds=0.0)

    assert summary["resources"]["measured"] is False
    markdown = report.write_markdown_report(summary, FIGURES, TABLES, tmp_path / "placeholder.md").read_text(encoding="utf-8")
    html_text = report.write_html_report(summary, FIGURES, TABLES, tmp_path / "placeholder.html").read_text(encoding="utf-8")
    parser = _AuditParser()
    parser.feed(html_text)

    assert PLACEHOLDER_DISCLOSURE in markdown
    assert PLACEHOLDER_DISCLOSURE in "".join(parser.text)


def test_measured_resource_status_uses_measurement_wording(valid_sidecar: Path, tmp_path: Path):
    report = _report_module()
    summary = _summary(valid_sidecar, resources_measured=True)
    markdown = report.write_markdown_report(summary, FIGURES, TABLES, tmp_path / "measured.md").read_text(encoding="utf-8")
    html_text = report.write_html_report(summary, FIGURES, TABLES, tmp_path / "measured.html").read_text(encoding="utf-8")
    parser = _AuditParser()
    parser.feed(html_text)

    assert "以上为调用方提供的实测资源值。" in markdown
    assert "以上为调用方提供的实测资源值。" in "".join(parser.text)
    assert PLACEHOLDER_DISCLOSURE not in markdown


def test_writers_share_sections_and_render_required_content(valid_sidecar: Path, tmp_path: Path):
    report = _report_module()
    summary = _official_summary(valid_sidecar)
    sections = report.build_report_sections(summary, FIGURES, TABLES)
    markdown_path = report.write_markdown_report(summary, FIGURES, TABLES, tmp_path / "report.md")
    html_path = report.write_html_report(summary, FIGURES, TABLES, tmp_path / "report.html")
    markdown = markdown_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")

    assert len(sections) == 10
    for section in sections:
        assert section.heading in markdown and section.heading in html
        for paragraph in section.paragraphs:
            assert paragraph in markdown and paragraph in html
        for figure in section.figures:
            assert figure.caption in markdown and figure.caption in html
    for required in ("6,168", "27,539", SMOKE_LABEL, ALLOWED, *EXCLUDED):
        assert required in markdown and required in html
    assert summary["provenance"]["caveat"] in markdown and summary["provenance"]["caveat"] in html
    for path in FIGURES:
        assert markdown.count(path) == 1
        assert html.count(path) == 1
    for path in (*TABLES, "summary.json"):
        assert path in markdown and path in html
    assert markdown.endswith("\n") and html.endswith("\n")


def test_html_is_offline_semantic_and_escapes_dynamic_values(valid_sidecar: Path, tmp_path: Path):
    report = _report_module()
    attack = '<script>alert(1)</script><img src="https://evil.invalid/x">'
    summary = _official_summary(valid_sidecar, annotations_root=attack)
    path = report.write_html_report(summary, FIGURES, TABLES, tmp_path / "safe.html")
    text = path.read_text(encoding="utf-8")
    parser = _AuditParser()
    parser.feed(text)

    assert parser.counts.get("html") == parser.counts.get("head") == parser.counts.get("body") == 1
    assert parser.counts.get("style") == 1 and parser.counts.get("script", 0) == 0
    assert parser.counts.get("figure") == parser.counts.get("figcaption") == 8
    assert [image["src"] for image in parser.images] == list(FIGURES)
    assert all(image.get("alt") and any("\u4e00" <= char <= "\u9fff" for char in image["alt"] or "") for image in parser.images)
    assert set((*TABLES, "summary.json")).issubset(parser.links)
    assert '<script>alert(1)</script>' not in text and "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "http://" not in text and "https://" not in text and "data:" not in text.lower()
    assert not any(name.lower().startswith("on") for _, attrs in parser.attrs for name, _ in attrs)
    assert '<meta charset="utf-8">' in text and '<html lang="zh-CN">' in text


def test_uri_bearing_dynamic_metadata_preserves_shared_visible_text(valid_sidecar: Path, tmp_path: Path):
    report = _report_module()
    benign = "元数据 <A&B> https://example.invalid/a http://example.invalid/b data:text/plain,ok"
    summary = _official_summary(valid_sidecar, annotations_root=benign)
    sections = report.build_report_sections(summary, FIGURES, TABLES)
    source_paragraph = sections[1].paragraphs[0]
    markdown = report.write_markdown_report(summary, FIGURES, TABLES, tmp_path / "benign.md").read_text(encoding="utf-8")
    raw_html = report.write_html_report(summary, FIGURES, TABLES, tmp_path / "benign.html").read_text(encoding="utf-8")
    parser = _AuditParser()
    parser.feed(raw_html)

    assert source_paragraph in markdown
    assert source_paragraph in "".join(parser.text)
    assert "https://" not in raw_html and "http://" not in raw_html and "data:" not in raw_html.lower()
    assert parser.counts.get("script", 0) == 0
    assert len(parser.images) == 8


@pytest.mark.parametrize(
    ("figures", "tables"),
    [
        (FIGURES[:-1], TABLES),
        (("/absolute.png", *FIGURES[1:]), TABLES),
        (("../escape.png", *FIGURES[1:]), TABLES),
        (("https://evil.invalid/x.png", *FIGURES[1:]), TABLES),
        ((FIGURES[0], FIGURES[0], *FIGURES[2:]), TABLES),
        ((FIGURES[1], FIGURES[0], *FIGURES[2:]), TABLES),
        (FIGURES, TABLES[:-1]),
        (FIGURES, ("tables/wrong.csv", *TABLES[1:])),
        (FIGURES, (TABLES[0], TABLES[0], *TABLES[2:])),
    ],
)
def test_invalid_paths_raise_before_writing(valid_sidecar: Path, tmp_path: Path, figures, tables):
    report = _report_module()
    destination = tmp_path / "must-not-exist.html"
    with pytest.raises(ValueError, match="path|figure|table|expected|order|duplicate"):
        report.write_html_report(_summary(valid_sidecar), figures, tables, destination)
    assert not destination.exists()


def test_summary_json_is_utf8_deterministic_and_round_trips(valid_sidecar: Path, tmp_path: Path):
    report = _report_module()
    summary = _summary(valid_sidecar)
    first = report.write_summary_json(summary, tmp_path / "one" / "summary.json")
    second = report.write_summary_json(summary, tmp_path / "two" / "summary.json")

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")
    assert b"\\u" not in first.read_bytes()
    with first.open(encoding="utf-8") as handle:
        assert json.load(handle) == summary
