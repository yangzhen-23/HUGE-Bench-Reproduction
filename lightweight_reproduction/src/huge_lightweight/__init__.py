"""Typed, offline loaders for the HUGE-Bench annotation sidecar."""

from .analysis import AnalysisEpisodeRow, AnalysisResult, AnalysisStageRow, RepresentativeEpisode, analyze, write_csv_tables
from .loader import JsonlParseError, iter_jsonl, load_annotations, load_manifest
from .metric_smoke import MetricSmokeResult, run_metric_smoke
from .plots import (
    TimelineRow,
    TimelineSegment,
    create_all_figures,
    representative_timeline_rows,
    stage_duration_p99,
    stages_per_episode_proportions,
    task_scene_array,
)
from .models import (
    AnnotationDataset,
    AnnotationManifest,
    EpisodeRecord,
    ManifestFile,
    SplitManifest,
    StageRecord,
)
from .validation import ValidationCheck, ValidationResult, validate_annotations
from .report import (
    ReportFigure,
    ReportLink,
    ReportSection,
    build_report_sections,
    build_summary,
    write_html_report,
    write_markdown_report,
    write_summary_json,
)

__all__ = [
    "AnnotationDataset",
    "AnnotationManifest",
    "AnalysisEpisodeRow",
    "AnalysisResult",
    "AnalysisStageRow",
    "EpisodeRecord",
    "JsonlParseError",
    "ManifestFile",
    "MetricSmokeResult",
    "RepresentativeEpisode",
    "ReportFigure",
    "ReportLink",
    "ReportSection",
    "SplitManifest",
    "StageRecord",
    "TimelineRow",
    "TimelineSegment",
    "ValidationCheck",
    "ValidationResult",
    "analyze",
    "build_report_sections",
    "build_summary",
    "create_all_figures",
    "iter_jsonl",
    "load_annotations",
    "load_manifest",
    "run_metric_smoke",
    "representative_timeline_rows",
    "stage_duration_p99",
    "stages_per_episode_proportions",
    "task_scene_array",
    "validate_annotations",
    "write_csv_tables",
    "write_html_report",
    "write_markdown_report",
    "write_summary_json",
]
