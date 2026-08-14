# Remove Figure Footers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the two approved bottom explanatory sentences from all generated figures without changing analytical annotations, data, filenames, or report scope.

**Architecture:** Keep the existing eight-plot pipeline and remove footer rendering at its source. Add a figure-object regression test that observes real Matplotlib text artists before changing production code, then regenerate the official outputs with `my_env` and inspect the rendered images.

**Tech Stack:** Python 3.12, Matplotlib, NumPy, Pillow, pytest, PowerShell.

## Global Constraints

- Remove `Source: official HUGE-Bench stage-annotation sidecar` from all eight figures.
- Remove `Recovered boundaries apply to released obstacle actions; they are not original human annotations.` from the annotation-provenance figure.
- Preserve titles, axes, legends, values, median labels, the p99 note, timeline labels, provenance categories, eight filenames, PNG format, ordering, and 180 DPI.
- Preserve provenance caveats in Markdown/HTML reports and structured data.
- Do not modify the official `HUGE-Bench/` repository.
- Do not create a Git commit unless the user explicitly requests one.

---

### Task 0: Restore a Clean Python Baseline

**Files:**
- Modify: `lightweight_reproduction/src/huge_lightweight/analysis.py`
- Modify: `lightweight_reproduction/src/huge_lightweight/cli.py`
- Modify: `lightweight_reproduction/src/huge_lightweight/loader.py`
- Modify: `lightweight_reproduction/src/huge_lightweight/metric_smoke.py`
- Modify: `lightweight_reproduction/src/huge_lightweight/plots.py`
- Modify: `lightweight_reproduction/src/huge_lightweight/report.py`

**Interfaces:**
- Consumes: the existing English module docstring and the user-added Chinese explanatory string in each affected module.
- Produces: one legal combined module docstring per file, followed immediately by `from __future__ import annotations`; no runtime interface or explanatory content changes.

- [ ] **Step 1: Preserve the observed RED baseline evidence**

The pre-implementation baseline command already failed during collection:

```powershell
& 'D:\anaconda3\envs\my_env\python.exe' -m pytest lightweight_reproduction/tests -q
```

Expected/observed failure: `SyntaxError: from __future__ imports must occur at the beginning of the file` because six modules contain two string literals before the future import.

- [ ] **Step 2: Merge each pair of top-level strings**

For each affected file, retain all English and Chinese explanatory text inside a single leading triple-quoted module docstring. Ensure `from __future__ import annotations` is the first statement after that one docstring. Do not edit executable statements.

- [ ] **Step 3: Verify compilation and the clean baseline**

```powershell
& 'D:\anaconda3\envs\my_env\python.exe' -m compileall -q lightweight_reproduction/src
& 'D:\anaconda3\envs\my_env\python.exe' -m pytest lightweight_reproduction/tests -q
```

Expected: compile exit 0 and the pre-footer baseline suite passes 76 tests.

---

### Task 1: Remove Footer Text at the Figure-Object Boundary

**Files:**
- Modify: `lightweight_reproduction/tests/test_plots.py`
- Modify: `lightweight_reproduction/src/huge_lightweight/plots.py`

**Interfaces:**
- Consumes: the existing private plotters `_split_overview`, `_task_distribution`, `_task_scene_heatmap`, `_episode_length_distribution`, `_stages_per_episode`, `_stage_duration_by_task`, `_annotation_provenance`, and `_example_stage_timeline`, each taking `AnalysisResult` and returning `matplotlib.pyplot.Figure`.
- Produces: the same eight figures with unchanged public `create_all_figures(result, output_dir) -> tuple[Path, ...]`, but without either removed sentence.

- [ ] **Step 1: Add the failing text-artist regression test**

In `test_plots.py`, import the plots module as `plots_module`, construct each real figure from `synthetic_result`, collect `figure.texts` and every `axis.texts`, close each figure in `finally`, and assert both forbidden sentences are absent:

```python
from huge_lightweight import plots as plots_module


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
```

- [ ] **Step 2: Run the focused test and verify RED**

Run from `HUGE-Bench Reproduction/lightweight_reproduction`:

```powershell
& 'D:\anaconda3\envs\my_env\python.exe' -m pytest tests/test_plots.py::test_all_figure_objects_omit_explanatory_footers -v
```

Expected: FAIL because at least `_split_overview` contains the shared source footer.

- [ ] **Step 3: Apply the minimal production change**

In `plots.py`:

- delete the `FOOTER` constant;
- delete `_footer(figure)` and its layout reservation;
- delete all eight `_footer(figure)` calls;
- delete only the bottom `figure.text(...)` sentence inside `_annotation_provenance`;
- keep the p99 `axis.text(...)`, medians, bar labels, heatmap values, timeline labels, and `recovered` suffix untouched.

- [ ] **Step 4: Run focused plotting tests and verify GREEN**

```powershell
& 'D:\anaconda3\envs\my_env\python.exe' -m pytest tests/test_plots.py -v
```

Expected: all plotting tests PASS with no unclosed-figure warning.

- [ ] **Step 5: Run the full regression suite**

If pytest is absent from `my_env`, install the test-only dependency first:

```powershell
& 'D:\anaconda3\envs\my_env\python.exe' -m pip install "pytest>=7.4"
```

Then run:

```powershell
& 'D:\anaconda3\envs\my_env\python.exe' -m pytest tests -q
& 'D:\anaconda3\envs\my_env\python.exe' -m compileall -q src
```

Expected: 77 tests PASS after adding the new regression test; compile exit 0.

---

### Task 2: Regenerate and Inspect Official Outputs

**Files:**
- Regenerate: `lightweight_reproduction/outputs/figures/*.png`
- Regenerate: `lightweight_reproduction/outputs/tables/*.csv`
- Regenerate: `lightweight_reproduction/outputs/HUGE_Bench_轻量复现报告.md`
- Regenerate: `lightweight_reproduction/outputs/HUGE_Bench_轻量复现报告.html`
- Regenerate: `lightweight_reproduction/outputs/summary.json`
- Regenerate: `lightweight_reproduction/outputs/run_manifest.json`

**Interfaces:**
- Consumes: official sidecar at `../HUGE-Bench/trajectory_generation/stage_annotations` relative to repository root and the unchanged CLI module.
- Produces: the same 17-file output contract, with eight footer-free PNG images.

- [ ] **Step 1: Run the official pipeline from the repository root**

```powershell
$env:PYTHONPATH = (Resolve-Path '.\lightweight_reproduction\src').Path
try {
    & 'D:\anaconda3\envs\my_env\python.exe' -m huge_lightweight.cli `
      --annotations-root '..\HUGE-Bench\trajectory_generation\stage_annotations' `
      --repo-root '..\HUGE-Bench' `
      --output '.\lightweight_reproduction\outputs'
} finally {
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
}
```

Expected terminal gate:

```text
LIGHTWEIGHT_REPRODUCTION_OK episodes=6168 segments=27539
```

- [ ] **Step 2: Verify the output contract and PNG metadata**

Use a read-only Python probe with Pillow to require:

- exactly eight approved PNG filenames;
- each image is RGB/RGBA, at least 800 × 450, approximately 180 DPI, and nonblank;
- `summary.json` reports 6,168 episodes, 27,539 stage segments, and 147,263 PASS / 0 FAIL;
- `run_manifest.json` records SUCCESS and 16 output fingerprints.

- [ ] **Step 3: Perform visual inspection**

Build a temporary contact sheet outside `outputs/`, inspect it at original resolution, and verify:

- neither bottom sentence is visible;
- no title, axis label, legend, number, p99 note, median label, or timeline label is clipped;
- reclaimed bottom whitespace does not create overlap or imbalance.

- [ ] **Step 4: Review the final diff**

```powershell
git status --short
git diff -- lightweight_reproduction/src/huge_lightweight/plots.py lightweight_reproduction/tests/test_plots.py
git -C '..\HUGE-Bench' diff --stat
```

Expected: source/test changes are limited to footer removal and its regression test; the official nested repository has no tracked diff.
