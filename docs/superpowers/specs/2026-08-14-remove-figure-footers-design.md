# Remove Figure Footers — Design

## Goal

Remove the two user-visible explanatory footer sentences from generated HUGE-Bench figures while preserving every analytical label, value, legend, title, and data series.

## Scope

Remove:

1. `Source: official HUGE-Bench stage-annotation sidecar` from all eight figures.
2. `Recovered boundaries apply to released obstacle actions; they are not original human annotations.` from the annotation-provenance figure.

Preserve:

- figure titles, axes, legends, bar labels, heatmap values, median labels, p99 display note, timeline labels, and provenance categories;
- the scientific provenance caveat in Markdown/HTML reports and structured output data;
- the eight approved filenames, PNG format, 180 DPI, and output ordering.

## Implementation

Use the minimal source-level removal:

- delete the shared `FOOTER` constant and `_footer()` helper;
- remove each `_footer(figure)` call;
- remove the provenance figure's standalone bottom explanatory `figure.text(...)` call;
- allow constrained layout to use the recovered vertical space naturally.

Do not add a configuration flag because the requested behavior is unconditional and no current caller needs the old footer.

## Verification

Follow test-driven development:

1. Add a test that constructs all eight real figure objects and asserts neither removed sentence appears in any figure-level or axes-level text.
2. Run that test before the production change and require the expected failure.
3. Apply the minimal production change and rerun the focused and full test suites.
4. Regenerate the official outputs with `my_env`.
5. Verify eight PNG files, exact filenames, nonblank content, minimum dimensions, and approximately 180 DPI.
6. Inspect a contact sheet to confirm that removing footer space causes no clipping or overlap.

## Non-goals

- No changes to dataset loading, validation, statistics, metrics, report claims, or provenance values.
- No removal of useful in-plot analytical annotations.
- No changes to the official `HUGE-Bench/` repository.
