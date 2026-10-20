# Architecture

## Modules

- `signals.py`: `SignalParams` (what the Signal page edits) and `make_signal`, returning a `Signal`: the trace,
  its sampling rate, the ground truth when known, and event times to mark.
- `morlet.py`: pywt's `cmorB-C`: `name`, `response` (the Gaussian frequency response of one row, for the band
  gain curve) and `cwt`, which is exactly the call `Denoiser._cwt` makes.
- `pipeline.py`: `Settings` (every parameter, grouped by the stage it first affects; the cluster, reduce and mask
  groups are vnoiser's own `ClusteringConfig`, `cwtReducerConfig` and `thresConfig`), `compute`, `Run`, `Band`.
- `plotting.py`: colors, `begin_plot` (legend above the plot), `decimate` / `line` for long traces,
  `shade_intervals`, `hz_ticks`.
- `config.py`: `~/.imgui_vnoiser/settings.json` and the UI scale.
- `app.py`: `App`, the window. `controls_<page>` draws a page's parameters in the rail, `page_<page>` its plots.

## Stages

`STAGES = signal, cwt, cluster, reduce, mask, final`. `Settings.first_change(other)` names the earliest stage whose
values differ; `compute(settings, start, previous)` copies every stage before `start` from `previous` and runs the
rest. A stage that raises sets `Run.error` / `Run.failed`; the next compute restarts at the failed stage at the
latest.

Each stage calls vnoiser's class for it (`FrequencyClusterer`, `WaveletReducer`, `AdaptiveThreshold`,
`fir_lowpass`). What those classes compute and discard, a band's trace before smoothing, its open and close levels,
its windows, a mask before the blur, is recomputed beside them in `band_details` and `mask_details`.
`tests/test_pipeline.py` checks those against what the classes return, and the final trace against `Denoiser.run`,
so a change to vnoiser that the mirrors miss fails a test.

The final stage sums each kept band's masked trace itself (rather than taking `rescaled_signal`) so bands can be
left out; with none left out the two agree to 1e-15.

## Threads

`App.poll()` runs at the top of each frame. One worker thread runs `compute`; while it runs the window keeps
drawing the previous `Run`. When it finishes and the settings have moved on, the next run starts from the earliest
changed stage, so dragging a slider coalesces into at most one queued run. Runs never share mutable state: `Band`s
are copied before the mask stage fills them, arrays are read-only once made.

## Plots

All time plots link their x axis to two `implot.BoxedValue`s on the App (`t0`, `t1`), so zooming one zooms all.
Lines are cut to that window and drawn as per-bin min/max past 4000 points. The scalogram and mask images are
cut to the window and max-pooled to about 1200 columns, cached per run, mode and window. Frequency axes on the
images are log10(Hz) with Hz tick labels, since a heatmap's cells are uniform in plot units.

## UI scale

`style.font_scale_main` is the scale (imgui 1.92 rasterizes glyphs at the size drawn, so text stays sharp);
`style.scale_all_sizes` and a list of ImPlot paddings get the ratio to the previous scale; `plotting.SCALE`
multiplies line weights and marker sizes. `App.apply_scale` does this at the top of a frame, never mid-frame.

## Gotchas

- ImPlot's bindings want 1-D contiguous arrays: `x.real` of a complex array and slices of `welch`'s output are
  strided, so they go through `np.ascontiguousarray`.
- Headless tests run on hello_imgui's null backend, which needs `renderer_has_textures` set in `post_init`.
- `implot.setup_axis_links` must come before anything that locks the axis in the same plot.
