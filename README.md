# imgui_vnoiser

Interactive viewer for [vnoiser](https://github.com/FlynnOConnell/vnoiser)'s Morlet wavelet denoiser. Each page is
one stage of `vnoiser.Denoiser.run`, with its parameters on the left and its plots on the right. Changing a
parameter reruns that stage and the ones after it. The stages call vnoiser's classes directly, and the output
matches `Denoiser.run` (`tests/test_pipeline.py`).

Built on [imgui-bundle](https://github.com/pthom/imgui_bundle) (ImGui + ImPlot).

![Scalogram page](docs/scalogram.png)

## Install

Needs a vnoiser checkout next to this one (`../vnoiser`, see `[tool.uv.sources]` in `pyproject.toml`).

```bash
git clone git@github.com:FlynnOConnell/vnoiser.git
git clone git@github.com:FlynnOConnell/imgui_vnoiser.git
cd imgui_vnoiser
uv venv -p 3.12 && uv pip install -e ".[dev]"
imgui-vnoiser
```

## Text size

GLFW reports a content scale of 1 under XWayland, so the desktop scale has no effect on the window. Set it in the
app instead:

- Ctrl + / Ctrl - to resize, Ctrl 0 to reset to the detected scale
- the text size slider at the bottom of the left panel
- `imgui-vnoiser --scale 2` or `IMGUI_VNOISER_SCALE=2`

The last value is saved in `~/.imgui_vnoiser/settings.json`. With nothing saved, the default comes from
`GDK_SCALE`, then Hyprland's monitor scale, then 1.

## Pages

| page | plots | parameters |
|------|-------|------------|
| 1 Signal | input trace, ground truth, power spectrum | source, fs, duration, noise SD, file and column |
| 2 Morlet wavelet | wavelet at one row, time-frequency tiles, frequency response of every row, summed gain | B, C, scale range, number of rows |
| 3 Scalogram | \|W\|, \|W\| / row median, standardized \|W\|, or Re W; one row against the input; sum of all rows | color mode, row |
| 4 Frequency bands | band membership over frequency, band-sum gain, PCA scatter, explained variance, Ward tree | PCA components, PCs clustered, linkage, number of bands |
| 5 Events and masks | band trace with open/close levels and windows, mask before and after blur, masked band, all masks | open levels, savgol windows, complex fast bands, soft/hard, floors, blur SD |
| 6 Reconstruction | per-band contributions, denoised vs raw vs ground truth, removed signal, spectra | baseline filter, add baseline, bands in the sum |

All time plots share one x range (drag to pan, scroll to zoom, double-click to fit). The buttons at the bottom of
the left panel reset to `Denoiser()` defaults, switch to `Denoiser.upstream`, or copy the equivalent `Denoiser(...)`
call to the clipboard.

## Things to try

1. Compare the bands for Spikes and bumps vs JEDI3sub. Band edges change with the trace, and fast bands can
   interleave.
2. The band-sum gain on the Frequency bands page is about 0.5 to 0.9, because each band averages its rows instead of
   summing them. Spikes in the reconstruction are smaller than in the ground truth.
3. Switch masks from soft to hard: the floor drops to 0.05, removing more noise between events and more of the slow
   components.
4. Untick bands on the Reconstruction page to see what each contributes.

## Layout

| file | contents |
|------|----------|
| `src/imgui_vnoiser/signals.py` | synthetic traces, test signals, file loading |
| `src/imgui_vnoiser/morlet.py` | cmor wavelet shape, widths and frequency response |
| `src/imgui_vnoiser/pipeline.py` | `Settings`, staged `compute`, per-band details |
| `src/imgui_vnoiser/plotting.py` | ImPlot helpers |
| `src/imgui_vnoiser/config.py` | saved UI scale and its detection |
| `src/imgui_vnoiser/app.py` | the window |

See [ARCHITECTURE.md](ARCHITECTURE.md).
