# imgui_vnoiser

A desktop window for seeing what [vnoiser](https://github.com/FlynnOConnell/vnoiser)'s Morlet wavelet denoiser does
to a voltage-imaging trace, one stage at a time. Every page is one stage of `vnoiser.Denoiser.run`, with that stage's
parameters on the left and its plots on the right. Change a parameter and only that stage and the ones after it
rerun. The stages call vnoiser's own classes, so what you see is what the package does: the final trace matches
`Denoiser.run` to float precision (`tests/test_pipeline.py`).

Built on [imgui-bundle](https://github.com/pthom/imgui_bundle) (ImGui + ImPlot).

![Scalogram page](docs/scalogram.png)

## Install

vnoiser is installed from a checkout next to this one (`../vnoiser`, see `[tool.uv.sources]` in `pyproject.toml`):

```bash
git clone git@github.com:FlynnOConnell/vnoiser.git
git clone <this repo> imgui_vnoiser
cd imgui_vnoiser
uv venv -p 3.12 && uv pip install -e ".[dev]"
imgui-vnoiser              # or: python -m imgui_vnoiser.app
```

### Text size

On some HiDPI setups (Hyprland through XWayland, for one) GLFW reports a scale of 1, so the window cannot see that
the monitor is scaled and changing the desktop scale has no effect on it. The window has its own scale:

- **Ctrl +** / **Ctrl -** grow and shrink everything, **Ctrl 0** goes back to the detected scale.
- **text size** at the bottom of the left rail does the same.
- `imgui-vnoiser --scale 2`, or `IMGUI_VNOISER_SCALE=2`, sets it at start.

Without either, the window uses the last scale you picked (saved in `~/.imgui_vnoiser/settings.json`), else
`GDK_SCALE`, else Hyprland's monitor scale, else 1.

## The pages

| page | shows | parameters |
|------|-------|------------|
| 1 Signal | the input trace, its clean version when known, its spectrum | source, fs, duration, noise, file and column |
| 2 Morlet wavelet | one row's wavelet in time, the time-frequency tile of every 10th row, every row's frequency response and what summing them passes | B, C, the scale range and count |
| 3 Scalogram | \|W\|, \|W\| over its row's median, standardized \|W\| (what the clustering sees) or Re W; one row against the input; every row summed | the color mode, the row |
| 4 Frequency bands | which rows went to which band, the gain of summing the bands, rows in PC space, explained variance, the Ward tree with its cut | PCA components, PCs clustered, linkage, band count |
| 5 Events and masks | one band's trace with its open and close levels and windows, its mask before and after the blur, what it adds; every band's mask as an image | open levels, savgol windows, complex fast bands, soft/hard, floors, blur |
| 6 Reconstruction | each band's contribution stacked, the denoised trace with the baseline, what was removed, spectra | the baseline filter, adding it, which bands go in the sum |

Every time plot shares one time window: drag to pan, scroll to zoom, double-click to fit, or use the
buttons over the plots. **How it works** on each page says what the stage does in vnoiser's terms.

At the bottom of the rail: **Denoiser defaults**, **Denoiser.upstream** (the archive's settings), and
**Copy as Python**, which copies a `vnoiser.Denoiser(...)` call with the current settings.

## Things to try

1. **What a row is.** Signal: *Impulse*, noise 0. Scalogram, Re W: each row shows its own wavelet, short at the top
   and long at the bottom. Morlet wavelet: pick *Narrowband* and look again; the cone widens and events smear.
2. **Rows tile frequency.** Signal: *Chirp*. The rows light up one after another. Morlet wavelet: rows above fs / 2
   (yellow line) have nothing real to respond to.
3. **The bands come from the data.** Frequency bands: compare *Spikes and bumps* with *JEDI3sub*. The edges move,
   and fast bands can interleave (a band's rows need not be contiguous).
4. **Oscillations are dropped, transients kept.** Signal: *Sum of tones*, one tone (e.g. `20`), noise 0. A sine
   peaks at 1.41 SD, never reaching the 2 or 2.5 SD open level, so its band opens no window and is dropped.
   Reconstruction: the tone is gone.
5. **The sum does not preserve amplitude.** Frequency bands: the gain curve sits around 0.5 to 0.9 because each
   band averages its rows rather than summing them. Reconstruction: spikes come back smaller than in the clean
   trace.
6. **Soft vs hard masks.** Events and masks: switch to *hard* and watch the floor go to 0.05; Reconstruction: the
   noise between events drops and slow drifts go with it.
7. **What each band carries.** Reconstruction: untick bands one at a time.

## Files

| path | holds |
|------|-------|
| `src/imgui_vnoiser/signals.py` | input traces: synthetic, test signals, files |
| `src/imgui_vnoiser/morlet.py` | the cmor wavelet in seconds and hertz: shape, widths, frequency response |
| `src/imgui_vnoiser/pipeline.py` | `Settings`, the staged `compute`, per-band details |
| `src/imgui_vnoiser/plotting.py` | ImPlot helpers: decimated lines, shaded windows, colors |
| `src/imgui_vnoiser/config.py` | the saved UI scale and its detection |
| `src/imgui_vnoiser/app.py` | the window |

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the parts fit.
