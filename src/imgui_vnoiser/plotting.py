"""
ImPlot helpers shared by the pages: colors, line specs, a decimating time-series view of long traces, shaded
intervals and the log-frequency ticks.
"""

import math
from typing import Optional

import numpy as np
from imgui_bundle import imgui, implot

ImVec4 = imgui.ImVec4

ACCENT = ImVec4(0.40, 0.68, 1.00, 1.0)
DIM = ImVec4(0.62, 0.62, 0.65, 1.0)
TEXT = ImVec4(0.92, 0.92, 0.94, 1.0)
RAW = ImVec4(0.60, 0.60, 0.66, 0.70)
TRUTH = ImVec4(0.35, 0.85, 0.45, 1.0)
WARN = ImVec4(1.0, 0.75, 0.3, 1.0)
ERROR = ImVec4(1.0, 0.4, 0.4, 1.0)
OK = ImVec4(0.40, 0.90, 0.40, 1.0)
OPEN = ImVec4(1.00, 0.55, 0.30, 1.0)
CLOSE = ImVec4(0.95, 0.85, 0.35, 1.0)
WHITE = ImVec4(1, 1, 1, 0.85)

# band colors, in band order from slowest to fastest
BAND_COLORS = tuple(
    ImVec4(r / 255, g / 255, b / 255, 1.0)
    for r, g, b in (
        (78, 121, 167), (242, 142, 43), (225, 87, 89), (118, 183, 178), (89, 161, 79), (237, 201, 72),
        (176, 122, 161), (255, 157, 167), (156, 117, 95), (186, 176, 172), (31, 119, 180), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (188, 189, 34),
    )
)

SCALE = 1.0  # the UI scale; line weights and marker sizes are in pixels, so they grow with it
MAX_POINTS = 4000  # per series on screen; longer views are drawn as per-pixel-column min/max
NICE_HZ = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)


def begin_plot(title: str, size: imgui.ImVec2, flags: int = 0) -> bool:
    """implot.begin_plot with the legend in a row above the plot, so big text never covers the data."""
    if not implot.begin_plot(title, size, flags):
        return False
    if not flags & implot.Flags_.no_legend:
        implot.setup_legend(implot.Location_.north_west, implot.LegendFlags_.outside | implot.LegendFlags_.horizontal)
    return True


def band_color(index: int, alpha: float = 1.0) -> ImVec4:
    c = BAND_COLORS[index % len(BAND_COLORS)]
    return ImVec4(c.x, c.y, c.z, alpha)


def faded(c: ImVec4, alpha: float) -> ImVec4:
    return ImVec4(c.x, c.y, c.z, alpha)


def spec(color: Optional[ImVec4] = None, weight: float = 1.0, fill_alpha: float = 1.0, flags: int = 0) -> implot.Spec:
    return implot.Spec(line_color=color, fill_color=color, line_weight=weight * SCALE, fill_alpha=fill_alpha, flags=flags)


def marker(color: ImVec4, size: float = 4.0, edge: Optional[ImVec4] = None) -> implot.Spec:
    """Filled circles, size in unscaled pixels."""
    return implot.Spec(marker=implot.Marker_.circle, marker_size=size * SCALE, marker_fill_color=color, marker_line_color=edge or color)


def window_indices(n: int, fs: float, t0: float, t1: float) -> tuple[int, int]:
    """The sample range [i0, i1) covering t0..t1 seconds, one sample of margin each side, clipped to the trace."""
    i0 = max(0, int(math.floor(t0 * fs)) - 1)
    i1 = min(n, int(math.ceil(t1 * fs)) + 2)
    return i0, max(i1, min(i0 + 2, n))


def decimate(y: np.ndarray, fs: float, t0: float, t1: float, max_points: int = MAX_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """
    (t, y) of the part of y between t0 and t1 seconds, as float64. Past max_points samples each bin of samples is
    drawn as its min then its max, so a spike one sample wide still shows at any zoom.
    """
    i0, i1 = window_indices(y.size, fs, t0, t1)
    seg = np.ascontiguousarray(y[i0:i1], dtype=np.float64)
    if seg.size <= max_points:
        return np.arange(i0, i0 + seg.size, dtype=np.float64) / fs, seg
    width = int(math.ceil(seg.size / (max_points // 2)))
    whole = seg.size // width * width
    bins = seg[:whole].reshape(-1, width)
    out = np.empty(bins.shape[0] * 2)
    out[0::2], out[1::2] = bins.min(axis=1), bins.max(axis=1)
    t = (i0 + np.repeat(np.arange(bins.shape[0]) * width, 2) + np.tile([0, width - 1], bins.shape[0])) / fs
    return t.astype(np.float64), out


def line(label: str, y: np.ndarray, fs: float, t0: float, t1: float, color: Optional[ImVec4] = None, weight: float = 1.0, flags: int = 0) -> None:
    """y against time, decimated to the view."""
    t, v = decimate(y, fs, t0, t1)
    implot.plot_line(label, t, v, spec(color, weight, flags=flags))


def hline(label: str, value: float, color: ImVec4, weight: float = 1.0) -> None:
    implot.plot_inf_lines(label, np.array([value], dtype=np.float64), spec(color, weight, flags=implot.InfLinesFlags_.horizontal))


def vlines(label: str, values, color: ImVec4, weight: float = 1.0) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.size:
        implot.plot_inf_lines(label, values, spec(color, weight))


def shade_intervals(intervals: np.ndarray, fs: float, t0: float, t1: float, color: ImVec4) -> None:
    """Fill each (start, end) sample interval across the plot's full height."""
    if intervals is None or len(intervals) == 0:
        return
    starts, ends = intervals[:, 0] / fs, intervals[:, 1] / fs
    visible = (ends >= t0) & (starts <= t1)
    if not np.any(visible):
        return
    draw = implot.get_plot_draw_list()
    implot.push_plot_clip_rect()
    top = implot.get_plot_pos().y
    bottom = top + implot.get_plot_size().y
    col = imgui.color_convert_float4_to_u32(color)
    for a, b in zip(starts[visible], ends[visible]):
        x0 = implot.plot_to_pixels(float(a), 0.0).x
        x1 = implot.plot_to_pixels(float(b), 0.0).x
        draw.add_rect_filled(imgui.ImVec2(x0, top), imgui.ImVec2(max(x1, x0 + 1), bottom), col)
    implot.pop_plot_clip_rect()


def hz_ticks(lo_hz: float, hi_hz: float, axis=implot.ImAxis_.y1) -> None:
    """Ticks in Hz on an axis plotted as log10(Hz)."""
    values = [f for f in NICE_HZ if lo_hz <= f <= hi_hz]
    if values:
        implot.setup_axis_ticks(axis, [math.log10(f) for f in values], [f"{f:g}" for f in values], False)


def fit_scale(values: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> tuple[float, float]:
    """Color limits from percentiles, never equal."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-9
    return float(lo), float(hi)
