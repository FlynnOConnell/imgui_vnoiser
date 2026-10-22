"""
The window `imgui-vnoiser` opens: a rail of pages on the left, one per step of the denoiser, with that
stage's parameters under the rail and its plots on the right. Every time plot shares one time window, so zooming
one zooms them all. The pipeline runs on a worker thread and reruns only the stages a change affects.
"""

import math
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp, implot
from imgui_bundle import portable_file_dialogs as pfd

from imgui_vnoiser import __version__, config
from imgui_vnoiser import plotting as P
from imgui_vnoiser.pipeline import STAGES, Run, Settings, band_sum_gain, compute, fmt_hz, metrics
from imgui_vnoiser.signals import SOURCES, find_scans, scan_info

TITLE_WINDOW = "vnoiser"
SIZE_WINDOW = (1560, 980)
WIDTH_RAIL_EM = 20

# the rail: a page key and its name
PAGES = (
    ("signal", "1  Line scan traces"),
    ("scalogram", "2  Split by speed"),
    ("bands", "3  Group into bands"),
    ("events", "4  Keep events"),
    ("final", "5  Add back"),
)

# one line per page: what to check on it
LOOK_FOR = {
    "signal": "a bright stripe (the dendrite) in every line you include; lines without one only add noise. "
              "A stripe that wanders is motion. The baseline should follow the slow decline (bleaching) and nothing faster.",
    "scalogram": "events as bright vertical streaks. Spikes reach up into the fast rows, slow depolarizations "
                 "stay low. Rows that are even speckle everywhere hold only noise; rows above the orange line always do.",
    "bands": "bands that sit either on signal or on noise, not across both. In the spectrum, signal is where the "
             "trace rises above the flat noise floor. In the bars, a fast band kept a large share of the time is letting "
             "noise through.",
    "events": "shaded windows on real events and not on noise. Many windows in noise: raise the threshold. "
              "Events with no window: lower it.",
    "final": "events kept in the denoised trace, and nothing event-shaped in 'removed'. Events in 'removed' "
             "mean the denoiser is cutting signal.",
}

HEAT_MODES = ("amplitude", "amplitude / typical", "standardized", "real part")


def same_line_if_fits(label: str) -> None:
    """Stay on this line when a button labeled label still fits on it."""
    room = imgui.get_content_region_avail().x - imgui.get_item_rect_max().x + imgui.get_cursor_screen_pos().x
    need = imgui.calc_text_size(label).x + 2 * imgui.get_style().frame_padding.x + imgui.get_style().item_spacing.x
    if room > need:
        imgui.same_line()


def caption(text: str) -> None:
    imgui.push_style_color(imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled))
    imgui.text_wrapped(text)
    imgui.pop_style_color()


def heading(text: str) -> None:
    imgui.spacing()
    imgui.text_colored(P.ACCENT, text.upper())
    imgui.separator()


class App:
    """
    State behind the window: the settings being edited, the last finished run, the job computing the next one,
    and what each page has selected.
    """

    def __init__(self, settings: Optional[Settings] = None, scale: float = 1.0):
        self.settings = settings or Settings()
        self.scale = config.clamp(scale)
        self.scale_applied = 1.0  # what the style sizes are scaled by now
        self.scale_edit: Optional[float] = None  # the text size slider's value while it is dragged
        self.run: Optional[Run] = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future: Optional[Future] = None
        self.submitted: Optional[Settings] = None
        self.stage_running = ""
        self.page = "signal"
        self.t0 = implot.BoxedValue(0.0)
        self.t1 = implot.BoxedValue(self.settings.signal.duration_s)
        self.fit_on_next = True
        self.row = 40
        self.heat_mode = 1
        self.heat_cache = (None, None)
        self.pixel_cache = (None, None)
        self.style_set = False
        self.band_id: Optional[int] = None
        self.preset = 0
        self.picker = None
        self.scans: Optional[list] = None  # exported line scans found under ~/data
        self.copied = ""

    # ------------------------------------------------------------------
    # the worker
    # ------------------------------------------------------------------

    def poll(self) -> None:
        """Land a finished run, then start the next if the settings moved on."""
        if self.future is not None and self.future.done():
            run = self.future.result()
            new_signal = self.run is None or run.signal is not self.run.signal
            self.run, self.future, self.stage_running = run, None, ""
            if new_signal and run.signal is not None:
                self.fit_view()
            self.check_selection()
        if self.future is None and self.settings.first_change(self.submitted) is not None:
            start = self.settings.first_change(self.run.settings if self.run else None) or "final"
            self.submitted = self.settings.copy()
            self.future = self.executor.submit(compute, self.submitted, start, self.run, self.on_stage)
        if self.picker is not None and self.picker.ready():
            chosen = self.picker.result()
            self.picker = None
            if chosen and chosen[0].endswith(".npz"):
                self.settings.signal.scan_path = chosen[0]
                self.settings.signal.source = "recorded"
            elif chosen:
                self.settings.signal.path = chosen[0]
                self.settings.signal.source = "file"

    def on_stage(self, stage: str) -> None:
        self.stage_running = stage

    def wait(self) -> Run:
        """Block until the settings' run is in (for scripts and tests)."""
        while self.future is not None or self.settings.first_change(self.submitted) is not None:
            if self.future is not None:
                self.future.result()
            self.poll()
        return self.run

    def fit_view(self) -> None:
        sig = self.run.signal
        self.t0.value, self.t1.value = 0.0, sig.duration if sig.lines is not None else min(sig.duration, 4.0)

    def check_selection(self) -> None:
        run = self.run
        if run is None or not run.bands:
            return
        if run.coeff is not None:
            self.row = int(np.clip(self.row, 0, run.coeff.shape[0] - 1))
        if run.band(self.band_id) is None:
            kept = run.kept or run.bands
            self.band_id = kept[len(kept) // 2].id

    @property
    def ready(self) -> bool:
        return self.run is not None and self.run.signal is not None

    def has(self, stage: str) -> bool:
        """Whether the last run got through stage."""
        run = self.run
        if run is None:
            return False
        return stage in run.timing

    def band_order(self) -> dict:
        """Band id -> its color index (bands in frequency order)."""
        return {band.id: i for i, band in enumerate(self.run.bands)} if self.run else {}

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------

    def apply_scale(self) -> None:
        """Font, imgui and ImPlot sizes to self.scale. Called every frame; does work only when the scale moved."""
        if self.scale == self.scale_applied:
            return
        ratio = self.scale / self.scale_applied
        style = imgui.get_style()
        style.font_scale_main = self.scale
        style.scale_all_sizes(ratio)
        ps = implot.get_style()
        for name in ("plot_padding", "label_padding", "legend_padding", "legend_inner_padding", "legend_spacing",
                     "mouse_pos_padding", "annotation_padding", "major_tick_len", "minor_tick_len"):
            v = getattr(ps, name)
            setattr(ps, name, imgui.ImVec2(v.x * ratio, v.y * ratio))
        ps.plot_min_size = imgui.ImVec2(ps.plot_min_size.x * ratio, ps.plot_min_size.y * ratio)
        P.SCALE = self.scale
        self.scale_applied = self.scale

    def set_scale(self, scale: float) -> None:
        self.scale = config.clamp(scale)
        config.save({"ui_scale": self.scale})

    def scale_keys(self) -> None:
        """Ctrl + / Ctrl - / Ctrl 0, as in a browser."""
        if not imgui.get_io().key_ctrl:
            return
        if imgui.is_key_pressed(imgui.Key.equal) or imgui.is_key_pressed(imgui.Key.keypad_add):
            self.set_scale(self.scale * 1.1)
        elif imgui.is_key_pressed(imgui.Key.minus) or imgui.is_key_pressed(imgui.Key.keypad_subtract):
            self.set_scale(self.scale / 1.1)
        elif imgui.is_key_pressed(imgui.Key._0) or imgui.is_key_pressed(imgui.Key.keypad0):
            self.set_scale(config.detect_scale())

    def tighten_style(self) -> None:
        """Smaller ImPlot padding than the defaults, so stacked plots leave more room for data. Once, at scale 1."""
        ps = implot.get_style()
        ps.plot_padding = imgui.ImVec2(4, 4)
        ps.label_padding = imgui.ImVec2(3, 2)
        ps.legend_padding = imgui.ImVec2(4, 2)
        ps.legend_inner_padding = imgui.ImVec2(3, 1)
        imgui.get_style().item_spacing = imgui.ImVec2(imgui.get_style().item_spacing.x, 3)
        # black to green, like the green channel on the rig
        ramp = np.linspace(0, 1, 32, dtype=np.float32)
        cols = np.stack([0.15 * ramp**2, ramp, 0.25 * ramp**2, np.ones_like(ramp)], axis=1)
        self.fluo_map = implot.add_colormap("fluorescence", cols, False)
        self.style_set = True

    def gui(self) -> None:
        if not self.style_set:
            self.tighten_style()
        self.scale_keys()
        self.apply_scale()
        self.poll()
        em = imgui.get_font_size()
        # a third of the window, between 17 and 26 em: big text in a half-screen tile still leaves room for plots
        rail = min(max(0.2 * imgui.get_content_region_avail().x, 15 * em), WIDTH_RAIL_EM * em)
        if imgui.begin_child("rail", imgui.ImVec2(rail, 0), imgui.ChildFlags_.borders):
            self.draw_rail()
        imgui.end_child()
        imgui.same_line()
        if imgui.begin_child("page", imgui.ImVec2(0, 0)):
            self.draw_page()
        imgui.end_child()

    def draw_rail(self) -> None:
        imgui.text_colored(P.ACCENT, TITLE_WINDOW)
        imgui.same_line()
        imgui.text_disabled(f"v{__version__}")
        imgui.spacing()
        for key, label in PAGES:
            if imgui.selectable(label, self.page == key)[0]:
                self.page = key
        imgui.spacing()
        imgui.separator()
        if imgui.begin_child("controls", imgui.ImVec2(0, -imgui.get_frame_height_with_spacing() * 4.5)):
            # leave about 11 em for each widget's label
            em = imgui.get_font_size()
            imgui.push_item_width(max(6 * em, imgui.get_content_region_avail().x - 11 * em))
            getattr(self, f"controls_{self.page}")()
            imgui.pop_item_width()
        imgui.end_child()
        imgui.separator()
        if imgui.button("defaults"):
            self.settings.use_defaults()
        same_line_if_fits("lab settings")
        if imgui.button("lab settings"):
            self.settings.use_upstream()
        same_line_if_fits("copy code")
        if imgui.button("copy code"):
            imgui.set_clipboard_text(self.settings.python())
            self.copied = "copied"
        if self.copied:
            imgui.same_line()
            imgui.text_disabled(self.copied)
        imgui.set_next_item_width(8 * imgui.get_font_size())
        # applied on release: rescaling mid-drag would move the slider out from under the mouse
        shown = self.scale_edit if self.scale_edit is not None else self.scale
        changed, value = imgui.slider_float("text size", shown, config.SCALE_MIN, config.SCALE_MAX, "%.2fx")
        if changed:
            self.scale_edit = value
        if self.scale_edit is not None and not imgui.is_item_active():
            self.set_scale(self.scale_edit)
            self.scale_edit = None

    def draw_page(self) -> None:
        key = self.page
        self.draw_view_bar()
        imgui.push_style_color(imgui.Col_.text, P.WARN)
        imgui.text_wrapped(LOOK_FOR[key])
        imgui.pop_style_color()
        imgui.separator()
        status_h = imgui.get_frame_height_with_spacing()
        if imgui.begin_child("plots", imgui.ImVec2(0, -status_h)):
            if not self.ready:
                imgui.text_disabled("Computing..." if self.future else "No trace yet.")
            else:
                getattr(self, f"page_{key}")()
        imgui.end_child()
        self.draw_status()

    def draw_view_bar(self) -> None:
        if not self.ready:
            return
        sig = self.run.signal
        for i, (label, span) in enumerate((("all", sig.duration), ("4 s", 4.0), ("1 s", 1.0), ("200 ms", 0.2))):
            if i:
                imgui.same_line()
            if imgui.small_button(label):
                center = 0.5 * (self.t0.value + self.t1.value) if label != "all" else sig.duration / 2
                span = min(span, sig.duration)
                lo = float(np.clip(center - span / 2, 0, sig.duration - span))
                self.t0.value, self.t1.value = lo, lo + span

    def draw_status(self) -> None:
        if self.future is not None:
            imgui.text_colored(P.WARN, "...")
            imgui.same_line()
        run = self.run
        if run is not None and run.error:
            imgui.text_colored(P.ERROR, run.error)

    # ------------------------------------------------------------------
    # plot frames
    # ------------------------------------------------------------------

    def begin_time_plot(self, title: str, height: float, y_label: str = "", flags: int = 0, y_fit: bool = True,
                        x_axis: bool = True) -> bool:
        """A plot against time on the shared x range; x_axis False hides the time ticks (for plots stacked above
        another)."""
        if not P.begin_plot(title, imgui.ImVec2(-1, height), flags):
            return False
        y_flags = implot.AxisFlags_.auto_fit | implot.AxisFlags_.range_fit if y_fit else 0
        x_flags = 0 if x_axis else implot.AxisFlags_.no_tick_labels | implot.AxisFlags_.no_label
        implot.setup_axes("time (s)" if x_axis else "", y_label, x_flags, y_flags)
        implot.setup_axis_links(implot.ImAxis_.x1, self.t0, self.t1)
        return True

    def line(self, label: str, y: np.ndarray, color=None, weight: float = 1.0, flags: int = 0) -> None:
        P.line(label, y, self.run.signal.fs, self.t0.value, self.t1.value, color, weight, flags)

    def plot_height(self, share: float, rows: float = 1.0, minimum: float = 8.0) -> float:
        """share of the space left in the plots child, in pixels, at least minimum em."""
        avail = imgui.get_content_region_avail().y
        return max(minimum * imgui.get_font_size(), avail * share / rows)

    # ------------------------------------------------------------------
    # 1  signal
    # ------------------------------------------------------------------

    def controls_signal(self) -> None:
        sp = self.settings.signal
        heading("Source")
        for key, label, _ in SOURCES:
            if imgui.radio_button(label, sp.source == key):
                sp.source = key
        if sp.source == "recorded":
            self.controls_recorded()
            return
        heading("Shape")
        _, sp.fs = imgui.input_float("fs (Hz)", sp.fs, 100.0, 1000.0, "%.4g")
        if sp.source != "file":
            _, sp.duration_s = imgui.slider_float("duration (s)", sp.duration_s, 0.5, 120.0, "%.1f", imgui.SliderFlags_.logarithmic)
            if sp.source != "jedi":
                _, sp.noise_sd = imgui.slider_float("noise", sp.noise_sd, 0.0, 3.0, "%.2f")
            if imgui.small_button("new noise"):
                sp.seed += 1
        if sp.source == "file":
            _, sp.path = imgui.input_text("##path", sp.path)
            imgui.same_line()
            if imgui.button("Browse...") and self.picker is None:
                self.picker = pfd.open_file("Open a trace", sp.path or ".", ["traces", "*.npy *.csv *.txt *.tsv", "all", "*"])
            columns = self.run.signal.columns if self.ready and self.run.settings.signal.source == "file" else 1
            if columns > 1:
                _, value = imgui.slider_int("column", sp.column + 1, 1, columns)
                sp.column = value - 1
            _, sp.zscore = imgui.checkbox("z-score", sp.zscore)
        sp.fs = max(sp.fs, 1.0)

    def controls_recorded(self) -> None:
        sp = self.settings.signal
        heading("Scan")
        if self.scans is None:
            self.scans = find_scans()
            if not sp.scan_path and self.scans:
                sp.scan_path = self.scans[0]
        names = [Path(f).parent.parent.name + " / " + Path(f).stem for f in self.scans]
        current = self.scans.index(sp.scan_path) if sp.scan_path in self.scans else -1
        changed, i = imgui.combo("##scan", current, names)
        if changed:
            sp.scan_path = self.scans[i]
        if imgui.button("Browse...") and self.picker is None:
            self.picker = pfd.open_file("Open an exported line scan", sp.scan_path or ".", ["line scans", "*.npz"])
        if not sp.scan_path:
            return
        try:
            info = scan_info(sp.scan_path)
        except ValueError as e:
            imgui.text_colored(P.ERROR, str(e))
            return
        fs, total = info["fs"], info["frames"] / info["fs"]
        imgui.text_disabled(f"{info['unit']}  {fs:.0f} Hz  {total:.0f} s")
        heading("Lines in the domain")
        chosen = set(sp.rois)
        for k, (roi, width) in enumerate(sorted(info["widths"].items())):
            if k % 3:
                imgui.same_line()
            changed, on = imgui.checkbox(f"{roi} ({width} px)##roi{roi}", roi in chosen)
            if changed:
                chosen = chosen | {roi} if on else chosen - {roi}
        if chosen:
            sp.rois = tuple(sorted(chosen))
        heading("Window")
        sp.duration_s = float(min(sp.duration_s, total))
        _, sp.duration_s = imgui.slider_float("length (s)", sp.duration_s, 1.0, total, "%.1f", imgui.SliderFlags_.logarithmic)
        _, sp.start_s = imgui.slider_float("start (s)", sp.start_s, 0.0, max(total - sp.duration_s, 0.0), "%.1f")

    def page_signal(self) -> None:
        sig = self.run.signal
        if sig.lines is not None:
            self.draw_line_scan()
            return
        h = self.plot_height(0.62)
        if self.begin_time_plot("trace", h, "z"):
            self.line("raw", sig.x, P.RAW)
            if sig.truth is not None:
                self.line("ground truth", sig.truth, P.TRUTH, 1.5)
            for i, (name, times) in enumerate(sig.events_s.items()):
                P.vlines(name, times, P.band_color(i + 2, 0.45))
            implot.end_plot()
        run = self.run
        if run.psd and P.begin_plot("spectrum", imgui.ImVec2(-1, -1)):
            implot.setup_axes("frequency (Hz)", "power / Hz", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
            implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10)
            implot.plot_line("raw", run.psd["f"], run.psd["raw"], P.spec(P.RAW))
            if "truth" in run.psd:
                implot.plot_line("ground truth", run.psd["f"], run.psd["truth"], P.spec(P.TRUTH, 1.5))
            P.vlines("fastest possible", [sig.fs / 2], P.DIM)
            implot.end_plot()

    def pixel_image(self) -> tuple:
        """(rows x columns, t_start, t_end, lo, hi): the scan's pixels over the noiseless pixels, one blank row
        between, averaged to about 1200 columns of the view."""
        run = self.run
        fs, ls = run.signal.fs, run.signal.lines
        i0, i1 = P.window_indices(run.signal.n, fs, self.t0.value, self.t1.value)
        key = (id(run), i0, i1)
        if self.pixel_cache[0] == key:
            return self.pixel_cache[1]
        width = max(1, int(math.ceil((i1 - i0) / 1200)))
        whole = max((i1 - i0) // width * width, 1)

        def pooled(px):
            m = px[:, i0:i0 + whole].astype(np.float64)
            return m.reshape(m.shape[0], -1, width).mean(axis=2) if width > 1 else m

        noisy = pooled(ls.pixels)
        lo, hi = P.fit_scale(noisy, 1.0, 99.5)
        if ls.clean_pixels is None:
            m = noisy
        else:
            gap = np.full((1, noisy.shape[1]), np.nan)
            m = np.vstack([noisy, gap, pooled(ls.clean_pixels)])  # heatmap row 0 is drawn at the top
        value = (np.ascontiguousarray(np.nan_to_num(m, nan=lo)), i0 / fs, (i0 + whole) / fs, lo, hi)
        self.pixel_cache = (key, value)
        return value

    def draw_line_scan(self) -> None:
        sig = self.run.signal
        ls = sig.lines
        n_px = ls.pixels.shape[0]
        clean = ls.clean_pixels is not None
        name = "noisy" if clean else "recorded"
        rows = 2 * n_px + 1 if clean else n_px
        gap = imgui.get_style().item_spacing.y
        avail = imgui.get_content_region_avail().y - 3 * gap
        h_img, h = avail * 0.34, avail * 0.22
        m, ta, tb, lo, hi = self.pixel_image()
        implot.push_colormap(self.fluo_map)
        if self.begin_time_plot("##pixels", h_img, "", implot.Flags_.no_legend, y_fit=False, x_axis=False):
            implot.setup_axis_limits(implot.ImAxis_.y1, 0, rows, implot.Cond_.always)
            ticks = [rows - n_px / 2, n_px / 2] if clean else [n_px / 2]
            implot.setup_axis_ticks(implot.ImAxis_.y1, ticks, ["noisy", "noiseless"] if clean else [name], False)
            implot.plot_heatmap("##px", m, lo, hi, "", implot.Point(ta, 0), implot.Point(tb, rows))
            for k, (a, b) in enumerate(ls.edges):
                for top in (rows, n_px) if clean else (rows,):  # the noisy block, then the clean one
                    y0, y1 = top - b, top - a
                    implot.plot_shaded(f"##line{k}{top}", np.array([ta, ta + 0.006 * (tb - ta)]),
                                       np.full(2, float(y0)), np.full(2, float(y1)), P.spec(P.band_color(k), fill_alpha=1.0))
            implot.end_plot()
        implot.pop_colormap()
        if self.begin_time_plot("##F", h, "counts", x_axis=False):
            for k, mean in enumerate(ls.line_means):
                self.line(ls.names[k] if k < len(ls.names) else f"line {k + 1}", mean, P.band_color(k, 0.6))
            self.line("all pixels", ls.f, P.WHITE, 1.5)
            if clean:
                self.line("noiseless", ls.clean_f, P.TRUTH, 1.5)
            self.line(f"baseline ({ls.sigma_s[0]:.1f} s)", ls.f_baseline, P.OPEN, 2)
            implot.end_plot()
        if self.begin_time_plot("##dfof", h, "dF/F", x_axis=False):
            self.line(name, ls.dfof, P.RAW)
            if clean:
                self.line("noiseless", ls.clean_dfof, P.TRUTH, 1.5)
            self.line(f"slower baseline ({ls.sigma_s[1]:.1f} s)", ls.dfof_baseline, P.OPEN, 2)
            implot.end_plot()
        if self.begin_time_plot("##z", -1, "z (flipped)"):
            self.line(name, sig.x, P.ACCENT)
            if sig.truth is not None:
                self.line("noiseless", sig.truth, P.TRUTH, 1.5)
            implot.end_plot()

    # ------------------------------------------------------------------
    # 2  scalogram
    # ------------------------------------------------------------------

    def controls_scalogram(self) -> None:
        heading("Color")
        for i, label in enumerate(HEAT_MODES):
            if imgui.radio_button(label, self.heat_mode == i):
                self.heat_mode = i
        heading("Row")
        n = self.run.coeff.shape[0] if self.has("cwt") else int(self.settings.n_scales)
        _, self.row = imgui.slider_int("row", self.row, 0, n - 1)
        self.row = int(np.clip(self.row, 0, n - 1))
        if self.has("cwt"):
            imgui.text(f"{self.run.freqs[self.row]:.4g} Hz")

    def heat_matrix(self) -> tuple:
        """(values rows x columns, t_start, t_end, lo, hi) for the current view and color mode, cached."""
        run = self.run
        fs = run.signal.fs
        i0, i1 = P.window_indices(run.signal.n, fs, self.t0.value, self.t1.value)
        key = (id(run), self.heat_mode, i0, i1)
        if self.heat_cache[0] == key:
            return self.heat_cache[1]
        if self.heat_mode == 3:
            m = (run.coeff.real[:, i0:i1] / np.sqrt(run.scales)[:, None])
        elif self.heat_mode == 2:
            if not hasattr(run, "_row_mean"):
                run._row_mean = run.coeff.mean(axis=1, keepdims=True)
                run._row_sd = run.coeff.std(axis=1, keepdims=True)
            m = np.abs((run.coeff[:, i0:i1] - run._row_mean) / run._row_sd)
        else:
            m = run.amp[:, i0:i1].astype(np.float64)
            if self.heat_mode == 1:
                if not hasattr(run, "_row_median"):
                    run._row_median = np.median(run.amp, axis=1, keepdims=True) + 1e-12
                m = m / run._row_median
        width = max(1, int(math.ceil(m.shape[1] / 1200)))
        whole = m.shape[1] // width * width
        if width > 1:
            blocks = m[:, :whole].reshape(m.shape[0], -1, width)
            if self.heat_mode == 3:
                pick = np.abs(blocks).argmax(axis=2)[..., None]
                m = np.take_along_axis(blocks, pick, axis=2)[..., 0]
            else:
                m = blocks.max(axis=2)
        if self.heat_mode in (0, 1):
            m = np.log10(m + 1e-12)
        if self.heat_mode == 3:
            hi = float(np.percentile(np.abs(m), 99.5)) or 1e-9
            lo = -hi
        else:
            lo, hi = P.fit_scale(m, 2.0, 99.7)
        value = (np.ascontiguousarray(m, dtype=np.float64), i0 / fs, (i0 + max(whole, 1)) / fs, lo, hi)
        self.heat_cache = (key, value)
        return value

    def page_scalogram(self) -> None:
        if not self.has("cwt"):
            imgui.text_disabled("The transform has not run.")
            return
        run = self.run
        sig, freqs = run.signal, run.freqs
        m, ta, tb, lo, hi = self.heat_matrix()
        log_f = np.log10(freqs)
        step = (log_f[0] - log_f[-1]) / max(len(freqs) - 1, 1)
        top, bottom = log_f[0] + step / 2, log_f[-1] - step / 2
        em = imgui.get_font_size()
        h = self.plot_height(0.5)
        cmap = implot.Colormap_.rd_bu if self.heat_mode == 3 else implot.Colormap_.viridis
        implot.push_colormap(cmap)
        if self.begin_time_plot("##scalogram", h, "frequency (Hz)", implot.Flags_.no_legend, y_fit=False, x_axis=False):
            implot.setup_axis_limits(implot.ImAxis_.y1, bottom, top, implot.Cond_.always)
            P.hz_ticks(freqs.min(), freqs.max())
            implot.plot_heatmap("##W", m, lo, hi, "", implot.Point(ta, bottom), implot.Point(tb, top))
            # band edges, once the clustering has run on this transform
            if run.bands and run.bands[0].rows.size == len(freqs):
                for below, above in zip(run.bands, run.bands[1:]):
                    P.hline("##edge", math.log10(math.sqrt(below.hi_hz * above.lo_hz)), P.faded(P.WHITE, 0.55))
            if sig.fs / 2 < freqs.max():
                P.hline("##nyquist", math.log10(sig.fs / 2), P.WARN)
            changed, y, *_ = implot.drag_line_y(1, float(log_f[self.row]), P.WHITE, 1.5)
            if changed:
                self.row = int(np.argmin(np.abs(log_f - y)))
            if implot.is_plot_hovered():
                mouse = implot.get_plot_mouse_pos()
                k = int(np.argmin(np.abs(log_f - mouse.y)))
                imgui.set_tooltip(f"{freqs[k]:.4g} Hz (row {k}) at {mouse.x:.4f} s")
            implot.end_plot()
        imgui.same_line()
        implot.colormap_scale("##scale", lo, hi, imgui.ImVec2(4 * em, h))
        implot.pop_colormap()
        # the selected row over the input
        k = self.row
        caption("Below: the trace, the selected row (the part of the trace moving at that speed), and its envelope, "
                "the row's strength at each moment, traced over the row's peaks.")
        root = math.sqrt(run.scales[k])
        if self.begin_time_plot(f"one row: {freqs[k]:.4g} Hz (drag the white line to pick)", -1):
            self.line("trace", sig.x, P.RAW)
            self.line("this row", run.coeff[k].real / root, P.ACCENT, 1.5)
            self.line("envelope", run.amp[k], P.OPEN, 1.0)
            implot.end_plot()

    # ------------------------------------------------------------------
    # 3  bands
    # ------------------------------------------------------------------

    def controls_bands(self) -> None:
        c = self.settings.clust
        n_rows = int(self.settings.n_scales)
        heading("Compare rows")
        _, c.standardize = imgui.checkbox("standardize", c.standardize)
        _, c.n_components = imgui.slider_int("PCA components", c.n_components, 2, min(60, n_rows))
        _, c.n_comp_clu = imgui.slider_int("PCs used", c.n_comp_clu, 1, c.n_components)
        heading("Tree")
        methods = ["ward", "average", "complete", "single"]
        changed, i = imgui.combo("linkage", methods.index(c.method_linkage) if c.method_linkage in methods else 0, methods)
        if changed:
            c.method_linkage = methods[i]
        _, c.n_subclusters = imgui.slider_int("bands", c.n_subclusters, 2, min(30, n_rows - 1))
        c.n_comp_clu = min(c.n_comp_clu, c.n_components)
        if self.has("reduce"):
            heading("Bands")
            order = self.band_order()
            if imgui.begin_table("bands", 4, imgui.TableFlags_.row_bg | imgui.TableFlags_.sizing_fixed_fit):
                for col in ("", "Hz", "rows", "windows"):
                    imgui.table_setup_column(col)
                imgui.table_headers_row()
                for band in self.run.bands:
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.color_button(f"##c{band.id}", P.band_color(order[band.id]), 0, imgui.ImVec2(0.8 * imgui.get_font_size(), 0.8 * imgui.get_font_size()))
                    imgui.table_next_column()
                    imgui.text(band.span)
                    imgui.table_next_column()
                    imgui.text(str(int(band.rows.sum())))
                    imgui.table_next_column()
                    if band.kept:
                        imgui.text(str(len(band.windows)))
                    else:
                        imgui.text_colored(P.WARN, "0")
                imgui.end_table()

    def page_bands(self) -> None:
        if not self.has("cluster"):
            imgui.text_disabled("The clustering has not run.")
            return
        run = self.run
        order = self.band_order()
        freqs = run.freqs
        caption("Each colored block is a band, covering the rows (dots) it groups; faded blocks are dropped. The white line "
                "is the gain of the kept bands summed: 1 passes a frequency unchanged, 0 removes it.")
        avail = imgui.get_content_region_avail()
        h1 = max(8 * imgui.get_font_size(), avail.y * 0.28)
        if P.begin_plot("which speeds each band holds, and how much of each speed comes back (white)", imgui.ImVec2(-1, h1), implot.Flags_.no_legend):
            implot.setup_axes("frequency (Hz)", "", implot.AxisFlags_.auto_fit, implot.AxisFlags_.no_tick_labels | implot.AxisFlags_.lock)
            implot.setup_axis_limits(implot.ImAxis_.y1, 0, 1.05, implot.Cond_.always)
            implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
            implot.setup_axis(implot.ImAxis_.y2, "gain", implot.AxisFlags_.auto_fit | implot.AxisFlags_.opposite)
            ratio = math.sqrt(freqs[0] / freqs[1]) if len(freqs) > 1 else 1.0
            for band in run.bands:
                f = freqs[band.rows]
                xs = np.array([f.min() / ratio, f.max() * ratio])
                alpha = 0.45 if band.kept else 0.12
                implot.plot_shaded(f"{band.span}{'' if band.kept else ' (dropped)'}", xs, np.ones(2), 0.0, P.spec(P.band_color(order[band.id]), fill_alpha=alpha))
                c = P.band_color(order[band.id])
                implot.plot_scatter(f"##rows{band.id}", f, np.full(f.size, 0.5), P.marker(c, 3.5, P.faded(P.TEXT, 0.8)))
            P.vlines("fastest possible", [run.signal.fs / 2], P.WARN)
            if self.has("reduce"):
                fx = np.logspace(np.log10(freqs.min()), np.log10(min(freqs.max(), run.signal.fs / 2)), 500)
                implot.set_axes(implot.ImAxis_.x1, implot.ImAxis_.y2)
                implot.plot_line("gain", fx, band_sum_gain(run, fx), P.spec(P.WHITE, 2))
            implot.end_plot()
        spectrum = run.psd if self.has("final") else None
        if spectrum:
            caption("Power at each frequency in the trace and the denoised trace, with each band shaded behind the "
                    "frequencies it covers.")
        h2 = imgui.get_content_region_avail().y * 0.55
        if spectrum and P.begin_plot("the trace's spectrum, with the bands behind it", imgui.ImVec2(-1, h2)):
            implot.setup_axes("frequency (Hz)", "power / Hz", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
            implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10)
            f = spectrum["f"]
            top = float(max(spectrum["raw"].max(), spectrum.get("truth", spectrum["raw"]).max())) * 2
            low = float(min(spectrum["raw"].min(), spectrum["denoised"].min() + 1e-30))
            ratio = math.sqrt(freqs[0] / freqs[1]) if len(freqs) > 1 else 1.0
            for band in run.bands:
                fb = freqs[band.rows]
                xs = np.array([fb.min() / ratio, min(fb.max() * ratio, f.max())])
                implot.plot_shaded(f"##bg{band.id}", xs, np.full(2, top), low,
                                   P.spec(P.band_color(order[band.id]), fill_alpha=0.18 if band.kept else 0.05))
            implot.plot_line("trace", f, spectrum["raw"], P.spec(P.RAW, 1.5))
            if "truth" in spectrum:
                implot.plot_line("ground truth", f, spectrum["truth"], P.spec(P.TRUTH, 1.5))
            implot.plot_line("denoised", f, spectrum["denoised"] + 1e-30, P.spec(P.ACCENT, 1.5))
            implot.end_plot()
        if self.has("reduce"):
            caption("How much of the recording each band is kept for; the number on a bar is its event windows.")
        if self.has("reduce") and P.begin_plot("share of the time each band is kept", imgui.ImVec2(-1, -1), implot.Flags_.no_legend):
            n = run.signal.n
            bands = run.bands
            implot.setup_axes("", "kept (%)", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            implot.setup_axis_ticks(implot.ImAxis_.x1, [float(i) for i in range(len(bands))], [b.span for b in bands], False)
            for i, band in enumerate(bands):
                if band.label is not None:
                    kept = float(np.mean(band.label >= 1.0)) * 100
                else:
                    kept = float(np.sum(band.windows[:, 1] - band.windows[:, 0])) / n * 100 if band.kept else 0.0
                implot.plot_bars(f"##k{band.id}", np.array([float(i)]), np.array([kept]), 0.7,
                                 P.spec(P.band_color(order[band.id]), fill_alpha=1.0 if band.kept else 0.3))
                implot.plot_text(f"{len(band.windows) if band.kept else 0}", float(i), kept, imgui.ImVec2(0, -0.7 * imgui.get_font_size()))
            implot.end_plot()

    # ------------------------------------------------------------------
    # 4  events and masks
    # ------------------------------------------------------------------

    def controls_events(self) -> None:
        r, t = self.settings.reducer, self.settings.thres
        if self.has("reduce"):
            heading("Band")
            order = self.band_order()
            for band in reversed(self.run.bands):
                imgui.color_button(f"##b{band.id}", P.band_color(order[band.id]), 0, imgui.ImVec2(0.8 * imgui.get_font_size(), 0.8 * imgui.get_font_size()))
                imgui.same_line()
                label = band.span if band.kept else f"{band.span}  (dropped)"
                if imgui.selectable(f"{label}##{band.id}", self.band_id == band.id)[0]:
                    self.band_id = band.id
        heading("Threshold")
        _, r.slow_upthres = imgui.slider_float("SD, < 30 Hz", r.slow_upthres, 0.5, 6.0, "%.2f")
        _, r.fast_upthres = imgui.slider_float("SD, >= 30 Hz", r.fast_upthres, 0.5, 6.0, "%.2f")
        changed, windows = imgui.input_int3("smoothing", list(r.smooth_windows))
        if changed:
            r.smooth_windows = tuple(max(3, int(w)) for w in windows)
        _, r.complex_bands = imgui.checkbox("complex above 75 Hz", r.complex_bands)
        heading("Turn down")
        kinds = ["soft", "hard"]
        changed, i = imgui.combo("type", kinds.index(t.thres_type), kinds)
        if changed:
            t.thres_type = kinds[i]
        if t.thres_type == "soft":
            changed, levels = imgui.slider_float4("to", list(t.soft_levels), 0.0, 1.0, "%.2f")
            if changed:
                t.soft_levels = tuple(levels)
            changed, sigmas = imgui.input_float2("blur", list(t.soft_sigma), "%.1f")
            if changed:
                t.soft_sigma = tuple(max(0.1, v) for v in sigmas)
        else:
            _, t.hard_floor = imgui.slider_float("to", t.hard_floor, 0.0, 1.0, "%.3f")
            changed, sigmas = imgui.input_float2("blur", list(t.hard_sigma), "%.1f")
            if changed:
                t.hard_sigma = tuple(max(0.1, v) for v in sigmas)

    def page_events(self) -> None:
        if not self.has("reduce"):
            imgui.text_disabled("Bands not computed yet.")
            return
        run = self.run
        band = run.band(self.band_id)
        if band is None:
            return
        fs = run.signal.fs
        color = P.band_color(self.band_order()[band.id])
        gap = imgui.get_style().item_spacing.y
        h = max(6 * imgui.get_font_size(), (imgui.get_content_region_avail().y - 4 * gap) / 5)
        if self.begin_time_plot(f"{band.span}: its {int(band.rows.sum())} rows, and their average", h, x_axis=False):
            for k in np.flatnonzero(band.rows):
                self.line("##row", run.coeff[k].real, P.faded(P.TEXT, 0.35))
            self.line("average", band.average.real, color, 2)
            implot.end_plot()
        if self.begin_time_plot("where it sticks out", h, x_axis=False):
            if band.kept and band.extra:
                wide = np.clip(band.windows + np.array([-band.extra, band.extra]), 0, run.signal.n)
                P.shade_intervals(wide, fs, self.t0.value, self.t1.value, P.faded(color, 0.10))
            P.shade_intervals(band.windows, fs, self.t0.value, self.t1.value, P.faded(color, 0.22))
            if band.smooth_window:
                self.line("average", band.average.real, P.faded(P.TEXT, 0.35))
            self.line("smoothed", band.smoothed.real, color, 1.5)
            P.hline("start", band.up, P.OPEN, 1.5)
            P.hline("end", band.down, P.CLOSE, 1.0)
            implot.end_plot()
        if not band.kept:
            imgui.text_colored(P.WARN, "never sticks out: dropped")
        elif band.mask is not None:
            if self.begin_time_plot("keep (1) or turn down", h, y_fit=False, x_axis=False):
                implot.setup_axis_limits(implot.ImAxis_.y1, -0.05, 1.08, implot.Cond_.always)
                self.line("unblurred", band.label, P.faded(P.TEXT, 0.5))
                self.line("mask", band.mask, color, 2)
                implot.end_plot()
            if self.begin_time_plot("band x mask", h, x_axis=False):
                self.line("before", band.scaled.real, P.faded(P.TEXT, 0.35))
                self.line("after", band.masked.real, color, 1.5)
                implot.end_plot()
        self.draw_mask_image()

    def draw_mask_image(self) -> None:
        run = self.run
        kept = [band for band in run.bands if band.kept and band.mask is not None][::-1]
        if not kept:
            return
        fs = run.signal.fs
        i0, i1 = P.window_indices(run.signal.n, fs, self.t0.value, self.t1.value)
        m = np.stack([band.mask[i0:i1] for band in kept])
        width = max(1, int(math.ceil(m.shape[1] / 1200)))
        whole = m.shape[1] // width * width
        if width > 1:
            m = m[:, :whole].reshape(m.shape[0], -1, width).mean(axis=2)
        nb = len(kept)
        implot.push_colormap(implot.Colormap_.viridis)
        em = imgui.get_font_size()
        h = max(imgui.get_content_region_avail().y, (1.3 * nb + 5) * em)
        if self.begin_time_plot("every band's mask", h, flags=implot.Flags_.no_legend, y_fit=False):
            implot.setup_axis_limits(implot.ImAxis_.y1, 0, nb, implot.Cond_.always)
            implot.setup_axis_ticks(implot.ImAxis_.y1, [nb - k - 0.5 for k in range(nb)], [b.span for b in kept], False)
            implot.plot_heatmap("##masks", np.ascontiguousarray(m, dtype=np.float64), 0.0, 1.0, "",
                                implot.Point(i0 / fs, 0), implot.Point((i0 + max(whole, 1)) / fs, nb))
            implot.end_plot()
        implot.pop_colormap()

    # ------------------------------------------------------------------
    # 5  reconstruction
    # ------------------------------------------------------------------

    def controls_final(self) -> None:
        s = self.settings
        heading("Slow part")
        _, s.add_baseline = imgui.checkbox("add", s.add_baseline)
        _, s.lp_cutoff = imgui.slider_float("below (Hz)", s.lp_cutoff, 0.05, 20.0, "%.3g", imgui.SliderFlags_.logarithmic)
        if self.has("mask"):
            heading("Bands")
            order = self.band_order()
            excluded = set(s.excluded)
            for band in reversed(self.run.bands):
                imgui.color_button(f"##f{band.id}", P.band_color(order[band.id]), 0, imgui.ImVec2(0.8 * imgui.get_font_size(), 0.8 * imgui.get_font_size()))
                imgui.same_line()
                if not band.kept:
                    imgui.text_disabled(band.span)
                    continue
                changed, on = imgui.checkbox(f"{band.span}##inc{band.id}", band.id not in excluded)
                if changed:
                    excluded = excluded - {band.id} if on else excluded | {band.id}
            s.excluded = tuple(sorted(excluded))
        if self.has("final"):
            numbers = metrics(self.run)
            if "corr denoised vs ground truth" in numbers:
                heading("Match to ground truth")
                imgui.text(f"before {numbers['corr raw vs ground truth']:.2f}   after {numbers['corr denoised vs ground truth']:.2f}")

    def page_final(self) -> None:
        if not self.has("final"):
            imgui.text_disabled("The reconstruction has not run.")
            return
        run = self.run
        sig = run.signal
        order = self.band_order()
        excluded = set(run.settings.excluded)
        kept = [band for band in run.bands if band.kept and band.masked is not None]
        h = self.plot_height(0.33)
        spread = max((np.percentile(np.abs(b.masked.real), 99.5) for b in kept), default=1.0) * 2.2 or 1.0
        if self.begin_time_plot("each band, slowest at the bottom", h, flags=implot.Flags_.no_legend, y_fit=False, x_axis=False):
            implot.setup_axis(implot.ImAxis_.y1, None, implot.AxisFlags_.no_tick_labels)
            implot.setup_axis_limits(implot.ImAxis_.y1, -0.7 * spread, (len(kept) - 0.2) * spread, implot.Cond_.always)
            for k, band in enumerate(kept):
                alpha = 0.25 if band.id in excluded else 1.0
                self.line(band.span, band.masked.real + k * spread, P.band_color(order[band.id], alpha), 1.2)
                implot.plot_text(band.span, self.t0.value, k * spread, imgui.ImVec2(3 * imgui.get_font_size(), -0.6 * imgui.get_font_size()))
            implot.end_plot()
        if self.begin_time_plot("bands added together", h):
            self.line("raw", sig.x, P.RAW)
            if sig.truth is not None:
                self.line("ground truth", sig.truth, P.TRUTH, 1.5)
            if run.settings.add_baseline:
                self.line("slow part", run.baseline, P.CLOSE, 1.0)
            self.line("denoised", run.denoised, P.ACCENT, 1.8)
            implot.end_plot()
        avail = imgui.get_content_region_avail()
        half = (avail.x - imgui.get_style().item_spacing.x) / 2
        if P.begin_plot("removed", imgui.ImVec2(half, -1)):
            implot.setup_axes("time (s)", "", 0, implot.AxisFlags_.auto_fit | implot.AxisFlags_.range_fit)
            implot.setup_axis_links(implot.ImAxis_.x1, self.t0, self.t1)
            self.line("removed", sig.x - run.denoised, P.faded(P.ERROR, 0.8))
            implot.end_plot()
        imgui.same_line()
        if run.psd and P.begin_plot("spectrum", imgui.ImVec2(-1, -1)):
            implot.setup_axes("frequency (Hz)", "power / Hz", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
            implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10)
            f = run.psd["f"]
            implot.plot_line("raw", f, run.psd["raw"], P.spec(P.RAW))
            if "truth" in run.psd:
                implot.plot_line("ground truth", f, run.psd["truth"], P.spec(P.TRUTH, 1.5))
            implot.plot_line("denoised", f, run.psd["denoised"] + 1e-30, P.spec(P.ACCENT, 1.8))
            implot.end_plot()


def runner_params(app: App) -> hello_imgui.RunnerParams:
    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = TITLE_WINDOW
    params.app_window_params.window_geometry.size = tuple(int(v * min(app.scale, 1.5)) for v in SIZE_WINDOW)
    params.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.provide_full_screen_window
    params.imgui_window_params.tweaked_theme.theme = hello_imgui.ImGuiTheme_.darcula_darker
    params.fps_idling.fps_idle = 20
    params.ini_filename = "imgui_vnoiser.ini"
    params.callbacks.show_gui = app.gui
    return params


def run() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="imgui-vnoiser", description="See what vnoiser's wavelet denoiser does to a trace.")
    parser.add_argument("--scale", type=float, default=0.0, help="UI scale, e.g. 2 on a 2x HiDPI panel (default: saved, then detected)")
    args = parser.parse_args()
    app = App(scale=config.ui_scale(args.scale))
    immapp.run(runner_params(app), immapp.AddOnsParams(with_implot=True))
    app.executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    run()
