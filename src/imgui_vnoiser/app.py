"""
The window `imgui-vnoiser` opens: a rail of pages on the left, one per stage of vnoiser.Denoiser.run, with that
stage's parameters under the rail and its plots on the right. Every time plot shares one time window, so zooming
one zooms them all. The pipeline runs on a worker thread and reruns only the stages a change affects.
"""

import math
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp, implot
from imgui_bundle import portable_file_dialogs as pfd

from imgui_vnoiser import __version__, config, morlet
from imgui_vnoiser import plotting as P
from imgui_vnoiser.pipeline import STAGES, Run, Settings, band_sum_gain, compute, fmt_hz, metrics
from imgui_vnoiser.signals import SOURCES

TITLE_WINDOW = "vnoiser lab"
SIZE_WINDOW = (1560, 980)
WIDTH_RAIL_EM = 26

# the rail: a page key, its name, and the line under the name on its page
PAGES = (
    ("signal", "1  Signal", "input trace: synthetic (with ground truth), a test signal, or a file"),
    ("wavelet", "2  Morlet wavelet", "the filter every row of the transform applies, and how its scales tile frequency"),
    ("scalogram", "3  Scalogram", "the trace after the wavelet transform: one row per scale, one column per sample"),
    ("bands", "4  Frequency bands", "rows that rise and fall together are grouped: PCA, then Ward clustering"),
    ("events", "5  Events and masks", "each band's trace, the levels that open and close its windows, and its mask"),
    ("final", "6  Reconstruction", "masked bands summed, the 1 Hz baseline added back, and what was removed"),
)

# what each page's "How it works" header says; numbers are the defaults, the controls show the current values
ABOUT = {
    "signal": (
        "vnoiser expects a z-scored dF/F trace with depolarization positive (JEDI fluorescence falls on "
        "depolarization, so the pipeline flips its sign).",
        "Synthetic sources have a ground truth (green): the trace before noise was added. For the JEDI3sub "
        "simulation it is the indicator response without line-scan noise or AP artifacts. Reconstruction compares "
        "the denoised trace against it.",
        "Tones, the chirp, the impulse and the step are for seeing the transform itself: an impulse shows each "
        "row's wavelet, a chirp shows the rows lighting up one after another.",
    ),
    "wavelet": (
        "pywt's complex Morlet cmorB-C is a sine at C cycles per sample under a Gaussian envelope exp(-t^2/B). "
        "Stretching it by a scale s centers it on C fs / s Hz; vnoiser uses 100 scales log-spaced from 1 to 1000.",
        "B trades time for frequency: a wider envelope (larger B) holds more cycles and passes a narrower band, "
        "but smears an event over a longer time. vnoiser's cmor0.5-1.0 holds about one cycle: events stay sharp, "
        "bands overlap heavily.",
        "Every row has the same shape on a log-frequency axis (constant Q): fast rows are short and wide in "
        "frequency, slow rows long and narrow.",
        "Rows centered above fs / 2 (scale below 2 C) have their center frequency above Nyquist; they only pick up "
        "the lower tail of their band.",
    ),
    "scalogram": (
        "Each row is the trace filtered by one scaled wavelet. Its magnitude says how much of that frequency is "
        "present around each moment; its real part is the band-passed trace itself.",
        "pywt's coefficients grow as sqrt(scale); vnoiser divides by it later, so the plots here do too.",
        "A spike shows as a vertical streak (short in time, every fast row at once), a slow bump as a blob in the "
        "low rows. Row-normalized color shows each row against its own noise level.",
        "Standardized |W| is exactly what the clustering sees: each row minus its mean, over its SD, magnitude.",
    ),
    "bands": (
        "FrequencyClusterer standardizes each row over time, takes its magnitude and runs PCA with the rows as "
        "observations (30 components). Rows whose envelopes rise and fall together land close in PC space.",
        "Ward clustering on the first 10 PC scores builds the tree; cutting it at 10 clusters gives the bands "
        "(sublabels). The 5-cluster cut is computed but not used.",
        "Neighboring rows overlap in frequency, so bands are usually contiguous ranges, but rows near and above "
        "fs / 2 can interleave between bands. Band edges depend on the trace.",
        "The gain curve is what summing the bands back does to a sine with every mask at 1: each band averages "
        "its rows, so the sum passes less than the input (savgol smoothing lowers it further below 75 Hz).",
    ),
    "events": (
        "WaveletReducer averages a band's rows into one trace (real part), smooths it with a running line fit "
        "(savgol, 41 / 21 / 11 samples at 1075 Hz for bands topping out by 15 / 50 / 75 Hz) and rescales it to "
        "its old peak.",
        "Below 30 Hz a window opens when the trace rises through 2 SD and closes when it falls through 1 SD; "
        "faster bands open at 2.5 SD and close at 0. A band with no window is dropped from the sum.",
        "AdaptiveThreshold widens every window by the band's median window length times 2 (top <= 20 Hz), "
        "1.5 (<= 50 Hz) or 0.5, sets the mask to 1 inside and to a floor outside (soft: 0.7 / 0.5 / 0.2 / 0.1 "
        "by top frequency <5 / <30 / <80 / above; hard: 0.05), then blurs it with a Gaussian.",
        "The band trace is divided by sqrt(its mean scale) and multiplied by its mask: that is what it adds.",
    ),
    "final": (
        "The masked bands are summed (rescaled_signal), and a zero-phase FIR low-pass of the input (1 Hz, 2 s "
        "Hamming) is added back as the baseline: denoised = rescaled_signal + baseline.",
        "The archive's denoised_trace_scans.pkl holds the sum without the baseline; untick Add baseline to see it.",
        "Untick a band to remove it from the sum. The spectrum shows what was removed: the masked-down part of "
        "each band between events, and any band that opened no window.",
    ),
}

HEAT_MODES = (
    ("|W| / sqrt(scale), log", "how much of each frequency there is around each moment"),
    ("|W| / row median, log", "each row against its own typical level: events stand out at every frequency"),
    ("standardized |W|", "each row minus its mean, over its SD, magnitude: what PCA and the clustering see"),
    ("Re W / sqrt(scale)", "the band-passed trace itself, signed; this is what gets summed back"),
)


def hint(text: str) -> None:
    """A dim (?) after the previous item that shows text when hovered."""
    imgui.same_line()
    imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 30)
        imgui.text_unformatted(text)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()


def same_line_if_fits(label: str) -> None:
    """Stay on this line when a button labeled label still fits on it."""
    room = imgui.get_content_region_avail().x - imgui.get_item_rect_max().x + imgui.get_cursor_screen_pos().x
    need = imgui.calc_text_size(label).x + 2 * imgui.get_style().frame_padding.x + imgui.get_style().item_spacing.x
    if room > need:
        imgui.same_line()


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
        self.dendro_cache = (None, None)
        self.band_id: Optional[int] = None
        self.pc = [0, 1]
        self.preset = 0
        self.picker = None
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
            if chosen:
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
        self.t0.value, self.t1.value = 0.0, min(sig.duration, 4.0)

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

    def gui(self) -> None:
        self.scale_keys()
        self.apply_scale()
        self.poll()
        em = imgui.get_font_size()
        # a third of the window, between 17 and 26 em: big text in a half-screen tile still leaves room for plots
        rail = min(max(0.3 * imgui.get_content_region_avail().x, 17 * em), WIDTH_RAIL_EM * em)
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
        for key, label, _ in PAGES:
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
        if imgui.button("Denoiser defaults"):
            self.settings.use_defaults()
        hint("Every parameter back to vnoiser.Denoiser's constructor defaults; the signal stays.")
        same_line_if_fits("Denoiser.upstream (?)")
        if imgui.button("Denoiser.upstream"):
            self.settings.use_upstream()
        hint("The spatial JEDI archive's settings: complex band traces above 75 Hz, soft floor 0.01 above 80 Hz, an even FIR tap count.")
        if imgui.button("Copy as Python"):
            imgui.set_clipboard_text(self.settings.python())
            self.copied = "Denoiser call copied"
        hint("A vnoiser.Denoiser(...) call with these settings, to run the same thing in a notebook.")
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
        hint("Ctrl + and Ctrl - change it too, Ctrl 0 goes back to the desktop's scale. Saved in ~/.imgui_vnoiser/settings.json.")

    def draw_page(self) -> None:
        key, label, line = next(p for p in PAGES if p[0] == self.page)
        imgui.text(label[3:])
        imgui.same_line()
        imgui.text_disabled(f"  {line}")
        if imgui.collapsing_header("How it works"):
            for paragraph in ABOUT[key]:
                imgui.bullet()
                imgui.text_wrapped(paragraph)
        self.draw_view_bar()
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
        if not self.ready or self.page == "wavelet":
            return
        sig = self.run.signal
        imgui.text_disabled(f"View {self.t0.value:.3f} to {self.t1.value:.3f} s of {sig.duration:.2f} s")
        hint("Every time plot shares this window. Drag a plot to pan, scroll to zoom, double-click to fit it.")
        for label, span in (("Whole trace", sig.duration), ("4 s", 4.0), ("1 s", 1.0), ("200 ms", 0.2)):
            imgui.same_line()
            if imgui.small_button(label):
                center = 0.5 * (self.t0.value + self.t1.value) if label != "Whole trace" else sig.duration / 2
                span = min(span, sig.duration)
                lo = float(np.clip(center - span / 2, 0, sig.duration - span))
                self.t0.value, self.t1.value = lo, lo + span

    def draw_status(self) -> None:
        if self.future is not None:
            imgui.text_colored(P.WARN, f"Computing {self.stage_running or '...'}")
            imgui.same_line()
        run = self.run
        if run is None:
            return
        if run.error:
            imgui.text_colored(P.ERROR, run.error)
            return
        sig = run.signal
        times = "  ".join(f"{k} {v * 1000:.0f} ms" for k, v in run.timing.items())
        imgui.text_disabled(f"{sig.name}: {sig.note}.   {times}")

    # ------------------------------------------------------------------
    # plot frames
    # ------------------------------------------------------------------

    def begin_time_plot(self, title: str, height: float, y_label: str = "", flags: int = 0, y_fit: bool = True) -> bool:
        if not P.begin_plot(title, imgui.ImVec2(-1, height), flags):
            return False
        y_flags = implot.AxisFlags_.auto_fit | implot.AxisFlags_.range_fit if y_fit else 0
        implot.setup_axes("time (s)", y_label, 0, y_flags)
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
        for key, label, tip in SOURCES:
            if imgui.radio_button(label, sp.source == key):
                sp.source = key
            hint(tip)
        heading("Shape")
        _, sp.fs = imgui.input_float("fs (Hz)", sp.fs, 100.0, 1000.0, "%.4g")
        hint("The lab's line scans run at 1075.2688 Hz. Row frequency is C fs / scale, so the same scales cover different Hz at a different fs.")
        if sp.source != "file":
            _, sp.duration_s = imgui.slider_float("duration (s)", sp.duration_s, 0.5, 120.0, "%.1f", imgui.SliderFlags_.logarithmic)
            if sp.source != "jedi":
                _, sp.noise_sd = imgui.slider_float("noise SD", sp.noise_sd, 0.0, 2.0, "%.2f")
            imgui.text_disabled(f"seed {sp.seed}")
            imgui.same_line()
            if imgui.small_button("New noise draw"):
                sp.seed += 1
        if sp.source == "tones":
            _, sp.tones_hz = imgui.input_text("tones (Hz)", sp.tones_hz)
            hint("Comma separated. Each tone is on for its own stretch, overlapping its neighbors by half.")
        if sp.source == "chirp":
            _, sp.chirp_lo_hz = imgui.input_float("from (Hz)", sp.chirp_lo_hz, 0, 0, "%.3g")
            _, sp.chirp_hi_hz = imgui.input_float("to (Hz)", sp.chirp_hi_hz, 0, 0, "%.3g")
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
            hint("Subtract the mean and divide by the SD. Band thresholds are in SDs of each band, so this mostly changes the y units.")
        sp.fs = max(sp.fs, 1.0)

    def page_signal(self) -> None:
        sig = self.run.signal
        h = self.plot_height(0.62)
        if self.begin_time_plot("Input trace", h, "z"):
            self.line("raw", sig.x, P.RAW)
            if sig.truth is not None:
                self.line("ground truth", sig.truth, P.TRUTH, 1.5)
            for i, (name, times) in enumerate(sig.events_s.items()):
                P.vlines(name, times, P.band_color(i + 2, 0.45))
            implot.end_plot()
        run = self.run
        if run.psd and P.begin_plot("Power spectrum (Welch)", imgui.ImVec2(-1, -1)):
            implot.setup_axes("frequency (Hz)", "power / Hz", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
            implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10)
            implot.plot_line("raw", run.psd["f"], run.psd["raw"], P.spec(P.RAW))
            if "truth" in run.psd:
                implot.plot_line("ground truth", run.psd["f"], run.psd["truth"], P.spec(P.TRUTH, 1.5))
            P.vlines("fs / 2", [sig.fs / 2], P.DIM)
            implot.end_plot()

    # ------------------------------------------------------------------
    # 2  wavelet
    # ------------------------------------------------------------------

    def controls_wavelet(self) -> None:
        s = self.settings
        heading("Wavelet")
        for i, (label, b, c, tip) in enumerate(morlet.PRESETS):
            if imgui.radio_button(label, math.isclose(s.b, b) and math.isclose(s.c, c)):
                s.b, s.c = b, c
            hint(tip)
        _, s.b = imgui.slider_float("B (bandwidth)", s.b, 0.1, 20.0, "%.3g", imgui.SliderFlags_.logarithmic)
        hint("Envelope exp(-t^2 / B): larger B, longer wavelet, narrower band.")
        _, s.c = imgui.slider_float("C (center)", s.c, 0.2, 4.0, "%.3g")
        hint("Cycles per sample at scale 1. Row frequency = C fs / scale.")
        imgui.text_disabled(f"pywt name: {morlet.name(s.b, s.c)}")
        heading("Scales")
        _, s.scale_min = imgui.slider_float("smallest", s.scale_min, 0.5, 50.0, "%.3g", imgui.SliderFlags_.logarithmic)
        _, s.scale_max = imgui.slider_float("largest", s.scale_max, 10.0, 20000.0, "%.4g", imgui.SliderFlags_.logarithmic)
        _, s.n_scales = imgui.slider_int("rows", s.n_scales, 12, 200)
        s.scale_max = max(s.scale_max, s.scale_min * 1.5)
        fs = s.signal.fs
        imgui.text_disabled(f"rows span {fmt_hz(s.c * fs / s.scale_max)} to {fmt_hz(s.c * fs / s.scale_min)} Hz at fs {fs:g} Hz")
        heading("Inspect")
        _, self.row = imgui.slider_int("row", self.row, 0, int(s.n_scales) - 1)
        self.row = int(np.clip(self.row, 0, int(s.n_scales) - 1))
        scale = float(s.scales[self.row])
        w = morlet.widths(s.b, s.c, scale, fs)
        for name, value in (
            ("scale", f"{scale:.3g}"),
            ("center", f"{w.freq_hz:.4g} Hz"),
            ("envelope SD", f"{w.sigma_t_s * 1000:.3g} ms"),
            ("half-power band", f"{w.half_power_lo_hz:.3g} to {w.half_power_hi_hz:.3g} Hz"),
            ("Q (f / bandwidth)", f"{w.q:.2f}  (every row)"),
            ("cycles within +-1 SD", f"{w.cycles:.2f}"),
        ):
            imgui.text_disabled(f"{name:<22}")
            imgui.same_line()
            imgui.text(value)
        if w.freq_hz > fs / 2:
            imgui.text_colored(P.WARN, "Center frequency is above fs / 2.")
        above = int(np.sum(s.c * fs / s.scales > fs / 2))
        if above:
            imgui.text_wrapped(f"{above} of {int(s.n_scales)} rows sit above fs / 2 = {fs / 2:g} Hz.")

    def page_wavelet(self) -> None:
        s = self.settings
        fs, scales = s.signal.fs, s.scales
        scale = float(scales[self.row])
        avail = imgui.get_content_region_avail()
        half = (avail.x - imgui.get_style().item_spacing.x) / 2
        h = max(14 * imgui.get_font_size(), avail.y * 0.48)
        t, psi = morlet.wavelet_in_time(s.b, s.c, scale, fs)
        if P.begin_plot("The wavelet at this row", imgui.ImVec2(half, h)):
            implot.setup_axes("time (ms)", "", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            ms = t * 1000
            implot.plot_line("real (what gets summed)", ms, np.ascontiguousarray(psi.real), P.spec(P.ACCENT, 2))
            implot.plot_line("imaginary", ms, np.ascontiguousarray(psi.imag), P.spec(P.OPEN, 1))
            implot.plot_line("envelope", ms, np.abs(psi), P.spec(P.DIM, 1))
            implot.plot_line("##-envelope", ms, -np.abs(psi), P.spec(P.DIM, 1))
            sd = morlet.widths(s.b, s.c, scale, fs).sigma_t_s * 1000
            P.vlines("+-1 SD", [-sd, sd], P.faded(P.DIM, 0.5))
            implot.end_plot()
        imgui.same_line()
        if P.begin_plot("Time-frequency tiles (+-1 SD) of every 10th row", imgui.ImVec2(-1, h)):
            implot.setup_axes("time (ms)", "frequency (Hz)", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10)
            for k in list(range(0, len(scales), 10)) + [self.row]:
                w = morlet.widths(s.b, s.c, float(scales[k]), fs)
                if w.freq_hz > fs / 2 and k != self.row:
                    continue
                lo, hi = max(w.freq_hz - w.sigma_f_hz, w.freq_hz / 20), w.freq_hz + w.sigma_f_hz
                ts = w.sigma_t_s * 1000
                xs = np.array([-ts, ts, ts, -ts, -ts])
                ys = np.array([lo, lo, hi, hi, lo])
                color = P.ACCENT if k == self.row else P.faded(P.TEXT, 0.4)
                implot.plot_line("selected row" if k == self.row else "##tile", xs, ys, P.spec(color, 2 if k == self.row else 1))
            implot.end_plot()
        f = np.logspace(np.log10(max(0.05, s.c * fs / scales.max() / 4)), np.log10(fs / 2), 600)
        if P.begin_plot("Frequency response of every row", imgui.ImVec2(-1, -1)):
            implot.setup_axes("frequency (Hz)", "gain", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
            implot.setup_axis(implot.ImAxis_.y2, "summed gain", implot.AxisFlags_.auto_fit | implot.AxisFlags_.opposite)
            for k, sc in enumerate(scales):
                if k % 2 == 0 and k != self.row:
                    implot.plot_line("##row", f, morlet.response(s.b, s.c, sc, fs, f), P.spec(P.faded(P.TEXT, 0.25)))
            implot.plot_line("selected row", f, morlet.response(s.b, s.c, scale, fs, f), P.spec(P.ACCENT, 2.5))
            P.vlines("fs / 2", [fs / 2], P.WARN)
            implot.set_axes(implot.ImAxis_.x1, implot.ImAxis_.y2)
            implot.plot_line("every row summed (right axis)", f, morlet.bank_sum(s.b, s.c, scales, fs, f), P.spec(P.TRUTH, 2))
            if self.has("reduce") and self.run.settings.first_change(s) not in ("signal", "cwt"):
                implot.plot_line("bands summed (right axis)", f, band_sum_gain(self.run, f, include_all=True), P.spec(P.OPEN, 2))
            implot.end_plot()

    # ------------------------------------------------------------------
    # 3  scalogram
    # ------------------------------------------------------------------

    def controls_scalogram(self) -> None:
        heading("Color")
        for i, (label, tip) in enumerate(HEAT_MODES):
            if imgui.radio_button(label, self.heat_mode == i):
                self.heat_mode = i
            hint(tip)
        heading("Row")
        n = self.run.coeff.shape[0] if self.has("cwt") else int(self.settings.n_scales)
        _, self.row = imgui.slider_int("row", self.row, 0, n - 1)
        self.row = int(np.clip(self.row, 0, n - 1))
        if self.has("cwt"):
            run = self.run
            w = morlet.widths(run.settings.b, run.settings.c, float(run.scales[self.row]), run.signal.fs)
            imgui.text_disabled(f"{w.freq_hz:.4g} Hz, scale {run.scales[self.row]:.3g}")
            imgui.text_disabled(f"envelope SD {w.sigma_t_s * 1000:.3g} ms")
            imgui.text_disabled(f"passes {w.half_power_lo_hz:.3g} to {w.half_power_hi_hz:.3g} Hz at half power")
            imgui.text_wrapped("Drag the white line on the scalogram, or this slider, to pick the row plotted below it.")
        heading("Wavelet")
        imgui.text_wrapped(f"{morlet.name(self.settings.b, self.settings.c)}, {int(self.settings.n_scales)} rows. Change them on the Morlet wavelet page.")

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
        if self.begin_time_plot("Scalogram", h, "frequency (Hz)", implot.Flags_.no_legend, y_fit=False):
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
        root = math.sqrt(run.scales[k])
        rest = imgui.get_content_region_avail().y
        if self.begin_time_plot(f"Row {k}: {freqs[k]:.4g} Hz", rest * 0.5):
            self.line("input", sig.x, P.RAW)
            self.line("Re W / sqrt(scale)", run.coeff[k].real / root, P.ACCENT, 1.5)
            self.line("|W| / sqrt(scale)", run.amp[k], P.OPEN, 1.0)
            implot.end_plot()
        mid = float(np.sqrt(freqs.min() * min(freqs.max(), sig.fs / 2)))
        gain = morlet.bank_sum(run.settings.b, run.settings.c, run.scales, sig.fs, np.array([mid]))[0]
        if self.begin_time_plot(f"Sum of Re W / sqrt(scale) over all rows, no bands or masks (gain about {gain:.2g})", -1):
            self.line("input", sig.x, P.RAW)
            if sig.truth is not None:
                self.line("ground truth", sig.truth, P.TRUTH)
            self.line("sum of Re W / sqrt(scale)", run.all_rows, P.ACCENT, 1.5)
            implot.end_plot()

    # ------------------------------------------------------------------
    # 4  bands
    # ------------------------------------------------------------------

    def controls_bands(self) -> None:
        c = self.settings.clust
        n_rows = int(self.settings.n_scales)
        heading("PCA")
        _, c.standardize = imgui.checkbox("standardize rows", c.standardize)
        hint("Subtract each row's mean and divide by its SD before taking the magnitude. Off, the slow rows' large coefficients dominate the PCA.")
        _, c.n_components = imgui.slider_int("PCA components", c.n_components, 2, min(60, n_rows))
        _, c.n_comp_clu = imgui.slider_int("PCs clustered", c.n_comp_clu, 1, c.n_components)
        heading("Tree")
        methods = ["ward", "average", "complete", "single"]
        changed, i = imgui.combo("linkage", methods.index(c.method_linkage) if c.method_linkage in methods else 0, methods)
        if changed:
            c.method_linkage = methods[i]
        _, c.n_subclusters = imgui.slider_int("bands (sublabels)", c.n_subclusters, 2, min(30, n_rows - 1))
        _, c.n_clusters = imgui.slider_int("coarse clusters", c.n_clusters, 2, min(30, n_rows - 1))
        hint("vnoiser computes this cut (labels) too but bands come from the sublabels.")
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
                        imgui.text_colored(P.WARN, "0, dropped")
                imgui.end_table()

    def page_bands(self) -> None:
        if not self.has("cluster"):
            imgui.text_disabled("The clustering has not run.")
            return
        run = self.run
        order = self.band_order()
        freqs = run.freqs
        sub = run.cluster["sublabels"]
        avail = imgui.get_content_region_avail()
        half = (avail.x - imgui.get_style().item_spacing.x) / 2
        h1 = max(12 * imgui.get_font_size(), avail.y * 0.36)
        if P.begin_plot("Bands over frequency, and the gain of summing them", imgui.ImVec2(-1, h1), implot.Flags_.no_legend):
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
            P.vlines("fs / 2", [run.signal.fs / 2], P.WARN)
            if self.has("reduce"):
                fx = np.logspace(np.log10(freqs.min()), np.log10(min(freqs.max(), run.signal.fs / 2)), 500)
                implot.set_axes(implot.ImAxis_.x1, implot.ImAxis_.y2)
                implot.plot_line("band sum gain, kept bands (right)", fx, band_sum_gain(run, fx), P.spec(P.WHITE, 2))
            implot.end_plot()
        h2 = imgui.get_content_region_avail().y * 0.5
        pca = run.cluster["pca_coeff"]
        n_pc = pca.shape[1]
        if P.begin_plot("Rows in PC space (one point per row)", imgui.ImVec2(half, h2), implot.Flags_.no_legend):
            px, py = (min(p, n_pc - 1) for p in self.pc)
            implot.setup_axes(f"PC{px + 1}", f"PC{py + 1}", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            for band in run.bands:
                rows = sub == band.id
                implot.plot_scatter(band.span, pca[rows, px].astype(np.float64), pca[rows, py].astype(np.float64),
                                    P.marker(P.band_color(order[band.id])))
            if implot.is_plot_hovered():
                mouse = implot.get_plot_mouse_pos()
                d = (pca[:, px] - mouse.x) ** 2 + (pca[:, py] - mouse.y) ** 2
                k = int(np.argmin(d))
                imgui.set_tooltip(f"row {k}: {freqs[k]:.4g} Hz, band {fmt_hz(freqs[sub == sub[k]].min())} to {fmt_hz(freqs[sub == sub[k]].max())} Hz")
            implot.end_plot()
        imgui.same_line()
        if P.begin_plot("Explained variance", imgui.ImVec2(-1, h2)):
            implot.setup_axes("component", "fraction", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            ex = run.cluster["explained"].astype(np.float64)
            xs = np.arange(1, ex.size + 1, dtype=np.float64)
            implot.plot_bars("each", xs, ex, 0.7, P.spec(P.ACCENT))
            implot.plot_line("cumulative", xs, np.cumsum(ex), P.spec(P.OPEN, 2))
            P.vlines("clustered", [run.settings.clust.n_comp_clu + 0.5], P.WARN)
            implot.end_plot()
        imgui.text_disabled("PC axes")
        for i, name in enumerate(("x", "y")):
            imgui.same_line()
            imgui.set_next_item_width(7 * imgui.get_font_size())
            _, value = imgui.slider_int(f"##pc{name}", self.pc[i] + 1, 1, n_pc, f"{name}: PC%d")
            self.pc[i] = value - 1
        self.draw_dendrogram(order, sub)

    def draw_dendrogram(self, order: dict, sub: np.ndarray) -> None:
        run = self.run
        if self.dendro_cache[0] is not id(run):
            d = run.dendro
            leaves = np.asarray(d["leaves"])
            segments = []
            for xs, ys in zip(d["icoord"], d["dcoord"]):
                k0, k1 = int(math.ceil((min(xs) - 5) / 10)), int(math.floor((max(xs) - 5) / 10))
                bands = set(sub[leaves[k0:k1 + 1]].tolist())
                segments.append((np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), bands.pop() if len(bands) == 1 else None))
            self.dendro_cache = (id(run), (segments, leaves))
        segments, leaves = self.dendro_cache[1]
        if P.begin_plot("Ward tree of the rows, cut into bands", imgui.ImVec2(-1, -1), implot.Flags_.no_legend):
            implot.setup_axes("rows, in tree order", "merge distance", implot.AxisFlags_.auto_fit | implot.AxisFlags_.no_tick_labels, implot.AxisFlags_.auto_fit)
            for xs, ys, band in segments:
                color = P.band_color(order[band]) if band is not None and band in order else P.faded(P.TEXT, 0.6)
                implot.plot_line("##seg", xs, ys, P.spec(color, 1.5))
            P.hline("cut", run.cut_height, P.WARN)
            if implot.is_plot_hovered():
                mouse = implot.get_plot_mouse_pos()
                k = int(np.clip(round((mouse.x - 5) / 10), 0, len(leaves) - 1))
                imgui.set_tooltip(f"leaf {k}: row {leaves[k]}, {run.freqs[leaves[k]]:.4g} Hz")
            implot.end_plot()

    # ------------------------------------------------------------------
    # 5  events and masks
    # ------------------------------------------------------------------

    def controls_events(self) -> None:
        r, t = self.settings.reducer, self.settings.thres
        if self.has("reduce"):
            heading("Band")
            order = self.band_order()
            for band in reversed(self.run.bands):
                imgui.color_button(f"##b{band.id}", P.band_color(order[band.id]), 0, imgui.ImVec2(0.8 * imgui.get_font_size(), 0.8 * imgui.get_font_size()))
                imgui.same_line()
                label = f"{band.span}" + (f"   {len(band.windows)} windows" if band.kept else "   dropped")
                if imgui.selectable(f"{label}##{band.id}", self.band_id == band.id)[0]:
                    self.band_id = band.id
        heading("Windows (WaveletReducer)")
        _, r.slow_upthres = imgui.slider_float("open SD, < 30 Hz", r.slow_upthres, 0.5, 6.0, "%.2f")
        hint("Close level is 1 SD for these bands.")
        _, r.fast_upthres = imgui.slider_float("open SD, >= 30 Hz", r.fast_upthres, 0.5, 6.0, "%.2f")
        hint("Close level is 0 for these bands: a window ends when the trace crosses zero.")
        changed, windows = imgui.input_int3("savgol windows", list(r.smooth_windows))
        if changed:
            r.smooth_windows = tuple(max(3, int(w)) for w in windows)
        hint(f"For bands topping out at <= 15, <= 50 and <= 75 Hz, in samples at {r.smooth_windows_fs:g} Hz; rescaled to the trace's rate, odd, at least 3.")
        _, r.complex_bands = imgui.checkbox("complex bands above 75 Hz", r.complex_bands)
        hint("Denoiser.upstream keeps fast bands complex through the thresholds, as the archive was made.")
        heading("Masks (AdaptiveThreshold)")
        kinds = ["soft", "hard"]
        changed, i = imgui.combo("type", kinds.index(t.thres_type), kinds)
        if changed:
            t.thres_type = kinds[i]
        if t.thres_type == "soft":
            changed, levels = imgui.slider_float4("floors", list(t.soft_levels), 0.0, 1.0, "%.2f")
            if changed:
                t.soft_levels = tuple(levels)
            hint("Mask value outside windows for bands topping out below 5, below 30, below 80 Hz, and above.")
            changed, sigmas = imgui.input_float2("blur SD", list(t.soft_sigma), "%.1f")
            if changed:
                t.soft_sigma = tuple(max(0.1, v) for v in sigmas)
            hint("Gaussian blur of the mask in samples, for bands topping out at or below 120 Hz, and above.")
        else:
            _, t.hard_floor = imgui.slider_float("floor", t.hard_floor, 0.0, 1.0, "%.3f")
            changed, sigmas = imgui.input_float2("blur SD", list(t.hard_sigma), "%.1f")
            if changed:
                t.hard_sigma = tuple(max(0.1, v) for v in sigmas)
        imgui.text_wrapped("Window widening (median window x 2 / 1.5 / 0.5 for bands topping at <= 20 / <= 50 Hz / above) is fixed in vnoiser.")

    def page_events(self) -> None:
        if not self.has("reduce"):
            imgui.text_disabled("The bands have not been made.")
            return
        run = self.run
        band = run.band(self.band_id)
        if band is None:
            return
        fs = run.signal.fs
        color = P.band_color(self.band_order()[band.id])
        what = f"smoothed, savgol {band.smooth_window}" if band.smooth_window else "not smoothed (top above 75 Hz)"
        h = self.plot_height(0.3)
        if self.begin_time_plot(f"Band {band.span}: mean of its {int(band.rows.sum())} rows, {what}", h):
            if band.kept and band.extra:
                wide = np.clip(band.windows + np.array([-band.extra, band.extra]), 0, run.signal.n)
                P.shade_intervals(wide, fs, self.t0.value, self.t1.value, P.faded(color, 0.10))
            P.shade_intervals(band.windows, fs, self.t0.value, self.t1.value, P.faded(color, 0.22))
            if band.smooth_window:
                self.line("band mean", band.average.real, P.faded(P.TEXT, 0.35))
            self.line("thresholded trace", band.smoothed.real, color, 1.5)
            P.hline(f"open {band.up:.3g}", band.up, P.OPEN, 1.5)
            P.hline(f"close {band.down:.3g}", band.down, P.CLOSE, 1.0)
            implot.end_plot()
        if not band.kept:
            imgui.text_colored(P.WARN, "No window opened in this band, so WaveletReducer drops it: it adds nothing to the sum.")
        elif band.mask is not None:
            if self.begin_time_plot(f"Mask: floor {band.floor:g}, windows widened {band.extra} samples, blur SD {band.sigma:g}", h * 0.7, y_fit=False):
                implot.setup_axis_limits(implot.ImAxis_.y1, -0.05, 1.08, implot.Cond_.always)
                self.line("before blur", band.label, P.faded(P.TEXT, 0.5))
                self.line("mask", band.mask, color, 2)
                implot.end_plot()
            if self.begin_time_plot("What the band adds: trace / sqrt(mean scale), times the mask", h):
                self.line("unmasked", band.scaled.real, P.faded(P.TEXT, 0.35))
                self.line("masked", band.masked.real, color, 1.5)
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
        if self.begin_time_plot("Every kept band's mask (fastest on top)", h, flags=implot.Flags_.no_legend, y_fit=False):
            implot.setup_axis_limits(implot.ImAxis_.y1, 0, nb, implot.Cond_.always)
            implot.setup_axis_ticks(implot.ImAxis_.y1, [nb - k - 0.5 for k in range(nb)], [b.span for b in kept], False)
            implot.plot_heatmap("##masks", np.ascontiguousarray(m, dtype=np.float64), 0.0, 1.0, "",
                                implot.Point(i0 / fs, 0), implot.Point((i0 + max(whole, 1)) / fs, nb))
            implot.end_plot()
        implot.pop_colormap()

    # ------------------------------------------------------------------
    # 6  reconstruction
    # ------------------------------------------------------------------

    def controls_final(self) -> None:
        s = self.settings
        heading("Baseline")
        _, s.add_baseline = imgui.checkbox("Add baseline", s.add_baseline)
        hint("Denoiser.run adds it; the archive's denoised_trace_scans.pkl did not.")
        _, s.lp_cutoff = imgui.slider_float("cutoff (Hz)", s.lp_cutoff, 0.05, 20.0, "%.3g", imgui.SliderFlags_.logarithmic)
        _, s.fir_window_ms = imgui.slider_float("FIR window (ms)", s.fir_window_ms, 50.0, 10000.0, "%.0f", imgui.SliderFlags_.logarithmic)
        _, s.fir_odd_taps = imgui.checkbox("odd tap count", s.fir_odd_taps)
        if self.has("mask"):
            heading("Bands in the sum")
            order = self.band_order()
            excluded = set(s.excluded)
            for band in reversed(self.run.bands):
                imgui.color_button(f"##f{band.id}", P.band_color(order[band.id]), 0, imgui.ImVec2(0.8 * imgui.get_font_size(), 0.8 * imgui.get_font_size()))
                imgui.same_line()
                if not band.kept:
                    imgui.text_disabled(f"{band.span}  (dropped)")
                    continue
                changed, on = imgui.checkbox(f"{band.span}##inc{band.id}", band.id not in excluded)
                if changed:
                    excluded = excluded - {band.id} if on else excluded | {band.id}
            s.excluded = tuple(sorted(excluded))
        if self.has("final"):
            heading("Numbers")
            for name, value in metrics(self.run).items():
                imgui.text_disabled(f"{name:<24}")
                imgui.same_line()
                imgui.text(f"{value:.4g}")
            hint_text = "RMSE and correlation subtract each trace's mean first. With Add baseline off, the ground truth has its own 1 Hz low-pass removed before comparing."
            hint(hint_text)

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
        if self.begin_time_plot("Each band's contribution, stacked (slowest at the bottom)", h, flags=implot.Flags_.no_legend, y_fit=False):
            implot.setup_axis(implot.ImAxis_.y1, None, implot.AxisFlags_.no_tick_labels)
            implot.setup_axis_limits(implot.ImAxis_.y1, -0.7 * spread, (len(kept) - 0.2) * spread, implot.Cond_.always)
            for k, band in enumerate(kept):
                alpha = 0.25 if band.id in excluded else 1.0
                self.line(band.span, band.masked.real + k * spread, P.band_color(order[band.id], alpha), 1.2)
                implot.plot_text(band.span, self.t0.value, k * spread, imgui.ImVec2(3 * imgui.get_font_size(), -0.6 * imgui.get_font_size()))
            implot.end_plot()
        if self.begin_time_plot("Denoised trace", h):
            self.line("raw", sig.x, P.RAW)
            if sig.truth is not None:
                self.line("ground truth", sig.truth, P.TRUTH, 1.5)
            if run.settings.add_baseline:
                self.line("baseline (1 Hz FIR)", run.baseline, P.CLOSE, 1.0)
            self.line("denoised", run.denoised, P.ACCENT, 1.8)
            implot.end_plot()
        avail = imgui.get_content_region_avail()
        half = (avail.x - imgui.get_style().item_spacing.x) / 2
        if P.begin_plot("Removed: raw minus denoised", imgui.ImVec2(half, -1)):
            implot.setup_axes("time (s)", "", 0, implot.AxisFlags_.auto_fit | implot.AxisFlags_.range_fit)
            implot.setup_axis_links(implot.ImAxis_.x1, self.t0, self.t1)
            self.line("removed", sig.x - run.denoised, P.faded(P.ERROR, 0.8))
            implot.end_plot()
        imgui.same_line()
        if run.psd and P.begin_plot("Power spectrum", imgui.ImVec2(-1, -1)):
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
