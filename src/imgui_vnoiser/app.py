"""
The `imgui-vnoiser` window, in imgui_data_loader's shell: a rail of pages on the left, one per stage of vnoiser's
denoiser, with the stage's settings under it, and the stage's plots on the right. Every time plot shares one time
range, so zooming one zooms them all. The pipeline runs on a worker thread and reruns only the stages a change
affects.
"""

import argparse
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import imgui_data_loader as idl
import numpy as np
from imgui_bundle import hello_imgui, imgui, imgui_ctx, immapp, implot
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import portable_file_dialogs as pfd

from imgui_vnoiser import __version__, config, style
from imgui_vnoiser import plotting as P
from imgui_vnoiser.pipeline import STAGES, Run, Settings, band_sum_gain, compute, metrics, noise_floor_hz
from imgui_vnoiser.signals import SOURCES, find_scans, scan_info

ACCENT = style.vec4(style.ACCENT)
DIM = style.vec4(style.DIM)
TEXT = style.vec4(style.TEXT)
CARD = style.vec4(style.CARD)
CARD_HOVER = style.vec4(style.CARD_HOVER)
EDGE = style.vec4(style.EDGE)
OK = style.vec4(style.OK)
WARN = style.vec4(style.WARN)
ERROR = style.vec4(style.ERROR)
LIT = style.vec4(style.ACCENT, 0.22)
CLEAR = imgui.ImVec4(0, 0, 0, 0)

TITLE_WINDOW = "vnoiser"
SIZE_WINDOW = (1560, 980)
WIDTH_RAIL_EM = 19
WIDTH_CAPTION_EM = 6.5

# the rail: a page key, its icon, its name, and the line under the name on its page
PAGES = (
    ("signal", fa.ICON_FA_GRIP_LINES, "Line scan traces", "the scan's pixels averaged into F, then dF/F, then z"),
    ("scalogram", fa.ICON_FA_CHART_AREA, "Scalogram", "the trace's Morlet transform: one row per frequency"),
    ("bands", fa.ICON_FA_LAYER_GROUP, "Bands", "rows that vary together over time, grouped by PCA and Ward clustering"),
    ("events", fa.ICON_FA_FILTER, "Masks", "per band: the mean of its rows, events over a threshold, and a mask"),
    ("final", fa.ICON_FA_WAVE_SQUARE, "Denoised", "the masked bands added up, plus the input's slow part"),
)
# what the header and footer call each stage
STAGE_NAMES = {"signal": "trace", "cwt": "transform", "cluster": "grouping", "reduce": "bands", "mask": "masks",
               "final": "sum"}
HEAT_MODES = ("amplitude", "amplitude / median", "standardized", "real part")
VIEWS = (("all", 0.0), ("4 s", 4.0), ("1 s", 1.0), ("200 ms", 0.2))
SECONDS_COPIED = 3.0  # how long the footer says the code was copied


def u32(color: imgui.ImVec4) -> int:
    return imgui.color_convert_float4_to_u32(color)


def section(text: str) -> None:
    """A dim upper-case heading over a rule, as in the masknmf launcher's rail."""
    imgui.dummy(imgui.ImVec2(0, 0.4 * imgui.get_font_size()))
    imgui.text_disabled(text.upper())
    imgui.separator()


def caption(text: str) -> None:
    """A dim line over the plot it explains."""
    idl.text_wrapped_colored(DIM, text)


def begin_form(name: str) -> bool:
    """A two column table: dim captions in a fixed column, widgets filling the rest; end it with imgui.end_table."""
    if not imgui.begin_table(f"##form_{name}", 2, imgui.TableFlags_.sizing_stretch_prop):
        return False
    imgui.table_setup_column("##caption", imgui.TableColumnFlags_.width_fixed, WIDTH_CAPTION_EM * imgui.get_font_size())
    imgui.table_setup_column("##widget", imgui.TableColumnFlags_.width_stretch)
    return True


def row(caption: str, hint: str = "") -> None:
    """Start a form row: the caption, then the cursor in the widget column, the next widget as wide as it."""
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.align_text_to_frame_padding()
    imgui.text_disabled(caption)
    if hint and imgui.is_item_hovered():
        idl.wrapped_tooltip(hint)
    imgui.table_next_column()
    imgui.set_next_item_width(-1)


def swatch(key: str, color: imgui.ImVec4) -> None:
    em = imgui.get_font_size()
    imgui.color_button(f"##{key}", color, imgui.ColorEditFlags_.no_tooltip, imgui.ImVec2(0.8 * em, 0.8 * em))
    imgui.same_line()


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
        self.row = 40
        self.heat_mode = 1
        self.heat_cache = (None, None)
        self.pixel_cache = (None, None)
        self.colormaps: Optional[dict] = None
        self.band_id: Optional[int] = None
        self.picker = None
        self.scans: Optional[list] = None  # exported line scans found under ~/data
        self.time_copied = -math.inf
        self.dialog = idl.FileDialog(idl.FileDialogConfig(
            title=TITLE_WINDOW,
            buttons=[],
            theme=style.THEME,
            header_draw=self.draw_header,
            body_draw=self.draw_body,
            footer_draw=self.draw_footer,
            show_options_button=False,
            close_on_select=False,
            quit_on_escape=False,
            on_cancel=self.quit,
        ))

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

    def quit(self) -> None:
        hello_imgui.get_runner_params().app_shall_exit = True

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
        return self.run is not None and stage in self.run.timing

    def band_order(self) -> dict:
        """Band id -> its color index (bands in frequency order)."""
        return {band.id: i for i, band in enumerate(self.run.bands)} if self.run else {}

    # ------------------------------------------------------------------
    # scale
    # ------------------------------------------------------------------

    def apply_scale(self) -> None:
        """Font, imgui and ImPlot sizes to self.scale. Called every frame; does work only when the scale moved."""
        if self.scale == self.scale_applied:
            return
        ratio = self.scale / self.scale_applied
        style_imgui = imgui.get_style()
        style_imgui.font_scale_main = self.scale
        style_imgui.scale_all_sizes(ratio)
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

    def setup_plots(self) -> None:
        """Smaller ImPlot padding than the defaults, so stacked plots leave more room for data, and the colormaps.
        Once, at scale 1."""
        ps = implot.get_style()
        ps.plot_padding = imgui.ImVec2(4, 4)
        ps.label_padding = imgui.ImVec2(3, 2)
        ps.legend_padding = imgui.ImVec2(4, 2)
        ps.legend_inner_padding = imgui.ImVec2(3, 1)
        self.colormaps = style.add_colormaps()

    # ------------------------------------------------------------------
    # the shell: header, body, footer
    # ------------------------------------------------------------------

    def gui(self) -> None:
        if self.colormaps is None:
            self.setup_plots()
        self.dialog.render()

    def state(self) -> tuple:
        """(color, text) for the header: what the pipeline is doing or how it ended."""
        run = self.run
        if run is not None and run.error:
            return ERROR, f"failed at the {STAGE_NAMES.get(run.failed, run.failed)}"
        if self.future is not None:
            return WARN, f"running: {STAGE_NAMES.get(self.stage_running, 'starting')}"
        if self.has("final"):
            return OK, f"{len(run.kept)} of {len(run.bands)} bands kept"
        return DIM, "starting"

    def source_name(self) -> str:
        sp = self.settings.signal
        if sp.source == "recorded" and sp.scan_path:
            return Path(sp.scan_path).stem
        if sp.source == "file" and sp.path:
            return Path(sp.path).name
        return next(label for key, label, _ in SOURCES if key == sp.source)

    def draw_header(self) -> None:
        """Name, version and the pipeline's state."""
        em = imgui.get_font_size()
        imgui.dummy(imgui.ImVec2(0, 0.2 * em))
        imgui.push_font(None, 1.5 * em)
        imgui.text_colored(ACCENT, f"{fa.ICON_FA_WAVE_SQUARE}  vnoiser")
        imgui.pop_font()
        imgui.same_line(0, 0.6 * em)
        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 0.45 * em)
        imgui.text_disabled(f"v{__version__}")
        color, state = self.state()
        text = f"{fa.ICON_FA_GRIP_LINES}  {self.source_name()}    {fa.ICON_FA_CIRCLE}  {state}"
        imgui.same_line()
        room = imgui.get_content_region_avail().x - imgui.calc_text_size(text).x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, room))
        imgui.text_colored(color, text)
        idl.text_wrapped_colored(DIM, "What vnoiser's Morlet wavelet denoiser does to a voltage trace, stage by stage")

    def draw_body(self) -> None:
        """The rail and the page side by side, filling the room above the footer."""
        self.scale_keys()
        self.apply_scale()
        self.poll()
        em = imgui.get_font_size()
        avail = imgui.get_content_region_avail()
        # a fifth of the window, between 15 and 19 em: big text in a half-screen tile still leaves room for plots
        width = min(max(0.2 * avail.x, 15 * em), WIDTH_RAIL_EM * em)
        dl = imgui.get_window_draw_list()
        p = imgui.get_cursor_screen_pos()
        with imgui_ctx.begin_child("##rail", imgui.ImVec2(width, avail.y), 0, imgui.WindowFlags_.no_scrollbar):
            self.draw_rail()
        x = p.x + width + 0.4 * em
        dl.add_line(imgui.ImVec2(x, p.y), imgui.ImVec2(x, p.y + avail.y), u32(EDGE), 1.0)
        imgui.same_line(0, 0.8 * em)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.8 * em, 0.2 * em))
        with imgui_ctx.begin_child("##content", imgui.ImVec2(0, avail.y), imgui.ChildFlags_.always_use_window_padding,
                                   imgui.WindowFlags_.no_scrollbar):
            self.draw_content()
        imgui.pop_style_var()

    def draw_footer(self, dialog: idl.FileDialog) -> None:
        """Status on the left, the text size and Quit on the right."""
        em = imgui.get_font_size()
        imgui.separator()
        imgui.spacing()
        text_quit = f"{fa.ICON_FA_XMARK}  Quit"
        width_quit = imgui.calc_text_size(text_quit).x + 2 * imgui.get_style().frame_padding.x + 1.5 * em
        width_scale = 9 * em
        if not imgui.begin_table("##footer", 3, imgui.TableFlags_.sizing_stretch_prop):
            return
        imgui.table_setup_column("##status", imgui.TableColumnFlags_.width_stretch)
        imgui.table_setup_column("##scale", imgui.TableColumnFlags_.width_fixed, width_scale)
        imgui.table_setup_column("##quit", imgui.TableColumnFlags_.width_fixed, width_quit)
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.align_text_to_frame_padding()
        run = self.run
        if run is not None and run.error:
            idl.text_wrapped_colored(ERROR, f"{fa.ICON_FA_CIRCLE_EXCLAMATION}  {run.error}")
        elif self.future is not None:
            imgui.text_colored(WARN, f"{fa.ICON_FA_SPINNER}  vnoiser: {STAGE_NAMES.get(self.stage_running, 'starting')}")
        elif time.monotonic() - self.time_copied < SECONDS_COPIED:
            imgui.text_colored(DIM, f"{fa.ICON_FA_COPY}  copied the code for these settings")
        elif run is not None and run.timing:
            steps = ", ".join(f"{STAGE_NAMES[k]} {1000 * v:.0f}" for k, v in run.timing.items())
            imgui.text_colored(DIM, f"{fa.ICON_FA_CIRCLE_CHECK}  ran in {1000 * sum(run.timing.values()):.0f} ms: {steps}")
        imgui.table_next_column()
        imgui.set_next_item_width(-1)
        # applied on release: rescaling mid-drag would move the slider out from under the mouse
        shown = self.scale_edit if self.scale_edit is not None else self.scale
        changed, value = imgui.slider_float("##scale", shown, config.SCALE_MIN, config.SCALE_MAX, "text %.2fx")
        if changed:
            self.scale_edit = value
        if imgui.is_item_hovered():
            idl.wrapped_tooltip("Text size. Ctrl + and Ctrl - change it too, Ctrl 0 resets it.")
        if self.scale_edit is not None and not imgui.is_item_active():
            self.set_scale(self.scale_edit)
            self.scale_edit = None
        imgui.table_next_column()
        idl.push_button_style(dialog.theme, primary=False)
        if imgui.button(text_quit, imgui.ImVec2(width_quit, 1.7 * em)):
            dialog.cancel()
        idl.pop_button_style()
        imgui.end_table()

    def draw_rail(self) -> None:
        """Page buttons, the page's settings, then the presets."""
        em = imgui.get_font_size()
        width = imgui.get_content_region_avail().x
        imgui.push_style_var(imgui.StyleVar_.button_text_align, imgui.ImVec2(0.0, 0.5))
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 5.0)
        for key, icon, name, hint in PAGES:
            lit = key == self.page
            imgui.push_style_color(imgui.Col_.button, LIT if lit else CLEAR)
            imgui.push_style_color(imgui.Col_.button_hovered, LIT if lit else CARD_HOVER)
            imgui.push_style_color(imgui.Col_.button_active, LIT)
            imgui.push_style_color(imgui.Col_.text, ACCENT if lit else TEXT)
            if imgui.button(f"{icon}   {name}##rail_{key}", imgui.ImVec2(width, 2.0 * em)):
                self.page = key
            imgui.pop_style_color(4)
            if imgui.is_item_hovered():
                idl.wrapped_tooltip(f"{name}: {hint}")
        imgui.pop_style_var(2)
        bottom = 2 * imgui.get_frame_height_with_spacing() + 0.8 * em
        with imgui_ctx.begin_child("##settings", imgui.ImVec2(0, -bottom)):
            getattr(self, f"controls_{self.page}")()
        imgui.separator()
        imgui.dummy(imgui.ImVec2(0, 0.2 * em))
        self.draw_presets(width)

    def draw_presets(self, width: float) -> None:
        """vnoiser's defaults, the lab's settings, and the code for the current ones."""
        half = (width - imgui.get_style().item_spacing.x) / 2
        idl.push_button_style(style.THEME, primary=False)
        if imgui.button(f"{fa.ICON_FA_ARROW_ROTATE_LEFT}  Defaults", imgui.ImVec2(half, 0)):
            self.settings.use_defaults()
        if imgui.is_item_hovered():
            idl.wrapped_tooltip("Denoiser(): vnoiser's defaults")
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_FLASK}  Lab settings", imgui.ImVec2(half, 0)):
            self.settings.use_upstream()
        if imgui.is_item_hovered():
            idl.wrapped_tooltip("Denoiser.upstream: the lab's original settings. Bands above 75 Hz stay complex, and "
                                "the floor above 80 Hz is 0.01 instead of 0.1.")
        if imgui.button(f"{fa.ICON_FA_COPY}  Copy code", imgui.ImVec2(width, 0)):
            imgui.set_clipboard_text(self.settings.python())
            self.time_copied = time.monotonic()
        if imgui.is_item_hovered():
            idl.wrapped_tooltip("The Python that runs vnoiser with these settings")
        idl.pop_button_style()

    def draw_content(self) -> None:
        """The page's title, the view buttons and the line under the title; the plots; the page's note."""
        em = imgui.get_font_size()
        key, icon, name, hint = next(page for page in PAGES if page[0] == self.page)
        imgui.push_font(None, 1.35 * em)
        imgui.text_colored(ACCENT, f"{icon}  {name}")
        imgui.pop_font()
        if self.ready:
            self.draw_views()
        idl.text_wrapped_colored(DIM, hint)
        imgui.separator()
        note = self.note() if self.ready else ""
        width = imgui.get_content_region_avail().x
        # the text, the dummy over it, and the spacing before and inside the note's child
        height_note = imgui.calc_text_size(note, wrap_width=width).y + 0.2 * em + 2 * imgui.get_style().item_spacing.y if note else 0.0
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(0.6 * em, 0.25 * em))
        with imgui_ctx.begin_child("##plots", imgui.ImVec2(0, -height_note)):
            if self.ready:
                getattr(self, f"page_{key}")()
            else:
                self.draw_waiting()
        imgui.pop_style_var()
        if note:
            with imgui_ctx.begin_child("##note", imgui.ImVec2(0, 0), 0, imgui.WindowFlags_.no_scrollbar):
                imgui.dummy(imgui.ImVec2(0, 0.2 * em))
                idl.text_wrapped_colored(TEXT, note)

    def draw_views(self) -> None:
        """Chips at the right of the title row that set how much time the plots show; the current one lit."""
        duration = self.run.signal.duration
        span_now = self.t1.value - self.t0.value
        texts = [text for text, _ in VIEWS]
        width = sum(imgui.calc_text_size(t).x + 2 * imgui.get_style().frame_padding.x for t in texts)
        width += imgui.get_style().item_spacing.x * (len(texts) - 1)
        imgui.same_line()
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, imgui.get_content_region_avail().x - width))
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 5.0)
        for i, (text, span) in enumerate(VIEWS):
            span = min(span or duration, duration)
            lit = abs(span_now - span) < 0.01 * span
            if i:
                imgui.same_line()
            imgui.push_style_color(imgui.Col_.button, LIT if lit else CARD)
            imgui.push_style_color(imgui.Col_.button_hovered, LIT if lit else CARD_HOVER)
            imgui.push_style_color(imgui.Col_.button_active, LIT)
            imgui.push_style_color(imgui.Col_.text, ACCENT if lit else DIM)
            if imgui.button(f"{text}##view"):
                center = 0.5 * (self.t0.value + self.t1.value) if span < duration else duration / 2
                lo = float(np.clip(center - span / 2, 0, duration - span))
                self.t0.value, self.t1.value = lo, lo + span
            imgui.pop_style_color(4)
        imgui.pop_style_var()

    def draw_waiting(self) -> None:
        """The stages, the done ones checked, until the first run is in."""
        em = imgui.get_font_size()
        imgui.dummy(imgui.ImVec2(0, 2 * em))
        current = STAGES.index(self.stage_running) if self.stage_running in STAGES else -1
        for i, stage in enumerate(STAGES):
            if i < current:
                color, icon = OK, fa.ICON_FA_CIRCLE_CHECK
            elif i == current:
                color, icon = WARN, fa.ICON_FA_SPINNER
            else:
                color, icon = DIM, fa.ICON_FA_CIRCLE
            idl.center_text(f"{icon}  {STAGE_NAMES[stage]}", color)

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

    def plot_height(self, share: float, minimum: float = 8.0) -> float:
        """share of the space left in the plots child, in pixels, at least minimum em."""
        return max(minimum * imgui.get_font_size(), imgui.get_content_region_avail().y * share)

    def begin_spectrum(self, title: str, size: imgui.ImVec2) -> bool:
        if not P.begin_plot(title, size):
            return False
        implot.setup_axes("frequency (Hz)", "power / Hz", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
        implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
        implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10)
        return True

    # ------------------------------------------------------------------
    # notes: one paragraph per page, with this trace's numbers in it
    # ------------------------------------------------------------------

    def note(self) -> str:
        return getattr(self, f"note_{self.page}")()

    def note_signal(self) -> str:
        sig = self.run.signal
        ls = sig.lines
        if ls is None:
            sp = self.run.settings.signal
            return (f"{sig.n} samples at {sig.fs:g} Hz, {'z-scored' if sp.zscore else 'as read'}. The denoiser expects "
                    "z-scored dF/F with depolarization up.")
        text = (f"F is the mean of all {ls.pixels.shape[0]} pixels. dF/F is F minus its baseline, over the baseline. "
                "z is dF/F minus a slower baseline, over its SD, with the sign flipped so depolarization points up. "
                f"The first {ls.n_startup / sig.fs:.1f} s are set to the mean while the baselines settle.")
        if sig.truth is not None:
            r = float(np.corrcoef(sig.x[ls.n_startup:], sig.truth[ls.n_startup:])[0, 1])
            return text + f" Here z correlates {r:.2f} with the noiseless z."
        return text + (" A line without a bright stripe adds only noise to F: untick it. A stripe that wanders is "
                       "motion. The baseline should follow the slow decline (bleaching) and nothing faster.")

    def note_scalogram(self) -> str:
        run = self.run
        freqs, k = run.freqs, self.row
        text = (f"Row {k} is the trace filtered at {freqs[k]:.3g} Hz, the white line. The {freqs.size} rows run from "
                f"{freqs.min():.2g} Hz at the bottom to {freqs.max():.0f} Hz at the top. ")
        if self.heat_mode == 1:
            text += "Color is each row's amplitude over its median, so a row lights up where it is above its usual level. "
        text += ("Events are vertical streaks: spikes reach up into the fast rows, slow depolarizations stay low. A "
                 "row with even speckle everywhere holds noise.")
        if run.signal.fs / 2 < freqs.max():
            text += f" Rows above the orange line ({run.signal.fs / 2:.0f} Hz) are faster than the sampling rate allows."
        return text

    def note_bands(self) -> str:
        run = self.run
        if not run.bands:
            return ""
        text = (f"The {run.freqs.size} rows form {len(run.bands)} bands; {len(run.kept)} have events and stay in the "
                "sum. A band should sit on signal or on noise, not across both.")
        edge = noise_floor_hz(run.psd, run.signal.fs) if run.psd else None
        if edge is None:
            return text
        text += f" The input's spectrum reaches its noise floor near {edge:.0f} Hz"
        above = [band for band in run.kept if band.lo_hz > edge]
        if not above:
            return text + "."
        events = sum(len(band.windows) for band in above)
        return text + (f", so the {len(above)} kept bands above it hold mostly noise, and their {events} events are "
                       "mostly noise crossing the threshold. Untick them on the Denoised page to see what they add.")

    def note_events(self) -> str:
        run = self.run
        band = run.band(self.band_id)
        if band is None:
            return ""
        smoothed = f", smoothed over {band.smooth_window} samples" if band.smooth_window else ""
        text = f"{band.span}: the mean of its {int(band.rows.sum())} rows{smoothed}. "
        if not band.kept:
            return text + f"It never reaches {band.up:.2f}, so it is left out of the sum."
        text += (f"An event opens at {band.up:.2f} and closes at {band.down:.2f}: {len(band.windows)} here. ")
        if band.mask is not None:
            text += (f"The mask is 1 on each event, widened by {1000 * band.extra / run.signal.fs:.0f} ms, and "
                     f"{band.floor:g} elsewhere, then blurred. ")
        return text + "Windows on noise mean the threshold is too low; events without one mean it is too high."

    def note_final(self) -> str:
        run = self.run
        s = run.settings
        kept = [band for band in run.kept if band.id not in s.excluded]
        text = f"Denoised is the {len(kept)} masked bands added up"
        text += f", plus the input below {s.lp_cutoff:g} Hz. " if s.add_baseline else ". "
        text += "Bottom left is what the denoiser took out: an event there was cut."
        numbers = metrics(run)
        if "corr denoised vs ground truth" in numbers:
            text += (f" Correlation with the noiseless trace: {numbers['corr raw vs ground truth']:.2f} for the input, "
                     f"{numbers['corr denoised vs ground truth']:.2f} denoised.")
        return text

    # ------------------------------------------------------------------
    # Line scan traces
    # ------------------------------------------------------------------

    def controls_signal(self) -> None:
        sp = self.settings.signal
        section("Source")
        for key, label, hint in SOURCES:
            if imgui.radio_button(label, sp.source == key):
                sp.source = key
            if imgui.is_item_hovered():
                idl.wrapped_tooltip(hint)
        if sp.source == "recorded":
            self.controls_recorded()
            return
        section("Scan" if sp.source == "lines" else "File")
        if not begin_form("signal"):
            return
        if sp.source == "file":
            row("path")
            _, sp.path = imgui.input_text("##path", sp.path)
            columns = self.run.signal.columns if self.ready and self.run.settings.signal.source == "file" else 1
            if columns > 1:
                row("column")
                _, value = imgui.slider_int("##column", sp.column + 1, 1, columns)
                sp.column = value - 1
            row("z-score", "Subtract the mean and divide by the SD before the denoiser")
            _, sp.zscore = imgui.checkbox("##zscore", sp.zscore)
        row("rate")
        _, sp.fs = imgui.input_float("##fs", sp.fs, 100.0, 1000.0, "%.4g Hz")
        if sp.source != "file":
            row("length")
            _, sp.duration_s = imgui.slider_float("##duration", sp.duration_s, 0.5, 120.0, "%.1f s",
                                                  imgui.SliderFlags_.logarithmic)
            row("noise", "Shot noise; 1 is about what the recordings have")
            _, sp.noise_sd = imgui.slider_float("##noise", sp.noise_sd, 0.0, 3.0, "%.2f x")
        imgui.end_table()
        sp.fs = max(sp.fs, 1.0)
        idl.push_button_style(style.THEME, primary=False)
        if sp.source == "file":
            if imgui.button(f"{fa.ICON_FA_FOLDER_OPEN}  Browse...") and self.picker is None:
                self.picker = pfd.open_file("Open a trace", sp.path or ".", ["traces", "*.npy *.csv *.txt *.tsv", "all", "*"])
        elif imgui.button(f"{fa.ICON_FA_DICE}  New noise"):
            sp.seed += 1
        idl.pop_button_style()

    def controls_recorded(self) -> None:
        sp = self.settings.signal
        section("Scan")
        if self.scans is None:
            self.scans = find_scans()
            if not sp.scan_path and self.scans:
                sp.scan_path = self.scans[0]
        names = [Path(f).parent.parent.name + " / " + Path(f).stem for f in self.scans]
        current = self.scans.index(sp.scan_path) if sp.scan_path in self.scans else -1
        imgui.set_next_item_width(-1)
        changed, i = imgui.combo("##scan", current, names)
        if changed:
            sp.scan_path = self.scans[i]
        idl.push_button_style(style.THEME, primary=False)
        if imgui.button(f"{fa.ICON_FA_FOLDER_OPEN}  Browse...") and self.picker is None:
            self.picker = pfd.open_file("Open an exported line scan", sp.scan_path or ".", ["line scans", "*.npz"])
        idl.pop_button_style()
        if not sp.scan_path:
            return
        try:
            info = scan_info(sp.scan_path)
        except ValueError as e:
            idl.text_wrapped_colored(ERROR, str(e))
            return
        fs, total = info["fs"], info["frames"] / info["fs"]
        imgui.text_colored(DIM, f"{info['unit']}, {fs:.0f} Hz, {total:.0f} s")
        section("Lines in the domain")
        chosen = set(sp.rois)
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        for k, (roi, width) in enumerate(sorted(info["widths"].items())):
            label = f"{roi} ({width} px)"
            if k:
                imgui.same_line()
                need = imgui.get_frame_height() + imgui.get_style().item_inner_spacing.x + imgui.calc_text_size(label).x
                if imgui.get_cursor_pos_x() + need > right:
                    imgui.new_line()
            changed, on = imgui.checkbox(f"{label}##roi{roi}", roi in chosen)
            if changed:
                chosen = chosen | {roi} if on else chosen - {roi}
        if chosen:
            sp.rois = tuple(sorted(chosen))
        section("Window")
        sp.duration_s = float(min(sp.duration_s, total))
        if begin_form("window"):
            row("length")
            _, sp.duration_s = imgui.slider_float("##length", sp.duration_s, 1.0, total, "%.1f s",
                                                  imgui.SliderFlags_.logarithmic)
            row("start")
            _, sp.start_s = imgui.slider_float("##start", sp.start_s, 0.0, max(total - sp.duration_s, 0.0), "%.1f s")
            imgui.end_table()

    def page_signal(self) -> None:
        sig = self.run.signal
        if sig.lines is not None:
            self.draw_line_scan()
            return
        if self.begin_time_plot("##trace", self.plot_height(0.62), "z"):
            self.line("input", sig.x, P.INPUT)
            if sig.truth is not None:
                self.line("noiseless", sig.truth, P.NOISELESS, 1.5)
            implot.end_plot()
        run = self.run
        if run.psd and self.begin_spectrum("##spectrum", imgui.ImVec2(-1, -1)):
            implot.plot_line("input", run.psd["f"], run.psd["raw"], P.spec(P.INPUT))
            if "truth" in run.psd:
                implot.plot_line("noiseless", run.psd["f"], run.psd["truth"], P.spec(P.NOISELESS, 1.5))
            P.vlines("fs / 2", [sig.fs / 2], P.DIM)
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
        implot.push_colormap(self.colormaps["scan"])
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
        if self.begin_time_plot("##F", h, "F (counts)", x_axis=False):
            for k, mean in enumerate(ls.line_means):
                self.line(ls.names[k] if k < len(ls.names) else f"line {k + 1}", mean, P.band_color(k, 0.6))
            self.line("all pixels", ls.f, P.TEXT, 1.5)
            if clean:
                self.line("noiseless", ls.clean_f, P.NOISELESS, 1.5)
            self.line(f"baseline ({ls.sigma_s[0]:.1f} s)", ls.f_baseline, P.BASELINE, 2)
            implot.end_plot()
        if self.begin_time_plot("##dfof", h, "dF/F", x_axis=False):
            self.line(name, ls.dfof, P.INPUT)
            if clean:
                self.line("noiseless", ls.clean_dfof, P.NOISELESS, 1.5)
            self.line(f"baseline ({ls.sigma_s[1]:.1f} s)", ls.dfof_baseline, P.BASELINE, 2)
            implot.end_plot()
        if self.begin_time_plot("##z", -1, "z"):
            self.line(name, sig.x, P.INPUT)
            if sig.truth is not None:
                self.line("noiseless", sig.truth, P.NOISELESS, 1.5)
            implot.end_plot()

    # ------------------------------------------------------------------
    # Scalogram
    # ------------------------------------------------------------------

    def controls_scalogram(self) -> None:
        section("Scalogram")
        if not begin_form("scalogram"):
            return
        row("color", "amplitude: |row|. amplitude / median: each row over its own median, so every row's usual "
                     "level looks the same. standardized: each complex row minus its mean, over its SD. real part: "
                     "the filtered trace itself.")
        _, self.heat_mode = imgui.combo("##heat", self.heat_mode, HEAT_MODES)
        n = self.run.coeff.shape[0] if self.has("cwt") else int(self.settings.n_scales)
        row("row", "Or drag the white line on the scalogram")
        freq = f"  ({self.run.freqs[self.row]:.3g} Hz)" if self.has("cwt") else ""
        _, self.row = imgui.slider_int("##row", self.row, 0, n - 1, f"%d{freq}")
        self.row = int(np.clip(self.row, 0, n - 1))
        imgui.end_table()

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
        h = self.plot_height(0.62)
        implot.push_colormap(self.colormaps["diverging" if self.heat_mode == 3 else "sequential"])
        if self.begin_time_plot("##scalogram", h, "frequency (Hz)", implot.Flags_.no_legend, y_fit=False, x_axis=False):
            implot.setup_axis_limits(implot.ImAxis_.y1, bottom, top, implot.Cond_.always)
            P.hz_ticks(freqs.min(), freqs.max())
            implot.plot_heatmap("##W", m, lo, hi, "", implot.Point(ta, bottom), implot.Point(tb, top))
            # band edges, once the clustering has run on this transform
            if run.bands and run.bands[0].rows.size == len(freqs):
                for below, above in zip(run.bands, run.bands[1:]):
                    P.hline("##edge", math.log10(math.sqrt(below.hi_hz * above.lo_hz)), P.faded(P.TEXT, 0.4))
            if sig.fs / 2 < freqs.max():
                P.hline("##nyquist", math.log10(sig.fs / 2), P.WARN)
            changed, y, *_ = implot.drag_line_y(1, float(log_f[self.row]), P.TEXT, 1.5)
            if changed:
                self.row = int(np.argmin(np.abs(log_f - y)))
            if implot.is_plot_hovered():
                mouse = implot.get_plot_mouse_pos()
                k = int(np.argmin(np.abs(log_f - mouse.y)))
                idl.wrapped_tooltip(f"row {k}, {freqs[k]:.3g} Hz, at {mouse.x:.3f} s")
            implot.end_plot()
        imgui.same_line()
        implot.colormap_scale("##scale", lo, hi, imgui.ImVec2(4 * em, h))
        implot.pop_colormap()
        k = self.row
        caption("Below: the trace, the selected row (the part of the trace moving at that speed), and its envelope, "
                "the row's strength at each moment, traced over the row's peaks.")
        root = math.sqrt(run.scales[k])
        if self.begin_time_plot("##row", -1):
            self.line("input", sig.x, P.INPUT)
            self.line(f"row {k}, {freqs[k]:.3g} Hz", run.coeff[k].real / root, P.ACCENT, 1.5)
            self.line("envelope", run.amp[k], P.LEVEL_UP, 1.0)
            implot.end_plot()

    # ------------------------------------------------------------------
    # Bands
    # ------------------------------------------------------------------

    def controls_bands(self) -> None:
        c = self.settings.clust
        n_rows = int(self.settings.n_scales)
        section("Grouping")
        if begin_form("bands"):
            row("standardize", "Each row minus its mean, over its SD, before the PCA")
            _, c.standardize = imgui.checkbox("##standardize", c.standardize)
            row("PCA", "Principal components computed from the rows")
            _, c.n_components = imgui.slider_int("##pca", c.n_components, 2, min(60, n_rows), "%d components")
            row("clustered on", "How many of those components the clustering uses")
            _, c.n_comp_clu = imgui.slider_int("##pcs", c.n_comp_clu, 1, c.n_components, "first %d")
            methods = ["ward", "average", "complete", "single"]
            row("linkage")
            changed, i = imgui.combo("##linkage", methods.index(c.method_linkage) if c.method_linkage in methods else 0, methods)
            if changed:
                c.method_linkage = methods[i]
            row("bands", "How many bands the tree is cut into, at most")
            _, c.n_subclusters = imgui.slider_int("##n_bands", c.n_subclusters, 2, min(30, n_rows - 1))
            imgui.end_table()
        c.n_comp_clu = min(c.n_comp_clu, c.n_components)
        if not self.has("reduce"):
            return
        section("Bands")
        order = self.band_order()
        if imgui.begin_table("##band_table", 4, imgui.TableFlags_.row_bg | imgui.TableFlags_.sizing_fixed_fit):
            for col in ("", "Hz", "rows", "events"):
                imgui.table_setup_column(col)
            imgui.table_headers_row()
            for band in self.run.bands:
                imgui.table_next_row()
                imgui.table_next_column()
                swatch(f"c{band.id}", P.band_color(order[band.id]))
                imgui.table_next_column()
                imgui.text(band.span)
                imgui.table_next_column()
                imgui.text(str(int(band.rows.sum())))
                imgui.table_next_column()
                if band.kept:
                    imgui.text(str(len(band.windows)))
                else:
                    imgui.text_colored(WARN, "0")
            imgui.end_table()

    def page_bands(self) -> None:
        if not self.has("cluster"):
            imgui.text_disabled("The grouping has not run.")
            return
        run = self.run
        order = self.band_order()
        freqs = run.freqs
        caption("Each colored block is a band, covering the rows (dots) it groups; faded blocks are dropped. The white "
                "line is the gain of the kept bands summed: 1 passes a frequency unchanged, 0 removes it.")
        h1 = self.plot_height(0.28)
        title = f"{freqs.size} rows in {len(run.bands)} bands"
        if P.begin_plot(title, imgui.ImVec2(-1, h1), implot.Flags_.no_legend):
            implot.setup_axes("frequency (Hz)", "", implot.AxisFlags_.auto_fit, implot.AxisFlags_.no_tick_labels | implot.AxisFlags_.lock)
            implot.setup_axis_limits(implot.ImAxis_.y1, 0, 1.05, implot.Cond_.always)
            implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
            implot.setup_axis(implot.ImAxis_.y2, "gain of the sum", implot.AxisFlags_.auto_fit | implot.AxisFlags_.opposite)
            ratio = math.sqrt(freqs[0] / freqs[1]) if len(freqs) > 1 else 1.0
            for band in run.bands:
                f = freqs[band.rows]
                xs = np.array([f.min() / ratio, f.max() * ratio])
                c = P.band_color(order[band.id])
                implot.plot_shaded(f"##{band.id}", xs, np.ones(2), 0.0, P.spec(c, fill_alpha=0.45 if band.kept else 0.12))
                implot.plot_scatter(f"##rows{band.id}", f, np.full(f.size, 0.5), P.marker(c, 3.5, P.faded(P.TEXT, 0.8)))
            P.vlines("##nyquist", [run.signal.fs / 2], P.WARN)
            if self.has("reduce"):
                fx = np.logspace(np.log10(freqs.min()), np.log10(min(freqs.max(), run.signal.fs / 2)), 500)
                implot.set_axes(implot.ImAxis_.x1, implot.ImAxis_.y2)
                implot.plot_line("##gain", fx, band_sum_gain(run, fx), P.spec(P.TEXT, 2))
            implot.end_plot()
        spectrum = run.psd if self.has("final") else None
        if spectrum:
            caption("Power at each frequency in the trace and the denoised trace, with each band shaded behind the "
                    "frequencies it covers.")
        h2 = imgui.get_content_region_avail().y * 0.55
        if spectrum and self.begin_spectrum("##spectrum", imgui.ImVec2(-1, h2)):
            f = spectrum["f"]
            top = float(max(spectrum["raw"].max(), spectrum.get("truth", spectrum["raw"]).max())) * 2
            low = float(min(spectrum["raw"].min(), spectrum["denoised"].min() + 1e-30))
            ratio = math.sqrt(freqs[0] / freqs[1]) if len(freqs) > 1 else 1.0
            for band in run.bands:
                fb = freqs[band.rows]
                xs = np.array([fb.min() / ratio, min(fb.max() * ratio, f.max())])
                implot.plot_shaded(f"##bg{band.id}", xs, np.full(2, top), low,
                                   P.spec(P.band_color(order[band.id]), fill_alpha=0.18 if band.kept else 0.05))
            implot.plot_line("input", f, spectrum["raw"], P.spec(P.INPUT, 1.5))
            if "truth" in spectrum:
                implot.plot_line("noiseless", f, spectrum["truth"], P.spec(P.NOISELESS, 1.5))
            implot.plot_line("denoised", f, spectrum["denoised"] + 1e-30, P.spec(P.DENOISED, 1.5))
            edge = noise_floor_hz(spectrum, run.signal.fs)
            if edge is not None:
                P.vlines(f"noise floor from {edge:.0f} Hz", [edge], P.DIM, 1.5)
            implot.end_plot()
        if self.has("reduce"):
            caption("How much of the recording each band is kept for; the number on a bar is its event windows.")
        if self.has("reduce") and P.begin_plot("##kept", imgui.ImVec2(-1, -1), implot.Flags_.no_legend):
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
    # Masks
    # ------------------------------------------------------------------

    def controls_events(self) -> None:
        r, t = self.settings.reducer, self.settings.thres
        if self.has("reduce"):
            section("Band")
            order = self.band_order()
            for band in reversed(self.run.bands):
                swatch(f"b{band.id}", P.band_color(order[band.id]))
                label = band.span if band.kept else f"{band.span}, dropped"
                if imgui.selectable(f"{label}##{band.id}", self.band_id == band.id)[0]:
                    self.band_id = band.id
        section("Events")
        if begin_form("events"):
            row("under 30 Hz", "Where an event opens, in SDs of the band")
            _, r.slow_upthres = imgui.slider_float("##slow", r.slow_upthres, 0.5, 6.0, "%.2f SD")
            row("30 Hz and up")
            _, r.fast_upthres = imgui.slider_float("##fast", r.fast_upthres, 0.5, 6.0, "%.2f SD")
            row("smoothing", "savgol windows in samples, for bands at or below 75 Hz, slowest band first")
            changed, windows = imgui.input_int3("##smoothing", list(r.smooth_windows))
            if changed:
                r.smooth_windows = tuple(max(3, int(w)) for w in windows)
            row("complex", "Bands above 75 Hz keep their complex mean: its real part is what's added back")
            _, r.complex_bands = imgui.checkbox("above 75 Hz##complex", r.complex_bands)
            imgui.end_table()
        section("Between events")
        if begin_form("mask"):
            kinds = ["soft", "hard"]
            row("type", "soft: a floor per band speed. hard: one floor for every band")
            changed, i = imgui.combo("##type", kinds.index(t.thres_type), kinds)
            if changed:
                t.thres_type = kinds[i]
            if t.thres_type == "soft":
                row("floor", "Under 5 Hz, under 30, under 80, 80 and up")
                changed, levels = imgui.slider_float4("##levels", list(t.soft_levels), 0.0, 1.0, "%.2f")
                if changed:
                    t.soft_levels = tuple(levels)
                row("blur", "Gaussian SD of the mask in samples: up to 120 Hz, above")
                changed, sigmas = imgui.input_float2("##blur", list(t.soft_sigma), "%.1f")
                if changed:
                    t.soft_sigma = tuple(max(0.1, v) for v in sigmas)
            else:
                row("floor")
                _, t.hard_floor = imgui.slider_float("##floor", t.hard_floor, 0.0, 1.0, "%.3f")
                row("blur", "Gaussian SD of the mask in samples: up to 120 Hz, above")
                changed, sigmas = imgui.input_float2("##blur", list(t.hard_sigma), "%.1f")
                if changed:
                    t.hard_sigma = tuple(max(0.1, v) for v in sigmas)
            imgui.end_table()

    def page_events(self) -> None:
        if not self.has("reduce"):
            imgui.text_disabled("The bands have not run.")
            return
        run = self.run
        band = run.band(self.band_id)
        if band is None:
            return
        fs = run.signal.fs
        color = P.band_color(self.band_order()[band.id])
        gap = imgui.get_style().item_spacing.y
        h = max(5 * imgui.get_font_size(), (imgui.get_content_region_avail().y - 4 * gap) * 0.17)
        if self.begin_time_plot("##rows", h, x_axis=False):
            for k in np.flatnonzero(band.rows):
                self.line(f"{int(band.rows.sum())} rows", run.coeff[k].real, P.faded(P.TEXT, 0.35))
            self.line(f"{band.span}, their mean", band.average.real, color, 2)
            implot.end_plot()
        if self.begin_time_plot("##levels", h, x_axis=False):
            if band.kept and band.extra:
                wide = np.clip(band.windows + np.array([-band.extra, band.extra]), 0, run.signal.n)
                P.shade_intervals(wide, fs, self.t0.value, self.t1.value, P.faded(color, 0.10))
            P.shade_intervals(band.windows, fs, self.t0.value, self.t1.value, P.faded(color, 0.22))
            if band.smooth_window:
                self.line("mean", band.average.real, P.faded(P.TEXT, 0.35))
            self.line("smoothed" if band.smooth_window else "mean", band.smoothed.real, color, 1.5)
            P.hline(f"opens {band.up:.2f}", band.up, P.LEVEL_UP, 1.5)
            P.hline(f"closes {band.down:.2f}", band.down, P.LEVEL_DOWN, 1.0)
            implot.end_plot()
        if not band.kept:
            imgui.text_colored(WARN, f"{fa.ICON_FA_CIRCLE_INFO}  never reaches {band.up:.2f}: left out of the sum")
        elif band.mask is not None:
            if self.begin_time_plot("##mask", h, y_fit=False, x_axis=False):
                implot.setup_axis_limits(implot.ImAxis_.y1, -0.05, 1.08, implot.Cond_.always)
                self.line("before the blur", band.label, P.faded(P.TEXT, 0.5))
                self.line("mask", band.mask, color, 2)
                implot.end_plot()
            if self.begin_time_plot("##masked", h, x_axis=False):
                self.line("band", band.scaled.real, P.faded(P.TEXT, 0.35))
                self.line("band x mask", band.masked.real, color, 1.5)
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
        implot.push_colormap(self.colormaps["sequential"])
        em = imgui.get_font_size()
        h = max(imgui.get_content_region_avail().y, (nb + 3) * em)
        if self.begin_time_plot("##masks", h, flags=implot.Flags_.no_legend, y_fit=False):
            implot.setup_axis_limits(implot.ImAxis_.y1, 0, nb, implot.Cond_.always)
            implot.setup_axis_ticks(implot.ImAxis_.y1, [nb - k - 0.5 for k in range(nb)], [b.span for b in kept], False)
            implot.plot_heatmap("##all_masks", np.ascontiguousarray(m, dtype=np.float64), 0.0, 1.0, "",
                                implot.Point(i0 / fs, 0), implot.Point((i0 + max(whole, 1)) / fs, nb))
            implot.end_plot()
        implot.pop_colormap()

    # ------------------------------------------------------------------
    # Denoised
    # ------------------------------------------------------------------

    def controls_final(self) -> None:
        s = self.settings
        section("Slow part")
        if begin_form("final"):
            row("add", "The input below the cutoff, through a zero-phase FIR low-pass, added to the bands")
            _, s.add_baseline = imgui.checkbox("##add", s.add_baseline)
            row("below")
            _, s.lp_cutoff = imgui.slider_float("##cutoff", s.lp_cutoff, 0.05, 20.0, "%.3g Hz", imgui.SliderFlags_.logarithmic)
            imgui.end_table()
        if not self.has("mask"):
            return
        section("Bands in the sum")
        order = self.band_order()
        excluded = set(s.excluded)
        for band in reversed(self.run.bands):
            if not band.kept:
                imgui.text_disabled(f"{band.span}, dropped")
                continue
            imgui.push_style_color(imgui.Col_.check_mark, P.band_color(order[band.id]))
            changed, on = imgui.checkbox(f"{band.span}##inc{band.id}", band.id not in excluded)
            imgui.pop_style_color()
            if changed:
                excluded = excluded - {band.id} if on else excluded | {band.id}
        s.excluded = tuple(sorted(excluded))

    def page_final(self) -> None:
        if not self.has("final"):
            imgui.text_disabled("The sum has not run.")
            return
        run = self.run
        sig = run.signal
        order = self.band_order()
        excluded = set(run.settings.excluded)
        kept = [band for band in run.bands if band.kept and band.masked is not None]
        h = self.plot_height(0.33)
        spread = max((np.percentile(np.abs(b.masked.real), 99.5) for b in kept), default=1.0) * 2.2 or 1.0
        if self.begin_time_plot("##stack", h, "bands", implot.Flags_.no_legend, y_fit=False, x_axis=False):
            implot.setup_axis(implot.ImAxis_.y1, "bands", implot.AxisFlags_.no_tick_labels)
            implot.setup_axis_limits(implot.ImAxis_.y1, -0.7 * spread, (len(kept) - 0.2) * spread, implot.Cond_.always)
            for k, band in enumerate(kept):
                alpha = 0.25 if band.id in excluded else 1.0
                self.line(band.span, band.masked.real + k * spread, P.band_color(order[band.id], alpha), 1.2)
                implot.plot_text(band.span, self.t0.value, k * spread, imgui.ImVec2(3 * imgui.get_font_size(), -0.6 * imgui.get_font_size()))
            implot.end_plot()
        if self.begin_time_plot("##sum", h, "z"):
            self.line("input", sig.x, P.INPUT)
            if sig.truth is not None:
                self.line("noiseless", sig.truth, P.NOISELESS, 1.5)
            if run.settings.add_baseline:
                self.line("slow part", run.baseline, P.BASELINE, 1.0)
            self.line("denoised", run.denoised, P.DENOISED, 1.8)
            implot.end_plot()
        avail = imgui.get_content_region_avail()
        half = (avail.x - imgui.get_style().item_spacing.x) / 2
        if P.begin_plot("##removed", imgui.ImVec2(half, -1)):
            implot.setup_axes("time (s)", "", 0, implot.AxisFlags_.auto_fit | implot.AxisFlags_.range_fit)
            implot.setup_axis_links(implot.ImAxis_.x1, self.t0, self.t1)
            self.line("input - denoised", sig.x - run.denoised, P.REMOVED)
            implot.end_plot()
        imgui.same_line()
        if run.psd and self.begin_spectrum("##final_spectrum", imgui.ImVec2(-1, -1)):
            f = run.psd["f"]
            implot.plot_line("input", f, run.psd["raw"], P.spec(P.INPUT))
            if "truth" in run.psd:
                implot.plot_line("noiseless", f, run.psd["truth"], P.spec(P.NOISELESS, 1.5))
            implot.plot_line("denoised", f, run.psd["denoised"] + 1e-30, P.spec(P.DENOISED, 1.8))
            implot.end_plot()


def runner_params(app: App, hidden: bool = False) -> hello_imgui.RunnerParams:
    idl.ensure_assets()
    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = TITLE_WINDOW
    params.app_window_params.window_geometry.size = tuple(int(v * min(app.scale, 1.5)) for v in SIZE_WINDOW)
    params.app_window_params.hidden = hidden
    params.imgui_window_params.tweaked_theme.theme = hello_imgui.ImGuiTheme_.darcula_darker
    params.callbacks.default_icon_font = hello_imgui.DefaultIconFont.font_awesome6  # hello_imgui loads 4 otherwise
    params.ini_disable = True
    params.fps_idling.fps_idle = 20
    app.dialog.apply_host_theme(params)
    params.callbacks.show_gui = app.gui
    return params


def screenshot_gui(app: App, frames: int = 4):
    """A show_gui that waits for the run, draws frames more, then closes the window."""
    drawn = [0]

    def gui() -> None:
        app.gui()
        if app.future is not None or not app.ready:
            app.wait()
            return
        drawn[0] += 1
        if drawn[0] > frames:
            app.quit()

    return gui


def run() -> None:
    """Open the window and block until it is closed."""
    parser = argparse.ArgumentParser(prog="imgui-vnoiser", description="See what vnoiser's wavelet denoiser does to a trace.")
    parser.add_argument("--scale", type=float, default=0.0, help="UI scale, e.g. 2 on a 2x HiDPI panel (default: saved, then detected)")
    parser.add_argument("--scan", default="", help="a line scan .npz from scripts/export_mesc_linescan.py")
    parser.add_argument("--lines", type=int, nargs="+", default=[0], help="with --scan: the lines averaged into F")
    parser.add_argument("--page", default="signal", choices=[key for key, *_ in PAGES])
    parser.add_argument("--screenshot", default="", help="draw the page in a hidden window, save this PNG, exit")
    args = parser.parse_args()
    app = App(scale=config.ui_scale(args.scale))
    app.page = args.page
    if args.scan:
        sp = app.settings.signal
        sp.source, sp.scan_path, sp.rois = "recorded", str(Path(args.scan).expanduser()), tuple(args.lines)
    params = runner_params(app, hidden=bool(args.screenshot))
    if args.screenshot:
        params.callbacks.show_gui = screenshot_gui(app)
    immapp.run(params, immapp.AddOnsParams(with_implot=True))
    app.executor.shutdown(wait=False, cancel_futures=True)
    if args.screenshot:
        from PIL import Image  # only screenshots need it: pip install pillow

        image = np.asarray(hello_imgui.final_app_window_screenshot())
        Image.fromarray(np.ascontiguousarray(image[..., :3])).save(args.screenshot)
        print(f"{args.screenshot}: {app.page}")


if __name__ == "__main__":
    run()
