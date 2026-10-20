"""
vnoiser.Denoiser.run, split into the stages the window shows, each kept so a change reruns only what it affects.

The stages call vnoiser's own classes (FrequencyClusterer, WaveletReducer, AdaptiveThreshold, fir_lowpass), so what
the window shows is what the package does. What those classes keep to themselves (a band's trace before smoothing,
its open and close levels, a mask before its Gaussian blur) is recomputed beside them in `band_details` and
`mask_details`; tests/test_pipeline.py checks they agree with what the classes return.
"""

import copy
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, welch
from vnoiser import (
    AdaptiveThreshold,
    ClusteringConfig,
    FrequencyClusterer,
    WaveletReducer,
    cwtReducerConfig,
    fir_lowpass,
    thresConfig,
    trigger_detect,
)
from vnoiser.denoiser import UPSTREAM_SOFT_LEVELS, _normalize_crossing_area

from imgui_vnoiser import morlet
from imgui_vnoiser.signals import Signal, SignalParams, make_signal

# in run order; a change to a stage's settings reruns it and every stage after it
STAGES = ("signal", "cwt", "cluster", "reduce", "mask", "final")


@dataclass
class Settings:
    """Every parameter of the run, grouped by the stage it first affects."""

    signal: SignalParams = field(default_factory=SignalParams)
    # cwt
    b: float = 0.5
    c: float = 1.0
    scale_min: float = 1.0
    scale_max: float = 1000.0
    n_scales: int = 100
    # cluster, reduce, mask: vnoiser's own configs
    clust: ClusteringConfig = field(default_factory=ClusteringConfig)
    reducer: cwtReducerConfig = field(default_factory=cwtReducerConfig)
    thres: thresConfig = field(default_factory=lambda: thresConfig(thres_type="soft"))
    # final
    lp_cutoff: float = 1.0
    fir_window_ms: float = 2000.0
    fir_odd_taps: bool = True
    add_baseline: bool = True
    excluded: tuple = ()  # band ids left out of the sum

    @property
    def scales(self) -> np.ndarray:
        return np.logspace(np.log10(self.scale_min), np.log10(self.scale_max), int(self.n_scales))

    def stage_groups(self) -> dict:
        """The values each stage reads, for spotting which stage a change starts at."""
        return {
            "signal": asdict(self.signal),
            "cwt": (self.b, self.c, self.scale_min, self.scale_max, int(self.n_scales)),
            "cluster": asdict(self.clust),
            "reduce": asdict(self.reducer),
            "mask": asdict(self.thres),
            "final": (self.lp_cutoff, self.fir_window_ms, self.fir_odd_taps, self.add_baseline, tuple(self.excluded)),
        }

    def first_change(self, other: Optional["Settings"]) -> Optional[str]:
        """The earliest stage whose values differ from other's, or None when they are the same."""
        if other is None:
            return STAGES[0]
        mine, theirs = self.stage_groups(), other.stage_groups()
        return next((stage for stage in STAGES if mine[stage] != theirs[stage]), None)

    def copy(self) -> "Settings":
        return copy.deepcopy(self)

    def use_upstream(self) -> None:
        """Denoiser.upstream: the lab's original settings (complex fast bands, lab soft levels, even taps)."""
        self.reducer.complex_bands = True
        self.thres = thresConfig(thres_type="soft", soft_levels=UPSTREAM_SOFT_LEVELS)
        self.fir_odd_taps = False

    def use_defaults(self) -> None:
        """Denoiser's constructor defaults, keeping the signal."""
        signal = self.signal
        for f in fields(Settings):
            setattr(self, f.name, getattr(Settings(), f.name))
        self.signal = signal

    def python(self) -> str:
        """The Denoiser call that reproduces this run outside the window."""
        def cfg(obj) -> str:
            args = ", ".join(f"{k}={v!r}" for k, v in asdict(obj).items() if getattr(type(obj)(), k) != v)
            return f"{type(obj).__name__}({args})"

        thres = asdict(self.thres)
        thres_default = asdict(thresConfig(thres_type="soft"))
        thres_args = ", ".join(f"{k}={v!r}" for k, v in thres.items() if k == "thres_type" or thres_default[k] != v)
        return "\n".join((
            "import numpy as np",
            "from vnoiser import Denoiser, ClusteringConfig, cwtReducerConfig, thresConfig",
            "",
            "denoiser = Denoiser(",
            f"    fs={self.signal.fs!r},",
            f"    wavelet={morlet.name(self.b, self.c)!r},",
            f"    freq_scales=np.logspace(np.log10({self.scale_min!r}), np.log10({self.scale_max!r}), num={int(self.n_scales)}),",
            f"    lp_cutoff={self.lp_cutoff!r},",
            f"    fir_window_ms={self.fir_window_ms!r},",
            f"    fir_odd_taps={self.fir_odd_taps!r},",
            f"    cfg_clust={cfg(self.clust)},",
            f"    cfg_reducer={cfg(self.reducer)},",
            f"    cfg_thres=thresConfig({thres_args}),",
            ")",
            "denoised, event_starts = denoiser.run(trace)",
        ))


@dataclass
class Band:
    """One frequency band (a `sublabels` cluster) and what WaveletReducer does with it."""

    id: int
    position: int  # its place in np.unique(sublabels): the i in vnoiser's "clu{i}_{max f}Hz"
    lo_hz: float
    hi_hz: float
    rows: np.ndarray  # which CWT rows belong to it
    mean_scale: float
    average: np.ndarray  # mean of its rows (real part unless complex_bands keeps it complex)
    smoothed: np.ndarray  # what the levels apply to; average itself above 75 Hz
    smooth_window: int  # savgol window in samples, 0 when not smoothed
    up: float  # open level
    down: float  # close level
    windows: np.ndarray  # (n, 2) start and end samples
    kept: bool  # whether it had a window, and so stays in the sum
    extra: int = 0  # samples each window is widened by on both sides, once masked
    sigma: float = 0  # Gaussian blur of the mask in samples
    floor: float = 0  # mask value outside windows
    label: Optional[np.ndarray] = None  # the mask before the blur
    mask: Optional[np.ndarray] = None  # after
    scaled: Optional[np.ndarray] = None  # smoothed / sqrt(mean scale), before the mask
    masked: Optional[np.ndarray] = None  # after: what the band adds to the sum

    @property
    def name(self) -> str:
        return f"clu{self.position}_{np.around(self.hi_hz, 1)}Hz"

    @property
    def span(self) -> str:
        return f"{fmt_hz(self.lo_hz)} to {fmt_hz(self.hi_hz)} Hz"


@dataclass
class Run:
    """Every stage's output for one settings; stages past `done` are None."""

    settings: Settings
    signal: Optional[Signal] = None
    error: str = ""
    failed: str = ""  # the stage that raised
    timing: dict = field(default_factory=dict)
    # cwt
    coeff: Optional[np.ndarray] = None
    freqs: Optional[np.ndarray] = None
    scales: Optional[np.ndarray] = None
    amp: Optional[np.ndarray] = None  # |coeff| / sqrt(scale), float32
    # cluster
    cluster: Optional[dict] = None
    # reduce
    bands: list = field(default_factory=list)
    reduced: Optional[dict] = None
    # mask
    thres: Optional[dict] = None
    # final
    baseline: Optional[np.ndarray] = None
    wavelet_sum: Optional[np.ndarray] = None
    denoised: Optional[np.ndarray] = None
    psd: Optional[dict] = None

    @property
    def kept(self) -> list:
        return [band for band in self.bands if band.kept]

    def band(self, band_id: int) -> Optional[Band]:
        return next((band for band in self.bands if band.id == band_id), None)


def fmt_hz(f: float) -> str:
    return f"{f:.0f}" if f >= 10 else f"{f:.2g}" if f < 1 else f"{f:.1f}"


def compute(settings: Settings, start: str = "signal", previous: Optional[Run] = None, log=None) -> Run:
    """
    Run every stage from start on, taking earlier stages from previous.

    Args:
        settings (Settings): The parameters
        start (str): The first stage to compute; earlier ones are copied from previous
        previous (Run | None): The run to reuse; None computes everything
        log (callable | None): Called with each stage's name before it runs

    Returns:
        Run: With `error` set (and later stages None) when a stage raised
    """
    s = settings.copy()
    if previous is not None and previous.failed:
        start = min(start, previous.failed, key=STAGES.index)
    if previous is None or previous.signal is None:
        start = "signal"
    run = Run(settings=s)
    if start != "signal":
        reuse = STAGES.index(start)
        run.signal, run.timing = previous.signal, {k: v for k, v in previous.timing.items() if STAGES.index(k) < reuse}
        if reuse > 1:
            run.coeff, run.freqs, run.scales, run.amp = previous.coeff, previous.freqs, previous.scales, previous.amp
        if reuse > 2:
            run.cluster = previous.cluster
        if reuse > 3:
            run.bands, run.reduced = [copy.copy(band) for band in previous.bands], previous.reduced
        if reuse > 4:
            run.thres = previous.thres
    for stage in STAGES[STAGES.index(start):]:
        if log:
            log(stage)
        t0 = time.perf_counter()
        try:
            STEPS[stage](run)
        except Exception as e:  # shown in the window; the run keeps what finished
            run.error, run.failed = f"{stage}: {e}", stage
            return run
        run.timing[stage] = time.perf_counter() - t0
    return run


def step_signal(run: Run) -> None:
    run.signal = make_signal(run.settings.signal)


def step_cwt(run: Run) -> None:
    s, x = run.settings, run.signal.x
    if not 0 < s.scale_min < s.scale_max:
        raise ValueError("need 0 < smallest scale < largest scale")
    run.scales = s.scales
    run.coeff, run.freqs = morlet.cwt(x, run.scales, run.signal.fs, s.b, s.c)
    root = np.sqrt(run.scales)[:, None]
    run.amp = (np.abs(run.coeff) / root).astype(np.float32)


def step_cluster(run: Run) -> None:
    clust = run.settings.clust
    n_rows = run.coeff.shape[0]
    if clust.n_components > min(n_rows, run.signal.n):
        raise ValueError(f"PCA components ({clust.n_components}) above the number of rows ({n_rows})")
    if clust.n_comp_clu > clust.n_components:
        raise ValueError(f"components clustered ({clust.n_comp_clu}) above PCA components ({clust.n_components})")
    run.cluster = FrequencyClusterer(run.coeff, run.freqs, config=clust).run()


def band_details(run: Run) -> list:
    """WaveletReducer.reduce_and_threshold one band at a time, keeping what it throws away."""
    reducer = WaveletReducer(run.cluster, run.coeff, run.freqs, run.scales, config=run.settings.reducer, fs=run.signal.fs)
    cfg = reducer.config
    slow_w, mid_w, fast_w = reducer.smoothing_windows()
    bands = []
    for i, band_id in enumerate(np.unique(run.cluster[cfg.cluster_label])):
        freq_range = run.cluster[cfg.cluster_freq][band_id]
        rows = np.isin(run.freqs, freq_range)
        if not np.any(rows):
            continue
        average = np.mean(run.coeff[rows, :], axis=0)
        if not cfg.complex_bands:
            average = average.real
        lo, hi = float(min(freq_range)), float(max(freq_range))
        window = slow_w if hi <= 15 else mid_w if hi <= 50 else fast_w if hi <= 75 else 0
        smoothed = average
        if window:
            peak = np.max(average.real)
            smoothed = savgol_filter(average.real, window_length=window, polyorder=1)
            peak_smoothed = np.max(smoothed)
            if peak_smoothed != 0:
                smoothed = smoothed * (peak / peak_smoothed)
        sd = np.std(smoothed)
        up, down = (sd * cfg.slow_upthres, sd) if hi < 30 else (sd * cfg.fast_upthres, 0.0)
        windows = _normalize_crossing_area(trigger_detect(smoothed, up, down)["Area"])
        bands.append(Band(
            id=int(band_id), position=i, lo_hz=lo, hi_hz=hi, rows=rows, mean_scale=float(np.mean(run.scales[rows])),
            average=average, smoothed=smoothed, smooth_window=window, up=float(up), down=float(down),
            windows=windows, kept=windows.size > 0,
        ))
    bands.sort(key=lambda band: band.lo_hz)
    return bands


def step_reduce(run: Run) -> None:
    run.bands = band_details(run)
    run.reduced = WaveletReducer(
        run.cluster, run.coeff, run.freqs, run.scales, config=run.settings.reducer, fs=run.signal.fs,
    ).reduce_and_threshold()


def mask_details(run: Run) -> None:
    """Fill each kept band's mask fields the way AdaptiveThreshold.run makes them."""
    cfg = run.settings.thres
    n_frame = run.signal.n
    for band in run.bands:
        if not band.kept:
            band.label = band.mask = band.scaled = band.masked = None
            continue
        f = band.hi_hz
        if cfg.thres_type == "soft":
            low, mid, high, top = cfg.soft_levels
            band.floor = low if f < 5 else mid if f < 30 else high if f < 80 else top
        else:
            band.floor = cfg.hard_floor
        median_iei = np.median(band.windows[:, 1] - band.windows[:, 0])
        band.extra = int(median_iei * (2.0 if f <= 20 else 1.5 if f <= 50 else 0.5))
        label = np.full(n_frame, float(band.floor))
        for st, ed in band.windows:
            label[max(st - band.extra, 0):min(ed + band.extra, n_frame)] = 1
        sigmas = cfg.soft_sigma if cfg.thres_type == "soft" else cfg.hard_sigma
        band.sigma = sigmas[0] if f <= 120 else sigmas[1]
        band.label = label
        band.mask = gaussian_filter1d(label, sigma=band.sigma)
        band.scaled = band.smoothed / np.sqrt(band.mean_scale)
        band.masked = band.scaled * band.mask


def step_mask(run: Run) -> None:
    mask_details(run)
    if run.kept:
        run.thres = AdaptiveThreshold(run.reduced, config=run.settings.thres).run()
    else:
        run.thres = None


def step_final(run: Run) -> None:
    s, sig = run.settings, run.signal
    run.baseline = fir_lowpass(sig.x, sig.fs, s.lp_cutoff, s.fir_window_ms, odd_taps=s.fir_odd_taps)
    total = np.zeros(sig.n)
    for band in run.kept:
        if band.id not in s.excluded:
            total = total + band.masked.real
    run.wavelet_sum = total
    run.denoised = total + run.baseline if s.add_baseline else total
    nper = int(min(sig.n, max(256, 2 ** int(np.log2(sig.fs * 2)))))
    psd = {}
    for key, y in (("raw", sig.x), ("denoised", run.denoised), ("truth", sig.truth)):
        if y is not None:
            f, p = welch(y, fs=sig.fs, nperseg=nper)
            psd[key] = np.ascontiguousarray(p[1:], dtype=np.float64)
            psd["f"] = np.ascontiguousarray(f[1:], dtype=np.float64)
    run.psd = psd


STEPS = {
    "signal": step_signal,
    "cwt": step_cwt,
    "cluster": step_cluster,
    "reduce": step_reduce,
    "mask": step_mask,
    "final": step_final,
}


def metrics(run: Run) -> dict:
    """How the denoised trace compares with the raw one, and with the ground truth when known."""
    sig, out = run.signal, {}
    if run.denoised is None:
        return out
    x, y = sig.x, run.denoised
    out["raw SD"] = float(np.std(x))
    out["removed SD"] = float(np.std(x - y))
    if sig.truth is not None:
        truth = sig.truth
        if not run.settings.add_baseline:
            truth = truth - fir_lowpass(truth, sig.fs, run.settings.lp_cutoff, run.settings.fir_window_ms)
        # compare shapes: the denoiser does not promise the input's offset or gain
        out["RMSE raw vs ground truth"] = float(np.sqrt(np.mean((x - x.mean() - (truth - truth.mean())) ** 2)))
        out["RMSE denoised vs ground truth"] = float(np.sqrt(np.mean((y - y.mean() - (truth - truth.mean())) ** 2)))
        out["corr raw vs ground truth"] = float(np.corrcoef(x, truth)[0, 1])
        out["corr denoised vs ground truth"] = float(np.corrcoef(y, truth)[0, 1]) if np.std(y) > 0 else 0.0
    return out


def band_sum_gain(run: Run, f_hz: np.ndarray, include_all: bool = False) -> np.ndarray:
    """
    Gain of the band sum for a unit sine at each f_hz with every mask at 1 and no smoothing: each band averages its
    rows (row r passes sqrt(scale_r) / 2 of its response) and divides by sqrt(its mean scale). The savgol smoothing of
    bands below 75 Hz lowers this further near their top; the masks lower it outside events.
    """
    s, gain = run.settings, np.zeros_like(np.asarray(f_hz, dtype=float))
    for band in run.bands if include_all else run.kept:
        if band.id in s.excluded and not include_all:
            continue
        rows = run.scales[band.rows]
        part = sum(np.sqrt(scale) * morlet.response(s.b, s.c, scale, run.signal.fs, f_hz) for scale in rows)
        gain += 0.5 * part / (rows.size * np.sqrt(band.mean_scale))
    return gain
