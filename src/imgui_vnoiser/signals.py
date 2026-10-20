"""
Input traces for the window: a simulated AOD line scan (pixels to z, the way vnoiser.preprocess does it) next to
the same scan without noise, a recorded line scan exported from a .mesc (scripts/export_mesc_linescan.py), or a
trace read from a file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

# the source key, its name in the source list, and a note
SOURCES = (
    ("lines", "Simulated line scan", "3 lines across one dendrite, raw counts as the .mesc stores them"),
    ("recorded", "Recorded line scan", "a .npz from scripts/export_mesc_linescan.py"),
    ("file", "From a file", ".npy, .csv, .txt or .tsv, one trace per column"),
)


@dataclass
class SignalParams:
    """Everything that picks and shapes the input trace."""

    source: str = "lines"
    fs: float = 1075.2688
    duration_s: float = 30.0
    noise_sd: float = 1.0  # shot noise, 1 = like the recordings
    seed: int = 7
    path: str = ""
    column: int = 0
    zscore: bool = True
    scan_path: str = ""  # recorded: the exported .npz
    rois: tuple = (0,)  # recorded: the lines averaged into one domain
    start_s: float = 0.0  # recorded: where the window starts


@dataclass
class LineScan:
    """
    The steps from pixels to the z-scored trace, each kept for plotting, for the noisy scan and the same scan
    without noise (`clean_*`).

    Args:
        pixels (np.ndarray): (pixels, samples), every line's pixels stacked, raw counts
        clean_pixels (np.ndarray | None): the same without shot noise or motion; None for a recording
        edges (list): (first, last + 1) pixel of each line in pixels
        line_means (np.ndarray): (lines, samples), each line's mean
        f (np.ndarray): the mean over every pixel
        f_baseline (np.ndarray): Gaussian-smoothed f
        dfof_raw (np.ndarray): (f - f_baseline) / f_baseline
        dfof (np.ndarray): dfof_raw with the first n_startup samples replaced by the mean of the rest
        dfof_baseline (np.ndarray): Gaussian-smoothed dfof
        n_startup (int): samples replaced
        clean_f, clean_dfof (np.ndarray | None): f and dfof of clean_pixels
    """

    pixels: np.ndarray
    clean_pixels: Optional[np.ndarray]
    edges: list
    line_means: np.ndarray
    f: np.ndarray
    f_baseline: np.ndarray
    dfof_raw: np.ndarray
    dfof: np.ndarray
    dfof_baseline: np.ndarray
    n_startup: int
    clean_f: Optional[np.ndarray] = None
    clean_dfof: Optional[np.ndarray] = None
    names: list = field(default_factory=list)  # each line's name in the legend
    sigma_s: tuple = (0.0, 0.0)  # the two Gaussian baselines' SDs in seconds: F's, then dF/F's


@dataclass
class Signal:
    """
    One trace in the units the denoiser expects (z-scored dF/F, depolarization positive).

    Args:
        name (str): What the plots call it
        fs (float): Sampling rate in Hz
        x (np.ndarray): The trace the denoiser runs on
        truth (np.ndarray | None): Ground truth in the same units, when known
        note (str): One line about where it came from
        columns (int): How many columns the source file had
    """

    name: str
    fs: float
    x: np.ndarray
    truth: Optional[np.ndarray] = None
    note: str = ""
    columns: int = 1
    events_s: dict = field(default_factory=dict)
    lines: Optional[LineScan] = None

    @property
    def n(self) -> int:
        return int(self.x.size)

    @property
    def duration(self) -> float:
        return self.n / self.fs

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n) / self.fs


def make_signal(p: SignalParams) -> Signal:
    """The trace p describes. Raises ValueError with a message for the window when it cannot be made."""
    if p.source == "file":
        return load_file(p.path, p.fs, p.column, p.zscore)
    if p.source == "recorded":
        return recorded_scan(p)
    if p.fs <= 0:
        raise ValueError("The sampling rate must be above 0 Hz.")
    n = int(round(p.duration_s * p.fs))
    if n < 64:
        raise ValueError("Need at least 64 samples; raise the duration or the sampling rate.")
    rng = np.random.default_rng(p.seed)
    if p.source != "lines":
        raise ValueError(f"Unknown source {p.source!r}.")
    return line_scan(p, rng)


# like the JEDI dendrite scans: 3 lines per domain, 10 or 20 pixels wide
LINE_WIDTHS = (10, 20, 10)
LINE_CENTERS = (4.0, 11.0, 5.5)  # where each line crosses the dendrite, in pixels
BACKGROUND = 115.0  # counts off the dendrite
DENDRITE = (45.0, 60.0, 35.0)  # extra counts at the dendrite's center
DENDRITE_SD_PX = 1.8
OFFSET = 1053.0  # the conversion offset the raw uint16 carries
SIGNAL_SCALE = 0.3  # the simulated dF/F is for membrane-bound dye; scaled to the 2-7 % seen on the lines
GAIN = 0.8  # counts per photon: shot noise SD = sqrt(counts / GAIN) * GAIN


REFERENCE_FS = 1075.2688  # the rate vnoiser's sample counts were set at


def dfof_config(fs: float):
    """vnoiser's DfofConfig with its sample counts scaled from 1075 Hz to fs, as mbo_utilities' pipeline does, so
    the baselines keep their length in seconds."""
    from vnoiser.preprocess import DfofConfig

    cfg = DfofConfig()
    k = fs / REFERENCE_FS
    if abs(k - 1) < 1e-6:
        return cfg
    return DfofConfig(sigma_dfof=cfg.sigma_dfof * k, sigma_baseline=cfg.sigma_baseline * k,
                      n_startup=int(round(cfg.n_startup * k)), negative=cfg.negative)


def baseline_seconds(fs: float) -> tuple:
    """The SDs of the two Gaussian baselines in seconds: the one F is divided by, and the slower one taken off dF/F."""
    cfg = dfof_config(fs)
    return cfg.sigma_dfof / fs, cfg.sigma_baseline / fs


def preprocess_steps(f: np.ndarray, fs: float = REFERENCE_FS):
    """vnoiser.preprocess.domain_zscore on one mean trace, keeping every step: (f_baseline, dfof_raw, dfof,
    dfof_baseline, n_startup)."""
    cfg = dfof_config(fs)
    f_baseline = gaussian_filter1d(f, sigma=cfg.sigma_dfof)
    dfof_raw = (f - f_baseline) / f_baseline
    dfof = dfof_raw.copy()
    n = int(cfg.n_startup) if dfof.size > cfg.n_startup else 0
    if n:
        dfof[:n] = dfof[n:].mean()
    return f_baseline, dfof_raw, dfof, gaussian_filter1d(dfof, sigma=cfg.sigma_baseline), n


def line_scan(p: SignalParams, rng: np.random.Generator) -> Signal:
    """
    Raw counts of three lines across one dendrite: a flat background, a dendrite a few pixels wide that dims with
    vnoiser's JEDI3sub simulation, slow bleaching, a little motion along the line, shot noise and the conversion
    offset. Then the same without noise or motion, and both averaged and turned into z the way vnoiser.preprocess
    does it.
    """
    from vnoiser import Jedi3SubConfig, simulate_jedi3sub_trace

    n = int(round(p.duration_s * p.fs))
    sim = simulate_jedi3sub_trace(n / p.fs, Jedi3SubConfig(fs_hz=p.fs), seed=int(rng.integers(1 << 31)))
    t = sim.t
    dff = SIGNAL_SCALE * sim.dff_clean
    bleach = 1.0 - 0.1 * (1 - np.exp(-t / 20.0))
    shift = 0.8 * np.sin(2 * np.pi * t / 11.0) + np.cumsum(rng.normal(0, 0.01, t.size))
    noisy, clean, edges = [], [], []
    for width, center, peak in zip(LINE_WIDTHS, LINE_CENTERS, DENDRITE):
        x = np.arange(width)[:, None]
        still = np.exp(-((x - center) ** 2) / (2 * DENDRITE_SD_PX**2))
        moving = np.exp(-((x - center - shift[None]) ** 2) / (2 * DENDRITE_SD_PX**2))
        c = (BACKGROUND + peak * still * (1 + dff[None])) * bleach[None]
        m = (BACKGROUND + peak * moving * (1 + dff[None])) * bleach[None]
        m = m + p.noise_sd * GAIN * np.sqrt(np.clip(m / GAIN, 0, None)) * rng.standard_normal(m.shape)
        clean.append(c + OFFSET)
        noisy.append(np.round(np.clip(m + OFFSET, 0, 65535)))
        start = edges[-1][1] if edges else 0
        edges.append((start, start + width))
    pixels, clean_pixels = np.concatenate(noisy), np.concatenate(clean)
    f = pixels.mean(axis=0)
    f_baseline, dfof_raw, dfof, dfof_baseline, n_startup = preprocess_steps(f, p.fs)
    sd = dfof.std()
    x = -(dfof - dfof_baseline) / sd
    clean_f = clean_pixels.mean(axis=0)
    _, _, clean_dfof, clean_base, _ = preprocess_steps(clean_f, p.fs)
    truth = -(clean_dfof - clean_base) / sd
    lines = LineScan(
        pixels=pixels.astype(np.float32), clean_pixels=clean_pixels.astype(np.float32), edges=edges,
        line_means=np.stack([a.mean(axis=0) for a in noisy]), f=f, f_baseline=f_baseline, dfof_raw=dfof_raw,
        dfof=dfof, dfof_baseline=dfof_baseline, n_startup=n_startup, clean_f=clean_f, clean_dfof=clean_dfof,
        names=[f"line {k + 1}" for k in range(len(edges))], sigma_s=baseline_seconds(p.fs),
    )
    events = {"AP": sim.events["ap_times_s"]}
    note = f"{n:,} samples at {p.fs:g} Hz, {pixels.shape[0]} pixels"
    return Signal(name="Line scan", fs=p.fs, x=x, truth=truth, note=note, events_s=events, lines=lines)


_SCANS: dict = {}  # path -> (the open .npz, its info), so changing lines or the window does not reread it


def open_scan(path: str):
    """The exported .npz at path and its info: fs, unit, frames and each ROI's width."""
    file = Path(path).expanduser()
    if not file.is_file():
        raise ValueError(f"No file at {file}.")
    key = str(file.resolve())
    if key not in _SCANS:
        d = np.load(file)
        names = sorted((k for k in d.files if k.startswith("roi_")), key=lambda k: int(k[4:]))
        if not names:
            raise ValueError(f"{file.name} holds no roi_<i> arrays; export it with scripts/export_mesc_linescan.py.")
        rois = {int(k[4:]): d[k] for k in names}
        info = {
            "fs": float(d["fs"]), "unit": str(d["unit"]), "frames": int(rois[int(names[0][4:])].shape[0]),
            "widths": {i: int(a.shape[1]) for i, a in rois.items()},
        }
        _SCANS[key] = (rois, info)
    return _SCANS[key]


def scan_info(path: str) -> dict:
    return open_scan(path)[1]


def find_scans(root: str = "~/data") -> list:
    """Exported scans under root, for the source list."""
    base = Path(root).expanduser()
    return sorted(str(f) for f in base.glob("**/linescans/*.npz")) if base.is_dir() else []


def recorded_scan(p: SignalParams) -> Signal:
    """A window of an exported .mesc line scan: the chosen lines' raw pixels, averaged and turned into z."""
    if not p.scan_path:
        raise ValueError("Choose an exported line scan (.npz).")
    data, info = open_scan(p.scan_path)
    fs = info["fs"]
    rois = [int(r) for r in p.rois if int(r) in data]
    if not rois:
        raise ValueError("Pick at least one line.")
    n_all = info["frames"]
    i0 = int(np.clip(round(p.start_s * fs), 0, max(n_all - 64, 0)))
    i1 = int(min(n_all, i0 + round(p.duration_s * fs)))
    if i1 - i0 < 64:
        raise ValueError("The window holds fewer than 64 frames.")
    blocks, edges = [], []
    for roi in rois:
        px = data[roi][i0:i1].T.astype(np.float64)  # (width, frames)
        blocks.append(px)
        start = edges[-1][1] if edges else 0
        edges.append((start, start + px.shape[0]))
    pixels = np.concatenate(blocks)
    f = pixels.mean(axis=0)
    f_baseline, dfof_raw, dfof, dfof_baseline, n_startup = preprocess_steps(f, fs)
    x = -(dfof - dfof_baseline) / dfof.std()
    lines = LineScan(
        pixels=pixels.astype(np.float32), clean_pixels=None, edges=edges,
        line_means=np.stack([b.mean(axis=0) for b in blocks]), f=f, f_baseline=f_baseline, dfof_raw=dfof_raw,
        dfof=dfof, dfof_baseline=dfof_baseline, n_startup=n_startup,
        names=[f"line {roi}" for roi in rois], sigma_s=baseline_seconds(fs),
    )
    note = f"{info['unit']}, lines {', '.join(str(r) for r in rois)}, {i0 / fs:.1f} to {i1 / fs:.1f} s at {fs:.1f} Hz"
    return Signal(name=Path(p.scan_path).name, fs=fs, x=x, note=note, lines=lines)


def read_columns(path: Path) -> np.ndarray:
    """A file's numbers as (samples, columns)."""
    if path.suffix == ".npy":
        data = np.load(path)
    else:
        rows = []
        for line in path.read_text().splitlines():
            parts = line.replace(",", " ").replace(";", " ").split()
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue  # a header or a comment
        if not rows:
            raise ValueError(f"No numeric rows in {path.name}.")
        width = min(len(r) for r in rows)
        data = np.array([r[:width] for r in rows])
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"{path.name} holds a {data.ndim}-d array; need one trace per column.")
    if data.shape[0] < data.shape[1]:
        data = data.T  # traces stored as rows
    return data


def load_file(path: str, fs: float, column: int, zscore: bool) -> Signal:
    """One column of a file as a Signal, z-scored when asked."""
    if not path:
        raise ValueError("Choose a file.")
    file = Path(path).expanduser()
    if not file.is_file():
        raise ValueError(f"No file at {file}.")
    if fs <= 0:
        raise ValueError("The sampling rate must be above 0 Hz.")
    data = read_columns(file)
    column = min(max(column, 0), data.shape[1] - 1)
    x = data[:, column].copy()
    x = x[np.isfinite(x)]
    if x.size < 64:
        raise ValueError(f"Column {column + 1} has {x.size} finite samples; need at least 64.")
    if zscore:
        x = (x - x.mean()) / (x.std() or 1.0)
    note = f"{x.size:,} samples at {fs:g} Hz, column {column + 1} of {data.shape[1]}" + (", z-scored" if zscore else "")
    return Signal(name=file.name, fs=fs, x=x, truth=None, note=note, columns=data.shape[1])
