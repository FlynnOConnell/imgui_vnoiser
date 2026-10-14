"""
Input traces for the window: two synthetic traces with a ground truth, and traces read from a file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# the source key, its name in the Signal page's list, and the line under it
SOURCES = (
    ("spikes", "Spikes and bumps", "Gaussian bumps (60 to 180 ms) and spikes (3 SD, 4 ms decay), plus white noise; repeats every 4 s"),
    ("jedi", "JEDI3sub simulation", "vnoiser.simulate_jedi3sub_trace, sign-flipped and z-scored"),
    ("file", "From a file", ".npy, .csv, .txt or .tsv, one trace per column"),
)


@dataclass
class SignalParams:
    """Everything that picks and shapes the input trace."""

    source: str = "spikes"
    fs: float = 1000.0
    duration_s: float = 4.0
    noise_sd: float = 0.3
    seed: int = 7
    path: str = ""
    column: int = 0
    zscore: bool = True


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
    if p.fs <= 0:
        raise ValueError("The sampling rate must be above 0 Hz.")
    n = int(round(p.duration_s * p.fs))
    if n < 64:
        raise ValueError("Need at least 64 samples; raise the duration or the sampling rate.")
    rng = np.random.default_rng(p.seed)
    if p.source == "jedi":
        return jedi3sub(p, rng)
    if p.source != "spikes":
        raise ValueError(f"Unknown source {p.source!r}.")
    truth, events = spikes_and_bumps(np.arange(n) / p.fs)
    x = truth + p.noise_sd * rng.standard_normal(n)
    note = f"{n:,} samples at {p.fs:g} Hz, noise SD {p.noise_sd:g}"
    return Signal(name="Spikes and bumps", fs=p.fs, x=x, truth=truth, note=note, events_s=events)


def spike_shape(d: np.ndarray) -> np.ndarray:
    """A fast-rising, 4 ms decaying spike at delay d seconds, peak 1."""
    def raw(d):
        d = np.clip(d, 0, None)
        return np.exp(-d / 0.004) * (1 - np.exp(-d / 0.0008))

    peak = raw(np.linspace(0, 0.02, 4001)).max()
    return np.where(d < 0, 0.0, raw(d)) / peak


def spikes_and_bumps(t: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    A synthetic voltage trace, any length: Gaussian bumps of 60 to 180 ms
    (subthreshold depolarizations and hyperpolarizations) and 3 SD spikes, alone and in bursts. Repeats every 4 s.
    """
    period = 4.0
    bumps = ((0.4, 0.09, 1.5), (1.15, 0.13, 1.6), (1.75, 0.07, -1.2), (2.6, 0.18, 1.3), (3.45, 0.06, -1.0))
    spikes = (0.33, 0.39, 0.44, 0.49, 0.8, 1.02, 1.09, 1.15, 1.22, 1.28, 2.48, 2.57, 2.68, 3.1)
    y = np.zeros_like(t)
    starts = np.arange(0.0, t[-1] + 1e-9, period)
    for start in starts:
        for center, width, amplitude in bumps:
            y += amplitude * np.exp(-((t - start - center) ** 2) / (2 * width * width))
        for s in spikes:
            y += 3.0 * spike_shape(t - start - s)
    spike_times = np.array([start + s for start in starts for s in spikes])
    return y, {"spikes": spike_times[spike_times <= t[-1]]}


def jedi3sub(p: SignalParams, rng: np.random.Generator) -> Signal:
    """vnoiser's JEDI3sub simulation, sign-flipped (depolarization up) and z-scored the way the pipeline does."""
    from vnoiser import Jedi3SubConfig, simulate_jedi3sub_trace

    cfg = Jedi3SubConfig(fs_hz=p.fs)
    sim = simulate_jedi3sub_trace(p.duration_s, cfg, seed=int(rng.integers(1 << 31)))
    mean, sd = sim.dff_observed.mean(), sim.dff_observed.std() or 1.0
    x = -(sim.dff_observed - mean) / sd
    truth = -(sim.dff_clean - sim.dff_clean.mean()) / sd
    events = {name[:-2]: sim.events[name] for name in ("ap_times_s", "slow_depol_times_s", "ripple_times_s")}
    note = f"{x.size:,} samples at {p.fs:g} Hz, {len(sim.events['ap_times_s'])} APs"
    return Signal(name="JEDI3sub simulation", fs=p.fs, x=x, truth=truth, note=note, events_s=events)


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
