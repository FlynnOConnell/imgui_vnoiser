"""
The complex Morlet wavelet pywt calls `cmorB-C`, in seconds and hertz, and the filter bank its scales make.

    psi(t) = exp(-t^2 / B) * exp(2j pi C t) / sqrt(pi B)        (t in samples at scale 1)
    Psi(f) = exp(-pi^2 B (f - C)^2)                            (f in cycles per sample at scale 1)

At scale s and sampling rate fs a row is centered on f_c = C fs / s Hz. pywt's coefficients grow as sqrt(s);
vnoiser divides each band by sqrt(mean scale), after which a unit sine at f_c gives |W| = 1/2 (half its power goes
to the negative frequency the complex wavelet ignores).
"""

from dataclasses import dataclass

import numpy as np
import pywt

# a preset's name, B, C and why it is in the list
PRESETS = (
    ("vnoiser (cmor0.5-1.0)", 0.5, 1.0, "what Denoiser uses: about one cycle under the envelope, sharp in time, wide in frequency"),
    ("Classic Morlet, w0 = 6", 2.0, 6 / (2 * np.pi), "the textbook Morlet: about 2 cycles"),
    ("Narrowband (cmor8-1.5)", 8.0, 1.5, "many cycles: a narrow band-pass that smears events in time"),
)


def name(b: float, c: float) -> str:
    """The pywt wavelet name, e.g. cmor0.5-1."""
    return f"cmor{b:g}-{c:g}"


def frequencies(scales: np.ndarray, fs: float, c: float) -> np.ndarray:
    """Center frequency in Hz of each scale (what pywt.cwt returns as freqs)."""
    return c * fs / np.asarray(scales, dtype=float)


@dataclass
class Widths:
    """How long and how wide one row of the transform is."""

    freq_hz: float
    sigma_t_s: float  # SD of the envelope |psi| in seconds
    sigma_f_hz: float  # SD of the frequency response Psi in Hz
    half_power_lo_hz: float
    half_power_hi_hz: float
    cycles: float  # cycles of the carrier within +-1 SD of the envelope

    @property
    def q(self) -> float:
        """Center frequency over half-power bandwidth: the same for every row of a cmor wavelet."""
        return self.freq_hz / max(self.half_power_hi_hz - self.half_power_lo_hz, 1e-12)


def widths(b: float, c: float, scale: float, fs: float) -> Widths:
    """The time and frequency widths of the row at scale."""
    sigma_t = np.sqrt(b / 2) * scale / fs
    sigma_f = 1 / (np.pi * np.sqrt(2 * b)) * fs / scale
    f_c = c * fs / scale
    # Psi(f)^2 = 1/2  <=>  |f - C| = sqrt(ln 2 / (2 B)) / pi, at scale 1
    half = np.sqrt(np.log(2) / (2 * b)) / np.pi * fs / scale
    return Widths(f_c, sigma_t, sigma_f, max(f_c - half, 0.0), f_c + half, 2 * np.sqrt(b / 2) * c)


def wavelet_in_time(b: float, c: float, scale: float, fs: float, n_sd: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """(t in seconds, psi(t) complex) of the row at scale, over +-n_sd envelope SDs, unit peak envelope."""
    sigma = np.sqrt(b / 2)
    u = np.linspace(-n_sd * sigma, n_sd * sigma, 801)  # in scale-1 samples
    psi = np.exp(-(u**2) / b) * np.exp(2j * np.pi * c * u)
    return u * scale / fs, psi


def response(b: float, c: float, scale: float, fs: float, f_hz: np.ndarray) -> np.ndarray:
    """Gain of the row at scale for a sine at each f_hz, peak 1."""
    return np.exp(-(np.pi**2) * b * (np.asarray(f_hz) * scale / fs - c) ** 2)


def bank_sum(b: float, c: float, scales: np.ndarray, fs: float, f_hz: np.ndarray) -> np.ndarray:
    """
    Gain of summing every row's real part after dividing by sqrt(scale): what adding all bands back with every mask
    at 1 does to a unit sine at each f_hz. Each row passes 1/2 of its response.
    """
    return 0.5 * sum(response(b, c, s, fs, f_hz) for s in np.asarray(scales, dtype=float))


def cwt(x: np.ndarray, scales: np.ndarray, fs: float, b: float, c: float) -> tuple[np.ndarray, np.ndarray]:
    """(coefficients rows x samples complex, frequencies), exactly as vnoiser.Denoiser._cwt computes them."""
    return pywt.cwt(x, scales=scales, wavelet=name(b, c), sampling_period=1 / fs, method="fft")
