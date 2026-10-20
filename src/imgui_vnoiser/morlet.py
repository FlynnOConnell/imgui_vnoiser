"""
The complex Morlet wavelet pywt calls `cmorB-C`, in seconds and hertz.

    psi(t) = exp(-t^2 / B) * exp(2j pi C t) / sqrt(pi B)        (t in samples at scale 1)
    Psi(f) = exp(-pi^2 B (f - C)^2)                            (f in cycles per sample at scale 1)

At scale s and sampling rate fs a row is centered on f_c = C fs / s Hz. pywt's coefficients grow as sqrt(s);
vnoiser divides each band by sqrt(mean scale), after which a unit sine at f_c gives |W| = 1/2 (half its power goes
to the negative frequency the complex wavelet ignores).
"""

import numpy as np
import pywt


def name(b: float, c: float) -> str:
    """The pywt wavelet name, e.g. cmor0.5-1."""
    return f"cmor{b:g}-{c:g}"


def response(b: float, c: float, scale: float, fs: float, f_hz: np.ndarray) -> np.ndarray:
    """Gain of the row at scale for a sine at each f_hz, peak 1."""
    return np.exp(-(np.pi**2) * b * (np.asarray(f_hz) * scale / fs - c) ** 2)


def cwt(x: np.ndarray, scales: np.ndarray, fs: float, b: float, c: float) -> tuple[np.ndarray, np.ndarray]:
    """(coefficients rows x samples complex, frequencies), exactly as vnoiser.Denoiser._cwt computes them."""
    return pywt.cwt(x, scales=scales, wavelet=name(b, c), sampling_period=1 / fs, method="fft")
