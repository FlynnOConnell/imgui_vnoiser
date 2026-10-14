"""See what vnoiser's Morlet wavelet denoiser does to a trace, one stage at a time."""

from imgui_vnoiser.pipeline import STAGES, Band, Run, Settings, compute, metrics
from imgui_vnoiser.signals import SOURCES, Signal, SignalParams, make_signal

__version__ = "0.1.0"

__all__ = ["STAGES", "SOURCES", "Band", "Run", "Settings", "Signal", "SignalParams", "compute", "make_signal", "metrics"]
