"""
Colors for the window and its plots: imgui_lab4_interface's palette, one color per trace the pages share, and
colormaps that fade into the window's background.
"""

import imgui_data_loader as idl
import numpy as np
from imgui_bundle import imgui, implot

# imgui_lab4_interface's palette
ACCENT = (0.40, 0.68, 1.00)
KEY = (1.00, 0.80, 0.20)
DIM = (0.62, 0.62, 0.65)
TEXT = (0.92, 0.92, 0.94)
CARD = (0.17, 0.18, 0.21)
CARD_HOVER = (0.21, 0.22, 0.26)
EDGE = (0.35, 0.35, 0.37)
OK = (0.40, 0.90, 0.40)
WARN = (1.00, 0.75, 0.30)
ERROR = (1.00, 0.40, 0.40)
BG = idl.Theme.dark().bg[:3]

# one color per trace, the same on every page
INPUT = (0.70, 0.70, 0.74)
NOISELESS = (0.45, 0.82, 0.74)
DENOISED = KEY
REMOVED = (0.78, 0.58, 1.00)
BASELINE = ACCENT

THEME = idl.Theme.dark().replace(
    accent=(*ACCENT, 1.0),
    accent_hover=(0.50, 0.75, 1.00, 1.0),
    accent_active=(0.32, 0.60, 0.95, 1.0),
    text=(*TEXT, 1.0),
    text_dim=(*DIM, 1.0),
    border=(*EDGE, 0.8),
    button_face=(*CARD, 1.0),
    button_face_hover=(*CARD_HOVER, 1.0),
)

# a line scan's darkest pixels are the plot's background; a signed map's 0 is a card
RAMPS = {
    "scan": (BG, (0.55, 0.56, 0.60), (0.97, 0.97, 0.98)),
    "sequential": (BG, (0.20, 0.38, 0.66), ACCENT, (0.86, 0.94, 1.00)),
    "diverging": ((0.30, 0.58, 1.00), CARD, (1.00, 0.66, 0.22)),
}


def vec4(color: tuple, alpha: float = 1.0) -> imgui.ImVec4:
    return imgui.ImVec4(color[0], color[1], color[2], alpha)


def ramp(stops: tuple, n: int = 64) -> np.ndarray:
    """(n, 4) float32 RGBA, linear between evenly spaced stops."""
    x = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, n)
    rgb = np.stack([np.interp(t, x, [s[i] for s in stops]) for i in range(3)], axis=1)
    return np.concatenate([rgb, np.ones((n, 1))], axis=1).astype(np.float32)


def add_colormaps() -> dict:
    """Register the ramps with ImPlot; name -> colormap id. Once per ImPlot context."""
    return {name: implot.add_colormap(name, ramp(stops), False) for name, stops in RAMPS.items()}
