"""
Screenshots for the README: the window's own framebuffer, from a hidden window (`--screenshot`), one process per
page. Each runs the default simulated line scan and saves docs/<page>.png. Needs a desktop session for the GL context.

    python scripts/screenshot.py         # every page
    python scripts/screenshot.py bands   # one
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGES = ("signal", "scalogram", "bands", "events", "final")


def main() -> None:
    for page in sys.argv[1:] or PAGES:
        filepath_png = REPO / "docs" / f"{page}.png"
        command = [sys.executable, "-m", "imgui_vnoiser.app", "--scale", "1", "--page", page, "--screenshot", str(filepath_png)]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(filepath_png)


if __name__ == "__main__":
    main()
