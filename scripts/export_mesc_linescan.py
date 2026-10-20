"""
Save the raw pixels of AOD line-scan units in a .mesc as .npz files the window opens ("Recorded line scan").

Needs mbo_utilities (pml_utilities), so run it with that environment:

    cd ../pml_utilities
    uv run --no-sync python ../imgui_vnoiser/scripts/export_mesc_linescan.py ~/data/turi/9_14_2026/session1.mesc

Writes <mesc folder>/linescans/<unit>.npz per line-scan unit: `roi_<i>` (frames, width) raw uint16 counts for
every ROI (the line's rows averaged when it has more than one), `fs`, `unit`, `channel`, `offset` (the file's
conversion offset: raw + offset is zero with no photons), `source`.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from mbo_utilities.arrays.mesc import MescArray, list_mesc_units
from mbo_utilities.roi_workflow import as_movie


def export_unit(path: Path, key: str, channel: int, out: Path, batch: int = 20000) -> Path:
    arr = MescArray(path, unit=key)
    md = arr.metadata
    rois = {}
    for i, ext in enumerate(md["mesc_roi_extents"]):
        h, w = int(ext["height"]), int(ext["width"])
        movie = as_movie(arr, z=int(ext["index"]), c=channel)
        n = movie.shape[0]
        px = np.empty((n, w), np.uint16)
        for t0 in range(0, n, batch):
            t1 = min(n, t0 + batch)
            block = movie.frames(t0, t1, slice(0, h), slice(0, w))
            px[t0:t1] = np.round(block.reshape(t1 - t0, h, w).mean(axis=1))
        rois[f"roi_{i}"] = px
    conv = md.get("mesc_channel_conversion") or []
    offset = float(conv[channel]["offset"]) if channel < len(conv) else 0.0
    unit = key.replace("/", "_")
    dest = out / f"{unit}.npz"
    np.savez(dest, fs=float(md["fs"]), unit=key, channel=channel,
             offset=offset, source=str(path), **rois)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mesc", type=Path)
    parser.add_argument("--unit", action="append", help="MSession_0/MUnit_30; every line-scan unit when left out")
    parser.add_argument("--channel", type=int, default=0, help="0 = green (functional)")
    args = parser.parse_args()
    out = args.mesc.parent / "linescans"
    out.mkdir(exist_ok=True)
    units = args.unit or [
        u["key"] for u in list_mesc_units(args.mesc) if u["modality_name"] == "linescan" and u["role"] == "measurement"
    ]
    for key in units:
        t0 = time.time()
        dest = export_unit(args.mesc, key, args.channel, out)
        print(f"{key} -> {dest} ({dest.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
