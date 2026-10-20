import numpy as np
import pytest

from imgui_vnoiser import SOURCES, SignalParams, make_signal


@pytest.mark.parametrize("source", [key for key, *_ in SOURCES if key not in ("file", "recorded")])
def test_every_source_makes_a_trace(source):
    sig = make_signal(SignalParams(source=source, fs=500.0, duration_s=2.0))
    assert sig.n == 1000 and np.all(np.isfinite(sig.x))
    assert sig.truth is not None and sig.truth.shape == sig.x.shape


def test_line_scan_looks_like_raw_mesc_counts():
    sig = make_signal(SignalParams(source="lines", duration_s=4.0))
    px = sig.lines.pixels
    assert np.all(px == np.round(px)) and 1100 < px.mean() < 1300  # integers, conversion offset left in
    assert px.shape[0] == sum(b - a for a, b in sig.lines.edges)


def test_without_noise_the_trace_follows_the_noiseless_one():
    sig = make_signal(SignalParams(source="lines", duration_s=8.0, noise_sd=0.0))
    assert np.corrcoef(sig.x[1000:], sig.truth[1000:])[0, 1] > 0.9


def test_file_columns_and_zscore(tmp_path):
    path = tmp_path / "trace.csv"
    t = np.arange(200) / 100
    rows = ["time,a,b"] + [f"{ti},{np.sin(ti)},{2 + 3 * np.cos(7 * ti)}" for ti in t]
    path.write_text("\n".join(rows))
    sig = make_signal(SignalParams(source="file", path=str(path), fs=100.0, column=2))
    assert sig.columns == 3 and sig.n == 200
    assert sig.x.mean() == pytest.approx(0, abs=1e-9) and sig.x.std() == pytest.approx(1)


def test_missing_file_is_a_message():
    with pytest.raises(ValueError, match="No file"):
        make_signal(SignalParams(source="file", path="/nope/trace.npy"))


def test_line_scan_trace_is_what_vnoiser_preprocess_makes():
    from vnoiser.preprocess import domain_zscore

    sig = make_signal(SignalParams(source="lines", duration_s=12.0))
    ls = sig.lines
    traces = {k: ls.pixels[a:b].astype(float).mean(axis=0) for k, (a, b) in enumerate(ls.edges)}
    weights = {k: b - a for k, (a, b) in enumerate(ls.edges)}
    z = domain_zscore(traces, {"dendrite": list(traces)}, weights=weights).z[0]
    np.testing.assert_allclose(sig.x, z, atol=1e-5)
    assert ls.n_startup == 1000 and np.all(ls.dfof[:1000] == ls.dfof[1000:].mean())


def test_recorded_scan_reads_the_chosen_lines_and_window(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "unit.npz"
    rois = {f"roi_{i}": rng.integers(1150, 1250, size=(3000, w), dtype=np.uint16) for i, w in enumerate((19, 20, 10))}
    np.savez(path, fs=1587.3, unit="MSession_0/MUnit_1", channel=0, offset=-1053.0, source="x.mesc", **rois)
    sig = make_signal(SignalParams(source="recorded", scan_path=str(path), rois=(0, 2), start_s=0.5, duration_s=1.0))
    i0 = round(0.5 * 1587.3)
    expected = np.concatenate([rois["roi_0"][i0:i0 + 1587].T, rois["roi_2"][i0:i0 + 1587].T]).astype(float)
    np.testing.assert_allclose(sig.lines.pixels, expected)
    assert sig.fs == pytest.approx(1587.3) and sig.lines.edges == [(0, 19), (19, 29)] and sig.truth is None
