import numpy as np
import pytest

from imgui_vnoiser import SOURCES, SignalParams, make_signal


@pytest.mark.parametrize("source", [key for key, *_ in SOURCES if key != "file"])
def test_every_source_makes_a_trace(source):
    sig = make_signal(SignalParams(source=source, fs=500.0, duration_s=2.0))
    assert sig.n == 1000 and np.all(np.isfinite(sig.x))
    assert sig.truth is not None and sig.truth.shape == sig.x.shape


def test_an_isolated_spike_peaks_at_three():
    sig = make_signal(SignalParams(source="spikes", fs=20000.0, duration_s=4.0, noise_sd=0.0))
    near = (sig.t > 3.05) & (sig.t < 3.15)  # the spike at 3.1 s, far from every bump
    assert sig.truth[near].max() == pytest.approx(3.0, abs=0.05)


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
