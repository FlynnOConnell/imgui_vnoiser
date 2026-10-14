"""The staged pipeline gives what vnoiser.Denoiser.run gives, and its per-band details agree with vnoiser's classes."""

import numpy as np
import pytest
from vnoiser import Denoiser

from imgui_vnoiser import Settings, compute, morlet
from imgui_vnoiser.pipeline import STAGES


def make(source="spikes", fs=1075.2688, duration=4.0, **edits) -> Settings:
    s = Settings()
    s.signal.source, s.signal.fs, s.signal.duration_s = source, fs, duration
    for key, value in edits.items():
        setattr(s, key, value)
    return s


@pytest.fixture(scope="module")
def run():
    return compute(make())


def test_every_stage_runs(run):
    assert run.error == ""
    assert list(run.timing) == list(STAGES)


@pytest.mark.parametrize("upstream", [False, True])
def test_matches_denoiser_run(upstream):
    s = make(source="jedi")
    if upstream:
        s.use_upstream()
    r = compute(s)
    d = Denoiser.upstream(s.signal.fs) if upstream else Denoiser(s.signal.fs)
    denoised, starts = d.run(r.signal.x)
    np.testing.assert_allclose(r.denoised, denoised.real, atol=1e-10)
    np.testing.assert_allclose(r.baseline, d.lp_dfof_, atol=1e-12)


def test_band_details_agree_with_wavelet_reducer(run):
    reduced = run.reduced
    by_position = {band.position: band for band in run.kept}
    for row, (name, onsets) in enumerate(reduced["event_onsets"].items()):
        position = reduced["selected_clusters"][row]
        band = by_position[position]
        assert band.name == name
        np.testing.assert_array_equal(band.windows, onsets["crossings"])
        np.testing.assert_allclose(band.smoothed, reduced["reduced_coeff"][row])
        np.testing.assert_allclose(band.mean_scale, reduced["reduced_scale"][row])


def test_mask_details_agree_with_adaptive_threshold(run):
    thres = run.thres
    kept = sorted(run.kept, key=lambda band: band.position)
    for row, band in enumerate(kept):
        np.testing.assert_allclose(band.label, thres["clu_label"][row])
        np.testing.assert_allclose(band.masked.real, thres["all_scaled"][row].real, atol=1e-12)


def test_a_later_change_reuses_earlier_stages(run):
    s = run.settings.copy()
    s.thres.soft_levels = (0.5, 0.4, 0.1, 0.0)
    assert s.first_change(run.settings) == "mask"
    partial = compute(s, "mask", run)
    assert partial.coeff is run.coeff and partial.cluster is run.cluster
    full = compute(s)
    np.testing.assert_allclose(partial.denoised, full.denoised)


def test_excluding_a_band_removes_its_contribution(run):
    s = run.settings.copy()
    band = run.kept[0]
    s.excluded = (band.id,)
    assert s.first_change(run.settings) == "final"
    r = compute(s, "final", run)
    np.testing.assert_allclose(run.denoised - r.denoised, band.masked.real, atol=1e-12)


def test_a_bad_setting_reports_the_stage_and_recovers(run):
    s = run.settings.copy()
    s.clust.n_comp_clu = s.clust.n_components + 5
    bad = compute(s, "cluster", run)
    assert bad.failed == "cluster" and "components" in bad.error
    s.clust.n_comp_clu = 10
    good = compute(s, "cluster", bad)
    assert good.error == "" and good.denoised is not None


def test_python_export_reproduces_the_run():
    s = make(source="tones")
    s.b, s.c, s.n_scales = 2.0, 6 / (2 * np.pi), 60
    s.clust.n_subclusters = 7
    s.thres.soft_levels = (0.6, 0.4, 0.2, 0.05)
    r = compute(s)
    scope = {"trace": r.signal.x}
    exec(s.python(), scope)
    np.testing.assert_allclose(scope["denoised"].real, r.denoised, atol=1e-10)


def test_morlet_response_matches_pywt():
    fs, b, c = 1000.0, 0.5, 1.0
    scales = np.logspace(0, 3, 60)
    t = np.arange(6000) / fs
    f0 = 40.0
    coeff, _ = morlet.cwt(np.sin(2 * np.pi * f0 * t), scales, fs, b, c)
    measured = np.abs(coeff[:, 2000:4000]).mean(axis=1) / np.sqrt(scales)
    predicted = 0.5 * np.array([morlet.response(b, c, s, fs, f0) for s in scales])
    near = predicted > 0.05
    np.testing.assert_allclose(measured[near], predicted[near], rtol=0.06, atol=0.01)


def test_widths_are_constant_q():
    a = morlet.widths(0.5, 1.0, 10, 1000)
    b = morlet.widths(0.5, 1.0, 100, 1000)
    assert a.q == pytest.approx(b.q)
    assert a.sigma_t_s * 10 == pytest.approx(b.sigma_t_s)
    assert a.cycles == pytest.approx(1.0)
