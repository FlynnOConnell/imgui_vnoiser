"""
Draw every page for a few frames on hello_imgui's null backend: no window or GPU. Skips when the backend cannot
start here.
"""

import numpy as np
import pytest
from imgui_bundle import hello_imgui, imgui, immapp

from imgui_vnoiser.app import PAGES, App, runner_params

FRAMES_PER_PAGE = 4


def enable_textures() -> None:
    # the null renderer uploads no font atlas; imgui 1.92 asserts in NewFrame unless the backend claims textures
    imgui.get_io().backend_flags |= imgui.BackendFlags_.renderer_has_textures


class Driver:
    """Waits for the first run, then visits every page; then the same for each further source."""

    def __init__(self, app: App, sources):
        self.app = app
        self.sources = list(sources)
        self.frame = 0
        self.seen = []

    def __call__(self) -> None:
        app = self.app
        app.gui()
        if app.future is not None or not app.ready:
            app.wait()
            return
        self.frame += 1
        page = self.frame // FRAMES_PER_PAGE
        if page < len(PAGES):
            app.page = PAGES[page][0]
            if (app.settings.signal.source, app.page) not in self.seen:
                self.seen.append((app.settings.signal.source, app.page))
        elif self.sources:
            app.settings.signal.source, app.settings.signal.path = self.sources.pop(0)
            self.frame = 0
        else:
            hello_imgui.get_runner_params().app_shall_exit = True


def test_every_page_renders(tmp_path):
    trace = tmp_path / "trace.npy"
    np.save(trace, np.random.default_rng(0).standard_normal(4000))
    app = App()
    app.settings.signal.duration_s = 4.0
    driver = Driver(app, [("file", str(trace))])
    params = runner_params(app)
    params.platform_backend_type = hello_imgui.PlatformBackendType.null
    params.renderer_backend_type = hello_imgui.RendererBackendType.null
    params.ini_folder_type = hello_imgui.IniFolderType.temp_folder
    params.callbacks.post_init = enable_textures
    params.callbacks.show_gui = driver
    try:
        immapp.run(params, immapp.AddOnsParams(with_implot=True))
    except Exception as e:
        if "backend" in str(e).lower():
            pytest.skip(f"null backend unavailable: {e}")
        raise
    app.executor.shutdown(wait=True)
    assert app.run.error == ""
    for source in ("lines", "file"):
        assert [page for s, page in driver.seen if s == source] == [key for key, *_ in PAGES]


def test_a_view_past_a_short_line_draws():
    """The startup line on the first page ends at 1 s; a view after that must not hand ImPlot an empty array."""
    app = App()
    frames = [0]

    def gui() -> None:
        if app.ready and app.future is None:
            app.t0.value, app.t1.value = 10.0, 11.0
        app.gui()
        if app.future is not None or not app.ready:
            app.wait()
            return
        frames[0] += 1
        if frames[0] > FRAMES_PER_PAGE:
            hello_imgui.get_runner_params().app_shall_exit = True

    params = runner_params(app)
    params.platform_backend_type = hello_imgui.PlatformBackendType.null
    params.renderer_backend_type = hello_imgui.RendererBackendType.null
    params.ini_folder_type = hello_imgui.IniFolderType.temp_folder
    params.callbacks.post_init = enable_textures
    params.callbacks.show_gui = gui
    try:
        immapp.run(params, immapp.AddOnsParams(with_implot=True))
    except Exception as e:
        if "backend" in str(e).lower():
            pytest.skip(f"null backend unavailable: {e}")
        raise
    app.executor.shutdown(wait=True)
    assert frames[0] > FRAMES_PER_PAGE
