"""
Draw every page for a few frames on hello_imgui's null backend: no window or GPU. Skips when the backend cannot
start here.
"""

import pytest
from imgui_bundle import hello_imgui, imgui, immapp

from imgui_vnoiser.app import PAGES, App, runner_params

FRAMES_PER_PAGE = 4


def enable_textures() -> None:
    # the null renderer uploads no font atlas; imgui 1.92 asserts in NewFrame unless the backend claims textures
    imgui.get_io().backend_flags |= imgui.BackendFlags_.renderer_has_textures


class Driver:
    """Waits for the first run, then visits every page; then the same for one more signal source."""

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
            app.settings.signal.source = self.sources.pop(0)
            self.frame = 0
        else:
            hello_imgui.get_runner_params().app_shall_exit = True


@pytest.mark.parametrize("first, then", [("spikes", ["jedi"]), ("jedi", ["spikes"])])
def test_every_page_renders(first, then):
    app = App()
    app.settings.signal.source = first
    app.settings.signal.duration_s = 3.0
    driver = Driver(app, then)
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
    assert [page for source, page in driver.seen if source == first] == [key for key, *_ in PAGES]
    assert [page for source, page in driver.seen if source == then[0]] == [key for key, *_ in PAGES]
