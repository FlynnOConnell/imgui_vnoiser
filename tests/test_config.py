from imgui_vnoiser import config


def test_scale_order_override_env_saved_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("IMGUI_VNOISER_HOME", str(tmp_path))
    monkeypatch.delenv("IMGUI_VNOISER_SCALE", raising=False)
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    monkeypatch.setenv("GDK_SCALE", "2")
    assert config.ui_scale() == 2.0
    config.save({"ui_scale": 1.25})
    assert config.ui_scale() == 1.25
    monkeypatch.setenv("IMGUI_VNOISER_SCALE", "3")
    assert config.ui_scale() == 3.0
    assert config.ui_scale(1.5) == 1.5
    assert config.ui_scale(99) == config.SCALE_MAX


def test_no_hint_means_one(tmp_path, monkeypatch):
    monkeypatch.setenv("IMGUI_VNOISER_HOME", str(tmp_path))
    for var in ("IMGUI_VNOISER_SCALE", "GDK_SCALE", "HYPRLAND_INSTANCE_SIGNATURE"):
        monkeypatch.delenv(var, raising=False)
    assert config.ui_scale() == 1.0
