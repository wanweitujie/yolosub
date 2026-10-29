from unittest.mock import MagicMock, patch

from modal import ModelExportConfig
from export_v.export_ncnn import transform_torchscript_ncnn


def _run_transform(cfg, tmp_path):
    ts_path = tmp_path / "m.torchscript"
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(stdout="")

    # PNNX_PATH 由 import 时按环境解析(Windows 上通常为 None),此处占位避免
    # logger.info 的 " ".join(cmd) 撞 None;subprocess.run 已被 mock,真实路径不影响断言。
    with patch("export_v.export_ncnn.PNNX_PATH", "pnnx"), \
         patch("export_v.export_ncnn.subprocess.run", side_effect=fake_run):
        transform_torchscript_ncnn(cfg, None, str(ts_path), MagicMock())
    return captured["cmd"]


def test_transform_half_none_defaults_to_fp16_0(tmp_path):
    """请求未传 half 时(config.half 默认 None),transform 不崩溃且命令含 fp16=0。

    e2e(Task4)暴露:export_ncnn.py `int(config.half)` 遇 None 抛
    TypeError("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'"),
    导致 v5/v8 ncnn-standard 与 v5 ncnn 三条回归全部失败。
    """
    cfg = ModelExportConfig(model_path="/tmp/m.pt", version="v5",
                            format="ncnn-standard", imgsz=640)
    assert cfg.half is None  # 复现前置:ModelExportConfig.half 默认 None

    cmd = _run_transform(cfg, tmp_path)

    assert "fp16=0" in cmd


def test_transform_half_true_keeps_fp16_1(tmp_path):
    """half=True 显式传入时行为不变:fp16=1(回归守卫)。"""
    cfg = ModelExportConfig(model_path="/tmp/m.pt", version="v5",
                            format="ncnn-standard", imgsz=640, half=True)

    cmd = _run_transform(cfg, tmp_path)

    assert "fp16=1" in cmd
