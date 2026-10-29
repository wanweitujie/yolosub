"""routes.export_model 的 onnx classes.txt 分支测试。

覆盖:version 判定矩阵(§9.2)、异常不阻断、extra_contents 透传。
全部外部依赖 mock,不依赖真实模型文件与 GPU。
"""
from unittest.mock import MagicMock, patch

from modal import ModelExportConfig
from export_v.routes import export_model, init_export

init_export(MagicMock())  # export_model 内引用全局 logger,经 init_export 绑定


def _run_onnx(version, names=None, names_side_effect=None):
    """跑一次 onnx 导出,返回传给 zip_with_max_compression 的 kwargs。"""
    cfg = ModelExportConfig(model_path="/tmp/m.pt", version=version, format="onnx")
    captured = {}

    def fake_zip(src, out, **kwargs):
        captured.update(kwargs)
        return out

    read_patch = (
        patch("export_v.routes._read_class_names", side_effect=names_side_effect)
        if names_side_effect
        else patch("export_v.routes._read_class_names", return_value=names)
    )

    with patch("export_v.routes.get_available_gpu", return_value=0), \
         patch("export_v.routes.export_ultralytics", return_value="/tmp/m.onnx"), \
         read_patch, \
         patch("export_v.routes.os.path.exists", return_value=True), \
         patch("export_v.routes.os.path.isfile", return_value=True), \
         patch("export_v.routes.os.remove"), \
         patch("export_v.routes.zip_with_max_compression", side_effect=fake_zip):
        result = export_model(cfg)
    return captured, result


def test_onnx_v11_includes_classes_txt():
    """onnx + v11 → extra_contents 含 classes.txt,内容为换行连接的标签名。"""
    captured, result = _run_onnx("v11", names=["person", "bicycle", "car"])

    assert captured["extra_contents"] == {"classes.txt": "person\nbicycle\ncar"}
    assert result["success"] is True


def test_onnx_v8_includes_classes_txt():
    """onnx + v8 → 同样含 classes.txt。"""
    captured, _ = _run_onnx("v8", names=["person"])

    assert captured["extra_contents"] == {"classes.txt": "person"}


def test_onnx_version_none_includes_classes_txt():
    """onnx + version=None → 也含(判定为 version != 'v5')。"""
    captured, _ = _run_onnx(None, names=["person"])

    assert captured["extra_contents"] == {"classes.txt": "person"}


def test_onnx_v5_excludes_classes_txt():
    """onnx + v5 → extra_contents 为 None(本次范围外)。"""
    cfg = ModelExportConfig(model_path="/tmp/m.pt", version="v5", format="onnx")
    captured = {}

    def fake_zip(src, out, **kwargs):
        captured.update(kwargs)
        return out

    with patch("export_v.routes.get_available_gpu", return_value=0), \
         patch("export_v.routes.export_v5", return_value="/tmp/m.onnx"), \
         patch("export_v.routes._read_class_names") as mock_read, \
         patch("export_v.routes.os.path.exists", return_value=True), \
         patch("export_v.routes.os.path.isfile", return_value=True), \
         patch("export_v.routes.os.remove"), \
         patch("export_v.routes.zip_with_max_compression", side_effect=fake_zip):
        export_model(cfg)

    mock_read.assert_not_called()                      # v5 不应触发读类名
    assert captured["extra_contents"] is None


def test_onnx_read_names_failure_does_not_break_export():
    """读类名抛异常 → extra_contents 为 None,但导出仍成功(告警不阻断)。"""
    captured, result = _run_onnx("v11", names_side_effect=RuntimeError("boom"))

    assert captured["extra_contents"] is None
    assert result["success"] is True


def test_onnx_empty_names_yields_empty_string():
    """空 names → 写空字符串(与 ncnn 路径一致写出 0 字节 classes.txt),不跳过。"""
    captured, _ = _run_onnx("v11", names=[])

    assert captured["extra_contents"] == {"classes.txt": ""}
