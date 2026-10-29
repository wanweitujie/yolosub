# tests/unit/test_export_v5_onnx_dispatch.py
from unittest.mock import patch

from modal import ModelExportConfig
from export_v.export_v5 import export_v5
from utils_v.index import YOLOV5_PYTHON_PATH, ULTRALYTICS_PYTHON_PATH

MARKER = '[yolov5.export.start$$$]["/tmp/out"][$$$yolov5.export.end]'


def _run_export(cfg):
    calls = {}

    def fake(args, **kw):
        calls["args"] = args
        return MARKER

    with patch("export_v.export_v5.subprocess.check_output", side_effect=fake), \
         patch("export_v.export_v5.os.path.exists", return_value=True):
        export_v5(None, cfg, None)
    return calls["args"]


def test_onnx_dispatches_to_main_python_and_main_wrapper():
    args = _run_export(ModelExportConfig(model_path="/tmp/m.pt", version="v5",
                                         format="onnx", imgsz=640))
    assert args[0] == ULTRALYTICS_PYTHON_PATH
    # 分隔符归一化:Windows 上 Path.resolve() 产出反斜杠,断言按 POSIX 分隔符写
    assert "v5_wrap_main.py" in args[1].replace("\\", "/")
    assert args[0] != YOLOV5_PYTHON_PATH


def test_non_onnx_keeps_v5_venv_and_original_wrapper():
    args = _run_export(ModelExportConfig(model_path="/tmp/m.pt", version="v5",
                                         format="torchscript", imgsz=640))
    assert args[0] == YOLOV5_PYTHON_PATH
    assert "process/v5_wrap.py" in args[1].replace("\\", "/")
