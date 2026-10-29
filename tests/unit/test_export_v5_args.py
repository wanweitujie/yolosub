from unittest.mock import MagicMock, patch

from modal import ModelExportConfig
from export_v.export_v5 import export_v5


def test_export_v5_does_not_pass_v5_data_flag(tmp_path):
    cfg = ModelExportConfig(model_path="/tmp/m.pt", version="v5",
                            format="ncnn", imgsz=640, v5_data="/tmp/names.yaml")
    calls = {}

    def fake_check_output(args, **kw):
        calls["args"] = args
        return "[yolov5.export.start$$$][\"/tmp/out\"][$$$yolov5.export.end]"

    with patch("export_v.export_v5.subprocess.check_output", side_effect=fake_check_output), \
         patch("export_v.export_v5.os.path.exists", return_value=True):
        export_v5(MagicMock(), cfg, None)

    assert "--v5_data" not in calls["args"]
    assert "--weights" in calls["args"]