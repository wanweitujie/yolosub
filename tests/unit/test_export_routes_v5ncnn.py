from unittest.mock import MagicMock, patch

from modal import ModelExportConfig
from export_v.routes import export_model, init_export

init_export(MagicMock())  # export_model 内引用全局 logger,经 init_export 绑定


def test_routes_v5_ncnn_routes_to_transform():
    cfg = ModelExportConfig(model_path="/tmp/m.pt", version="v5",
                            format="ncnn", imgsz=640, half=False)
    calls = {"export": [], "transform": []}

    def fake_export_v5(logger, config, gpu_id):
        calls["export"].append((config.format, config.half))
        return "/tmp/m.torchscript"

    def fake_transform(config, gpu_id, result, logger):
        calls["transform"].append((config.format, config.half))
        return "/tmp/m"

    with patch("export_v.routes.get_available_gpu", return_value=0), \
         patch("export_v.routes.export_v5", side_effect=fake_export_v5), \
         patch("export_v.routes.transform_torchscript_ncnn", side_effect=fake_transform), \
         patch("export_v.routes._write_ncnn_classes"), \
         patch("export_v.routes.os.path.exists", return_value=True), \
         patch("export_v.routes.os.path.isfile", return_value=True), \
         patch("export_v.routes.os.remove"), \
         patch("export_v.routes.zip_with_max_compression", return_value="/tmp/m.zip"):
        export_model(cfg)

    assert calls["export"] == [("torchscript", None)]   # 导出阶段:format→torchscript, half 剥离
    # 转换阶段:format 恢复 ncnn;half 恢复原值 = 入参 half=False(非 None)。见 task-3-report 偏差说明。
    assert calls["transform"] == [("ncnn", False)]