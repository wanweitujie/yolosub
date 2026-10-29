# v5 onnx 导出专用薄 wrapper:在【主环境】跑 yolov5 导出。
# 与 v5_wrap.py 的差异:不调 init_root_folder/set_yolo_env —— 前者会把 /venv_yolov5 的
# site-packages 前插 sys.path 并 chdir,令主环境解释器的 numpy 被劫持到 venv 的 1.23.1
# (coremltools 6.0 锁),import onnx(经 ml_dtypes,需 numpy>=2)必崩。
# 本文件只复刻其 chdir 与 yolov5【代码路径】注入,使 numpy/onnx/ml-dtypes 解析到主环境
# 自洽一套(2.2.6/1.22.0/0.6.0)。marker 协议与 include 改写逻辑与 v5_wrap.py 同构。
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))   # src(供 utils_v 导入)

from utils_v.index import YOLOV5_PACKAGE_DIR, YOLOV5_DIR

import json

if __name__ == "__main__":
    # 复刻 init_root_folder 的 chdir(导出链相对路径依赖);root = 本文件上四级 = 仓库根
    os.chdir(str(Path(__file__).resolve().parent.parent.parent.parent))
    if YOLOV5_PACKAGE_DIR not in sys.path:
        sys.path.insert(0, YOLOV5_PACKAGE_DIR)
    if YOLOV5_DIR not in sys.path:
        sys.path.insert(0, YOLOV5_DIR)
    os.environ["YOLOv5_OFFLINE"] = "true"
    os.environ["YOLOV5_AUTOINSTALL"] = "false"

    # [spec §9.1 裁决(b)] 强制 legacy 导出器:torch 2.11 dynamo 默认会把传入 opset=12 静默提升至 18
    # (version-converter 缺 Resize 的 18→12 adapter,降不回),产物与 8/18 前不一致。
    # 注入 dynamo=False 保 opset=12——实证与旧镜像 ground truth(opset 12/输出名 output/
    # shape [1,25200,85])完全一致。
    import torch.onnx as _torch_onnx
    _orig_onnx_export = _torch_onnx.export
    def _legacy_onnx_export(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return _orig_onnx_export(*args, **kwargs)
    _torch_onnx.export = _legacy_onnx_export

    from yolov5.export import parse_opt, run
    try:
        opt = parse_opt()
        opt.weights = opt.weights[0] if isinstance(opt.weights, list) else opt.weights

        is_ncnn = opt.include[0] == "ncnn"
        if is_ncnn:
            opt.include[0] = "onnx"

        result = run(**vars(opt))
        prev = "yolov5.export"
        print(f"[{prev}.start$$$]{json.dumps(result)}[$$${prev}.end]")
    except Exception as e:
        print(f"export error in process:{e}")
