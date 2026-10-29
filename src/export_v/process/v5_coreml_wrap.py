"""
v5 mlmodel(coreml) 导出 wrapper。

第三方 yolov5_convert_weight_to_coreml 目录保持 upstream 原样不改,本 wrapper 在外层
修正输入数据后以 runpy 执行原版 convert.py:

  data yaml 的 names 写成数字(如 names: [1, 2, ...])时,加载出的 model.names 值为 int,
  而 CoreML NMS 的 stringClassLabels.vector 只接受 str,会在 createNmsModelSpec 处抛
  TypeError: "N has type int, but expected one of: bytes, unicode"。

处理方式:monkeypatch models.experimental.attempt_load,在其返回处把 names 统一规范化为
字符串——这是 convert.py 中标签的唯一来源(labels = model.names),数据一进第三方代码
即为干净形态,后续任何新增消费点自动受益。

CLI 参数与 convert.py 完全一致(--yolov5-repo/--weight/--device/--img-size/--fp16/--int8)。
"""

import os
import runpy
import sys
from pathlib import Path

# 项目 src 根目录加入 sys.path,供 __main__ 中导入 utils_v(同 v5_wrap.py 模式)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def normalize_class_labels(names):
    """把 model.names 的类别标签统一转为 str(dict 值 / list、tuple 元素均可)"""
    if isinstance(names, dict):
        return {k: str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return [str(x) for x in names]
    return names


if __name__ == "__main__":
    from utils_v.index import YOLOV5_PACKAGE_DIR

    convert_path = os.path.join(YOLOV5_PACKAGE_DIR, 'yolov5_convert_weight_to_coreml', 'convert.py')
    package_dir = os.path.dirname(convert_path)

    # convert.py 以 runpy 执行时不会像直接运行脚本那样自动把脚本目录加进 sys.path,
    # 不补会导致 convert.py 内 from conversion_modules import * 报 ModuleNotFoundError
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

    # 从 CLI 提取 --yolov5-repo(convert.py 的 argparse 必填参数),yolov5 包需可导入
    yolov5_repo = sys.argv[sys.argv.index('--yolov5-repo') + 1]
    if yolov5_repo not in sys.path:
        sys.path.insert(0, yolov5_repo)

    import models.experimental
    _orig_attempt_load = models.experimental.attempt_load

    def _attempt_load(*args, **kwargs):
        model = _orig_attempt_load(*args, **kwargs)
        if hasattr(model, 'names'):
            model.names = normalize_class_labels(model.names)
        return model

    models.experimental.attempt_load = _attempt_load

    # runpy 以 __main__ 执行 convert.py,其 argparse 从 sys.argv 解析,
    # 故把 argv[0] 换成 convert.py 自身路径、透传其余参数
    sys.argv = [convert_path] + sys.argv[1:]
    runpy.run_path(convert_path, run_name="__main__")
