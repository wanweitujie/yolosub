# 本地修改说明(Local changes)

本目录相对上游 ultralytics/yolov5(该 vendored 副本自带 GPL-3.0 LICENSE)有以下本地修改,
按 GPL-3.0 §5(a) 显著声明如下。原修改日期不可考,统一登记于 2026-09-20。

1. `__init__.py` —— 新增(上游 v6.2 无此文件)。
2. `requirements.txt:7` —— numpy 锁定 `>=1.18.5,<1.23.2`(coremltools 6 兼容)。
3. `models/experimental.py:81` —— `torch.load(... weights_only=False)`。
4. `utils/general.py:289` —— `YOLOv5_OFFLINE` 环境变量断言(离线自检)。
5. `train.py:248,338` + `utils/loggers/__init__.py:122` —— 自定义 `nis` 参数(3 处)。
6. `setup.py:16` —— `yolov5_local` Custom Local Version 标记(本地重打包);该文件位于 vendored 目录之外:`docker-create/files/mlmodels/setup.py`。
