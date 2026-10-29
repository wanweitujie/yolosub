# tests/unit/test_log_redact.py
"""训练日志读时脱敏单测:路径/显卡型号纯函数 + manager.log 出口 + status 约束守卫。

约束(spec 2026-10-15 训练日志接口脱敏):
  - 只脱日志读接口的响应;/train/status 的 output/last/best **必须**保持容器内真实路径
    ——中间件靠它定位真实产物,故脱敏不得下沉到公共层(见 test_status_paths_stay_raw)。
  - 磁盘 train.log 保持原样,仅读时替换(见 test_log_does_not_rewrite_disk_file)。
  - 显卡型号(`CUDA:0 (Quadro RTX 5000, 15929MiB)`)同样只在日志出口过滤。

按项目约定单测在 Linux 容器内跑:
    PYTHONPATH=src pytest tests/unit/test_log_redact.py -v
"""
import json
import pathlib
from unittest.mock import patch

from utils_v.index import (
    VENV_YOLOV5_PATH, VENV_PATH_EXPORT_MLPACKAGE_IOS14, redact_container_paths,
    redact_gpu_model, strip_ansi,
)

# 打桩前置:manager 模块级 Path(YOLOV5_PYTHON_PATH).exists() 在 Windows 直接 RuntimeError,
# 不打桩会中断整轮 pytest 收集(照抄 test_train_start_validation.py:9-11 既有约定)
with patch.object(pathlib.Path, "exists", return_value=True):
    from train_v.manager import TrainingManager

EXP = "exp20260915071031_990238"


# ===== 纯函数 redact_container_paths =====

def test_data_root_with_trailing_slash():
    """容器数据根 + 尾斜杠 → 只去掉目录名,路径仍是合法绝对路径。"""
    assert redact_container_paths(
        "/docker-data-yolo/runs/train/202609/exp20260915071031_990238/output"
    ) == "/runs/train/202609/exp20260915071031_990238/output"


def test_bare_prefix_without_slash_removed():
    """裸前缀(后跟空白或行尾)整体删除——目录名本身不得残留。"""
    assert redact_container_paths("Results saved to /docker-data-yolo") == "Results saved to "
    assert redact_container_paths("root=/docker-data-yolo done") == "root= done"


def test_multiple_occurrences_in_one_line():
    """一行多次出现须全部替换(日志里 Scanning images/labels 同行)。"""
    src = ("Scanning /docker-data-yolo/train_data/x/images and "
           "/docker-data-yolo/train_data/x/labels")
    assert redact_container_paths(src) == \
        "Scanning /train_data/x/images and /train_data/x/labels"


def test_venv_prefixes_redacted():
    """v5 / iOS14 导出 venv 同属自建布局,一并脱(前缀取自常量,防改名后漏更)。"""
    assert redact_container_paths(f"{VENV_YOLOV5_PATH}/mlmodels/yolov5/train.py") == \
        "/mlmodels/yolov5/train.py"
    assert redact_container_paths(f"{VENV_PATH_EXPORT_MLPACKAGE_IOS14}/bin/python") == \
        "/bin/python"


def test_generic_linux_paths_untouched():
    """通用 Linux 路径不暴露容器身份,且大面积替换会损伤 traceback 可读性,保持原样。"""
    for src in ('File "/usr/lib/python3.10/multiprocessing/process.py", line 108',
                "/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py"):
        assert redact_container_paths(src) == src


def test_similar_names_not_redacted():
    """防误伤:前缀后必须跟 / 或空白/行尾,近似名不得命中。"""
    src = "/docker-data-yolo-backup/x /venv_yolov5_extra/y"
    assert redact_container_paths(src) == src


def test_no_hit_returns_text_unchanged():
    src = "Epoch 1/1  box_loss 1.234  cls_loss 0.567\n"
    assert redact_container_paths(src) == src


def test_path_wrapped_in_ansi_codes_redacted():
    """真实形态:ultralytics 末行 `Results saved to {colorstr('bold', path)}` 把路径包在
    ANSI 转义码里(实测 87 行日志有 15 行含转义码),前缀替换不得被转义码打断。"""
    raw = "Results saved to \x1b[1m/docker-data-yolo/runs/train/202610/exp_x/output\x1b[0m"
    out = redact_container_paths(raw)
    assert "/docker-data-yolo" not in out
    assert out == "Results saved to \x1b[1m/runs/train/202610/exp_x/output\x1b[0m"


# ===== 纯函数 strip_ansi =====

def test_strip_ansi_real_sequences():
    """实测日志里出现过的四种序列:加粗/重置/蓝色/擦行([K 无数字参数,易被正则漏掉)。"""
    assert strip_ansi("\x1b[1mbold\x1b[0m") == "bold"
    assert strip_ansi("\x1b[34mblue\x1b[0m") == "blue"
    assert strip_ansi("progress\x1b[K done") == "progress done"


def test_strip_ansi_real_log_line():
    """真实形态(实测 train.log):擦行 + 蓝色 + 加粗包裹前缀标签。"""
    raw = "\x1b[K\x1b[34m\x1b[1mtrain: \x1b[0mScanning /x/labels.cache... 666 images"
    assert strip_ansi(raw) == "train: Scanning /x/labels.cache... 666 images"


def test_strip_ansi_no_escape_returns_unchanged():
    src = "Epoch 1/1  box_loss 1.234\nSpeed: 0.3ms inference per image\n"
    assert strip_ansi(src) == src


def test_strip_ansi_keeps_carriage_return():
    """\\r 必须保留,不得一并剥掉。

    实测 87 行日志含 39 个 \\r,其中仅 5 个在行首(\\r\\x1b[K 擦行残留),另 34 个是
    tqdm 进度条的**原地重绘**分隔符(形如 `320: 0% ███ 0/42 0.5s\\r   1/1  0.539G ...`)。
    剥掉会把几十次重绘挤成一整行;保留则由浏览器在解析 HTML 时归一为换行,每次更新
    各成一行、可读性更好。
    """
    assert strip_ansi("a\rb") == "a\rb"


# ===== 纯函数 redact_gpu_model =====

def test_gpu_model_redacted_ultralytics_banner():
    """实测 ultralytics banner:删掉显卡型号括注,保留 CUDA:0(设备序号仍有排查价值)。"""
    src = ("Ultralytics 8.4.121 🚀 Python-3.10.12 torch-2.11.0+cu128 "
           "CUDA:0 (Quadro RTX 5000, 15929MiB)")
    assert redact_gpu_model(src) == \
        "Ultralytics 8.4.121 🚀 Python-3.10.12 torch-2.11.0+cu128 CUDA:0"


def test_gpu_model_redacted_v5_banner():
    """实测 v5 banner(日期版本号形态)同样命中。"""
    src = ("YOLOv5 🚀 2025-11-5 Python-3.10.12 torch-2.11.0+cu128 "
           "CUDA:0 (Quadro RTX 5000, 15929MiB)")
    assert redact_gpu_model(src) == \
        "YOLOv5 🚀 2025-11-5 Python-3.10.12 torch-2.11.0+cu128 CUDA:0"


def test_gpu_model_redacted_multi_gpu_lines():
    """多卡时框架每卡一行 CUDA:{d},前导空格用于对齐,须逐行全部替换。"""
    src = ("CUDA:0 (NVIDIA A30, 24576MiB)\n"
           "                 CUDA:1 (NVIDIA A30, 24576MiB)\n")
    assert redact_gpu_model(src) == "CUDA:0\n                 CUDA:1\n"


def test_gpu_model_no_hit_returns_text_unchanged():
    """防误伤:不带 MiB 的括注、纯文本、CUDA 之外的括号一律不动。"""
    for src in ("Epoch 1/1  box_loss 1.234\n",
                "CUDA:0 (something else)",
                "CUDA:0 (Quadro RTX 5000)",
                "speed (0.3ms, 12.1ms) per image"):
        assert redact_gpu_model(src) == src


# ===== manager.log 出口 =====

def _mgr(tmp_path):
    """与 test_train_start_validation._fresh_mgr 同款:单例重置目录与状态。"""
    mgr = TrainingManager()
    mgr.run_dir = tmp_path / "runs" / "train"
    mgr.run_dir.mkdir(parents=True, exist_ok=True)
    mgr.trainings = {}
    mgr.transport_yaml_files = []
    return mgr


def _write_train_log(mgr, text):
    d = mgr.run_dir / "202609" / EXP
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.log").write_text(text, encoding="utf-8")
    return d / "train.log"


def test_log_returns_redacted_text(tmp_path):
    """日志接口出口已脱敏:容器目录名不出现,剩余路径仍可用。"""
    mgr = _mgr(tmp_path)
    _write_train_log(mgr, "Results saved to /docker-data-yolo/runs/train/202609/%s/output\n" % EXP)

    out = mgr.log(EXP)
    assert "/docker-data-yolo" not in out
    assert f"/runs/train/202609/{EXP}/output" in out


def test_log_strips_ansi_and_redacts_path_together(tmp_path):
    """出口同时做两件事:剥 ANSI + 路径脱敏(实测末行两者叠加)。"""
    mgr = _mgr(tmp_path)
    _write_train_log(
        mgr,
        "Results saved to \x1b[1m/docker-data-yolo/runs/train/202610/exp_x/output\x1b[0m\n")

    out = mgr.log(EXP)
    assert "\x1b" not in out
    assert "/docker-data-yolo" not in out
    assert out == "Results saved to /runs/train/202610/exp_x/output\n"


def test_log_returns_text_without_gpu_model(tmp_path):
    """日志出口已过滤显卡型号:型号与显存总量不出现,`CUDA:0` 保留。"""
    mgr = _mgr(tmp_path)
    _write_train_log(
        mgr,
        "YOLOv5 🚀 2025-11-5 Python-3.10.12 torch-2.11.0+cu128 "
        "CUDA:0 (Quadro RTX 5000, 15929MiB)\n")

    out = mgr.log(EXP)
    assert "Quadro RTX 5000" not in out
    assert "15929MiB" not in out
    assert out.endswith("CUDA:0\n")


def test_log_does_not_rewrite_disk_file(tmp_path):
    """守住「只读时脱敏」:磁盘内容保持原样,容器内排查仍需真实路径。"""
    mgr = _mgr(tmp_path)
    raw = ("saved to /docker-data-yolo/runs/x\n"
           "CUDA:0 (Quadro RTX 5000, 15929MiB)\n")
    f = _write_train_log(mgr, raw)

    mgr.log(EXP)
    assert f.read_text(encoding="utf-8") == raw


def test_log_missing_file_still_returns_empty(tmp_path):
    """无日志文件时行为不变(不因脱敏改动破坏空态)。"""
    mgr = _mgr(tmp_path)
    assert mgr.log(EXP) == ""


def test_status_paths_stay_raw(tmp_path):
    """约束守卫:status 的 output/last/best 保持真实路径。

    中间件依赖这三个字段定位真实产物,脱敏只能挂日志出口;若日后有人把
    redact_container_paths 下沉到公共层,本用例会红。
    """
    mgr = _mgr(tmp_path)
    d = mgr.run_dir / "202609" / EXP
    d.mkdir(parents=True, exist_ok=True)
    raw_output = f"/docker-data-yolo/runs/train/202609/{EXP}/output"
    (d / "status.json").write_text(json.dumps({
        "status": "finished", "exp": EXP, "version": "v8",
        "output": raw_output,
        "last": raw_output + "/weights/last.pt",
        "best": raw_output + "/weights/best.pt",
    }), encoding="utf-8")

    data = mgr.status(EXP)
    assert data["output"] == raw_output
    assert data["last"].startswith("/docker-data-yolo/")
    assert data["best"].startswith("/docker-data-yolo/")
