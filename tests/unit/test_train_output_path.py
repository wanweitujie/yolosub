# tests/unit/test_train_output_path.py
"""回归:训练子进程写入 status.json 的 output 字段必须是真实产物目录。

2026-10-15 实测产物 status.json(exp20261015065909_990752,v5)拿到:
    "output": ".../exp20261015065909_990752/status.json/output"   ← 多了一层 status.json
而 v8 侧用 trainer.save_dir,本就是 ".../exp20261015065909_990752/output"。

该字段是中间件定位训练产物的契约(output/last/best 三者并列,见
docs/superpowers/specs/2026-10-15-训练日志接口脱敏-design.md 第一节:脱敏不得下沉到
这三个字段,因为中间件依赖它们),路径不存在会直接击穿下游下载/读 csv。

按项目约定单测在 Linux 容器内跑:
    PYTHONPATH=src pytest tests/unit/test_train_output_path.py -v
"""
from pathlib import Path

from train_v.process.base_train import resolve_output_dir


def test_output_dir_is_sibling_of_status_file(tmp_path):
    """{exp_dir}/status.json + name="output" → {exp_dir}/output(而非 {exp_dir}/status.json/output)。"""
    exp_dir = tmp_path / "runs" / "train" / "202610" / "exp20261015065909_990752"
    exp_dir.mkdir(parents=True)

    got = Path(resolve_output_dir(exp_dir / "status.json", "output"))

    assert got.name == "output"
    assert got.parent.name == "exp20261015065909_990752"   # 父目录是 exp_dir,不是 status.json
    assert "status.json" not in got.parts                  # 原缺陷:路径里混进 status.json


def test_output_dir_honors_name(tmp_path):
    """name 由 manager 固定传 "output",但函数不得写死——换名时目录应跟着变。"""
    exp_dir = tmp_path / "exp_x"
    exp_dir.mkdir(parents=True)

    got = Path(resolve_output_dir(exp_dir / "status.json", "output2"))

    assert got.name == "output2"
    assert got.parent.name == "exp_x"
