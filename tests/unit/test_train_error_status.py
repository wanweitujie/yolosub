# tests/unit/test_train_error_status.py
"""回归:训练子进程写 status=error 时,字段集必须与 running/finished 对齐。

背景(2026-10-15 核实):全仓 6 个 error 写入点里 **4 个**只写 `{status, message}`
—— 显存预检三处 + train_v5/train_v8 顶层 except;另 2 个(看门狗 _write_error_and_exit)
合并了 pid/epoch/total_epochs/progress。于是同一个 v5 任务,预检失败与训练中崩溃的
error 响应字段数不同,客户端无条件读 data["progress"] 会在预检失败时 KeyError。

统一为:error 恒带 status/message/pid/epoch/total_epochs/progress 六字段。
取值优先级 = 调用方显式传入 > status.json 里最后一次的进度 > 零值兜底。

按项目约定单测在 Linux 容器内跑:
    PYTHONPATH=src pytest tests/unit/test_train_error_status.py -v
"""
import json

from train_v.process import base_train

PROGRESS_KEYS = ("pid", "epoch", "total_epochs", "progress")


def _status_path(tmp_path, monkeypatch, payload=None):
    """把 STATUS_FILE 指到 tmp,并按需预置一份 status.json。返回该 Path。"""
    sf = tmp_path / "status.json"
    if payload is not None:
        sf.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(base_train, "STATUS_FILE", sf)
    return sf


def test_write_error_full_fields_when_no_prior_status(tmp_path, monkeypatch):
    """训练尚未开始(无 status.json)也要写全 4 个进度字段——原缺陷即此处。"""
    sf = _status_path(tmp_path, monkeypatch)

    base_train.write_error("显存预检:预算超限")

    data = json.loads(sf.read_text(encoding="utf-8"))
    assert data["status"] == "error"
    assert data["message"] == "显存预检:预算超限"
    for k in PROGRESS_KEYS:
        assert k in data, f"error 响应缺字段 {k}"


def test_write_error_keeps_last_progress(tmp_path, monkeypatch):
    """运行中崩溃:保留 status.json 里最后一次的进度,前端不丢上下文(原看门狗行为)。"""
    sf = _status_path(tmp_path, monkeypatch,
                      {"status": "running", "pid": 4242, "epoch": 7,
                       "total_epochs": 100, "progress": 0.07})

    base_train.write_error("训练失败:假死")

    data = json.loads(sf.read_text(encoding="utf-8"))
    assert (data["pid"], data["epoch"], data["total_epochs"], data["progress"]) == (4242, 7, 100, 0.07)


def test_write_error_explicit_fields_win(tmp_path, monkeypatch):
    """调用方显式传入优先于 status.json 旧值:预检失败时补上用户要的总轮数。"""
    sf = _status_path(tmp_path, monkeypatch, {"status": "running", "total_epochs": 3})

    base_train.write_error("显存预检:预算超限", total_epochs=100)

    assert json.loads(sf.read_text(encoding="utf-8"))["total_epochs"] == 100


def test_write_error_tolerates_corrupt_status_file(tmp_path, monkeypatch):
    """status.json 坏文件(撕裂/非对象 JSON)不得让 error 写入本身失败——否则连报错都丢了。"""
    sf = _status_path(tmp_path, monkeypatch)
    sf.write_text("[]", encoding="utf-8")

    base_train.write_error("boom")

    data = json.loads(sf.read_text(encoding="utf-8"))
    assert data["status"] == "error"
    for k in PROGRESS_KEYS:
        assert k in data


def test_write_error_and_exit_delegates_to_write_error(tmp_path, monkeypatch):
    """看门狗路径行为不变:仍写全字段并终止进程组(经 write_error 单点实现)。"""
    from unittest.mock import patch
    sf = _status_path(tmp_path, monkeypatch,
                      {"status": "running", "pid": 1, "epoch": 3,
                       "total_epochs": 50, "progress": 0.06})

    with patch("train_v.process.base_train._kill_group_and_exit") as mock_kill:
        base_train._write_error_and_exit("shm 分配失败")

    mock_kill.assert_called_once()
    data = json.loads(sf.read_text(encoding="utf-8"))
    assert data["status"] == "error" and data["epoch"] == 3 and data["total_epochs"] == 50
