# tests/unit/test_signal_log.py
"""回归:训练子进程 SIGTERM handler 必须区分主进程与 fork 出的 dataloader worker。

事故(2026-09,根因与处置见内部档案):v5 训练 --batch -1(autobatch)
建/销毁 loader 时 torch terminate worker,worker 继承的 SIGTERM handler 把主进程的
running 污染成假 stopped(收到中止信号),而主进程从未收到信号、继续训练到 finished,
上层却被假 stopped 误导。修复:worker 分支只 os._exit,绝不写共享 status.json。

按项目约定单测在 Linux 容器内跑:
    PYTHONPATH=src pytest tests/unit/test_signal_log.py -v
"""
import json
from unittest.mock import patch

from train_v.process import base_train


def test_main_process_writes_stopped(tmp_path, monkeypatch):
    """主进程收到 SIGTERM:running 态照旧写 stopped(原行为保留)。"""
    sf = tmp_path / "status.json"
    sf.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    monkeypatch.setattr(base_train, "STATUS_FILE", sf)

    # handler 调用必须在 with 内:出 with 即解除 patch,真实 sys.exit 会抛 SystemExit
    # (原 _capture_handler 抽函数在 with 内 return,补丁失效致本用例必败,code review 修复)
    with patch("train_v.process.base_train.signal.signal") as m_sig, \
         patch("train_v.process.base_train.sys.exit") as m_exit:
        base_train.setup_signal()
        handler = m_sig.call_args.args[1]
        handler(15, None)   # 15 = SIGTERM;测试进程即"主进程",pid 与注册时一致
        assert m_exit.call_count == 1

    assert json.loads(sf.read_text(encoding="utf-8"))["status"] == "stopped"


def test_worker_fork_must_not_write_status(tmp_path, monkeypatch):
    """fork 出的 dataloader worker 收到 SIGTERM:绝不写共享 status.json,经 os._exit 退出。"""
    sf = tmp_path / "status.json"
    sf.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    monkeypatch.setattr(base_train, "STATUS_FILE", sf)

    with patch("train_v.process.base_train.signal.signal") as m_sig, \
         patch("train_v.process.base_train.sys.exit") as m_exit, \
         patch("train_v.process.base_train.os._exit") as m_hard_exit:
        m_hard_exit.side_effect = SystemExit   # 模拟真实 os._exit 立即终止进程
        base_train.setup_signal()
        handler = m_sig.call_args.args[1]
        # 模拟 fork worker:setup_signal 注册后(getpid 已捕获为主进程),本进程 pid 变化
        monkeypatch.setattr(base_train.os, "getpid", lambda: 424242)
        exited = False
        try:
            handler(15, None)
        except SystemExit:
            exited = True
        assert exited, "worker 应经 os._exit 立即退出"

    assert json.loads(sf.read_text(encoding="utf-8"))["status"] == "running"  # 未被污染
    assert m_hard_exit.call_count == 1        # worker 走硬退出
    assert m_exit.call_count == 0             # 不走主进程 sys.exit 分支


def test_main_process_late_sigterm_keeps_terminal(tmp_path, monkeypatch):
    """终态(finished)后晚到 SIGTERM(主进程):不覆盖终态。"""
    sf = tmp_path / "status.json"
    sf.write_text(json.dumps({"status": "finished"}), encoding="utf-8")
    monkeypatch.setattr(base_train, "STATUS_FILE", sf)

    with patch("train_v.process.base_train.signal.signal") as m_sig, \
         patch("train_v.process.base_train.sys.exit"):
        base_train.setup_signal()
        handler = m_sig.call_args.args[1]
        handler(15, None)   # 同上:调用须在 with 内,防真实 sys.exit 抛 SystemExit

    assert json.loads(sf.read_text(encoding="utf-8"))["status"] == "finished"