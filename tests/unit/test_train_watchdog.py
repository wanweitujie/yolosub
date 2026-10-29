"""训练假死/致命错误看门狗单测。

背景:dataloader worker 的 queue feeder 线程在 /dev/shm 不足等场景静默崩溃
(异常仅打印到 stderr,不传播到主进程),主进程取首个 batch 时永久挂起,
yolo.train() 既不返回也不抛异常 -> try/except 捕获不到,status 永久卡 running。
本看门狗在训练子进程内旁路检测:命中已知致命日志关键字(且 status 已一周期无更新)
或 status.json 长时间无更新 -> 写 status=error(保留进度字段)+ 终止进程组。
"""
import json
import os
from unittest.mock import patch

import pytest

from train_v.process import base_train


@pytest.fixture(autouse=True)
def _reset_watchdog_state():
    """每用例复位模块级看门狗状态(事件 + 日志 tail 偏移),保证用例顺序无关。"""
    base_train._reset_watchdog_state()
    yield
    base_train._reset_watchdog_state()


def _set_status_mtime(tmp_path, mtime):
    """写 status.json 并把 mtime 钉在指定值,用于陈旧检测。"""
    sf = tmp_path / "status.json"
    sf.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    os.utime(sf, (mtime, mtime))
    return sf


def _wait_set_after(n):
    """返回 wait mock:前 n 次正常返回,之后置位 done(避免测试死循环)。"""
    state = {"count": 0}

    def _wait(timeout):
        state["count"] += 1
        if state["count"] > n:
            base_train._watchdog_done.set()
        return None

    return _wait


# ===== 致命日志模式检测(增量 tail)=====

def test_fatal_pattern_detected(tmp_path):
    lf = tmp_path / "train.log"
    lf.write_text("...\nunable to allocate shared memory(shm) for file </torch_1>...\n",
                  encoding="utf-8")
    old = base_train.LOG_FILE
    base_train.LOG_FILE = lf
    try:
        assert base_train._check_fatal_patterns() is True
    finally:
        base_train.LOG_FILE = old


def test_fatal_pattern_absent(tmp_path):
    lf = tmp_path / "train.log"
    lf.write_text("epoch 1/10 ...", encoding="utf-8")
    old = base_train.LOG_FILE
    base_train.LOG_FILE = lf
    try:
        assert base_train._check_fatal_patterns() is False
    finally:
        base_train.LOG_FILE = old


def test_fatal_pattern_missing_log_no_crash(tmp_path):
    """log 文件不存在 -> False,不抛。"""
    old = base_train.LOG_FILE
    base_train.LOG_FILE = tmp_path / "no_such.log"
    try:
        assert base_train._check_fatal_patterns() is False
    finally:
        base_train.LOG_FILE = old


def test_fatal_pattern_incremental_tail(tmp_path):
    """增量 tail:首次检查后新追加的致命日志也能被后续检查发现,不依赖全量重读。"""
    lf = tmp_path / "train.log"
    lf.write_text("epoch 1/10 ...", encoding="utf-8")
    old = base_train.LOG_FILE
    base_train.LOG_FILE = lf
    try:
        assert base_train._check_fatal_patterns() is False
        with open(lf, "a", encoding="utf-8") as f:      # 追加新内容
            f.write("\nunable to allocate shared memory...\n")
        assert base_train._check_fatal_patterns() is True
    finally:
        base_train.LOG_FILE = old


# ===== status.json 陈旧检测 =====

def test_status_stale_true(tmp_path):
    sf = _set_status_mtime(tmp_path, mtime=100)
    old = base_train.STATUS_FILE
    base_train.STATUS_FILE = sf
    try:
        assert base_train._status_stale(now=100 + 601, stale_after=600) is True
    finally:
        base_train.STATUS_FILE = old


def test_status_stale_false(tmp_path):
    sf = _set_status_mtime(tmp_path, mtime=100)
    old = base_train.STATUS_FILE
    base_train.STATUS_FILE = sf
    try:
        assert base_train._status_stale(now=100 + 300, stale_after=600) is False
    finally:
        base_train.STATUS_FILE = old


# ===== 写 error + 退出 =====

@patch("train_v.process.base_train._kill_group_and_exit")
@patch("train_v.process.base_train.write_status")
def test_write_error_and_exit_preserves_progress(mock_write, mock_exit, tmp_path):
    """error 态保留 running 的 pid/epoch/total_epochs/progress 字段,并终止进程组。"""
    sf = tmp_path / "status.json"
    sf.write_text(json.dumps({"status": "running", "pid": 9, "epoch": 3,
                              "total_epochs": 10, "progress": 0.3}), encoding="utf-8")
    old = base_train.STATUS_FILE
    base_train.STATUS_FILE = sf
    try:
        base_train._write_error_and_exit("boom")
    finally:
        base_train.STATUS_FILE = old
    kw = mock_write.call_args.kwargs
    assert kw["status"] == "error"
    assert kw["message"] == "boom"
    assert kw["pid"] == 9
    assert kw["epoch"] == 3
    assert kw["total_epochs"] == 10
    assert kw["progress"] == 0.3
    mock_exit.assert_called_once()


@patch("train_v.process.base_train.os._exit")
@patch("train_v.process.base_train.os.killpg", create=True)
@patch("train_v.process.base_train.os.getpgid", return_value=42, create=True)
def test_kill_group_and_exit(mock_getpgid, mock_killpg, mock_exit):
    """先杀自身进程组(含 dataloader worker)再退出,释放 CUDA context。

    create=True:Windows 上 os 为冻结模块、无 getpgid/killpg,须允许打桩(生产走 Linux 真实存在)。
    断言用 SIGKILL=9 字面量(Windows signal 模块亦无 SIGKILL)。"""
    base_train._kill_group_and_exit()
    mock_killpg.assert_called_once_with(42, 9)
    mock_exit.assert_called_once_with(1)


# ===== 看门狗主循环 =====

def test_watchdog_fires_on_fatal_pattern_when_stale(tmp_path):
    """命中致命日志关键字且 status 已一周期无更新(shm 挂起场景)-> 写 error 并退出。"""
    sf = _set_status_mtime(tmp_path, mtime=0)
    old = base_train.STATUS_FILE
    base_train.STATUS_FILE = sf
    try:
        with patch.object(base_train._watchdog_done, "wait", return_value=None), \
                patch("train_v.process.base_train._check_fatal_patterns", return_value=True), \
                patch("train_v.process.base_train.time.time", return_value=100), \
                patch("train_v.process.base_train._write_error_and_exit") as mock_exit:
            base_train.watchdog_loop()
        mock_exit.assert_called_once()
    finally:
        base_train.STATUS_FILE = old


def test_watchdog_no_fire_on_fatal_pattern_when_fresh(tmp_path):
    """致命日志存在但 status 新鲜(健康训练每批写 status)-> 不误杀。"""
    sf = _set_status_mtime(tmp_path, mtime=9_990)   # now=10000, 年龄=10 <= 周期,不算陈旧
    old = base_train.STATUS_FILE
    base_train.STATUS_FILE = sf
    try:
        with patch.object(base_train._watchdog_done, "wait", side_effect=_wait_set_after(1)), \
                patch("train_v.process.base_train._check_fatal_patterns", return_value=True), \
                patch("train_v.process.base_train.time.time", return_value=10_000), \
                patch("train_v.process.base_train._write_error_and_exit") as mock_exit:
            base_train.watchdog_loop()
        mock_exit.assert_not_called()
    finally:
        base_train.STATUS_FILE = old


def test_watchdog_fires_on_stale_status(tmp_path):
    """status.json 超阈值未更新 -> 写 error 并退出。"""
    sf = _set_status_mtime(tmp_path, mtime=0)
    old = base_train.STATUS_FILE
    base_train.STATUS_FILE = sf
    try:
        with patch.object(base_train._watchdog_done, "wait", return_value=None), \
                patch("train_v.process.base_train._check_fatal_patterns", return_value=False), \
                patch("train_v.process.base_train.time.time", return_value=10_000), \
                patch("train_v.process.base_train._write_error_and_exit") as mock_exit:
            base_train.watchdog_loop()
        mock_exit.assert_called_once()
    finally:
        base_train.STATUS_FILE = old


def test_watchdog_does_not_fire_when_fresh(tmp_path):
    """status 新鲜(有进度)且无致命日志 -> 不误触发。"""
    sf = _set_status_mtime(tmp_path, mtime=9_900)
    old = base_train.STATUS_FILE
    base_train.STATUS_FILE = sf
    try:
        with patch.object(base_train._watchdog_done, "wait", side_effect=_wait_set_after(1)), \
                patch("train_v.process.base_train._check_fatal_patterns", return_value=False), \
                patch("train_v.process.base_train.time.time", return_value=10_000), \
                patch("train_v.process.base_train._write_error_and_exit") as mock_exit:
            base_train.watchdog_loop()
        mock_exit.assert_not_called()
    finally:
        base_train.STATUS_FILE = old


def test_watchdog_stops_when_done():
    """on_train_end/except 置位 done 后,即使后续致命日志出现也不再触发。"""
    with patch.object(base_train._watchdog_done, "wait",
                      side_effect=_wait_set_after(0)), \
            patch("train_v.process.base_train._check_fatal_patterns", return_value=True), \
            patch("train_v.process.base_train._write_error_and_exit") as mock_exit:
        base_train.watchdog_loop()
    mock_exit.assert_not_called()


def test_watchdog_order_independent(tmp_path):
    """事件被置位后再触发 fire 场景(模拟乱序执行):fixture 已复位,fire 仍正常触发。"""
    base_train.stop_watchdog()                      # 模拟前序用例把事件置位
    base_train._reset_watchdog_state()              # 每用例前复位(fixture 语义)
    sf = _set_status_mtime(tmp_path, mtime=0)
    old = base_train.STATUS_FILE
    base_train.STATUS_FILE = sf
    try:
        with patch.object(base_train._watchdog_done, "wait", return_value=None), \
                patch("train_v.process.base_train._check_fatal_patterns", return_value=True), \
                patch("train_v.process.base_train.time.time", return_value=100), \
                patch("train_v.process.base_train._write_error_and_exit") as mock_exit:
            base_train.watchdog_loop()
        mock_exit.assert_called_once()
    finally:
        base_train.STATUS_FILE = old


# ===== init_status 携带 gpu_id(排查参考,非权威) =====

@patch("train_v.process.base_train.start_watchdog")
def test_init_status_writes_gpu_id_from_device(mock_watch, tmp_path):
    """--device 3 -> init_status 写 gpu_id=3;cpu/未传 -> None。"""
    sf = tmp_path / "status.json"
    old_sf, old_dev = base_train.STATUS_FILE, base_train.DEVICE
    base_train.STATUS_FILE = sf
    try:
        base_train.DEVICE = "3"
        base_train.init_status(pid=9, start_time=0, total_epochs=10)
        assert json.loads(sf.read_text(encoding="utf-8"))["gpu_id"] == 3
        base_train.DEVICE = "cpu"
        base_train.init_status(pid=9, start_time=0, total_epochs=10)
        assert json.loads(sf.read_text(encoding="utf-8"))["gpu_id"] is None
    finally:
        base_train.STATUS_FILE, base_train.DEVICE = old_sf, old_dev
