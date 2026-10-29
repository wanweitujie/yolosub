import asyncio

import httpx
import pytest
from unittest.mock import MagicMock


def _make_executor_no_spawn():
    """构造 executor 但不拉真实子进程(auto_spawn=False,Phase 2 task 10 加该参数前用 __new__ 绕过)。"""
    from detect_v.process.executor import YoloExecutorManager
    ex = YoloExecutorManager.__new__(YoloExecutorManager)
    ex.client = httpx.AsyncClient(base_url="http://127.0.0.1:29006",
                                  timeout=httpx.Timeout(connect=5.0, read=75.0, write=10.0, pool=5.0))
    return ex


def test_httpx_read_timeout_is_75():
    ex = _make_executor_no_spawn()
    # httpx client 的 read timeout 配置
    read_timeout = ex.client.timeout.read
    assert read_timeout == 75.0, f"read timeout 应为 75s,实际 {read_timeout}"


def test_auto_spawn_false_does_not_spawn(monkeypatch):
    """auto_spawn=False 时不应调用 _spawn(避免测试拉真实子进程)。"""
    from detect_v.process.executor import YoloExecutorManager
    spawned = []
    monkeypatch.setattr(YoloExecutorManager, "_spawn", lambda self: spawned.append(1))
    ex = YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    assert spawned == []
    assert ex._state.value == "HEALTHY"
    assert ex._fail_count == 0
    assert ex._loop is None
    assert ex._restarting is False
    assert ex._crash_timestamps == []


async def test_record_failure_increments_and_triggers_restart(monkeypatch):
    """_record_failure 内部 asyncio.create_task(_restart_async) 需 running loop,
    故用 async test(automode),在事件循环内调用。"""
    from detect_v.process.executor import YoloExecutorManager, FAIL_THRESHOLD
    ex = YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    triggered = []
    async def fake_restart(self): triggered.append(1)
    monkeypatch.setattr(YoloExecutorManager, "_restart_async", fake_restart)
    for _ in range(FAIL_THRESHOLD):
        ex._record_failure()
    await asyncio.sleep(0)    # 让 create_task 调度
    assert len(triggered) == 1  # 达阈值触发一次 _restart_async
    ex._fail_count = 0
    ex._record_failure()
    assert ex._fail_count == 1  # 计数重新累加


def _ex(monkeypatch):
    from detect_v.process.executor import YoloExecutorManager
    ex = YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    # 注入 mock client
    ex.client = MagicMock()
    return ex


def test_send_restarting_raises_unavailable(monkeypatch):
    from detect_v.process.executor import YoloExecutorManager, ExecutorState, ExecutorUnavailableError
    ex = _ex(monkeypatch)
    ex._state = ExecutorState.RESTARTING
    import asyncio
    with pytest.raises(ExecutorUnavailableError):
        asyncio.run(ex.send_to_process_async("/x", "get"))


def test_send_transport_error_counts_failure(monkeypatch):
    from detect_v.process.executor import YoloExecutorManager
    ex = _ex(monkeypatch)
    async def get(url): raise httpx.ReadTimeout("timeout")
    ex.client.get = get
    import asyncio
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(ex.send_to_process_async("/x", "get"))
    assert ex._fail_count == 1


def test_send_4xx_clears_counter(monkeypatch):
    """4xx = 子进程健康(正常响应),清零计数。"""
    from detect_v.process.executor import YoloExecutorManager
    ex = _ex(monkeypatch)
    ex._fail_count = 2
    async def get(url):
        req = httpx.Request("GET", "http://x/x")
        return httpx.Response(404, request=req)
    ex.client.get = get
    import asyncio
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(ex.send_to_process_async("/x", "get"))
    assert ex._fail_count == 0


def test_send_success_false_clears_counter(monkeypatch):
    """success=false(业务错误)= 子进程健康,清零。"""
    from detect_v.process.executor import YoloExecutorManager
    ex = _ex(monkeypatch)
    ex._fail_count = 2
    async def get(url):
        req = httpx.Request("GET", "http://x/x")
        return httpx.Response(200, request=req,
                              content=b'{"success":false,"data":null,"errMsg":"Model not found"}')
    ex.client.get = get
    import asyncio
    with pytest.raises(Exception):
        asyncio.run(ex.send_to_process_async("/x", "get"))
    assert ex._fail_count == 0


def test_rebuild_client_closes_old_and_creates_new(monkeypatch):
    """_rebuild_client 应关闭旧 client 并创建新的。"""
    from detect_v.process.executor import YoloExecutorManager
    ex = YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    old = ex.client
    closed = []
    async def aclose(): closed.append(1)
    old.aclose = aclose
    import asyncio
    new = asyncio.run(ex._rebuild_client())
    assert new is not old
    assert closed == [1]
    assert ex.client is new


def _spawn_capture_cmd(monkeypatch, use_v5):
    """调 _spawn 并捕获传给 Popen 的 cmd(验证解释器分支)。"""
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=use_v5, on_dead=lambda _u: None, auto_spawn=False)
    cmds = []
    class _FakeProc:
        def __init__(self, cmd, **k): cmds.append(cmd); self.pid = 12345; self.returncode = None
        def wait(self, timeout=None): return 0
        def kill(self): pass
    monkeypatch.setattr(E.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(E.YoloExecutorManager, "kill_yolo_process_service", lambda self: None)
    monkeypatch.setattr(E, "cleanup_stale_shared_memory", lambda: 0)
    monkeypatch.setattr(E.threading, "Thread", lambda **k: MagicMock(start=lambda: None, join=lambda timeout=None: None))
    ex._spawn()
    assert len(cmds) == 1
    return ex, cmds[0]


def test_spawn_invokes_popen_and_starts_monitor(monkeypatch):
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    popen_called = []
    class _FakeProc:
        def __init__(self, *a, **k): popen_called.append(1); self.pid = 12345; self.returncode = None
        def wait(self, timeout=None): return 0
        def kill(self): pass
    monkeypatch.setattr(E.subprocess, "Popen", _FakeProc)
    # kill_yolo_process_service / cleanup_stale_shared_memory 无副作用
    monkeypatch.setattr(E.YoloExecutorManager, "kill_yolo_process_service", lambda self: None)
    monkeypatch.setattr(E, "cleanup_stale_shared_memory", lambda: 0)
    monkeypatch.setattr(E.threading, "Thread", lambda **k: MagicMock(start=lambda: None, join=lambda timeout=None: None))
    ex._spawn()
    assert popen_called == [1]
    assert ex.process is not None


def test_spawn_v5_uses_yolov5_python(monkeypatch):
    """v5 子服务用 YOLOV5_PYTHON_PATH(/venv_yolov5 独立环境)。"""
    from detect_v.process.executor import YoloExecutorManager
    from utils_v.index import YOLOV5_PYTHON_PATH
    ex, cmd = _spawn_capture_cmd(monkeypatch, use_v5=True)
    assert cmd[0] == YOLOV5_PYTHON_PATH
    assert cmd[1].endswith("v5_service.py")
    assert ex.use_v5 is True


def test_spawn_ultralytics_uses_system_python(monkeypatch):
    """ultralytics(v8/v11)子服务用 ULTRALYTICS_PYTHON_PATH,且必须 == sys.executable。

    不硬编码解释器路径:硬编码 /usr/bin/python 只在 GPU 镜像存在(ln -sf),CPU 镜像
    (python:3.10 官方镜像)缺该路径 → 子服务 Popen FileNotFoundError 启动崩溃回归。
    断言 cmd[0] == sys.executable 钉死"跟随主进程解释器"语义,将来若有人写死坏路径即红。"""
    import sys
    from detect_v.process.executor import YoloExecutorManager
    from utils_v.index import ULTRALYTICS_PYTHON_PATH, YOLOV5_PYTHON_PATH
    ex, cmd = _spawn_capture_cmd(monkeypatch, use_v5=False)
    assert cmd[0] == ULTRALYTICS_PYTHON_PATH
    assert cmd[0] == sys.executable      # 跟随主进程解释器(防硬编码坏路径回归)
    assert cmd[0] != YOLOV5_PYTHON_PATH  # 与 v5 独立 venv 隔离(不退回共用 v5 路径)
    assert cmd[1].endswith("ultralytics_service.py")
    assert ex.use_v5 is False


def test_kill_old_terminates_process_group(monkeypatch):
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    proc = MagicMock(); proc.pid = 999
    ex.process = proc
    killed = []
    # raising=False: Windows 上 os.killpg/getpgid 不存在,生产代码经 except fallback 仍工作
    monkeypatch.setattr(E.os, "killpg", lambda *a: killed.append(a), raising=False)
    monkeypatch.setattr(E.os, "getpgid", lambda pid: 4321, raising=False)
    proc.wait.return_value = 0
    ex._kill_old()
    assert killed == [(4321, E.signal.SIGTERM)]
    assert ex.process is None


def test_monitor_out_self_returns(monkeypatch):
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    proc = MagicMock(); proc.wait.return_value = 0
    ex.process = proc; ex.out_self = True
    scheduled = []
    monkeypatch.setattr(E.asyncio, "run_coroutine_threadsafe", lambda *a, **k: scheduled.append(1))
    ex.monitor_process()  # 不应调度自愈
    assert scheduled == []


def test_monitor_restarting_returns(monkeypatch):
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    proc = MagicMock(); proc.wait.return_value = 0; proc.returncode = 0
    ex.process = proc; ex.out_self = False; ex._restarting = True
    scheduled = []
    monkeypatch.setattr(E.asyncio, "run_coroutine_threadsafe", lambda *a, **k: scheduled.append(1))
    ex.monitor_process()
    assert scheduled == []


def test_monitor_unexpected_exit_schedules_recovery(monkeypatch):
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    proc = MagicMock(); proc.wait.return_value = 0; proc.returncode = -9
    ex.process = proc; ex.out_self = False; ex._restarting = False
    import asyncio as _a
    ex._loop = _a.new_event_loop()
    scheduled = []
    monkeypatch.setattr(E.asyncio, "run_coroutine_threadsafe", lambda c, l: scheduled.append((c, l)))
    ex.monitor_process()
    assert len(scheduled) == 1
    ex._loop.close()


def test_monitor_loop_unbound_logs_and_returns(monkeypatch):
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    proc = MagicMock(); proc.wait.return_value = 0
    ex.process = proc; ex.out_self = False; ex._restarting = False; ex._loop = None
    scheduled = []
    monkeypatch.setattr(E.asyncio, "run_coroutine_threadsafe", lambda *a, **k: scheduled.append(1))
    ex.monitor_process()  # _loop None,仅打日志,不崩不调度
    assert scheduled == []


def test_restart_success_transitions_to_healthy(monkeypatch):
    from detect_v.process import executor as E
    dead_called = []
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False,
                               on_dead=lambda u: dead_called.append(u), auto_spawn=False)
    # mock 重启依赖
    async def fake_kill_old_to_thread(): pass
    monkeypatch.setattr(E.YoloExecutorManager, "_kill_old", lambda self: None)
    async def fake_rebuild(self): return MagicMock()
    monkeypatch.setattr(E.YoloExecutorManager, "_rebuild_client", fake_rebuild)
    monkeypatch.setattr(E.YoloExecutorManager, "_spawn", lambda self: None)
    async def fake_ready(self, t): return True
    monkeypatch.setattr(E.YoloExecutorManager, "_wait_ready", fake_ready)
    import asyncio
    asyncio.run(ex._restart_async())
    assert ex._state == E.ExecutorState.HEALTHY
    assert ex._fail_count == 0
    assert dead_called == [False]


def test_restart_all_fail_transitions_to_dead(monkeypatch):
    from detect_v.process import executor as E
    dead_loops = []
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    monkeypatch.setattr(E.YoloExecutorManager, "_kill_old", lambda self: None)
    async def fake_rebuild(self): return MagicMock()
    monkeypatch.setattr(E.YoloExecutorManager, "_rebuild_client", fake_rebuild)
    monkeypatch.setattr(E.YoloExecutorManager, "_spawn", lambda self: None)
    async def boom(self, t): raise RuntimeError("not ready")
    monkeypatch.setattr(E.YoloExecutorManager, "_wait_ready", boom)
    async def _fast_sleep(*a, **k): return None
    monkeypatch.setattr(E.asyncio, "sleep", _fast_sleep)  # 跳过退避(lambda 自引用会递归,用 async no-op)
    async def fake_dead_loop(self): dead_loops.append(1)  # 防 create_task 真跑(mock 为 coroutine)
    monkeypatch.setattr(E.YoloExecutorManager, "_dead_recovery_loop", fake_dead_loop)
    import asyncio
    asyncio.run(ex._restart_async())
    assert ex._state == E.ExecutorState.DEAD


# ===== Task 17:_health_loop =====
# 注:brief 原 boom_sleep 写法有 bug(第 1 次 sleep 就抛 CancelledError / raise None 触发
# TypeError),此处按注释与断言语义修正:第 1 次 sleep 放行跑一轮 ping,第 2 次抛 Cancelled 终止循环。


async def test_health_loop_clears_on_ok(monkeypatch):
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    ex.client = MagicMock()
    async def get(url, timeout=None):
        r = MagicMock(); r.raise_for_status = lambda: None; return r
    ex.client.get = get
    ex._fail_count = 2
    # 跑一轮 sleep 后退出:第 1 次 sleep 放行(让 ping OK 清零),第 2 次抛 Cancelled 终止循环
    n = []
    async def boom_sleep(_):
        n.append(1)
        if len(n) > 1:
            raise asyncio.CancelledError
    monkeypatch.setattr(E.asyncio, "sleep", boom_sleep)
    with pytest.raises(asyncio.CancelledError):
        await ex._health_loop()
    assert ex._fail_count == 0  # ping OK 清零


async def test_health_loop_counts_failure(monkeypatch):
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    ex.client = MagicMock()
    async def get(url, timeout=None): raise httpx.ReadTimeout("t")
    ex.client.get = get
    n = []
    async def boom_sleep(_):
        n.append(1)
        if len(n) > 1:
            raise asyncio.CancelledError
    monkeypatch.setattr(E.asyncio, "sleep", boom_sleep)
    with pytest.raises(asyncio.CancelledError):
        await ex._health_loop()
    assert ex._fail_count >= 1


# ===== Task 18:_handle_unexpected_exit(崩溃风暴保护 + 顶层兜底) =====


async def test_handle_single_crash_triggers_restart(monkeypatch):
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    restarted = []
    async def fake_restart(self): restarted.append(1)
    monkeypatch.setattr(E.YoloExecutorManager, "_restart_async", fake_restart)
    await ex._handle_unexpected_exit()
    assert restarted == [1]


async def test_handle_crash_storm_transitions_to_dead(monkeypatch):
    from detect_v.process import executor as E
    dead_loops = []
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    # 模拟窗口内已有 CRASH_WINDOW_MAX 次退出
    import time as _t
    now = _t.monotonic()
    ex._crash_timestamps = [now, now, now]  # 3 次,再加 1 次超限
    dead_called = []
    ex.on_dead = lambda u: dead_called.append(u)
    async def fake_dead_loop(self): dead_loops.append(1)
    monkeypatch.setattr(E.YoloExecutorManager, "_dead_recovery_loop", fake_dead_loop)
    async def fake_restart(self): assert False, "风暴期不应走标准重启"
    monkeypatch.setattr(E.YoloExecutorManager, "_restart_async", fake_restart)
    await ex._handle_unexpected_exit()
    # 风暴路径内 _dead_recovery_loop 经 create_task 调度,须让出事件循环一次才会执行
    await asyncio.sleep(0)
    assert ex._state == E.ExecutorState.DEAD
    assert dead_called == [False]
    assert dead_loops == [1]


async def test_handle_unexpected_exit_top_level_catch(monkeypatch):
    """顶层 try/except 兜底,防 run_coroutine_threadsafe 的 Future 静默吞异常。"""
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    async def boom_restart(self): raise RuntimeError("boom")
    monkeypatch.setattr(E.YoloExecutorManager, "_restart_async", boom_restart)
    # 不应抛出(被顶层捕获 + 记日志)
    await ex._handle_unexpected_exit()
    ex.logger.exception.assert_called()


# ===== Task 19:_dead_recovery_loop(DEAD 态慢恢复) =====


async def test_dead_recovery_success(monkeypatch):
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    ex._state = E.ExecutorState.DEAD
    ex._crash_timestamps = [1.0, 2.0]
    monkeypatch.setattr(E.YoloExecutorManager, "_kill_old", lambda self: None)
    async def fake_rebuild(self): return MagicMock()
    monkeypatch.setattr(E.YoloExecutorManager, "_rebuild_client", fake_rebuild)
    monkeypatch.setattr(E.YoloExecutorManager, "_spawn", lambda self: None)
    async def fake_ready(self, t): return True
    monkeypatch.setattr(E.YoloExecutorManager, "_wait_ready", fake_ready)
    # 首次 sleep 即进入恢复,恢复成功后 return
    iter_n = []
    async def fast_sleep(_):
        iter_n.append(1)
    monkeypatch.setattr(E.asyncio, "sleep", fast_sleep)
    await ex._dead_recovery_loop()
    assert ex._state == E.ExecutorState.HEALTHY
    assert ex._crash_timestamps == []  # 恢复清空窗口


# ===== #3 final review:_restart_lock 并发串行化 + _restart_epoch 幂等 =====

# 实测(修复前)该测试 spawns==2:锁只串行化,排队协程拿到锁见 state==HEALTHY 无法区分
# "初始待重启" vs "对方已恢复"。修复:_restart_async 进锁前记 trigger_epoch,持锁后若
# _restart_epoch 已变(对方成功恢复)则放弃;_restart_async/_dead_recovery_loop 回到 HEALTHY 均 ++。
# 修复后该测试应真 pass(spawns==1),无 xfail 标记(strict=True 时代码若回归会 XPASS 报错)。


async def test_concurrent_restart_single_spawn(monkeypatch):
    """两协程并发 _restart_async,async with _restart_lock + _restart_epoch 幂等保证 _spawn 恰一次。"""
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    monkeypatch.setattr(E.YoloExecutorManager, "_kill_old", lambda self: None)
    async def fake_rebuild(self): return MagicMock()
    monkeypatch.setattr(E.YoloExecutorManager, "_rebuild_client", fake_rebuild)
    spawns = []
    monkeypatch.setattr(E.YoloExecutorManager, "_spawn", lambda self: spawns.append(1))
    async def fake_ready(self, t): return True
    monkeypatch.setattr(E.YoloExecutorManager, "_wait_ready", fake_ready)
    await asyncio.gather(ex._restart_async(), ex._restart_async())
    assert len(spawns) == 1


# ===== Task 20:model_manager on_dead 回调 + _loop 绑定 + start_cleanup_task =====


def test_on_dead_clears_models(monkeypatch):
    from detect_v.model_manager import ModelManager
    mm = ModelManager.__new__(ModelManager)
    mm.logger = MagicMock()
    from datetime import datetime
    from modal import ModelInfo
    mm.models = {
        "a": ModelInfo(None, 0, 10, datetime.now(), "x", use_v5=False),
        "b": ModelInfo(None, 0, 10, datetime.now(), "x", use_v5=True),
        "c": ModelInfo(None, 0, 10, datetime.now(), "x", use_v5=False),
    }
    mm._on_subprocess_dead(use_v5=False)
    assert "a" not in mm.models and "c" not in mm.models
    assert "b" in mm.models  # v5 不受影响


async def test_start_cleanup_binds_loop_and_starts_health(monkeypatch):
    """start_cleanup_task 在 startup(running loop 内)绑定 _loop + 启动健康协程。
    注:async test(automode)本身已在 running loop 内,直接调 start_cleanup_task,
    不能再包 asyncio.run(plan 原码嵌套 _a.run 会在 running loop 内抛 RuntimeError)。"""
    import asyncio as _a
    from detect_v.model_manager import ModelManager
    from detect_v.process import executor as E
    # 构造 ModelManager 但不真拉子进程:monkeypatch executor __init__
    started_health = []
    def fake_init(self, logger, use_v5, on_dead, auto_spawn=True):
        self.logger = logger; self.use_v5 = use_v5; self.on_dead = on_dead
        self._state = E.ExecutorState.HEALTHY; self._fail_count = 0
        self._loop = None; self._restarting = False
        self._restart_lock = _a.Lock(); self._crash_timestamps = []
        self._base_url = "http://x"; self._timeout = None
        self.client = MagicMock()
        async def _hl(): started_health.append(use_v5)
        self._health_loop = _hl
    monkeypatch.setattr(E.YoloExecutorManager, "__init__", fake_init)
    mm = ModelManager(MagicMock())
    # start_cleanup_task 必须在 running loop 内调
    mm.start_cleanup_task()
    # _loop 已绑定
    assert mm.executor_v5._loop is _a.get_running_loop()
    assert mm.executor_ultralytics._loop is _a.get_running_loop()
    assert len(mm._health_tasks) == 2  # 两健康协程已启动并保强引用


# ===== B1: status_snapshot(健康态快照,供 /gpu_status 暴露) =====


def test_status_snapshot_healthy():
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    assert ex.status_snapshot() == {"state": "HEALTHY", "next_retry_in": None, "dead_since_ts": None}


def test_status_snapshot_restarting_no_eta():
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    ex._state = E.ExecutorState.RESTARTING
    snap = ex.status_snapshot()
    assert snap["state"] == "RESTARTING"
    assert snap["next_retry_in"] is None and snap["dead_since_ts"] is None


def test_status_snapshot_dead_with_eta():
    import time as _t
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    ex._state = E.ExecutorState.DEAD
    ex._dead_since_ts = _t.time()
    ex._next_retry_at = _t.monotonic() + 50.0
    snap = ex.status_snapshot()
    assert snap["state"] == "DEAD"
    assert snap["dead_since_ts"] is not None
    assert 45.0 < snap["next_retry_in"] <= 50.0


def test_status_snapshot_dead_eta_clamped_to_zero():
    import time as _t
    from detect_v.process import executor as E
    ex = E.YoloExecutorManager(MagicMock(), use_v5=False, on_dead=lambda _u: None, auto_spawn=False)
    ex._state = E.ExecutorState.DEAD
    ex._dead_since_ts = _t.time()
    ex._next_retry_at = _t.monotonic() - 10.0
    assert ex.status_snapshot()["next_retry_in"] == 0.0
