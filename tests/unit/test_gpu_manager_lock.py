import contextlib
import time
import threading
from datetime import datetime
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from detect_v.process.gpu_manager import GPUManager
from modal import ModelInfo


@pytest.fixture
def gm():
    logger = MagicMock()
    # detector_func 契约:返回 (框列表, speed_ms) 二元组(见 detect_v5/ultralytics_service)
    detector = MagicMock(return_value=({"ok": True}, {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5}))
    return GPUManager(logger, use_v5=False, detector_func=detector)


def test_gpu_lock_initial_state(gm):
    assert gm._gpu_lock is not None
    assert gm._lock_acquired_at is None


def test_gpu_locked_sets_and_clears_timestamp(gm):
    assert gm._lock_acquired_at is None
    with gm._gpu_locked():
        assert gm._lock_acquired_at is not None
    assert gm._lock_acquired_at is None


def test_gpu_locked_serializes_concurrent_access(gm):
    """两线程进 _gpu_locked,必须串行(第二线程等第一线程释放)。"""
    order = []
    lock = gm._gpu_lock

    def worker(name):
        with gm._gpu_locked():
            order.append(("enter", name))
            time.sleep(0.05)
            order.append(("exit", name))

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))
    t1.start(); t2.start()
    t1.join(); t2.join()
    # 串行:enterA...exitA...enterB...exitB(或 A/B 互换,但 enter-exit 不交错)
    assert order[0][0] == "enter" and order[1][0] == "exit"
    assert order[2][0] == "enter" and order[3][0] == "exit"


def _png_bytes():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_detect_signature_takes_model_name(gm):
    """detect 接收 model_name(非 model 对象),在锁内从 self.models 取。"""
    gm.models["m1"] = ModelInfo(
        model=MagicMock(), gpu_id=0, memory_usage=10,
        last_used=datetime.now(), path="x", use_v5=False)
    result = gm.detect("m1", _png_bytes(), None, 5)
    assert result == {"ok": True}
    # detector_func 收到的是 models["m1"].model(锁内取引用)
    gm.detector_func.assert_called_once()
    assert gm.detector_func.call_args[0][0] is gm.models["m1"].model
    # speed_ms 拆包后写入 model_info.last_speed_ms,detect 只返回框列表(响应结构不变)
    assert gm.models["m1"].last_speed_ms == {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5}


def test_detect_unknown_model_raises_404(gm):
    import pytest as _pytest
    from fastapi import HTTPException
    with _pytest.raises(HTTPException) as exc:
        gm.detect("nope", _png_bytes(), None, 5)
    assert exc.value.status_code == 404


def test_get_model_removed(gm):
    """get_model 已删除(detect 改锁内 self.models.get)。"""
    assert not hasattr(gm, "get_model")


# --- Task 4: _load_locked(锁内重验 + 加载 + 写入)---

import asyncio
import pytest as _pytest
from fastapi import HTTPException


def _mk_config(name, **kw):
    c = MagicMock()
    c.model_name = name
    c.model_path = kw.get("model_path", "/x.pt")
    c.preload = kw.get("preload", False)
    c.v5_data = kw.get("v5_data", None)
    c.v5_fp16 = kw.get("v5_fp16", False)
    c.version = kw.get("version", "v8")
    return c


def test_load_locked_rejects_duplicate(monkeypatch, gm):
    """并发 create 同名:第二个在锁内重验抛 400,且不调 init_model(无显存泄漏)。"""
    gm.models["dup"] = ModelInfo(
        model=MagicMock(), gpu_id=0, memory_usage=10,
        last_used=datetime.now(), path="x", use_v5=False)
    init_called = []
    monkeypatch.setattr(gm, "init_model",
                        lambda *a, **k: init_called.append(1) or MagicMock())
    monkeypatch.setattr(gm, "get_model_memory_usage", lambda m: 10)
    # format_yolo_modelpath / get_available_gpu 在 load_model 中为模块级函数,
    # 测试环境(Windows 无 GPU + /x.pt 不存在)需 mock 掉避免 RuntimeError。
    monkeypatch.setattr("detect_v.process.gpu_manager.format_yolo_modelpath",
                        lambda v, w, p: w)
    monkeypatch.setattr("detect_v.process.gpu_manager.get_available_gpu",
                        lambda req: None)

    async def run():
        await gm.load_model(_mk_config("dup"))
    with _pytest.raises(HTTPException) as exc:
        asyncio.run(run())
    assert exc.value.status_code == 400
    assert init_called == []  # 重验在加载前,未触发 init_model(无泄漏)


def test_load_locked_loads_and_writes(monkeypatch, gm):
    """正常加载:init + 写入字典,返回 ModelInfo。"""
    fake_model = MagicMock()
    monkeypatch.setattr(gm, "init_model", lambda *a, **k: fake_model)
    # _load_locked 持 _gpu_locked 后调 _get_model_memory_usage_unlocked(避免二次 acquire 死锁),
    # 故 mock 注入点应为 helper 而非公开 get_model_memory_usage。
    monkeypatch.setattr(gm, "_get_model_memory_usage_unlocked", lambda m: 42)
    monkeypatch.setattr("detect_v.process.gpu_manager.format_yolo_modelpath",
                        lambda v, w, p: w)
    monkeypatch.setattr("detect_v.process.gpu_manager.get_available_gpu",
                        lambda req: None)

    async def run():
        return await gm.load_model(_mk_config("new"))
    info = asyncio.run(run())
    assert "new" in gm.models
    assert gm.models["new"].model is fake_model
    assert info.memory_usage == 42


# --- Task 5: unload / get_model_memory_usage / update / preinit 入锁 ---

def test_get_model_memory_usage_locked(monkeypatch, gm):
    """get_model_memory_usage 持锁(遍历 parameters 与 forward 同级保护)。"""
    m = MagicMock()
    p = MagicMock(); p.is_cuda = True; p.numel.return_value = 1000; p.element_size.return_value = 4
    m.parameters.return_value = iter([p])
    m.buffers.return_value = iter([])
    captured = []
    orig_lock = gm._gpu_locked
    @contextlib.contextmanager
    def spy():
        captured.append(True)
        with orig_lock():
            yield
    monkeypatch.setattr(gm, "_gpu_locked", spy)
    gm.get_model_memory_usage(m)
    assert captured == [True]


def test_unload_model_locked(monkeypatch, gm):
    gm.models["m"] = ModelInfo(
        model=MagicMock(), gpu_id=0, memory_usage=10,
        last_used=datetime.now(), path="x", use_v5=False)
    import torch
    released = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: released.append("cache"))
    ok = gm.unload_model("m")
    assert ok is True
    assert "m" not in gm.models
    assert released == ["cache"]


def test_update_model_memory_usage_no_deadlock(monkeypatch, gm):
    """update_model_memory_usage 持 _gpu_locked 后调 _get_model_memory_usage_unlocked
    (非公开 get_model_memory_usage),避免二次 acquire 死锁。预填 mock ModelInfo,验证
    memory_usage 更新与 clone_to_main 返回。"""
    gm.models["m"] = ModelInfo(
        model=MagicMock(), gpu_id=0, memory_usage=10,
        last_used=datetime.now(), path="x", use_v5=False)
    monkeypatch.setattr(gm, "_get_model_memory_usage_unlocked", lambda m: 42)
    info = gm.update_model_memory_usage("m")
    assert info is not None                 # 不死锁,正常返回
    assert info.memory_usage == 42          # clone_to_main 透传 memory_usage
    assert gm.models["m"].memory_usage == 42  # 原地更新


# --- Task 6: 子进程路由层统一 to_thread ---

def test_detect_route_uses_to_thread_with_model_name(monkeypatch):
    """子进程 detect 路由必须经 to_thread 调 gpu_manager.detect,且传 model_name(非 model 对象)。"""
    from detect_v.process import routes as proutes
    proutes.init_detect_process(MagicMock(), False, MagicMock(return_value={"ok": True}))
    gm = proutes.gpu_manager
    gm.models["m1"] = ModelInfo(
        model=MagicMock(), gpu_id=0, memory_usage=10,
        last_used=datetime.now(), path="x", use_v5=False)

    called = {}

    async def fake_to_thread(fn, *a, **k):
        called["fn"] = fn
        called["args"] = a
        return fn(*a, **k)

    monkeypatch.setattr("asyncio.to_thread", fake_to_thread)
    # shm_size=0 时真实 SharedMemory 行为异常,monkeypatch 避开真实 shm
    monkeypatch.setattr("detect_v.process.routes.SharedMemory",
                        lambda **k: MagicMock(buf=bytearray(0), close=lambda: None))

    import asyncio

    class _Req:
        async def json(self):
            return {"shm_name": "x", "shm_size": 0, "decimals": 5}

    async def run():
        return await proutes.detect("m1", _Req())

    asyncio.run(run())
    assert called["fn"] == gm.detect
    assert called["args"][0] == "m1"  # model_name, 不是 model 对象


# --- Task 7: 删除对外 detect finally 的 update_model_memory_usage + 404 提示语 ---

def test_main_detect_route_no_update_call():
    """对外 detect 路由不再 finally 调 update_model_memory_usage(C:去请求放大)。"""
    # 直接读取源码文件避免导入触发 FastAPI 装饰器依赖
    with open("src/detect_v/routes.py", "r", encoding="utf-8") as f:
        src = f.read()
    # 精确验证:仅截取 detect 函数体(至下一个 @router 装饰器)检查不含 update_model_memory_usage。
    # 原实现检查整个文件:Task 6 新增 /model_memory 端点合法调用 update_model_memory_usage,
    # 会使全文件字符串命中,但与 detect 路由是否调用无关,故收窄到 detect 函数体。
    detect_fn = src.split('@router.post("/detect/{model_name}")', 1)[1]
    detect_fn = detect_fn.split("@router.", 1)[0]
    assert "update_model_memory_usage" not in detect_fn


# --- Task 16: /health 端点 + 锁持有 watchdog ---

def test_health_ok_when_no_lock(gm):
    from detect_v.process import routes as proutes
    proutes.init_detect_process(MagicMock(), False, MagicMock(return_value={"ok": True}))
    proutes.gpu_manager = gm
    gm._lock_acquired_at = None
    import asyncio
    resp = asyncio.run(proutes.health())
    assert resp == {"ok": True}


def test_health_503_when_lock_held_too_long(gm, monkeypatch):
    from detect_v.process import routes as proutes
    from detect_v.process import gpu_manager as pgm
    monkeypatch.setattr(pgm, "LOCK_HELD_TIMEOUT", 60.0)
    proutes.init_detect_process(MagicMock(), False, MagicMock(return_value={"ok": True}))
    proutes.gpu_manager = gm
    gm._lock_acquired_at = time.monotonic() - 90  # 持锁 90s(> 60s)
    import asyncio
    import pytest as _pytest
    from fastapi import HTTPException
    with _pytest.raises(HTTPException) as exc:
        asyncio.run(proutes.health())
    assert exc.value.status_code == 503


# --- Task: preinit 预热后 empty_cache(与 unload_model 对齐)---

def _preinit_cache_probe(gm, monkeypatch, fn_name):
    """跑 preinit_model_{v5,ultralytics}:mock init_model/preload_img 避免真加载,
    捕获 empty_cache/gc.collect 调用;返回 (released, collected)。"""
    import torch
    import gc
    released, collected = [], []
    monkeypatch.setattr(gm, "init_model", lambda *a, **k: MagicMock())
    monkeypatch.setattr(gm, "preload_img", lambda m: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: released.append(1))
    monkeypatch.setattr(gc, "collect", lambda: collected.append(1) or 0)
    getattr(gm, fn_name)(None)
    return released, collected


def test_preinit_ultralytics_releases_cache(monkeypatch, gm):
    """预热 del model 后必须 empty_cache + gc:否则小模型权重滞留缓存池不归还驱动,
    初始化后显存虚高(与 unload_model 的清理行为不一致)。"""
    released, collected = _preinit_cache_probe(gm, monkeypatch, "preinit_model_ultralytics")
    assert released == [1]
    assert collected == [1]


def test_preinit_v5_releases_cache(monkeypatch, gm):
    """同上,v5 预热路径同样必须 empty_cache + gc。"""
    released, collected = _preinit_cache_probe(gm, monkeypatch, "preinit_model_v5")
    assert released == [1]
    assert collected == [1]
