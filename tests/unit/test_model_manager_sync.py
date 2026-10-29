"""主进程 models 与子进程严格同步:释放后列表不残留。

背景:detect 报 404 时主进程列表仍显示已释放模型 —— 根因是 unload_model 的 del
在 close 成功后执行,子进程 close 失败(不可达/假死/已释放)时不删 → 残留
"列表有但实际没有"的模型。修复:del 移入 finally,无论 close 成败都移除。
"""
import asyncio
from unittest.mock import MagicMock
from datetime import datetime

from detect_v.model_manager import ModelManager
from modal import ModelInfo


def _mm_with_model(close_fail=False):
    """构造 ModelManager,预填一个 ultralytics 模型;可让子进程 close 抛异常模拟不可达/DEAD。"""
    mm = ModelManager.__new__(ModelManager)
    mm.logger = MagicMock()
    mm.models = {
        "m1": ModelInfo(None, 0, 10, datetime.now(), "/x", use_v5=False),
    }

    async def _close_fail(name):
        raise RuntimeError("subprocess DEAD")

    async def _close_ok(name):
        return {"success": True}

    ex = MagicMock()
    ex.close_model_async = _close_fail if close_fail else _close_ok
    mm.executor_ultralytics = ex
    mm.executor_v5 = MagicMock()
    return mm


def test_unload_ok_removes_from_models():
    """子进程 close 成功 → 主进程 dict 移除,返回 True。"""
    mm = _mm_with_model(close_fail=False)
    assert asyncio.run(mm.unload_model("m1")) is True
    assert "m1" not in mm.models


def test_unload_fail_still_removes_from_models():
    """子进程 close 失败(不可达/DEAD)→ 主进程 dict 仍移除(不留残留),返回 False。"""
    mm = _mm_with_model(close_fail=True)
    assert asyncio.run(mm.unload_model("m1")) is False   # close 失败反映为 False
    assert "m1" not in mm.models                         # 但列表已清(不残留)


def test_unload_unknown_model_returns_false():
    """dict 里没有的模型 → False,不报错。"""
    mm = _mm_with_model()
    assert asyncio.run(mm.unload_model("nope")) is False


def test_on_subprocess_dead_clears_only_that_version():
    """子进程判死/崩溃回调:只清对应版本(use_v5 匹配),另一版本不受影响。"""
    mm = ModelManager.__new__(ModelManager)
    mm.logger = MagicMock()
    mm.models = {
        "u1": ModelInfo(None, 0, 10, datetime.now(), "/x", use_v5=False),
        "u2": ModelInfo(None, 0, 10, datetime.now(), "/x", use_v5=False),
        "v1": ModelInfo(None, 0, 10, datetime.now(), "/x", use_v5=True),
    }
    mm._on_subprocess_dead(use_v5=False)   # ultralytics 崩溃
    assert "u1" not in mm.models and "u2" not in mm.models
    assert "v1" in mm.models               # v5 模型不受影响(子进程还在)
