"""服务端 allocator 显存统计单测(无 GPU:CPU 分支 + monkeypatch torch.cuda 的 CUDA 分支)。"""
from datetime import datetime
from unittest.mock import MagicMock

import pytest
import torch

from modal import ModelInfo
from detect_v.process.gpu_manager import GPUManager


class _CudaParam:
    """device/numel/element_size 都模拟 CUDA 张量,但本体不占真实显存。"""
    device = torch.device("cuda", 0)
    is_cuda = True

    def numel(self):
        return 4

    def element_size(self):
        return 4


class _CudaModel:
    def parameters(self):
        return iter([_CudaParam()])


def _gm():
    return GPUManager(MagicMock(), use_v5=False, detector_func=MagicMock())


def test_model_info_carries_allocator_fields():
    info = ModelInfo(None, 0, 100, datetime.now(), "/x/y.pt", False, None, 200, 300, 400)
    assert info.memory_allocated == 200
    assert info.memory_reserved == 300
    assert info.max_memory_allocated == 400
    c = info.clone_to_main()
    assert (c.memory_allocated, c.memory_reserved, c.max_memory_allocated) == (200, 300, 400)
    assert c.model is None


def test_model_info_carries_last_speed_ms():
    """ModelInfo 末尾可选字段 last_speed_ms 透传;旧位置参数构造不受影响(默认 None)。"""
    speed = {"preprocess": 1.5, "inference": 14.8, "postprocess": 1.1}
    info = ModelInfo(None, 0, 100, datetime.now(), "/x/y.pt", False, None, 200, 300, 400, speed)
    assert info.last_speed_ms == speed
    c = info.clone_to_main()
    assert c.last_speed_ms == speed
    assert c.model is None
    assert ModelInfo(None, 0, 100, datetime.now()).last_speed_ms is None


def test_gpu_manager_detect_stores_speed_ms_and_keeps_response_shape():
    """detector_func 返回 (框列表, speed_ms) 二元组:速度写入 model_info.last_speed_ms,
    detect 对外只返回框列表(响应结构不变)。"""
    speed = {"preprocess": 1.5, "inference": 14.8, "postprocess": 1.1}

    def detector(model, img, config, decimals):
        return ["box1"], speed

    gm = GPUManager(MagicMock(), use_v5=False, detector_func=detector)
    info = ModelInfo(_CudaModel(), 0, 100, datetime.now())
    gm.models["bench_x"] = info
    out = gm.detect("bench_x", b"FAKE", None, 5)
    assert out == ["box1"]
    assert info.last_speed_ms == speed


def test_allocator_mb_cpu_returns_none(monkeypatch):
    gm = _gm()
    cpu_model = torch.nn.Linear(2, 2)  # device = cpu
    assert gm._allocator_mb(cpu_model) == (None, None)


def test_allocator_mb_cuda(monkeypatch):
    gm = _gm()
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev: 200 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev: 500 * 1024 * 1024)
    assert gm._allocator_mb(_CudaModel()) == (200, 500)


def test_update_model_memory_usage_populates_and_reset(monkeypatch):
    gm = _gm()
    info = ModelInfo(_CudaModel(), 0, 100, datetime.now())
    gm.models["bench_x"] = info
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev: 200 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev: 500 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev: 700 * 1024 * 1024)
    reset_calls = []
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats",
                        lambda dev: reset_calls.append(dev))
    out = gm.update_model_memory_usage("bench_x", reset_peak=True)
    assert out.memory_allocated == 200
    assert out.memory_reserved == 500
    assert out.max_memory_allocated == 700
    assert len(reset_calls) == 1
    assert gm.update_model_memory_usage("bench_missing", reset_peak=True) is None
