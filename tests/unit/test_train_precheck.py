"""训练显存预检纯函数 + precheck_memory mock 测试。"""
from unittest.mock import MagicMock, patch, ANY

from train_v.process.base_train import _estimate_from_weights


def test_estimate_yolo11n_imgsz320():
    """yolo11n 权重 5MB(state_dict 实测), imgsz=320 batch=4 -> ~346MB(×1.2 上浮后)。"""
    budget = _estimate_from_weights(weights_bytes=5_000_000, imgsz=320, batch=4)
    assert 330 <= budget <= 360, f"yolo11n@320 预算应在 330-360MB,实际 {budget}"


def test_estimate_yolo11l_imgsz320():
    """yolo11l 权重 48.5MB, imgsz=320 batch=4 -> ~1270MB(激活按 scale=9.7 放大)。"""
    budget = _estimate_from_weights(weights_bytes=48_500_000, imgsz=320, batch=4)
    assert 1150 <= budget <= 1350, f"yolo11l@320 预算应在 1150-1350MB,实际 {budget}"


def test_estimate_yolo11n_imgsz640_batch16():
    """yolo11n 5MB, imgsz=640 batch=16 -> ~1624MB(激活主导,scale=1)。"""
    budget = _estimate_from_weights(weights_bytes=5_000_000, imgsz=640, batch=16)
    assert 1500 <= budget <= 1700, f"yolo11n@640b16 预算应在 1500-1700MB,实际 {budget}"


def test_estimate_yolo11x_imgsz640_batch8():
    """yolo11x 108.8MB(×21.76), imgsz=640 batch=8 -> ~15.2GB,对齐线上实测 ~15GB。

    验证大模型激活按 scale 放大后不再严重低估(原公式仅 1.1GB,低估 13×)。
    """
    budget = _estimate_from_weights(weights_bytes=108_800_000, imgsz=640, batch=8)
    assert 14000 <= budget <= 16000, f"yolo11x@640b8 预算应在 14-16GB,实际 {budget}"


def test_estimate_yolo11x_imgsz640_batch16():
    """yolo11x 108.8MB, imgsz=640 batch=16 -> ~30GB > 16GB,预检应拦 OOM(原 1.7GB 误放行)。"""
    budget = _estimate_from_weights(weights_bytes=108_800_000, imgsz=640, batch=16)
    assert 28000 <= budget <= 32000, f"yolo11x@640b16 预算应在 28-32GB,实际 {budget}"
    assert budget > 16384, f"yolo11x@640b16 应超过 16GB 卡容量被预检拦住,实际 {budget}MB"


def test_estimate_scales_with_imgsz():
    """imgsz 320->640(batch=4):激活 (640/320)²=4×,ratio~4.7。"""
    b320 = _estimate_from_weights(5_000_000, imgsz=320, batch=4)
    b640 = _estimate_from_weights(5_000_000, imgsz=640, batch=4)
    assert b640 > b320 * 1.6, f"imgsz 640 应明显 > 1.6× imgsz 320: {b640} vs {b320}"


def test_estimate_scales_with_batch():
    """batch 4->8:激活翻倍(线性),预算增大。"""
    b4 = _estimate_from_weights(5_000_000, imgsz=320, batch=4)
    b8 = _estimate_from_weights(5_000_000, imgsz=320, batch=8)
    assert b8 > b4, f"batch 8 应大于 batch 4: {b8} vs {b4}"


def test_estimate_scales_with_model_size():
    """同 imgsz/batch,11x(108.8MB)vs 11n(5MB):激活按 scale 放大 ~21.76×,预算远大。"""
    bn = _estimate_from_weights(5_000_000, imgsz=640, batch=8)
    bx = _estimate_from_weights(108_800_000, imgsz=640, batch=8)
    assert bx > bn * 10, f"11x 应远大于 11n(激活按规模放大): {bx} vs {bn}"


def test_estimate_imgsz_list_no_crash():
    """ultralytics 合法 list 型 imgsz=[640,480] 不崩,取 max 估算(等价 imgsz=640)。"""
    budget_list = _estimate_from_weights(5_000_000, imgsz=[640, 480], batch=16)
    budget_int = _estimate_from_weights(5_000_000, imgsz=640, batch=16)
    assert budget_list == budget_int, f"imgsz=[640,480] 应取 max=640,与 int 640 等价: {budget_list} vs {budget_int}"


def test_estimate_imgsz_invalid_fallback_default():
    """imgsz 非法(空 list / None)-> 回退默认 640,不崩。"""
    budget_empty = _estimate_from_weights(5_000_000, imgsz=[], batch=16)
    budget_none = _estimate_from_weights(5_000_000, imgsz=None, batch=16)
    budget_default = _estimate_from_weights(5_000_000, imgsz=640, batch=16)
    assert budget_empty == budget_default
    assert budget_none == budget_default


def test_estimate_amp_lower_than_fp32():
    """amp=True 激活减半:预算低于 FP32(权重/梯度/优化器仍 FP32,仅激活省一半),
    显著下降但非整降——选卡/预检不因 FP16 虚高预算误拒小卡训练。"""
    b_fp32 = _estimate_from_weights(5_000_000, imgsz=640, batch=16)
    b_amp = _estimate_from_weights(5_000_000, imgsz=640, batch=16, amp=True)
    assert b_amp < b_fp32, f"amp 预算应低于 FP32: {b_amp} vs {b_fp32}"
    assert b_amp < b_fp32 * 0.75, f"amp 应显著低于 FP32: {b_amp} vs {b_fp32}"
    assert b_amp > b_fp32 * 0.5, f"amp 不应整降(权重/优化器仍 FP32): {b_amp} vs {b_fp32}"


# ===== precheck_memory mock 测试 =====

def _fake_ckpt(n_params, element_size=4):
    t = MagicMock()
    t.numel.return_value = n_params
    t.element_size.return_value = element_size
    model = MagicMock()
    model.state_dict.return_value = {"w": t}
    return {"model": model}


@patch("train_v.process.base_train.write_status")
@patch("train_v.process.base_train.get_available_gpu")
@patch("torch.load")
def test_precheck_pass_when_enough(mock_torch_load, mock_gpu, mock_write):
    """GPU 够 -> precheck 正常返回(不 exit,不 write_status=error)。"""
    from train_v.process import base_train
    mock_torch_load.return_value = _fake_ckpt(2_600_000)
    mock_gpu.return_value = 0
    base_train.precheck_memory("/fake.pt", imgsz=320, batch=4)
    mock_write.assert_not_called()


@patch("train_v.process.base_train.write_status")
@patch("train_v.process.base_train.get_available_gpu")
@patch("torch.load")
def test_precheck_amp_lowers_budget(mock_torch_load, mock_gpu, mock_write):
    """amp=True 预算降低:选卡/校验按 FP16 激活估算,小卡装更大 batch 不被误拒。"""
    from train_v.process import base_train
    mock_torch_load.return_value = _fake_ckpt(2_600_000)
    mock_gpu.return_value = 0
    base_train.precheck_memory("/fake.pt", imgsz=320, batch=4, amp=True)
    amp_budget = mock_gpu.call_args.args[0]
    mock_gpu.reset_mock()
    base_train.precheck_memory("/fake.pt", imgsz=320, batch=4)
    fp32_budget = mock_gpu.call_args.args[0]
    assert amp_budget < fp32_budget, f"amp 预算应低于 FP32: {amp_budget} vs {fp32_budget}"


@patch("train_v.process.base_train.write_status")
@patch("train_v.process.base_train.sys")
@patch("train_v.process.base_train.get_available_gpu")
@patch("torch.load")
def test_precheck_exit_when_insufficient(mock_torch_load, mock_gpu, mock_sys, mock_write, tmp_path, monkeypatch):
    """GPU 不足 -> write_error(status=error) + sys.exit(1)。

    字段集须与 running/finished 对齐(2026-10-15 统一):预检失败时训练尚未开始,
    无历史进度可合并,靠 total_epochs 参数与零值兜底补齐——原先只写 {status, message},
    客户端无条件读 progress 会 KeyError。"""
    from train_v.process import base_train
    monkeypatch.setattr(base_train, "STATUS_FILE", tmp_path / "status.json")   # 隔离,防读到真实残留
    mock_torch_load.return_value = _fake_ckpt(25_400_000)
    mock_gpu.return_value = None
    base_train.precheck_memory("/fake.pt", imgsz=640, batch=16, total_epochs=100)
    mock_sys.exit.assert_called_once_with(1)
    mock_write.assert_called_once()
    kwargs = mock_write.call_args.kwargs
    assert kwargs.get("status") == "error"
    assert kwargs.get("total_epochs") == 100          # 调用方传入的总轮数透传进 error 响应
    for k in ("pid", "epoch", "progress"):
        assert k in kwargs, f"error 响应缺字段 {k}"


# ===== #3 autobatch(batch<=0)跳过预检 =====

@patch("train_v.process.base_train.write_status")
@patch("train_v.process.base_train.get_available_gpu")
@patch("torch.load")
def test_precheck_skip_when_autobatch(mock_torch_load, mock_gpu, mock_write):
    """batch<=0(autobatch)-> 跳过预检,不读权重不查 GPU,返回 None。"""
    from train_v.process import base_train
    result = base_train.precheck_memory("/fake.pt", imgsz=640, batch=-1)
    assert result is None
    mock_torch_load.assert_not_called()
    mock_gpu.assert_not_called()
    mock_write.assert_not_called()


@patch("train_v.process.base_train.write_status")
@patch("train_v.process.base_train.get_available_gpu")
@patch("torch.load")
def test_precheck_skip_when_batch_str_neg(mock_torch_load, mock_gpu, mock_write):
    """batch="-1"(train_v8 手动解析因负号保留字符串)-> int() 转 -1 -> 跳过,不 TypeError。"""
    from train_v.process import base_train
    result = base_train.precheck_memory("/fake.pt", imgsz=640, batch="-1")
    assert result is None
    mock_torch_load.assert_not_called()


# ===== #2 device 闭环 =====

@patch("train_v.process.base_train.write_status")
@patch("train_v.process.base_train._check_specific_gpu")
@patch("torch.load")
def test_precheck_device_specified_ok(mock_torch_load, mock_check, mock_write):
    """用户指定 device="1" 且该卡够 -> 返回 "1" 原样,不调 get_available_gpu。"""
    from train_v.process import base_train
    mock_torch_load.return_value = _fake_ckpt(2_600_000)
    mock_check.return_value = True
    result = base_train.precheck_memory("/fake.pt", imgsz=320, batch=4, device="1")
    assert result == "1"
    mock_check.assert_called_once()
    mock_write.assert_not_called()


@patch("train_v.process.base_train.write_status")
@patch("train_v.process.base_train.sys")
@patch("train_v.process.base_train._check_specific_gpu")
@patch("torch.load")
def test_precheck_device_specified_insufficient(mock_torch_load, mock_check, mock_sys, mock_write):
    """用户指定 device="1" 但该卡不够 -> write_status(error)+sys.exit(1)。"""
    from train_v.process import base_train
    mock_torch_load.return_value = _fake_ckpt(25_400_000)
    mock_check.return_value = False
    base_train.precheck_memory("/fake.pt", imgsz=640, batch=16, device="1")
    mock_sys.exit.assert_called_once_with(1)
    assert mock_write.call_args.kwargs.get("status") == "error"


@patch("train_v.process.base_train.write_status")
@patch("train_v.process.base_train._check_specific_gpu")
@patch("torch.load")
def test_precheck_device_zero_int_not_treated_as_unset(mock_torch_load, mock_check, mock_write):
    """device=0(int,用户显式指定 GPU 0)不被当未指定:校验 GPU 0,够则返回 0。

    防 falsy-zero 回归:原 `if device:` 对 0 为 falsy 跳过校验 + train_v8 覆盖成别的卡。
    """
    from train_v.process import base_train
    mock_torch_load.return_value = _fake_ckpt(2_600_000)
    mock_check.return_value = True
    result = base_train.precheck_memory("/fake.pt", imgsz=320, batch=4, device=0)
    assert result == 0   # 原样返回,不被改路由
    mock_check.assert_called_once_with(0, ANY)   # 校验 GPU 0
    mock_write.assert_not_called()


@patch("train_v.process.base_train.write_status")
@patch("train_v.process.base_train.get_available_gpu")
@patch("torch.load")
def test_precheck_device_none_picks_best(mock_torch_load, mock_gpu, mock_write):
    """device 未指定 -> get_available_gpu 选剩余最多,返回 str(gpu_id)。"""
    from train_v.process import base_train
    mock_torch_load.return_value = _fake_ckpt(2_600_000)
    mock_gpu.return_value = 2
    result = base_train.precheck_memory("/fake.pt", imgsz=320, batch=4)
    assert result == "2"


@patch("train_v.process.base_train.write_status")
@patch("train_v.process.base_train.get_available_gpu")
@patch("torch.load")
def test_precheck_device_cpu_skip_gpu(mock_torch_load, mock_gpu, mock_write):
    """device="cpu" -> 跳过 GPU 校验,返回 "cpu",不查 GPU 不读权重。"""
    from train_v.process import base_train
    result = base_train.precheck_memory("/fake.pt", imgsz=640, batch=16, device="cpu")
    assert result == "cpu"
    mock_torch_load.assert_not_called()
    mock_gpu.assert_not_called()


# ===== #6 check_gpu_memory 复用 utils_v.index(消除 GPUtil 遍历分叉)=====

def test_check_gpu_memory_true_when_enough():
    """check_gpu_memory 复用 _gpu_free_memory_mb,够 -> True。"""
    from utils_v.index import check_gpu_memory
    fake_gpu = MagicMock(); fake_gpu.id = 0; fake_gpu.memoryFree = 8000
    with patch("utils_v.index.GPUtil.getGPUs", return_value=[fake_gpu]):
        assert check_gpu_memory(0, 5000) is True

def test_check_gpu_memory_false_when_insufficient():
    """check_gpu_memory 不足 -> False(>语义与 get_available_gpu 一致)。"""
    from utils_v.index import check_gpu_memory
    fake_gpu = MagicMock(); fake_gpu.id = 0; fake_gpu.memoryFree = 3000
    with patch("utils_v.index.GPUtil.getGPUs", return_value=[fake_gpu]):
        assert check_gpu_memory(0, 5000) is False

def test_check_gpu_memory_false_when_gpu_not_found():
    """指定 GPU id 不存在 -> False。"""
    from utils_v.index import check_gpu_memory
    fake_gpu = MagicMock(); fake_gpu.id = 1; fake_gpu.memoryFree = 8000
    with patch("utils_v.index.GPUtil.getGPUs", return_value=[fake_gpu]):
        assert check_gpu_memory(0, 5000) is False   # 查 0,只有 1

def test_check_gpu_memory_false_on_exception():
    """GPUtil 异常 -> False(不抛)。"""
    from utils_v.index import check_gpu_memory
    with patch("utils_v.index.GPUtil.getGPUs", side_effect=RuntimeError("boom")):
        assert check_gpu_memory(0, 5000) is False
