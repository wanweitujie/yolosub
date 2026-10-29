"""train_v8 amp 透传回归测试(bug:amp 曾被双硬编码强制纯 FP32 训练)。

背景:train_v8.py 历史存在两处硬编码——`amp=False #禁止amp`(覆盖用户传值)
+ `yolo.train(..., amp=False, ...)`(调用处再写死)。用户配置 amp:true 被强制覆盖,
Ada 卡(Tensor Core FP16)算力浪费,是同配置训练慢 2 倍的根因
(根因见内部分析)。
修复:移除两处硬编码,amp 从 params 透传;不传默认 True 对齐 Ultralytics 官方默认
(default.yaml `amp: True`,AMP 混合精度是官方默认训练行为),显式 amp:false 仍可关闭。
code-review 后续:amp 值须经 _normalize_amp 归一(null/字符串 "False"/"" 原样透传
会在 ultralytics torch.tensor(amp) 抛 RuntimeError/TypeError),并有运行时测试兜底。
"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # tests/unit -> 仓库根(与 cwd 无关,防非根目录误红)


def _train_v8_src() -> str:
    with open(_ROOT / "src/train_v/process/train_v8.py", "r", encoding="utf-8") as f:
        return f.read()


def test_train_v8_amp_transparent_not_hardcoded():
    """amp 必须来自 params 透传,不得再被硬编码覆盖为 False。"""
    src = _train_v8_src()
    assert "#禁止amp" not in src            # 覆盖式硬编码已移除
    assert "args.pop('amp', True)" in src   # 不传 amp 默认 True(对齐 Ultralytics 官方)
    assert "amp=amp" in src                 # yolo.train 透传解析出的 amp 变量


def test_train_v8_train_call_does_not_force_amp_false():
    """yolo.train(...) 调用处不得再出现 amp=False 字面量硬编码。"""
    src = _train_v8_src()
    start = src.index("yolo.train(")
    call = src[start:src.index(")", start) + 1]
    assert "amp=False" not in call
    assert "amp=amp" in call


def test_train_v8_amp_normalized_not_raw():
    """amp 解析处必须经 _normalize_amp 归一(防 null/字符串透传崩溃)。"""
    src = _train_v8_src()
    assert "_normalize_amp(args.pop('amp', True))" in src


def test_normalize_amp_values():
    """amp 各种客户端值归一为严格 bool,杜绝 torch.tensor(None)/TypeError 崩溃。"""
    from utils_v.index import _normalize_amp
    assert _normalize_amp(None) is False      # json null
    assert _normalize_amp(True) is True
    assert _normalize_amp(False) is False
    assert _normalize_amp(1) is True          # json 数字 1
    assert _normalize_amp(0) is False
    assert _normalize_amp("true") is True     # json 字面量
    assert _normalize_amp("false") is False
    assert _normalize_amp("False") is False   # 大写非法 JSON -> 保留字符串,归一为 False
    assert _normalize_amp("") is False        # 空串
    assert _normalize_amp("1") is True
    assert _normalize_amp("0") is False
    assert _normalize_amp("yes") is True
    assert _normalize_amp("  TRUE  ") is True
    assert _normalize_amp("garbage") is False  # 非法字符串回退 False(显式传错值视为关)
