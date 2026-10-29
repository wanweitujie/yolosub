"""train_v8 argv 解析器回归测试(重构自 train_v8.py 手写解析循环)。

背景(code-review 发现 7):原手写循环对"非布尔参数后跟 '--' 开头 token"处理错误——
`--amp --x` 会把 amp 的值吞为 True,并把 `--x` 再当新标志注入 args 传给 yolo.train
→ TypeError unexpected keyword 'x'(改动前 amp 被硬编码覆盖使该边沿不可达)。
修复:解析逻辑提为 base_train.parse_v8_args,显式布尔白名单(仅 exist-ok),
非布尔参数一律消费一个值(即使以 '--' 开头),杜绝注入。
"""
from train_v.process.base_train import parse_v8_args


def test_parse_v8_args_basic():
    """常规参数:值按 JSON 解析(bool/数字/字符串),exist-ok 裸布尔标志。"""
    args = parse_v8_args(["train_v8.py", "--data", "/d.yaml", "--epochs", "10",
                          "--model", "m.pt", "--amp", "true", "--exist-ok"])
    assert args["data"] == "/d.yaml"
    assert args["epochs"] == 10
    assert args["model"] == "m.pt"
    assert args["amp"] is True
    assert args["exist_ok"] is True


def test_parse_v8_args_skips_status_and_log():
    """status_file/log_file 不进入参数字典(manager 内部参数)。"""
    args = parse_v8_args(["x", "--status_file", "/s.json", "--log_file", "/l.log", "--data", "/d.yaml"])
    assert "status_file" not in args and "log_file" not in args
    assert args["data"] == "/d.yaml"


def test_parse_v8_args_dash_dash_value_not_injected():
    """发现 7 回归:非布尔参数后跟 '--' 开头 token → 该 token 作值消费,
    不被当新标志注入(原实现注入 x=True 抛 TypeError)。"""
    args = parse_v8_args(["x", "--amp", "--x"])
    assert args["amp"] == "--x"      # 原样作值(后续 _normalize_amp 归一会兜底为 False)
    assert "x" not in args           # 不注入新标志
    # 值以 -- 开头的字符串参数(如特殊路径)也能正确消费
    args2 = parse_v8_args(["x", "--data", "--weird-path"])
    assert args2["data"] == "--weird-path"


def test_parse_v8_args_missing_value_fallback():
    """非布尔参数末尾缺值(越界用法)-> 兜底裸标志 True,不崩不注入。"""
    args = parse_v8_args(["x", "--amp"])
    assert args["amp"] is True


def test_parse_v8_args_numeric_and_negative():
    """数字字符串转 int/float(JSON 字面量):"-1" 经 json.loads 得 int -1(autobatch 语义)。"""
    args = parse_v8_args(["x", "--epochs", "100", "--lr0", "0.01", "--batch", "-1"])
    assert args["epochs"] == 100
    assert args["lr0"] == 0.01
    assert args["batch"] == -1


def test_parse_v8_args_adjacent_bool_then_value():
    """裸布尔 exist-ok 后紧跟其它参数:exist-ok 不吞后值,后续正常解析。"""
    args = parse_v8_args(["x", "--exist-ok", "--amp", "false", "--data", "/d.yaml"])
    assert args["exist_ok"] is True
    assert args["amp"] is False
    assert args["data"] == "/d.yaml"
