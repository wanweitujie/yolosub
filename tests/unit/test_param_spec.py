# tests/unit/test_param_spec.py
"""param_spec 纯函数单测:白名单校验分流、v5 CLI 拼装、hyp 默认合并。无 manager 依赖。"""
import pathlib

import pytest
import yaml as _yaml

from train_v.param_spec import (
    ULTRALYTICS_PARAM_KEYS, V5_HYP_DEFAULTS,
    split_params, build_v5_cli_args, build_v8_cli_args, build_hyp_data,
)

# PRD 2.4 逐键(spec §九⑤,用于反证白名单覆盖):47 项中开放 46 项,
# 第 47 项 resume 按 spec §5.1 刻意报错,不在本集合(走 check_resume 专用通道)
PRD_KEYS = """pretrained patience seed deterministic amp cos_lr compile close_mosaic save_period
nbs cls_remap channels_last profile fraction freeze single_cls rect multi_scale max_det workers
optimizer lr0 lrf momentum weight_decay warmup_epochs warmup_momentum warmup_bias_lr box cls dfl cls_pw
hsv_h hsv_s hsv_v degrees translate scale shear perspective flipud fliplr mosaic mixup bgr cutmix""".split()


def test_v8_whitelist_size_is_103():
    """115 − 静默3 − 报错9 = 103(单测断言规模防漂移,spec §5.1)。"""
    assert len(ULTRALYTICS_PARAM_KEYS) == 103


def test_prd_47_keys_all_in_ultralytics_whitelist():
    """PRD 47 项中 46 项对 v8/v11 开放零剔除;第 47 项 resume 按 spec §5.1 刻意报错(check_resume 会用 ckpt 的 train_args 整体替换 args,击穿 metrics 路径约定)。"""
    assert set(PRD_KEYS) <= ULTRALYTICS_PARAM_KEYS


def test_split_v8_accepts_and_filters():
    """v8 合法键进 cli、静默键丢弃、hyp 恒空。"""
    cli, hyp = split_params("v8", {"batch": -1, "imgsz": 640, "project": "x", "name": "y", "cos_lr": True})
    assert cli == {"batch": -1, "imgsz": 640, "cos_lr": True}
    assert hyp == {}


def test_split_v8_rejects_unknown_and_v5_only_keys():
    """v8 收 v5 专属键(obj)与拼错键 → 一次列全报错。"""
    with pytest.raises(ValueError) as e:
        split_params("v8", {"obj": 1.0, "iou_t": 0.2, "amp": True})
    msg = str(e.value)
    assert "不支持的参数(版本 v8)" in msg
    assert "obj" in msg and "iou_t" in msg   # 一次列全,sorted 顺序
    assert "amp" not in msg                  # amp 对 v8 合法


def test_split_v8_rejects_reserved_keys():
    """model/data/epochs/resume/exist_ok/task/mode/cfg/time 出现在 params 即报错(spec §5.1 报错键9)。"""
    for k in ("model", "data", "epochs", "resume", "exist_ok", "task", "mode", "cfg", "time"):
        with pytest.raises(ValueError, match=k):
            split_params("v8", {k: 1})


def test_split_v8_null_means_omitted_everywhere():
    """值为 null 一律视同未传(与 v5 对齐):白名单 103 键逐个传 None 均不进 cli。

    根因(2026-10-15 核实):ultralytics check_cfg 对 typed_keys
    (FLOAT/FRACTION/INT/BOOL/STR ∪ {scale,compile}) 中 DEFAULT_CFG_DICT 默认值非 None
    的键显式 raise——判据是「默认是否为 None」而非「默认是否假值」,故 channels_last
    (默认 False)同样 raise。白名单 103 键中 84 键属此类;设计文档「v8 的 null 恰好落在
    假值默认键上,无害」的前提不成立。"""
    cli, hyp = split_params("v8", {k: None for k in ULTRALYTICS_PARAM_KEYS})
    assert cli == {} and hyp == {}


def test_split_v8_channels_last_null_regression():
    """回归(2026-10-15 报告):channels_last:null 曾拼出 --channels_last null,
    子进程 check_cfg 抛 TypeError 致 status=error,且报错被上层截断不可诊断。"""
    assert split_params("v8", {"channels_last": None}) == ({}, {})


def test_split_v8_explicit_falsey_values_still_passed():
    """对照:显式 false/0 必须原样透传,不得与 null 一同被吞(防修复过度)。
    这些值是 ultralytics 合法 bool/number,check_cfg 不报错。"""
    cli, _ = split_params("v8", {"channels_last": False, "rect": False, "cos_lr": 0,
                                 "multi_scale": 0.0, "single_cls": False})
    assert cli == {"channels_last": False, "rect": False, "cos_lr": 0,
                   "multi_scale": 0.0, "single_cls": False}


def test_split_v8_null_on_reserved_key_omitted_not_rejected():
    """null 判定先于白名单判定(与 v5 同序):保留键传 null 视同未传、不报错——
    调用方按 PRD 默认集原样下发 resume:null 不应被拒;传非 null 值仍报错,由
    test_split_v8_rejects_reserved_keys 锚定。"""
    assert split_params("v8", {"resume": None, "exist_ok": None}) == ({}, {})


def test_split_v5_cli_vs_hyp_vs_reject():
    """v5:CLI 键/hyp 键分流;ultralytics 专属键与保留键报错。"""
    cli, hyp = split_params("v5", {"lr0": 0.01, "batch": 8, "workers": 4})
    assert cli == {"batch": 8, "workers": 4}
    assert hyp == {"lr0": 0.01}
    with pytest.raises(ValueError) as e:
        split_params("v5", {"amp": True, "compile": False, "resume": True, "exist_ok": False,
                            "evolve": 300, "epochs": 5})
    msg = str(e.value)
    for k in ("amp", "compile", "resume", "exist_ok", "evolve", "epochs"):
        assert k in msg


def test_split_v5_optimizer_choices_validated():
    """optimizer 值域 SGD/Adam/AdamW 在校验层同步报错(PRD 默认 auto 会让 v5 子进程 SystemExit,spec §八)。"""
    with pytest.raises(ValueError, match="optimizer"):
        split_params("v5", {"optimizer": "auto"})
    cli, _ = split_params("v5", {"optimizer": "AdamW"})
    assert cli == {"optimizer": "AdamW"}


def test_split_v5_cache_domain_validated():
    """cache 值域在校验层前置拦截(修拼装段在清理 try 外的残留缝隙)。"""
    with pytest.raises(ValueError, match="cache"):
        split_params("v5", {"cache": "ssd"})
    for ok in (True, "true", "ram", "disk", False, 0, "false", ""):
        cli, _ = split_params("v5", {"cache": ok})
        assert cli == {"cache": ok}
    # null 一律视同未传(spec §5.2,见 test_split_v5_null_means_omitted_everywhere):
    # 与其余带值/旗帜键一致,进不了 cli 字典(而非带 None 值透传)
    cli, _ = split_params("v5", {"cache": None})
    assert cli == {}


def test_split_v5_null_means_omitted_everywhere():
    """spec §5.2「值为 null 一律视同未传」:VALUE 键与 hyp 键统一落地(v8 侧同名规则见
    test_split_v8_null_means_omitted_everywhere)。"""
    cli, hyp = split_params("v5", {"patience": None, "lr0": None, "batch": 4})
    assert cli == {"batch": 4} and hyp == {}
    assert build_v5_cli_args({"patience": None}) == []
    assert build_hyp_data({"lr0": None})["lr0"] == 0.01   # null 视同未传,保默认


def test_split_v5_three_flags_accepted():
    """S18 收敛解除:noautoanchor/noplots/quad 为 vendored parse_opt 合法旗帜,补入白名单。"""
    cli, _ = split_params("v5", {"noplots": True, "quad": False, "noautoanchor": True})
    assert cli == {"noplots": True, "quad": False, "noautoanchor": True}
    argv = build_v5_cli_args(cli)
    assert "--noplots" in argv and "--noautoanchor" in argv and "--quad" not in argv


def test_build_v8_cli_args_null_omitted_defense():
    """防御直调路径:null 视同未传,不拼出 "--channels_last null"(与 build_v5_cli_args
    的带值键同款);非 null 值按 json.dumps 原样拼装,字符串直传(parse_v8_args 侧
    json.loads 还原,往返闭合)。"""
    assert build_v8_cli_args({"channels_last": None, "rect": None}) == []
    assert build_v8_cli_args({"batch": -1, "optimizer": "AdamW", "channels_last": True}) == \
        ["--batch", "-1", "--optimizer", "AdamW", "--channels_last", "true"]


def test_build_v5_cli_args_batch_and_values():
    """batch→--batch-size 映射;带值参数按 json.dumps 拼。"""
    argv = build_v5_cli_args({"batch": -1, "patience": 50, "optimizer": "SGD"})
    assert argv[argv.index("--batch-size") + 1] == "-1"
    assert argv[argv.index("--patience") + 1] == "50"
    assert "--batch" not in argv                    # 不得残留旧形态


def test_build_v5_cli_args_kebab_flags():
    """7 个下划线键转 kebab 旗帜;真值立旗。"""
    argv = build_v5_cli_args({"cos_lr": True, "image_weights": True, "multi_scale": True,
                               "single_cls": True, "save_period": 5, "label_smoothing": 0.1,
                               "sync_bn": True})
    assert "--cos-lr" in argv and "--image-weights" in argv and "--multi-scale" in argv
    assert "--single-cls" in argv and "--sync-bn" in argv
    assert argv[argv.index("--save-period") + 1] == "5"


def test_build_v5_cli_args_falsey_set_not_passed():
    """假值集合 {false,0,0.0,"",null,"false","0"} 全部不传(PRD 默认 multi_scale:0 是现实输入,spec §5.2)。"""
    argv = build_v5_cli_args({"rect": False, "cos_lr": 0, "multi_scale": 0.0,
                              "nosave": "false", "single_cls": None, "noval": "0"})
    assert argv == []


def test_build_v5_cli_args_cache_domain():
    """cache: true→裸旗(=ram);"ram"/"disk"→原样;false→不传;其他→报错(防 'false' 字符串真值反转,spec §5.2)。"""
    assert build_v5_cli_args({"cache": True}) == ["--cache"]
    assert build_v5_cli_args({"cache": "ram"}) == ["--cache", "ram"]
    assert build_v5_cli_args({"cache": "disk"}) == ["--cache", "disk"]
    assert build_v5_cli_args({"cache": False}) == []
    with pytest.raises(ValueError, match="cache"):
        build_v5_cli_args({"cache": "ssd"})


def test_build_v5_cli_args_freeze_null_and_list():
    """freeze: null/空列表视同未传;列表拆多 token;标量单 token(nargs='+')。"""
    assert build_v5_cli_args({"freeze": None}) == []
    assert build_v5_cli_args({"freeze": []}) == []
    argv = build_v5_cli_args({"freeze": [0, 1, 2]})
    assert argv[argv.index("--freeze") + 1: argv.index("--freeze") + 4] == ["0", "1", "2"]
    assert build_v5_cli_args({"freeze": 10}) == ["--freeze", "10"]


def test_build_hyp_data_merges_over_defaults():
    """hyp 合并:scratch-low 默认为底,用户键覆盖;未传键保持默认。"""
    data = build_hyp_data({"lr0": 0.02, "mosaic": 0.5})
    assert data["lr0"] == 0.02 and data["mosaic"] == 0.5
    assert data["lrf"] == 0.01 and data["hsv_h"] == 0.015   # 默认保留
    assert set(data) == set(V5_HYP_DEFAULTS)               # 不引入表外键


def test_hyp_defaults_no_drift_against_vendored():
    """常驻防漂移:内嵌 28 键默认值与 vendored hyp.scratch-low.yaml 逐键相等(spec §十二#6 转常驻)。"""
    vendored = pathlib.Path(__file__).resolve().parents[2] / \
        "docker-create/files/mlmodels/yolov5/data/hyps/hyp.scratch-low.yaml"
    expected = _yaml.safe_load(vendored.read_text(encoding="utf-8"))
    expected.pop("anchors", None)          # 注释键不计
    assert V5_HYP_DEFAULTS == expected
    assert len(V5_HYP_DEFAULTS) == 28
