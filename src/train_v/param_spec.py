# train_v/param_spec.py
"""版本化训练参数白名单与 v5 参数拼装(纯函数,无 IO/无进程)。

三张静态表 + 分流校验 + v5 CLI 拼装,为 manager.start 提供版本参数识别。
键集事实以此文件为准,来源:
- ultralytics 8.4.121 default.yaml(115 键,开发机实证;PRD 47 项中 46 项开放零剔除;
  resume 按设计报错(check_resume 会用 ckpt 的 train_args 整体替换 args,击穿 metrics 路径约定))
- vendored yolov5 7.0 train.py parse_opt + data/hyps/hyp.scratch-low.yaml
升级 ultralytics / 改 vendored v5 时须同步本表(白名单规模断言与防漂移单测兜底)。
"""
import json

# === ultralytics(v8/v11 共用)白名单:115 − 静默3 − 报错9 = 103 键 ===
# 静默过滤(3):device(manager start 顶部 pop)/project/name(manager 组装)
#   注:status_file/log_file 在 manager 过滤元组内但不在 default.yaml(对 v8 是空操作过滤)
# 报错键(9):model/data/epochs(TrainStartRequest 顶层字段,防 params 静默覆盖——
#   data 覆盖会绕过 transport yaml 路径改写)/resume(check_resume 用 ckpt 的 train_args
#   整体替换 args,save_dir 落回旧 exp 目录、metrics 路径约定被击穿)/exist_ok(可击穿
#   output 路径约定,触发 output-2 递增)/task/mode/cfg(非训练超参)/time(优先覆盖 epochs,
#   语义冲突)
ULTRALYTICS_PARAM_KEYS = frozenset("""
patience batch imgsz save save_period cache workers pretrained
cls_remap optimizer verbose seed deterministic single_cls rect cos_lr
close_mosaic amp fraction profile freeze multi_scale compile channels_last
overlap_mask mask_ratio dropout val split save_json conf iou
max_det quantize dnn plots end2end source vid_stride stream_buffer
visualize augment agnostic_nms classes retina_masks embed show save_frames
save_txt save_conf save_crop show_labels show_conf show_boxes line_width format
keras optimize dynamic simplify opset workspace nms lr0
lrf momentum weight_decay warmup_epochs warmup_momentum warmup_bias_lr distill_model dis
box cls cls_pw dfl pose kobj rle angle
dlog dgrad dlam nbs hsv_h hsv_s hsv_v degrees
translate scale shear perspective flipud fliplr bgr mosaic
mixup cutmix copy_paste copy_paste_mode auto_augment erasing tracker
""".split())

# v8/v11 循环拼装时跳过的保留键(与 manager 既有过滤元组对齐;device 在 start 顶部已 pop)
V8_SILENT_KEYS = frozenset({"project", "name", "status_file", "log_file"})

# === v5 白名单:CLI 表 + hyp 表(闭白名单,不在两表内的键一律报错) ===
# 静默过滤:manager 内部组装占用,不报错不透传
V5_SILENT_KEYS = frozenset({"project", "name", "status_file", "log_file"})
# 注:device 由 manager start 顶部 pop,不进本层;resume/evolve/cfg/local_rank/bucket/
#     weights/data/hyp/epochs/exist_ok 均不在下两表 → 自然落入报错分支

# v5 argparse 类型(spec §5.2):bool=store_true 旗帜;value=带值;cache=nargs'?' const'ram';
# freeze=nargs'+' 列表
_V5_BOOL, _V5_VALUE, _V5_CACHE, _V5_FREEZE = "bool", "value", "cache", "freeze"

V5_CLI_PARAMS = {
    "batch":           {"flag": "--batch-size",       "kind": _V5_VALUE},
    "imgsz":           {"flag": "--imgsz",            "kind": _V5_VALUE},
    "workers":         {"flag": "--workers",         "kind": _V5_VALUE},
    "patience":        {"flag": "--patience",         "kind": _V5_VALUE},
    "seed":            {"flag": "--seed",             "kind": _V5_VALUE},
    "save_period":     {"flag": "--save-period",      "kind": _V5_VALUE},
    "label_smoothing": {"flag": "--label-smoothing",  "kind": _V5_VALUE},
    "optimizer":       {"flag": "--optimizer",        "kind": _V5_VALUE,
                       "choices": ("SGD", "Adam", "AdamW")},
    "cache":           {"flag": "--cache",            "kind": _V5_CACHE},
    "freeze":          {"flag": "--freeze",          "kind": _V5_FREEZE},
    "rect":            {"flag": "--rect",             "kind": _V5_BOOL},
    "nosave":          {"flag": "--nosave",           "kind": _V5_BOOL},
    "noval":           {"flag": "--noval",            "kind": _V5_BOOL},
    "cos_lr":          {"flag": "--cos-lr",           "kind": _V5_BOOL},
    "image_weights":   {"flag": "--image-weights",   "kind": _V5_BOOL},
    "multi_scale":     {"flag": "--multi-scale",     "kind": _V5_BOOL},
    "single_cls":      {"flag": "--single-cls",       "kind": _V5_BOOL},
    "sync_bn":         {"flag": "--sync-bn",          "kind": _V5_BOOL},
    "noautoanchor":    {"flag": "--noautoanchor",     "kind": _V5_BOOL},
    "noplots":         {"flag": "--noplots",          "kind": _V5_BOOL},
    "quad":            {"flag": "--quad",             "kind": _V5_BOOL},
}

# v5 hyp 默认(静态内嵌 vendored hyp.scratch-low.yaml 28 键;防漂移有常驻单测逐键比对)
V5_HYP_DEFAULTS = {
    "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005,
    "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1,
    "box": 0.05, "cls": 0.5, "cls_pw": 1.0,
    "obj": 1.0, "obj_pw": 1.0, "iou_t": 0.20, "anchor_t": 4.0, "fl_gamma": 0.0,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0,
    "flipud": 0.0, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
}


def _is_falsey(value) -> bool:
    """布尔旗帜假值集合 {false,0,0.0,"",null,"false","0"}(spec §5.2)。
    不得按 truthy/字符串处理:PRD 默认集 multi_scale:0 是现实输入,
    拼出 --multi-scale 0 会立旗 + 0 落 unknown(实测)。"""
    if value is None or value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("", "false", "0")
    if isinstance(value, (int, float)):
        return value == 0
    return False


def _cache_value_ok(value) -> bool:
    """cache 合法值域:True/'true'(忽略大小写)/'ram'/'disk'/假值集合——与 build_v5_cli_args
    拼装分支一致。前置到校验层:拼装段在 manager 清理 try 之外,后置报错会残留空目录与
    transport 文件(spec §六.1 校验失败零残留)。"""
    if value is True or (isinstance(value, str) and value.lower() == "true"):
        return True
    if isinstance(value, str) and value in ("ram", "disk"):
        return True
    return _is_falsey(value)


def split_params(version: str, params: dict) -> tuple:
    """按版本分流并校验 params(校验时序:manager 已先 pop device)。

    返回 (cli_params, hyp_params);非法键/非法值 raise ValueError 且一次列全
    (errMsg 形如「不支持的参数(版本 v5): amp, compile」)。静默过滤键被丢弃不报错。
    v5+amp 不再静默跳过(旧行为)而是报错——v5 无该参数,由框架 check_amp 自动探测。"""
    if version not in ("v5", "v8", "v11"):
        raise ValueError(f"未知版本: {version}")
    cli, hyp, unknown = {}, {}, []
    if version in ("v8", "v11"):
        silent = V8_SILENT_KEYS
        for k, v in params.items():
            # spec §5.2「值为 null 一律视同未传」(两版通用):判定先于白名单,故保留键传 null
            # 亦视同未传(不报错),非 null 值仍按 §5.1 报错。
            # v8 侧原缺此规则,理由见设计文档「PRD 的 null 恰好落在 ultralytics 假值默认键上,
            # 无害」——该前提不成立:ultralytics check_cfg 的判据是 DEFAULT_CFG_DICT 默认值
            # **是否为 None**(非「是否假值」),channels_last 默认 False 同样 raise TypeError;
            # 白名单 103 键中 84 键属此类,任一传 null 即子进程崩(status=error)。
            if v is None:
                continue
            if k in silent:
                continue
            if k in ULTRALYTICS_PARAM_KEYS:
                cli[k] = v
            else:
                unknown.append(k)
    else:
        silent = V5_SILENT_KEYS
        for k, v in params.items():
            # spec §5.2「值为 null 一律视同未传」(两版通用):带值键/hyp 键/旗帜键统一在此
            # 落地——否则 json.dumps(None) 会拼出 "--patience null",子进程 argparse 报错走
            # SystemExit(2) 逃逸 except Exception;hyp 键则会把 null 写进 _hyp.yaml
            if v is None:
                continue
            if k in silent:
                continue
            if k in V5_CLI_PARAMS:
                spec = V5_CLI_PARAMS[k]
                if "choices" in spec and str(v) not in spec["choices"]:
                    # optimizer 等值域在提交侧同步报错:v5 子进程值错误走 argparse
                    # parser.error → SystemExit(2),不被 except Exception 捕获,
                    # 只能靠 manager 兜底补通用文案;在此提前拦截给出明确 errMsg
                    unknown.append(f"{k}={v!r}(合法值: {'/'.join(spec['choices'])})")
                elif spec["kind"] == _V5_CACHE and not _cache_value_ok(v):
                    # cache 值域前置拦截(同 optimizer choices 理由:v5 值错误后置会残留产物)
                    unknown.append(f"{k}={v!r}(合法: true / \"ram\" / \"disk\" / false)")
                else:
                    cli[k] = v
            elif k in V5_HYP_DEFAULTS:
                hyp[k] = v
            else:
                unknown.append(k)
    if unknown:
        raise ValueError(f"不支持的参数(版本 {version}): " + ", ".join(sorted(unknown)))
    return cli, hyp


def build_v5_cli_args(cli_params: dict) -> list:
    """v5 CLI 参数 → argv 片段(旗帜名按 V5_CLI_PARAMS 映射,值按 argparse 类型拼装)。

    规则(spec §5.2):布尔假值不传(实测 --rect false 会立旗且值落 unknown;
    --cache false 更糟,'false' 真值会打开 RAM 缓存,与意图相反);freeze 列表拆多
    token(单 token "0,1,2" 会被 int() 拒绝);null/空列表视同未传。"""
    argv = []
    for key, spec in V5_CLI_PARAMS.items():
        if key not in cli_params:
            continue
        value = cli_params[key]
        kind = spec["kind"]
        if kind == _V5_BOOL:
            if not _is_falsey(value):
                argv.append(spec["flag"])
        elif kind == _V5_VALUE:
            if value is None:
                continue                                # 防御直调路径:null 视同未传(与 cache/freeze 同款)
            argv += [spec["flag"], value if isinstance(value, str) else json.dumps(value)]
        elif kind == _V5_CACHE:
            if value is True or (isinstance(value, str) and value.lower() == "true"):
                argv.append("--cache")                    # 裸传 = const 'ram'
            elif isinstance(value, str) and value in ("ram", "disk"):
                argv += ["--cache", value]                # 字符串原样透传
            elif _is_falsey(value):
                pass
            else:
                raise ValueError(f"cache 值不合法: {value!r}(合法: true / \"ram\" / \"disk\" / false)")
        elif kind == _V5_FREEZE:
            if value is None or (isinstance(value, list) and not value):
                continue                                  # null/空列表视同未传
            items = value if isinstance(value, list) else [value]
            argv.append(spec["flag"])
            argv += [str(x) for x in items]               # 列表拆多 token
    return argv


def build_v8_cli_args(cli_params: dict) -> list:
    """v8/v11 CLI 参数 → argv 片段(键已白名单化,同名透传,值按 json.dumps 拼装)。

    防御直调路径:null 视同未传(与 build_v5_cli_args 带值键同款)——否则
    json.dumps(None) 会拼出 "--channels_last null",子进程 ultralytics check_cfg 对
    typed_keys 中默认非 None 的键显式 raise TypeError,训练直接 status=error。
    子进程 parse_v8_args 侧 json.loads 还原,与 json.dumps 往返闭合。"""
    argv = []
    for k, v in cli_params.items():
        if v is None:
            continue
        argv += [f"--{k}", v if isinstance(v, str) else json.dumps(v)]
    return argv


def build_hyp_data(hyp_params: dict) -> dict:
    """scratch-low 默认为底、用户 hyp 键覆盖,生成 hyp 字典(写 yaml 由调用方负责)。
    null 值键视同未传(spec §5.2):过滤后保默认,防 null 写进 _hyp.yaml。"""
    data = dict(V5_HYP_DEFAULTS)
    data.update({k: v for k, v in hyp_params.items() if v is not None})
    return data
