# train_v/metrics.py
"""results.csv 解析与训练结果摘要(纯函数,无 manager/进程依赖)。

兼容两代产物:
- v5 7.0:表头/数据行均 %20s/%20.5g 填充,列名无后缀(metrics/mAP_0.5),epoch 0 起;
- ultralytics 8.4:无填充,epoch 后多 time 列,detect 列名带 (B) 后缀,epoch 1 起。
输出统一 1-based epoch(v5 行整体 +1,按表头判代际,防曲线/status.json/PRD「第 N 轮」错位一格)。
"""
import csv
import io
import math
from pathlib import Path

_METRIC_KEYS = ("map50", "map50_95", "precision", "recall")

# 列名族:每指标一组候选(两代),匹配前 strip + 去任务后缀 (B)/(M)/(P)…
# 本服务恒 detect(schemas 无 task 字段、task 列报错键),放宽匹配防非 detect .pt 全列失配
_COLUMN_FAMILIES = {
    "metrics/mAP_0.5":     "map50",
    "metrics/mAP50":       "map50",
    "metrics/mAP_0.5:0.95": "map50_95",
    "metrics/mAP50-95":    "map50_95",
    "metrics/precision":   "precision",
    "metrics/recall":      "recall",
}


def _match_column(cell: str):
    """表头单元格 → 指标键;strip 后去 (X) 后缀比对候选族;非指标列返回 None。"""
    base = cell.strip().split("(", 1)[0].rstrip()
    return _COLUMN_FAMILIES.get(base)


def parse_results_csv(csv_path) -> list:
    """解析 results.csv → 升序行列表;文件不存在/空/表头缺任一指标列 → []。

    行级校验四件套(spec §七,防半写入行被当已完成轮致 since_epoch 客户端永久丢数据):
    ① 列数与表头一致;② 文件不以 \\n 结尾时丢弃最后一段(撕裂行必为无换行前缀);
    ③ 四指标字段全部 float() 成功且有限(nan/inf 行丢弃);④ 同 epoch 多行取最后(v5 COCO 场景重复回调)。"""
    try:
        text = Path(csv_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if not text.strip():
        return []
    if not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]          # ② 丢半行
    header, col_idx, is_v5, dedup = None, {}, False, {}
    for row in csv.reader(io.StringIO(text)):
        if header is None:
            header = [c.strip() for c in row]         # 所有单元格 strip(v5 填充)
            if "epoch" not in header:
                return []
            is_v5 = any(c == "metrics/mAP_0.5" for c in header)   # 按表头判代际,整文件统一归一
            col_idx["epoch"] = header.index("epoch")
            for i, cell in enumerate(header):
                key = _match_column(cell)
                if key and key not in col_idx:
                    col_idx[key] = i
            if any(k not in col_idx for k in _METRIC_KEYS):
                return []                             # 表头缺任一指标列,整体不可解析
            continue
        if len(row) != len(header):                   # ① 列数不齐
            continue
        try:                                          # ③ 四指标字段全部 float() 成功且有限
            rec = {k: float(row[col_idx[k]].strip()) for k in _METRIC_KEYS}
            epoch_raw = int(float(row[col_idx["epoch"]].strip()))
            if not all(math.isfinite(x) for x in rec.values()):
                continue                              # nan/inf 行丢弃:float("nan") 本身成功,
                                                     # 不滤会以裸 NaN 进响应破坏 JSON 序列化
        except (ValueError, TypeError, OverflowError):
            continue                                  # OverflowError:epoch 列 inf/超大数(如 1e999)
                                                       # 时 int(float()) 溢出,同样按坏行丢弃,防整体抛出
        rec["epoch"] = epoch_raw + 1 if is_v5 else epoch_raw
        dedup[rec["epoch"]] = rec                     # ④ 同 epoch 覆写=取最后
    return [dedup[e] for e in sorted(dedup)]


def detect_csv_generation(csv_path):
    """仅读表头行判定 csv 代际:"v5"/"v8"/None(不可读或无 epoch 列)。
    供 manager 在 status.json 缺 version 字段时回退分派 fitness 公式——
    历史任务(平铺窗口 2026-01~09)的 status.json 多无 version,静默按 v8
    公式算 best 会错行(与 parse_results_csv 的 is_v5 同判据,轻量重读表头)。"""
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
    except OSError:
        return None
    cells = [c.strip() for c in header.split(",")]
    if "metrics/mAP_0.5" in cells:
        return "v5"
    if "epoch" in cells:
        return "v8"
    return None


def summarize(rows: list, version: str) -> dict:
    """终态摘要 5 字段。fitness 按版本分派(与各代 best.pt 保存判定一致):
    v5 = 0.1×map50 + 0.9×map50_95;v8/v11 = map50_95(权重 [0,0,0,1])。
    并列取后(两代均为 best_fitness == fi 即保存);final_* = 最后一行。rows 空 → {}。"""
    if not rows:
        return {}
    if version == "v5":
        def fit(r):
            return 0.1 * r["map50"] + 0.9 * r["map50_95"]
    else:
        def fit(r):
            return r["map50_95"]
    best, best_f = rows[0], fit(rows[0])
    for r in rows[1:]:
        f = fit(r)
        if f >= best_f:                               # >= 并列取后
            best, best_f = r, f
    final = rows[-1]
    return {"best_epoch": best["epoch"], "best_map50": best["map50"],
            "best_map50_95": best["map50_95"],
            "final_precision": final["precision"], "final_recall": final["recall"]}


def parse_since_epoch(raw):
    """since_epoch 归一(str 声明防 FastAPI int 校验的 422 裸响应体破坏统一信封)。
    None/空 → (-1, None) 全量;非整数 → (None, errMsg)由调用方信封报错(不静默降级——
    增量游标静默回全量会让按增量语义 append 的客户端出现重复数据点);整数 → (max(-1, n), None)。"""
    if raw is None or not str(raw).strip():
        return -1, None
    try:
        n = int(str(raw).strip())
    except ValueError:
        return None, f"since_epoch 必须为整数: {raw!r}"
    return max(-1, n), None
