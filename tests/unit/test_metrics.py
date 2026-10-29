# tests/unit/test_metrics.py
"""metrics 纯函数单测:两代 csv 夹具解析、行级校验四件套、fitness 按版本分派、since_epoch 归一。"""
from train_v.metrics import parse_results_csv, summarize, parse_since_epoch


# ===== 夹具构造 =====
V5_KEYS = ["train/box_loss", "train/obj_loss", "train/cls_loss",
           "metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95",
           "val/box_loss", "val/obj_loss", "val/cls_loss", "x/lr0", "x/lr1", "x/lr2"]
V8_HEADER = ("epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
             "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),"
             "val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2\n")


def v5_csv_text(rows):
    """v5 夹具:表头与数据行均 %20s/%20.5g 填充(loggers/__init__.py:165-167 实证形态)。"""
    header = (("%20s," * 14) % tuple(["epoch"] + V5_KEYS)).rstrip(",") + "\n"
    body = "".join((("%20.5g," * 14) % tuple(r)).rstrip(",") + "\n" for r in rows)
    return header + body


def v8_csv_text(rows):
    """v8 夹具:无填充,epoch 后多 time 列(8.4.121 实跑实测)。rows: (epoch, p, r, m50, m50_95)。"""
    body = "".join(f"{e},1.5,2.0,3.0,4.0,{p},{r},{m50},{m95},1.1,2.2,3.3,9e-05,9e-05,9e-05\n"
                   for e, p, r, m50, m95 in rows)
    return V8_HEADER + body


def _v5_rows(n, m50=0.5, m95=0.3):
    """n 行 v5 数据(指标恒定),epoch 0 起。返回 (rows_text, 期望 1-based epoch 列表)。"""
    data = [(i, 2.0, 3.0, 1.0, 0.8, 0.6, m50, m95, 2.0, 3.0, 1.0, 1e-4, 1e-4, 1e-4) for i in range(n)]
    return v5_csv_text(data), list(range(1, n + 1))


# ===== 解析 =====
def test_parse_v5_padded_csv_and_epoch_normalized(tmp_path):
    """v5 填充表头/数据行解析 + epoch 0 起 → +1 归一 1-based。"""
    text, _ = _v5_rows(3)
    f = tmp_path / "results.csv"; f.write_text(text, encoding="utf-8")
    rows = parse_results_csv(f)
    assert [r["epoch"] for r in rows] == [1, 2, 3]
    assert rows[0]["map50"] == 0.5 and rows[0]["map50_95"] == 0.3


def test_parse_v8_csv_with_time_column(tmp_path):
    """v8 夹具解析,epoch 原样 1 起。"""
    f = tmp_path / "results.csv"
    f.write_text(v8_csv_text([(1, 0.0, 0.0, 0.0, 0.0), (2, 0.5, 0.4, 0.6, 0.35)]), encoding="utf-8")
    rows = parse_results_csv(f)
    assert [r["epoch"] for r in rows] == [1, 2]
    assert rows[1]["precision"] == 0.5 and rows[1]["map50"] == 0.6


def test_parse_v5_unsuffixed_precision_recall_matched(tmp_path):
    """v5 无后缀列名 metrics/precision 必须命中(只匹配带 (B) 形态会漏,spec §七)。"""
    text, _ = _v5_rows(2)
    f = tmp_path / "results.csv"; f.write_text(text, encoding="utf-8")
    rows = parse_results_csv(f)
    assert rows[0]["precision"] == 0.8        # 夹具第 5 列


def test_parse_non_detect_suffix_still_matched(tmp_path):
    """seg/pose 后缀 (M)/(P) 经列名族匹配仍命中(防非 detect .pt 全列失配)。"""
    header = ("epoch,time,metrics/precision(M),metrics/recall(M),metrics/mAP50(M),"
              "metrics/mAP50-95(M),x\n")
    f = tmp_path / "results.csv"; f.write_text(header + "1,1.5,0.7,0.6,0.5,0.3,0\n", encoding="utf-8")
    rows = parse_results_csv(f)
    assert rows and rows[0]["map50_95"] == 0.3


def test_parse_missing_file_or_empty_or_bad_header(tmp_path):
    f = tmp_path / "results.csv"
    assert parse_results_csv(f) == []                      # 不存在
    f.write_text("", encoding="utf-8")
    assert parse_results_csv(f) == []                      # 空
    f.write_text("epoch,foo\n1,2\n", encoding="utf-8")
    assert parse_results_csv(f) == []                      # 表头缺指标列 → 整体不可解析


def test_parse_truncated_last_line_dropped(tmp_path):
    """截断末行(无 \n 半行)被丢弃,已完成行不受影响(防 since_epoch 客户端永久丢轮,spec §七四件套②)。"""
    text, _ = _v5_rows(3)
    text += "               3,2.0,3.0,1.0,0.9,0.55,0.35"      # 无换行的半行(缺列)
    f = tmp_path / "results.csv"; f.write_text(text, encoding="utf-8")
    rows = parse_results_csv(f)
    assert [r["epoch"] for r in rows] == [1, 2, 3]
    assert rows[2]["map50"] == 0.5                           # 完整第 3 行保留,半行未污染


def test_parse_complete_last_line_without_newline_dropped(tmp_path):
    """四件套②独立触发:末行 14 字段完整但无尾换行仍被丢弃。撕裂行必无换行前缀,
    与①列数校验正交——完整行①拦不住,只有②能拦(不丢会多出 epoch 4 行)。"""
    text, _ = _v5_rows(3)
    text += "               3,2.0,3.0,1.0,0.9,0.6,0.62,0.35,2.0,3.0,1.0,1e-4,1e-4,1e-4"   # 14 字段,无 \n
    f = tmp_path / "results.csv"; f.write_text(text, encoding="utf-8")
    rows = parse_results_csv(f)
    assert [r["epoch"] for r in rows] == [1, 2, 3]           # 末行被②丢弃,未变 [1,2,3,4]


def test_parse_row_column_count_and_float_guarded(tmp_path):
    """列数不齐行丢弃(四件套①);指标 float() 失败行丢弃(四件套③)。"""
    text, _ = _v5_rows(2)
    text += "               3,2.0\n"                                        # 列数不齐
    text += "               4,2.0,3.0,1.0,abc,0.9,0.5,0.3,1,1,1,1,1,1\n"       # precision 非数
    f = tmp_path / "results.csv"; f.write_text(text, encoding="utf-8")
    rows = parse_results_csv(f)
    assert [r["epoch"] for r in rows] == [1, 2]


def test_parse_nan_inf_rows_dropped(tmp_path):
    """四件套③有限性:含 nan/inf 的指标行丢弃——float("nan") 本身成功,若不滤会以裸 NaN
    进入响应,json.dumps 产出非标准 NaN 字面量破坏 JSON 契约;正常行保留。"""
    text = v8_csv_text([(1, 0.3, 0.2, 0.4, 0.2)])
    # 列序同 V8_HEADER:epoch,time,3×train,4×metrics,3×val,3×lr——仅指标列填 nan/inf
    text += "2,1.5,2.0,3.0,4.0,0.4,nan,0.5,0.3,1.1,2.2,3.3,9e-05,9e-05,9e-05\n"
    text += "3,1.5,2.0,3.0,4.0,inf,0.3,0.5,0.3,1.1,2.2,3.3,9e-05,9e-05,9e-05\n"
    f = tmp_path / "results.csv"; f.write_text(text, encoding="utf-8")
    rows = parse_results_csv(f)
    assert [r["epoch"] for r in rows] == [1]           # nan 行与 inf 行都被丢,正常行保留


def test_parse_epoch_inf_overflow_row_dropped(tmp_path):
    """epoch 列 inf/溢出数(1e999→float inf)时 int(float()) 抛 OverflowError——不在旧
    except 元组内会整体抛出使该 exp 指标接口全不可用,按坏行丢弃(code-review 重审补漏)。"""
    text = v8_csv_text([(1, 0.3, 0.2, 0.4, 0.2)])
    text += "inf,1.5,2.0,3.0,4.0,0.4,0.3,0.5,0.3,1.1,2.2,3.3,9e-05,9e-05,9e-05\n"
    text += "1e999,1.5,2.0,3.0,4.0,0.4,0.3,0.5,0.3,1.1,2.2,3.3,9e-05,9e-05,9e-05\n"
    f = tmp_path / "results.csv"; f.write_text(text, encoding="utf-8")
    rows = parse_results_csv(f)
    assert [r["epoch"] for r in rows] == [1]           # epoch=inf 与 1e999 行被丢,正常行保留


def test_parse_dedup_same_epoch_keep_last(tmp_path):
    """同 epoch 多行取最后(v5 COCO 场景 train.py:427-428 实证)。"""
    data = [(0, 2.0, 3.0, 1.0, 0.8, 0.6, 0.5, 0.3, 1, 1, 1, 1, 1, 1),
            (1, 2.0, 3.0, 1.0, 0.9, 0.6, 0.6, 0.4, 1, 1, 1, 1, 1, 1),
            (1, 2.0, 3.0, 1.0, 0.7, 0.6, 0.55, 0.33, 1, 1, 1, 1, 1, 1)]         # 重复 epoch=1
    f = tmp_path / "results.csv"; f.write_text(v5_csv_text(data), encoding="utf-8")
    rows = parse_results_csv(f)
    assert len(rows) == 2 and rows[1]["map50"] == 0.55                     # 取最后一条


# ===== 摘要 =====
def test_summarize_v8_fitness_is_map50_95():
    """v8/v11 fitness ≡ map50_95(权重 [0,0,0,1]):map50 与 map50_95 argmax 异行时,best 取 map50_95 最大行。"""
    rows = [{"epoch": 1, "map50": 0.9, "map50_95": 0.30, "precision": 0.8, "recall": 0.7},
            {"epoch": 2, "map50": 0.7, "map50_95": 0.45, "precision": 0.6, "recall": 0.6},
            {"epoch": 3, "map50": 0.6, "map50_95": 0.40, "precision": 0.9, "recall": 0.9}]
    s = summarize(rows, "v8")
    assert s["best_epoch"] == 2 and s["best_map50"] == 0.7 and s["best_map50_95"] == 0.45
    assert s["final_precision"] == 0.9 and s["final_recall"] == 0.9        # final=最后一行


def test_summarize_v5_fitness_weighted():
    """v5 fitness = 0.1×map50 + 0.9×map50_95(vendored metrics.py 权重 [0,0,0.1,0.9])。"""
    rows = [{"epoch": 1, "map50": 0.95, "map50_95": 0.31, "precision": 0.8, "recall": 0.7},
            {"epoch": 2, "map50": 0.70, "map50_95": 0.44, "precision": 0.6, "recall": 0.6}]
    # epoch1 fitness=0.1*0.95+0.9*0.31=0.374; epoch2=0.1*0.70+0.9*0.44=0.466 → best=epoch2
    s = summarize(rows, "v5")
    assert s["best_epoch"] == 2


def test_summarize_tie_takes_later():
    """fitness 并列取后(两代 best.pt 判定均为 best_fitness == fi 即保存)。"""
    rows = [{"epoch": 1, "map50": 0.5, "map50_95": 0.3, "precision": 0.1, "recall": 0.1},
            {"epoch": 5, "map50": 0.5, "map50_95": 0.3, "precision": 0.9, "recall": 0.9}]
    assert summarize(rows, "v8")["best_epoch"] == 5


def test_summarize_all_zero_rows_no_crash():
    """全 0 指标行(实跑实证:欠训练首行全 0)不崩,best 取首行。"""
    rows = [{"epoch": 1, "map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0}]
    s = summarize(rows, "v8")
    assert s["best_epoch"] == 1 and s["best_map50"] == 0.0


def test_summarize_empty_rows_returns_empty_dict():
    assert summarize([], "v8") == {}


# ===== since_epoch 归一 =====
def test_parse_since_epoch_normalization():
    assert parse_since_epoch(None) == (-1, None)
    assert parse_since_epoch("") == (-1, None)
    assert parse_since_epoch("  ") == (-1, None)
    assert parse_since_epoch("0") == (0, None)            # 0 为合法全量游标
    assert parse_since_epoch("37") == (37, None)
    assert parse_since_epoch("-5") == (-1, None)           # 负值按 -1 全量
    val, err = parse_since_epoch("4O")                      # 笔误(字母O)
    assert val is None and "since_epoch" in err and "整数" in err
