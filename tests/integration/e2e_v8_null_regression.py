#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v8/v11「值为 null 一律视同未传」回归 E2E —— 2026-10-15

在 YoloSub 容器内执行(服务已在 29004 监听)。三段含义:
  A 负对照:ultralytics check_cfg 对 channels_last=None 确实 raise(修复前的崩溃点)
  B 正验证:提交 channels_last:null + freeze:null → 期望 status=finished
            (与历史问题报告中的失败作业同构;修复前该组合必 status=error)
  C 对照  :channels_last:false 显式假值须原样透传且不崩(防修复过度把假值一并吞掉)

透传证据取自服务端打印的拼装命令行(manager 内 print("[启动] ...") → log/nohup.out.log):
B 不应出现 --channels_last,C 应出现 --channels_last false。

训练参数刻意压小(fraction/imgsz/epochs)只为缩短耗时:崩溃点在 get_cfg 配置校验阶段,
先于数据加载,压小不影响结论。

用法(容器内): python /docker-data-yolo/tests/integration/e2e_v8_null_regression.py
"""
import json
import sys
import time
import urllib.request

API = "http://127.0.0.1:29004"
DATA = "/docker-data-yolo/train_data/demo/data.yaml"
SVC_LOG = "/docker-data-yolo/log/nohup.out.log"
BASE = {"fraction": 0.1, "imgsz": 320, "batch": 16}

try:  # 容器 stdout 可能非 UTF-8,中文证据行需显式转码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _req(path, body=None, timeout=30):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        API + path, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_ready():
    for i in range(1, 61):
        try:
            _req("/train/status", timeout=5)
            print("  服务就绪(第 %d 次探测)" % i)
            return True
        except Exception:
            time.sleep(3)
    return False


def wait_terminal(exp, timeout_s=900):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = (_req("/train/status/" + exp).get("data") or {}).get("status") or ""
        if st in ("finished", "error", "stopped"):
            return st
        time.sleep(5)
    return "timeout"


def last_cmdline():
    """服务端最近一条拼装命令行(透传证据)。"""
    try:
        with open(SVC_LOG, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f if "[启动]" in ln]
        return lines[-1] if lines else "(未取到)"
    except Exception as e:
        return "(读取失败: %s)" % e


def run_case(label, params):
    print("===== %s =====" % label)
    p = dict(BASE)
    p.update(params)
    print("  提交 params: %s" % json.dumps(p, ensure_ascii=False))
    r = _req("/train/start", {"version": "v8", "model": "yolov8n.pt",
                              "data": DATA, "epochs": 1, "params": p})
    print("  提交响应: %s" % json.dumps(r, ensure_ascii=False))
    exp = (r.get("data") or {}).get("exp") or ""
    if not exp:
        print("  FAIL: 未取得 exp\n")
        return ("", "no-exp")
    st = wait_terminal(exp)
    print("  exp=%s  终态=%s" % (exp, st))
    print("  服务端拼装命令行: %s" % last_cmdline())
    if st != "finished":
        try:
            lg = _req("/train/log/" + exp)
            print("  子进程日志尾部: ...%s" % json.dumps(lg, ensure_ascii=False)[-1200:])
        except Exception as e:
            print("  取日志失败: %s" % e)
    print()
    return (exp, st)


def main():
    print("===== A 负对照:ultralytics check_cfg 对 null 的行为 =====")
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
        for k in ("channels_last", "freeze"):
            try:
                get_cfg(DEFAULT_CFG_DICT, {k: None})
                print("  %s=None -> 未抛异常" % k)
            except TypeError as e:
                print("  %s=None -> TypeError: %s" % (k, e))
    except Exception as e:
        print("  (跳过: %s)" % e)
    print()

    print("===== 0 等待服务就绪 =====")
    if not wait_ready():
        print("  FAIL: 服务未就绪")
        return 1
    print()

    results = [
        run_case("B 正验证 channels_last:null + freeze:null(历史问题报告同构)",
                 {"channels_last": None, "freeze": None}),
        run_case("C 对照   channels_last:false(显式假值须透传且不崩)",
                 {"channels_last": False}),
    ]

    print("===== 汇总 =====")
    ok = True
    for (exp, st), want in zip(results, ("finished", "finished")):
        good = st == want
        ok = ok and good
        print("  %-30s 终态=%-9s 期望=%-9s %s"
              % (exp or "-", st, want, "OK" if good else "FAIL"))
    print("\n结果: %s" % ("全部通过" if ok else "存在失败"))
    print("本次实验目录: %s" % ", ".join(e for e, _ in results if e))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
