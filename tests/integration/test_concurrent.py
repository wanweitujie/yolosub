"""并发 detect:不止断 success,必须断事件循环未被阻塞(/gpu_status P95 < 1s)。

注:detect 在子进程内排队等锁是正常的(子进程内锁串行);本套件只验证主进程事件循环。
"""
import time
import pytest
import httpx
from concurrent.futures import ThreadPoolExecutor

from conftest import (M_COCO_V11N, M_COCO_V11L, M_COCO_V5N, M_BIZ_V5, M_BIZ_V8,
                      V5_DATA_COCO, MAIN_URL, load_preload, docker_exec)


def test_detect_concurrent_same_model(client, container):
    """同模型 20 并发 detect + watchdog ping /gpu_status:全成功 + P95 < 1s(事件循环未阻塞)。"""
    client.post("/create_model", json={"model_name":"cc","model_path":M_COCO_V11N,"version":"v11"})
    img = load_preload(container)
    latencies = []

    def watchdog():
        # ⚠️ 复用连接 client:httpx.get 每次新建连接,高并发下连接建立排队致 P95 虚高(非事件循环阻塞)
        with httpx.Client(base_url=MAIN_URL, timeout=15) as c:
            end = time.time() + 15
            while time.time() < end:
                t0 = time.time()
                try:
                    c.get("/gpu_status")
                    latencies.append(time.time() - t0)
                except Exception:
                    latencies.append(10)
                time.sleep(0.2)

    def one_detect(i):
        with httpx.Client(base_url=MAIN_URL, timeout=120) as c:   # 每线程新 client(线程安全)
            r = c.post("/detect/cc", files={"file":("a.webp",img,"image/webp")}, data={"decimals":"5"})
            return r.status_code, r.json()

    with ThreadPoolExecutor(max_workers=21) as pool:   # 21 = 20 detect + 1 watchdog
        wg = pool.submit(watchdog)
        dets = list(pool.map(one_detect, range(20)))
        wg.result()

    assert all(code == 200 and j["success"] for code, j in dets)
    assert all(len(j["data"]) > 0 for code, j in dets), "COCO 模型对 preload.webp 必有检出"
    assert sorted(latencies)[int(len(latencies)*0.95)] < 1.0, \
        f"/gpu_status P95={sorted(latencies)[int(len(latencies)*0.95)]:.2f}s,事件循环疑似被阻塞"


def test_detect_concurrent_multi_model(client, container):
    """v5 + ultralytics 并发 detect:两路独立,互不阻塞。"""
    img = load_preload(container)
    client.post("/create_model", json={"model_name":"m11","model_path":M_COCO_V11N,"version":"v11"})
    client.post("/create_model", json={
        "model_name":"m5","model_path":M_COCO_V5N,
        "version":"v5","v5_data":V5_DATA_COCO,"v5_fp16":False})
    def det(name):
        with httpx.Client(base_url=MAIN_URL, timeout=120) as c:
            return c.post(f"/detect/{name}", files={"file":("a.webp",img,"image/webp")}).json()["success"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        rs = list(pool.map(det, ["m11","m5"]))
    assert all(rs)


def test_create_detect_concurrent(client, container):
    """create 大模型(子进程持锁)+ detect 并发:主进程事件循环不阻塞(/gpu_status P95 < 1s)。

    detect 在子进程内排队等 create 锁是正常的(子进程内串行);本测试只验证主进程事件循环。
    max_workers=3:watchdog/create/detect 三任务真并发(worker 数 < 任务数致 detect 排队)。
    """
    client.post("/create_model", json={"model_name":"d1","model_path":M_COCO_V11N,"version":"v11"})
    img = load_preload(container)
    latencies = []

    def watchdog():
        # ⚠️ 复用连接 client:httpx.get 每次新建连接,高并发下连接建立排队致 P95 虚高(非事件循环阻塞)
        with httpx.Client(base_url=MAIN_URL, timeout=15) as c:
            end = time.time() + 20
            while time.time() < end:
                t0 = time.time()
                try:
                    c.get("/gpu_status")
                    latencies.append(time.time() - t0)
                except Exception:
                    latencies.append(10)
                time.sleep(0.2)

    def _create_thread(name, path):
        with httpx.Client(base_url=MAIN_URL, timeout=120) as c:
            return c.post("/create_model", json={"model_name":name,"model_path":path,"version":"v11"})
    def _detect_thread(name, img):
        with httpx.Client(base_url=MAIN_URL, timeout=120) as c:
            return c.post(f"/detect/{name}", files={"file":("a.webp",img,"image/webp")})

    with ThreadPoolExecutor(max_workers=3) as pool:
        wg = pool.submit(watchdog)
        time.sleep(0.5)   # 让 watchdog 进稳态
        f_create = pool.submit(_create_thread, "big", M_COCO_V11L)
        f_detect = pool.submit(_detect_thread, "d1", img)
        create_r, detect_r = f_create.result(), f_detect.result()
        wg.result()   # watchdog 20s 窗口覆盖 create 全程

    assert create_r.json()["success"]
    assert detect_r.json()["success"]
    assert len(detect_r.json()["data"]) > 0
    assert sorted(latencies)[int(len(latencies)*0.95)] < 1.0, \
        f"/gpu_status P95={sorted(latencies)[int(len(latencies)*0.95)]:.2f}s,主进程事件循环疑似被阻塞"


def test_detect_concurrent_4_models_mixed(client, container):
    """场景A:2 v5 + 2 ultralytics(含 demo 业务模型)同时在线,
    每模型并发多次 detect。验证多模型并发互不串台 + 主进程事件循环不阻塞。

    业务模型(demo-v5 / demo-v8)对 COCO 场景图应无检出(len==0,反向断言权重未串台);
    COCO 模型(yolo11n / yolov5n)对 preload 必有检出(len>0)。
    """
    img = load_preload(container)
    # 4 模型:2 ultralytics(COCO yolo11n + 业务 demo-v8)+ 2 v5(COCO yolov5n + 业务 demo-v5)
    models = [
        ("m11",  {"model_path":M_COCO_V11N, "version":"v11"}),
        ("biz8", {"model_path":M_BIZ_V8,    "version":"v8"}),
        ("m5",   {"model_path":M_COCO_V5N,  "version":"v5", "v5_data":V5_DATA_COCO, "v5_fp16":False}),
        ("biz5", {"model_path":M_BIZ_V5,    "version":"v5", "v5_data":V5_DATA_COCO, "v5_fp16":False}),
    ]
    for name, cfg in models:
        r = client.post("/create_model", json={"model_name":name, **cfg})
        assert r.json()["success"], f"create {name} 失败:{r.text}"
    coco_models = {"m11", "m5"}   # COCO 模型期望非空;业务模型期望空(反向断言)

    latencies = []
    def watchdog():
        # 复用连接 client,避免每次新建连接致 P95 虚高
        with httpx.Client(base_url=MAIN_URL, timeout=15) as c:
            end = time.time() + 20
            while time.time() < end:
                t0 = time.time()
                try:
                    c.get("/gpu_status"); latencies.append(time.time()-t0)
                except Exception:
                    latencies.append(10)
                time.sleep(0.2)

    N = 8   # 每模型 8 次 detect -> 4×8=32 并发 detect
    def one_detect(name):
        with httpx.Client(base_url=MAIN_URL, timeout=120) as c:   # 每线程独立 client(线程安全)
            r = c.post(f"/detect/{name}", files={"file":("a.webp",img,"image/webp")})
            return name, r.status_code, r.json()

    with ThreadPoolExecutor(max_workers=N*4+1) as pool:   # 32 detect + 1 watchdog
        wg = pool.submit(watchdog)
        time.sleep(0.5)   # 让 watchdog 进稳态
        tasks = [name for name, _ in models for _ in range(N)]
        rs = list(pool.map(one_detect, tasks))
        wg.result()

    # 全成功
    assert all(code == 200 and j["success"] for _, code, j in rs), f"有 detect 失败:{rs}"
    # COCO 模型非空 / 业务模型空(并发下权重未串台)
    for name, _, j in rs:
        if name in coco_models:
            assert len(j["data"]) > 0, f"{name}(COCO) 应有检出,实际 {len(j['data'])}"
        else:
            assert len(j["data"]) == 0, f"{name}(业务) 对 COCO 场景图应无检出,实际 {len(j['data'])}(疑似权重串台)"
    # 主进程事件循环未阻塞:32 并发 detect 瞬时施压,P95 阈值放宽到 1.5s
    # (真阻塞会是 5-10s+;此处验证事件循环未卡死,而非零延迟调度)
    p95 = sorted(latencies)[int(len(latencies)*0.95)]
    assert p95 < 1.5, f"/gpu_status P95={p95:.2f}s,事件循环疑似被阻塞"


def test_create_detect_close_cycles_concurrent(client, container):
    """场景B:多 worker 并发,每 worker 多轮 create->detect->close 完整生命周期。

    验证子进程模型加载/卸载在并发反复创建销毁下的稳定性(锁、显存回收、状态无残留)。
    4 worker 各用不同模型(2 v5 + 2 ultralytics,含 demo 业务模型),每 worker 3 轮。
    """
    img = load_preload(container)
    # 4 worker:2 ultralytics(COCO + 业务)+ 2 v5(COCO + 业务)
    worker_specs = [
        ("w11",  M_COCO_V11N, "v11", {}),
        ("wb8",  M_BIZ_V8,    "v8",  {}),
        ("w5",   M_COCO_V5N,  "v5",  {"v5_data":V5_DATA_COCO, "v5_fp16":False}),
        ("wb5",  M_BIZ_V5,    "v5",  {"v5_data":V5_DATA_COCO, "v5_fp16":False}),
    ]
    ROUNDS = 3
    latencies = []
    def watchdog():
        with httpx.Client(base_url=MAIN_URL, timeout=15) as c:
            end = time.time() + 30
            while time.time() < end:
                t0 = time.time()
                try:
                    c.get("/gpu_status"); latencies.append(time.time()-t0)
                except Exception:
                    latencies.append(10)
                time.sleep(0.2)

    def one_worker(spec):
        prefix, path, ver, extra = spec
        with httpx.Client(base_url=MAIN_URL, timeout=120) as c:   # 每线程独立 client
            results = []
            for k in range(ROUNDS):
                name = f"{prefix}_{k}"   # 每轮唯一名,避免并发同名 TOCTOU
                cr = c.post("/create_model",
                            json={"model_name":name,"model_path":path,"version":ver, **extra})
                dr = c.post(f"/detect/{name}", files={"file":("a.webp",img,"image/webp")})
                cl = c.delete(f"/close_model/{name}")
                results.append((cr.json().get("success"),
                                dr.json().get("success"),
                                cl.json().get("success")))
            return results

    with ThreadPoolExecutor(max_workers=4+1) as pool:   # 4 worker + 1 watchdog
        wg = pool.submit(watchdog)
        time.sleep(0.5)
        rs = list(pool.map(one_worker, worker_specs))
        wg.result()

    # 全成功:create/detect/close 三段 × 4 worker × 3 轮 = 36 个 True
    flat = [r for worker_rs in rs for r in worker_rs]
    assert all(flat), f"有失败轮次(create/detect/close):{flat}"
    # 收尾无残留模型(全部 close 成功,loaded 应为空)
    loaded = [m["name"] for m in client.get("/gpu_status").json()["data"]["loaded_models"]]
    assert loaded == [], f"close 后应无残留模型,实际 {loaded}"
    # 主进程事件循环未阻塞
    p95 = sorted(latencies)[int(len(latencies)*0.95)]
    assert p95 < 1.0, f"/gpu_status P95={p95:.2f}s,事件循环疑似被阻塞"


def test_train_concurrent_with_detect(client, container):
    """场景:1 个训练进行中 + 检测并发。

    验证训练子进程(TrainingManager)与检测子进程(executor)独立并行:
    GPU 显存竞争下检测仍成功 + 业务模型无串台 + 主进程事件循环不阻塞 + 训练未被检测拖垮。
    训练用 yolo11n + imgsz=320/batch=1 降显存加速;4 检测模型共 ~1.5GB + 训练 ~1GB < 8GB。
    """
    # 训练数据:/docker-data-yolo/train_data/demo(48 类,train 6663 / valid 865)
    data_yaml = "/docker-data-yolo/train_data/demo/data.yaml"
    if not docker_exec(container, f"test -f {data_yaml} && echo ok").strip():
        pytest.skip(f"缺少训练数据 {data_yaml},跳过训练+检测并发场景")

    # 训练单例:确保无残留(上次测试若崩溃未 stop,本次 start 会撞"已有任务")
    try:
        client.post("/train/stop", timeout=15)
    except Exception:
        pass

    # 启动训练(yolo11n,小参数降显存+加速;epochs=1)
    r = client.post("/train/start", json={
        "version":"v11","data":data_yaml,"epochs":1,
        "model":"/docker-data-yolo/models/ultralytics/yolo11n.pt",
        "params":{"imgsz":320,"batch":1}}, timeout=60)
    assert r.json()["success"], f"训练启动失败:{r.text}"
    exp = r.json()["data"]["exp"]

    try:
        # 轮询 status=running(init_status 在 yolo.train 前写,子进程已在线)
        # 多槽契约:单训练状态走 /train/status/{exp}(无参返回总览 {trainings, gpu_occupancy, cpu_busy})
        deadline = time.time() + 120
        running = False
        while time.time() < deadline:
            s = client.get(f"/train/status/{exp}").json().get("data", {}).get("status")
            if s == "running":
                running = True
                break
            if s in ("error", "stopped", "finished"):
                pytest.fail(f"训练未进入 running 即终态:{s}")
            time.sleep(2)
        assert running, "训练 120s 未进入 running(数据集扫描慢或预检失败)"

        # 等 progress>0(on_train_batch_end 触发,GPU 真在 forward);超时不强求(running 已代表在线)
        deadline2 = time.time() + 60
        while time.time() < deadline2:
            prog = client.get(f"/train/status/{exp}").json().get("data", {}).get("progress") or 0
            if prog > 0:
                break
            time.sleep(1)

        # 训练 running 中,并发 detect 4 模型(2 v5 + 2 ultralytics,含 demo 业务模型)
        img = load_preload(container)
        models = [
            ("m11",  {"model_path":M_COCO_V11N, "version":"v11"}),
            ("biz8", {"model_path":M_BIZ_V8,    "version":"v8"}),
            ("m5",   {"model_path":M_COCO_V5N,  "version":"v5", "v5_data":V5_DATA_COCO, "v5_fp16":False}),
            ("biz5", {"model_path":M_BIZ_V5,    "version":"v5", "v5_data":V5_DATA_COCO, "v5_fp16":False}),
        ]
        for name, cfg in models:
            cr = client.post("/create_model", json={"model_name":name, **cfg})
            assert cr.json()["success"], f"训练占显存致 create {name} 失败:{cr.text}"
        coco_models = {"m11", "m5"}

        latencies = []
        def watchdog():
            with httpx.Client(base_url=MAIN_URL, timeout=15) as c:
                end = time.time() + 20
                while time.time() < end:
                    t0 = time.time()
                    try:
                        c.get("/gpu_status"); latencies.append(time.time()-t0)
                    except Exception:
                        latencies.append(10)
                    time.sleep(0.2)

        N = 6   # 每模型 6 次 = 24 并发 detect(训练占 GPU,较场景A 降并发)
        def one_detect(name):
            with httpx.Client(base_url=MAIN_URL, timeout=120) as c:
                r = c.post(f"/detect/{name}", files={"file":("a.webp",img,"image/webp")})
                return name, r.status_code, r.json()

        with ThreadPoolExecutor(max_workers=N*4+1) as pool:   # 24 detect + 1 watchdog
            wg = pool.submit(watchdog)
            time.sleep(0.5)
            tasks = [name for name, _ in models for _ in range(N)]
            rs = list(pool.map(one_detect, tasks))
            wg.result()

        # detect 全成功(训练+检测显存竞争下不失败)
        assert all(code == 200 and j["success"] for _, code, j in rs), f"训练并发下 detect 失败:{rs}"
        # COCO 非空 / 业务空(训练占用 GPU 下权重未串台)
        for name, _, j in rs:
            if name in coco_models:
                assert len(j["data"]) > 0, f"{name}(COCO) 应有检出,实际 {len(j['data'])}"
            else:
                assert len(j["data"]) == 0, f"{name}(业务) 应无检出,实际 {len(j['data'])}(疑似串台)"
        # 主进程事件循环未阻塞(训练+检测并发,阈值 1.5s)
        p95 = sorted(latencies)[int(len(latencies)*0.95)]
        assert p95 < 1.5, f"/gpu_status P95={p95:.2f}s,事件循环疑似被阻塞"
        # 训练仍 running(检测并发未拖垮训练子进程)
        ts = client.get(f"/train/status/{exp}").json().get("data", {}).get("status")
        assert ts == "running", f"检测并发期间训练异常退出:{ts}"
    finally:
        # 停训练(清理本用例启动的训练,避免残留占用影响后续用例;多槽下无参停最后启动的活跃训练)
        try:
            client.post(f"/train/stop/{exp}", timeout=30)
        except Exception:
            pass
