"""train 测试:仅负向覆盖(参数校验/错误路径)。真实训练在 train_data 为空时 skip。

诚实声明:train 模块的正向训练路径在当前套件零覆盖(train_data/ 为空)。

⚠️ 字段(train_v/schemas.py):真实字段 = version / data / epochs / model / params
(无 model_path/imgsz/batch)。model 默认 "yolov8n.pt";imgsz/batch 等额外参数走 "params"
(manager.py:131 逐项透传)。
"""
import time
import pytest

from conftest import docker_exec


def test_train_unsupported_version_returns_422(client):
    """Literal['v5','v8','v11'] 校验在路由前抛 → HTTP 422(无 success 字段)。"""
    r = client.post("/train/start", json={
        "version":"v9","data":"/docker-data-yolo/train_data/dummy.yaml","epochs":1})
    assert r.status_code == 422
    # 不访问 r.json()["success"](可能无此字段)


def test_train_without_required_fields(client):
    """缺必填字段(如 data)→ 422。"""
    r = client.post("/train/start", json={"version":"v5","epochs":1})
    assert r.status_code == 422


@pytest.mark.slow
def test_train_real_run(client, container):
    """真实训练:train_data 为空 → skip(补数据后自动启用)。字段按真实 schema。"""
    data_yaml = docker_exec(container, "find /docker-data-yolo/train_data -name '*.yaml' | head -1").strip()
    if not data_yaml:
        pytest.skip("train_data 为空,跳过真实训练(正向路径当前无护栏覆盖)")
    r = client.post("/train/start", json={
        "version":"v11",
        "data":data_yaml,
        "epochs":1,
        "model":"/docker-data-yolo/models/ultralytics/yolo11n.pt",   # 真实字段是 model,非 model_path
        "params":{"imgsz":320,"batch":1}},                           # imgsz/batch 走 params 透传
        timeout=600)
    assert r.json()["success"]


@pytest.mark.slow
def test_train_autobatch_v8(client, container):
    """v8/v11 batch=-1 -> ultralytics autobatch 动态实测选 batch(对新架构准)。

    验证 batch=-1 全链路:train_v8.py 解析(json.loads("-1")=int -1,非 str)
    -> precheck 跳过(-1<=0) -> ultralytics auto_batch() 触发 -> 训练进入 running。
    锁住 autobatch 路径防回归(若 batch=-1 误解析为 str,ultralytics batch_size<1 会 TypeError)。
    """
    data_yaml = "/docker-data-yolo/train_data/demo/data.yaml"
    if not docker_exec(container, f"test -f {data_yaml} && echo ok").strip():
        pytest.skip(f"缺少训练数据 {data_yaml}")
    try:
        client.post("/train/stop", timeout=15)
    except Exception:
        pass
    r = client.post("/train/start", json={
        "version":"v11","data":data_yaml,"epochs":1,
        "model":"/docker-data-yolo/models/ultralytics/yolo11n.pt",
        "params":{"imgsz":320,"batch":-1}}, timeout=60)
    assert r.json()["success"], f"batch=-1 启动失败:{r.text}"
    exp = r.json()["data"]["exp"]
    try:
        # 等 running:init_status 在 yolo.train 前写;autobatch 在 yolo.train 内,完成后训练开始
        # 多槽契约:无参 /train/status 返回 {trainings, gpu_occupancy, cpu_busy},单训练状态走 /train/status/{exp}
        deadline = time.time() + 120
        running = False
        while time.time() < deadline:
            s = client.get(f"/train/status/{exp}").json().get("data", {}).get("status")
            if s == "running":
                running = True
                break
            if s in ("error", "stopped", "finished"):
                pytest.fail(f"batch=-1 未进入 running 即终态:{s}(可能 autobatch 路径回归)")
            time.sleep(2)
        assert running, "batch=-1 训练 120s 未进入 running(autobatch + 启动超时)"
    finally:
        try:
            client.post(f"/train/stop/{exp}", timeout=30)
        except Exception:
            pass


@pytest.mark.slow
def test_train_autobatch_v5(client, container):
    """v5 batch=-1 -> yolov5 autobatch 动态实测选 batch。

    params 用 "batch":manager 经 V5_CLI_PARAMS 映射为 --batch-size(v8/v11 仍同名透传)
    (argparse 不认 --batch_size 下划线,经验证 FAIL)。验证 autobatch 触发 + 训练进入 running。
    """
    data_yaml = "/docker-data-yolo/train_data/demo/data.yaml"
    if not docker_exec(container, f"test -f {data_yaml} && echo ok").strip():
        pytest.skip(f"缺少训练数据 {data_yaml}")
    try:
        client.post("/train/stop", timeout=15)
    except Exception:
        pass
    r = client.post("/train/start", json={
        "version":"v5","data":data_yaml,"epochs":1,
        "model":"/docker-data-yolo/models/v5/yolov5n.pt",
        "params":{"imgsz":320,"batch":-1}}, timeout=60)
    assert r.json()["success"], f"v5 batch=-1 启动失败:{r.text}"
    exp = r.json()["data"]["exp"]
    try:
        deadline = time.time() + 120
        running = False
        while time.time() < deadline:
            s = client.get(f"/train/status/{exp}").json().get("data", {}).get("status")
            if s == "running":
                running = True
                break
            if s in ("error", "stopped", "finished"):
                pytest.fail(f"v5 batch=-1 未进入 running 即终态:{s}")
            time.sleep(2)
        assert running, "v5 batch=-1 训练 120s 未进入 running"
    finally:
        try:
            client.post(f"/train/stop/{exp}", timeout=30)
        except Exception:
            pass
