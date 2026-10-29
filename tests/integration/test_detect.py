"""detect 模块:正向 + 边界 + 业务模型(数据完整性 + TOCTOU)。"""
import httpx
import json as _json
from concurrent.futures import ThreadPoolExecutor

from conftest import (M_COCO_V11N, M_COCO_V5N, M_BIZ_V5, M_BIZ_V8, V5_DATA_COCO,
                      load_preload, assert_detect_ok)


def test_create_detect_close_ultralytics(client, container):
    """yolo11n 全流程:COCO 模型 + preload 必检出 len>0。"""
    img = load_preload(container)
    r = client.post("/create_model", json={"model_name":"u","model_path":M_COCO_V11N,"version":"v11"})
    assert r.json()["success"], r.text
    assert_detect_ok(client.post("/detect/u", files={"file":("a.webp",img,"image/webp")}), require_nonempty=True)
    assert client.delete("/close_model/u").json()["success"]
    loaded = [m["name"] for m in client.get("/gpu_status").json()["data"]["loaded_models"]]
    assert "u" not in loaded


def test_create_detect_close_v5(client, container):
    """yolov5n(v5 独立环境 venv_yolov5/mlmodels,需 v5_data/v5_fp16)+ preload 必检出。"""
    img = load_preload(container)
    r = client.post("/create_model", json={
        "model_name":"v5","model_path":M_COCO_V5N,"version":"v5",
        "v5_data":V5_DATA_COCO,"v5_fp16":False})
    assert r.json()["success"], r.text
    assert_detect_ok(client.post("/detect/v5", files={"file":("a.webp",img,"image/webp")}), require_nonempty=True)
    assert client.delete("/close_model/v5").json()["success"]


def test_detect_with_options(client, container):
    """带 conf/iou 等;success + 结构正确(不断 len 随 conf 变化,preload 高置信度不稳健)。"""
    img = load_preload(container)
    client.post("/create_model", json={"model_name":"opt","model_path":M_COCO_V11N,"version":"v11"})
    for conf in (0.5, 0.9):
        r = client.post("/detect/opt", files={"file":("a.webp",img,"image/webp")},
                        data={"decimals":"3","options":_json.dumps({"conf":conf})})
        assert_detect_ok(r)


def test_detect_invalid_options_json(client, container):
    """options 非合法 JSON → success=false(经宽 try/except),不崩。"""
    img = load_preload(container)
    client.post("/create_model", json={"model_name":"bad","model_path":M_COCO_V11N,"version":"v11"})
    r = client.post("/detect/bad", files={"file":("a.webp",img,"image/webp")}, data={"options":"{not json"})
    j = r.json()
    assert j["success"] is False   # 不假设 HTTP 状态码(宽 try/except 吞成 200+success=false)


def test_detect_unknown_model(client):
    """detect 不存在模型 → success=false,errMsg 含 recreate 提示。"""
    r = client.post("/detect/nope", files={"file":("a.webp",b"fake","image/webp")})
    j = r.json()
    assert j["success"] is False
    assert "not found" in j["errMsg"].lower() or "recreate" in j["errMsg"].lower()


def test_create_duplicate_model_concurrent(client, container):
    """并发 create 同名:恰好 1 成功 + 1 失败(锁内重验 Task 4 TOCTOU)。"""
    def one_create(i):
        with httpx.Client(base_url="http://127.0.0.1:29004", timeout=60) as c:
            return c.post("/create_model",
                json={"model_name":"dup","model_path":M_COCO_V11N,"version":"v11"}).json()["success"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(one_create, range(2)))
    assert results.count(True) == 1, f"期望恰好 1 成功,实际 {results}"


def test_close_unknown_model(client):
    """close 不存在 → success=false,不崩。"""
    r = client.delete("/close_model/nope")
    assert r.json()["success"] is False


def test_gpu_status_format(client):
    """gpu_status 结构:gpu_status[].id/memory_total/.../gpu_load + loaded_models(顶层 key)。"""
    j = client.get("/gpu_status").json()["data"]
    assert "gpu_status" in j and isinstance(j["gpu_status"], list)
    g = j["gpu_status"][0]
    for k in ["id","memory_total","memory_used","memory_free","gpu_load","temperature"]:
        assert k in g, f"缺字段 {k}"
    assert "loaded_models" in j and isinstance(j["loaded_models"], list)


def test_biz_model_load_and_detect_v8(client, container):
    """业务模型 demo-v8:加载成功 + detect 不崩 + len==0(反向断言:业务权重对 COCO 场景图应无检出)。"""
    img = load_preload(container)
    r = client.post("/create_model", json={"model_name":"biz8","model_path":M_BIZ_V8,"version":"v8"})
    assert r.json()["success"], r.text
    r2 = client.post("/detect/biz8", files={"file":("a.webp",img,"image/webp")})
    j = r2.json()
    assert j["success"], j
    assert isinstance(j["data"], list)
    assert len(j["data"]) == 0, "业务模型对 COCO 场景图应无检出;若检出 person 等 COCO 类别,说明权重加载错误"
    assert client.delete("/close_model/biz8").json()["success"]


def test_biz_model_load_and_detect_v5(client, container):
    """业务模型 demo-v5(v5,含 v5_data/v5_fp16):加载 + detect + len==0 反向断言。"""
    img = load_preload(container)
    r = client.post("/create_model", json={
        "model_name":"biz5","model_path":M_BIZ_V5,"version":"v5",
        "v5_data":V5_DATA_COCO,"v5_fp16":False})
    assert r.json()["success"], r.text
    r2 = client.post("/detect/biz5", files={"file":("a.webp",img,"image/webp")})
    j = r2.json()
    assert j["success"]
    assert isinstance(j["data"], list)
    assert len(j["data"]) == 0, "业务模型对 COCO 场景图应无检出"
    assert client.delete("/close_model/biz5").json()["success"]


def test_multi_model_concurrent_load(client, container):
    """COCO + 业务模型共存(显存压力 + 权重切换不串)。业务模型在护栏的核心价值:加载/共存/显存,非业务类别检出。"""
    img = load_preload(container)
    r1 = client.post("/create_model", json={"model_name":"c","model_path":M_COCO_V11N,"version":"v11"})
    r2 = client.post("/create_model", json={"model_name":"b","model_path":M_BIZ_V8,"version":"v8"})
    assert r1.json()["success"] and r2.json()["success"]
    loaded = [m["name"] for m in client.get("/gpu_status").json()["data"]["loaded_models"]]
    assert set(["c","b"]).issubset(set(loaded))
    # 切换 detect 都不崩
    assert_detect_ok(client.post("/detect/c", files={"file":("a.webp",img,"image/webp")}))
    assert client.post("/detect/b", files={"file":("a.webp",img,"image/webp")}).json()["success"]


def test_detect_empty_file(client, container):
    """上传空文件(0 字节)-> success=false 友好 error,不触发 SharedMemory size=0 崩溃。"""
    img = load_preload(container)
    client.post("/create_model", json={"model_name":"empty","model_path":M_COCO_V11N,"version":"v11"})
    r = client.post("/detect/empty", files={"file":("empty.webp", b"", "image/webp")})
    j = r.json()
    assert j["success"] is False
    assert "空" in j["errMsg"] or "empty" in j["errMsg"].lower(), j["errMsg"]
