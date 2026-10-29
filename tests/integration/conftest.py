"""容器集成测试夹具与 helper。

复用运行中的 YoloSub 容器(按端口 29004 发现),主机 pytest 通过 httpx 访问 127.0.0.1:29004,
docker exec 注入故障/读日志。

环境要点:
- 子进程解释器按版本分支(executor.py _spawn):v5 用 /venv_yolov5/bin/python(v5 独立环境,靠
  init_root_folder(True) 注入 sys.path);ultralytics 用 /usr/bin/python(系统主环境,装 ultralytics 8.4.121)。
  历史:两者曾共用 /venv_yolov5/bin/python,ultralytics 靠 --system-site-packages 隐式链接主环境包,现已显式分支。
- v5_data 必须传 /venv_yolov5/mlmodels/yolov5/data/coco.yaml(容器实测可加载 COCO 与业务 v5 模型)。
- 测试模型完整容器路径(demo 模型不在 format_yolo_modelpath 简写规则内)。
"""
import os
import time
import base64
import subprocess
import httpx
import pytest


def _shq(s):
    """手动单引号包裹(bash 内字面量)。避免 Windows shlex.quote 对中文乱码。"""
    return "'" + s.replace("'", "'\\''") + "'"

# ===== 测试模型路径常量(完整容器路径)=====
PRELOAD_PATH = "/docker-data-yolo/src/assets/preload.webp"
M_COCO_V11N = "/docker-data-yolo/models/ultralytics/yolo11n.pt"
M_COCO_V11L = "/docker-data-yolo/models/ultralytics/yolo11l.pt"
M_COCO_V5N  = "/docker-data-yolo/models/v5/yolov5n.pt"
M_BIZ_V5    = "/docker-data-yolo/models/demo/v5/demo-v5.pt"
M_BIZ_V8    = "/docker-data-yolo/models/demo/v8/demo-v8.pt"
# v5 类别 yaml:yolov5 包在 venv_yolov5/mlmodels/ 下(v5 独立环境),实测确认此路径可加载 COCO 与业务 v5 模型
V5_DATA_COCO = "/venv_yolov5/mlmodels/yolov5/data/coco.yaml"

MAIN_URL = "http://127.0.0.1:29004"   # 硬编码 IPv4,避免 Windows localhost→IPv6


# ===== helper =====
def docker_exec(container, cmd, timeout=30):
    """docker exec <cid> bash -c '<cmd>'。用 bash -c(非 -lc,避免 source .bashrc 污染 stdout)。
    容器输出为 UTF-8,显式 encoding 避免 Windows gbk 解码 UTF-8 中文出错(UnicodeDecodeError → stdout=None)。"""
    r = subprocess.run(
        ["docker", "exec", container, "bash", "-c", cmd],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
        env={**os.environ, "MSYS_NO_PATHCONV": "1"})
    return r.stdout

def child_pids(container, service="ultralytics"):
    """返回该服务全部 PID 列表(返回全部,断言数量可抓双 spawn 回归)。"""
    out = docker_exec(container,
        f"ps aux | grep {service}_service | grep -v grep | awk '{{print $2}}'")
    return [p for p in out.split() if p.isdigit()]

def wait_log(container, pattern, timeout=60, interval=1):
    """轮询直到 main.txt 含 pattern。用 grep -F|tail -1(非 tail -50,避免 stack trace 刷掉窗口)。
    用手动单引号 _shq(不经 Windows shlex.quote,其会对中文乱码)。
    注:匹配整个文件,可能命中历史日志 —— 需要只查新日志时用 wait_log_since。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = docker_exec(container,
            f"grep -F {_shq(pattern)} /docker-data-yolo/log/main.txt | tail -1", timeout=5)
        if out.strip():
            return True
        time.sleep(interval)
    return False


def log_line_count(container):
    """main.txt 当前行数(测试开始前记录,供 wait_log_since 只查新日志)。"""
    out = docker_exec(container, "wc -l < /docker-data-yolo/log/main.txt").strip()
    return int(out) if out.isdigit() else 0


def wait_log_since(container, pattern, start_line, timeout=60, interval=1):
    """只匹配 start_line 之后的新日志行(避免 grep 命中历史日志污染)。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = docker_exec(container,
            f"tail -n +{start_line} /docker-data-yolo/log/main.txt | grep -F {_shq(pattern)} | tail -1",
            timeout=5)
        if out.strip():
            return True
        time.sleep(interval)
    return False

def wait_healthy(container, timeout=120):
    """等服务回 HEALTHY:create 一个小模型成功。自愈测试 cleanup 用。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.post(f"{MAIN_URL}/create_model",
                json={"model_name":"__health_probe__","model_path":M_COCO_V11N,"version":"v11"},
                timeout=30)
            if r.json().get("success"):
                httpx.delete(f"{MAIN_URL}/close_model/__health_probe__", timeout=10)
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def load_preload(container):
    """docker exec base64 取预热图 bytes(避免二进制经 docker exec 损坏)。"""
    return base64.b64decode(docker_exec(container, f"base64 -w0 {PRELOAD_PATH}"))

def assert_detect_ok(resp, require_nonempty=False):
    """统一 detect 断言:success + 结构正确;require_nonempty=True 时加 len(data)>0(仅 COCO 模型)。"""
    j = resp.json()
    assert j["success"], j
    assert "data" in j and isinstance(j["data"], list)
    if require_nonempty:
        assert len(j["data"]) > 0, f"期望非空检出,实际 {len(j['data'])}"


# ===== fixture =====
def _find_container_by_port(port=29004):
    """docker ps 按发布端口 + running 过滤,断言唯一;校验是 YoloSub 容器(防别服务误中)。"""
    r = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"publish={port}", "--filter", "status=running"],
        capture_output=True, text=True)
    cids = [c for c in r.stdout.strip().splitlines() if c]
    if len(cids) != 1:
        return None
    cid = cids[0]
    chk = subprocess.run(["docker", "exec", cid, "bash", "-c",
                          "test -f /docker-data-yolo/src/yolo_service.py && echo ok"],
                         capture_output=True, text=True,
                         env={**os.environ, "MSYS_NO_PATHCONV": "1"})
    return cid if chk.stdout.strip() == "ok" else None

@pytest.fixture(scope="session")
def container():
    cid = _find_container_by_port()
    if not cid:
        pytest.skip("未找到唯一运行中的 YoloSub 容器(端口 29004),跳过 integration")
    yield cid

@pytest.fixture(scope="session")
def client(container):
    # timeout=120s > executor read 75s,避免 client 先断致连接复用 race
    with httpx.Client(base_url=MAIN_URL, timeout=120) as c:
        yield c

@pytest.fixture(autouse=True)
def require_alive_container(container):
    """每测试前确认主服务可达;容器中途挂了 → skip 而非连串 fail。"""
    try:
        r = httpx.get(f"{MAIN_URL}/gpu_status", timeout=5)
        if r.status_code != 200:
            pytest.skip("容器主服务不可达")
    except Exception:
        pytest.skip("容器主服务不可达")

@pytest.fixture(autouse=True)
def clean_models(client):
    """setup + teardown 双向清空已加载模型(防残留/污染)。"""
    def _cleanup():
        try:
            names = [m["name"] for m in httpx.get(f"{MAIN_URL}/gpu_status", timeout=10)
                     .json().get("data", {}).get("loaded_models", [])]
        except Exception:
            return
        for name in names:
            try: httpx.delete(f"{MAIN_URL}/close_model/{name}", timeout=30)
            except Exception: pass
    _cleanup()
    yield
    _cleanup()

@pytest.fixture(scope="session", autouse=True)
def require_test_models(container):
    """校验测试模型文件存在,缺则 skip(而非让用例一个个 RuntimeError)。"""
    required = [M_COCO_V11N, M_COCO_V11L, M_COCO_V5N, M_BIZ_V5, M_BIZ_V8, PRELOAD_PATH, V5_DATA_COCO]
    missing = [p for p in required if not docker_exec(container, f"test -f {p} && echo ok").strip()]
    if missing:
        pytest.skip(f"缺少测试模型文件:{missing}")
