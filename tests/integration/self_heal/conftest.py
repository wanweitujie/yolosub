"""self_heal 子目录专用:可靠 cleanup(防污染其它套件)+ 前置门。

每个自愈测试后:① kill -CONT(救 STOP 态);② kill -9 残留;③ wait_healthy;
失败则 docker restart 终极兜底。确保下一测试从 HEALTHY 起步。
"""
import os
import sys
import subprocess

import pytest

# 从上层 integration/conftest.py 复用 helper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import docker_exec, child_pids, wait_healthy  # noqa: E402


@pytest.fixture(autouse=True)
def ensure_healthy_teardown(container):
    yield
    try:
        # 救 STOP 态 + 杀残留(ultralytics + v5)
        for svc in ("ultralytics", "yolov5"):
            for pid in child_pids(container, svc):
                docker_exec(container, f"kill -CONT {pid} 2>/dev/null; kill -9 {pid} 2>/dev/null", timeout=5)
        if not wait_healthy(container, timeout=60):
            subprocess.run(["docker", "restart", container], timeout=60)   # 终极兜底
            wait_healthy(container, timeout=120)
    except Exception:
        pass
