"""故障注入端到端:kill -STOP/-9/风暴→DEAD/DEAD 恢复/双 spawn 回归。

含治理核心 bug 回归断言:
- stderr 重定向回归:test_self_heal_unexpected_exit 断言 .stderr.log size 增长
- 双 spawn 回归:test_self_heal_double_spawn_regression 断言 PID 数 ≤1
"""
import time
import httpx

import pytest

from conftest import (docker_exec, child_pids, wait_log, wait_log_since,
                      log_line_count, MAIN_URL)


def _wait_dead_create(client, timeout=60):
    """轮询 create,直到 errMsg 含 subprocess DEAD(容忍先经历 RESTARTING 过渡)。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.post("/create_model",
            json={"model_name":"x","model_path":"/docker-data-yolo/models/ultralytics/yolo11n.pt","version":"v11"},
            timeout=10)
        if "subprocess DEAD" in r.json().get("errMsg",""):
            return True
        time.sleep(2)
    return False


def _stderr_size(container):
    out = docker_exec(container, "stat -c %s /docker-data-yolo/log/ultralytics_service.py.stderr.log").strip()
    return int(out) if out.isdigit() else 0


@pytest.mark.self_heal
@pytest.mark.slow
def test_self_heal_unexpected_exit(client, container):
    """kill -9 → monitor 第三类 → 自动重启 → 恢复。+ stderr.log size 增长(治理回归)。"""
    pids = child_pids(container, "ultralytics")
    assert len(pids) == 1, f"起点应有 1 个 ultralytics 进程,实际 {len(pids)}"
    pid = pids[0]
    before = _stderr_size(container)
    docker_exec(container, f"kill -9 {pid}")
    assert wait_log(container, "[restart] ultralytics_service.py 恢复成功", timeout=30)
    assert len(child_pids(container, "ultralytics")) == 1
    # 轮询等 stderr 增长(新子进程 uvicorn 启动/预热的 stderr 写入有时序,非立即)
    deadline = time.time() + 20
    grown = False
    while time.time() < deadline:
        if _stderr_size(container) > before:
            grown = True
            break
        time.sleep(1)
    assert grown, "stderr 治理回归:重启后 stderr.log 未增长 = 重定向失效"


@pytest.mark.self_heal
@pytest.mark.slow
def test_self_heal_event_loop_freeze(client, container):
    """kill -STOP 冻结 → /health ping 失败 3 次 → 判死 → 重启。"""
    pid = child_pids(container, "ultralytics")[0]
    docker_exec(container, f"kill -STOP {pid}")
    # 逐次确认(无 "×3" 字面值,实际是 count=1/2/3)
    assert wait_log(container, "ping failed, count=1", timeout=30)
    assert wait_log(container, "ping failed, count=2", timeout=20)
    assert wait_log(container, "ping failed, count=3", timeout=20)
    assert wait_log(container, "[restart] ultralytics_service.py 恢复成功", timeout=30)


@pytest.mark.self_heal
@pytest.mark.slow
def test_self_heal_crash_storm_to_dead(client, container):
    """快速连续 kill ×4(子进程一出现就 kill,不等恢复完成)→ 风暴保护 → DEAD。

    注:连续 kill 后重启可能变慢(>60s,GPU/显存碎片),故不"等恢复完成再 kill",而是
    轮询凑齐 4 次 kill(280s 窗口内,< 300s CRASH_WINDOW)。
    """
    start_line = log_line_count(container)   # 记录起始,只匹配本次 kill 之后的新日志
    kills = 0
    deadline = time.time() + 280
    while kills < 4 and time.time() < deadline:
        pids = child_pids(container, "ultralytics")
        if pids:
            docker_exec(container, f"kill -9 {pids[0]}")
            kills += 1
            time.sleep(1)
        else:
            time.sleep(1)
    assert kills == 4, f"280s 内只凑到 {kills} 次 kill"
    # DEAD 可能经"重启 3 次失败"(每次 wait_ready 60s + backoff ≈ 90s)才触发,需等 ~300s
    assert wait_log_since(container, "DEAD", start_line, timeout=300)  # 匹配 进入 DEAD(重启失败)或 转 DEAD(风暴)
    # 验证 create 返回 subprocess DEAD(此时已 DEAD,应立即返回)
    assert _wait_dead_create(client, timeout=30), "DEAD 后 create 应返回 subprocess DEAD"


@pytest.mark.self_heal
@pytest.mark.slow
def test_self_heal_dead_recovery(client, container):
    """触发 DEAD 后等恢复(>5min DEAD_RETRY_INTERVAL)。"""
    start_line = log_line_count(container)   # 记录起始,只匹配本次 kill 之后的新日志
    # 先触发 DEAD(轮询凑 4 次 kill,同 crash_storm)
    kills = 0
    deadline = time.time() + 280
    while kills < 4 and time.time() < deadline:
        pids = child_pids(container, "ultralytics")
        if pids:
            docker_exec(container, f"kill -9 {pids[0]}")
            kills += 1
            time.sleep(1)
        else:
            time.sleep(1)
    assert kills == 4, f"280s 内只凑到 {kills} 次 kill"
    # DEAD 可能经"重启 3 次失败"才触发,需等 ~300s
    assert wait_log_since(container, "DEAD", start_line, timeout=300)  # 匹配 进入 DEAD(重启失败)或 转 DEAD(风暴)
    # 等自动恢复(360s > 300s DEAD_RETRY_INTERVAL)
    assert wait_log_since(container, "[recover] ultralytics_service.py 从 DEAD 恢复", start_line, timeout=360)


@pytest.mark.self_heal
@pytest.mark.slow
def test_self_heal_double_spawn_regression(client, container):
    """kill + 并发 ping 失败:全程 PID 数 ≤1(双 spawn 回归)。"""
    pid = child_pids(container, "ultralytics")[0]
    docker_exec(container, f"kill -9 {pid}")
    # 重启过程中反复检查 PID 数
    deadline = time.time() + 60
    max_pids = 1
    while time.time() < deadline:
        n = len(child_pids(container, "ultralytics"))
        if n > max_pids:
            max_pids = n
        if wait_log(container, "[restart] ultralytics_service.py 恢复成功", timeout=2):
            break
        time.sleep(1)
    assert max_pids <= 1, f"双 spawn 回归:同时存在 {max_pids} 个 ultralytics 进程(应 ≤1)"
