# tests/unit/test_log_filter.py
"""回归:子服务 access log 过滤器只滤 /health 探活成功记录。

executor 假死自愈(_health_loop)每 10s ping 子进程 /health,成功(200)的 access 记录
纯刷屏(每子进程每天约 8600 行);非 200(503 gpu 锁超时)是假死线索必须保留,
业务请求(create_model/detect 等)的 access 记录全部保留。

按项目约定单测在 Linux 容器内跑:
    PYTHONPATH=src pytest tests/unit/test_log_filter.py -v
"""
import logging

from utils_v.log import HealthAccessLogFilter

# 与 uvicorn AccessFormatter 的 format 相同:'%s - "%s %s HTTP/%s" %d %s'
FMT = '%s - "%s %s HTTP/%s" %d %s'


def _record(method, path, status):
    return logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg=FMT, args=("127.0.0.1:43290", method, path, "1.1", status, "x"),
        exc_info=None)


def test_health_200_filtered():
    """探活成功记录被过滤(刷屏源头)。"""
    assert HealthAccessLogFilter().filter(_record("GET", "/health", 200)) is False


def test_health_503_kept():
    """/health 非 200 是假死自愈线索,必须保留。"""
    assert HealthAccessLogFilter().filter(_record("GET", "/health", 503)) is True


def test_business_requests_kept():
    """业务请求(含 200)的 access 记录不受影响。"""
    f = HealthAccessLogFilter()
    assert f.filter(_record("POST", "/create_model", 200)) is True
    assert f.filter(_record("POST", "/detect/m1", 200)) is True
    assert f.filter(_record("DELETE", "/close_model/m1", 200)) is True
    # 名字里恰好含 "health" 的业务模型不应误伤(路径须精确含 /health 段)
    assert f.filter(_record("POST", "/detect/health_model", 200)) is True
