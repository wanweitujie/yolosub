# tests/unit/test_train_stop_body.py
"""回归:POST /train/stop(无路径参版)必须消费 body/query 里的 exp,而不是恒落 _latest_active()。

背景(bug):旧 /train/stop 签名无 body 参数,上层规范"恒用 POST /train/stop + JSON
body {exp}"被 FastAPI 忽略(exp 丢失),路由落到 mgr.stop()(= _latest_active())
错杀"最后启动的活跃训练",与上层真正想停的 exp 不一致,甚至误停健康训练。

测法分层:
- resolve_stop_target 纯函数(只依赖 pydantic,轻环境可直接跑)
- 路由黑盒(mock 模块级 mgr,断言传入 mgr.stop 的目标 exp)

管理侧涉及 killpg/venv,完整路由测试按项目约定在 Linux 容器内跑:
    bash tests/run.sh            # 或
    PYTHONPATH=src pytest tests/unit/test_train_stop_body.py -v
"""
import asyncio
from unittest.mock import MagicMock, patch

from train_v import routes
from train_v.schemas import TrainStopRequest, resolve_stop_target

EXP_OTHER = "exp20260908145010_990102"   # 上层真正想停的训练
EXP_ACTIVE = "exp20260908145014_990103"  # 最后启动的活跃训练(旧实现的误停目标)


# ===== 纯函数 resolve_stop_target:不依赖 manager,轻环境可直接跑 =====

def test_target_body_exp_priority():
    assert resolve_stop_target(EXP_OTHER, EXP_ACTIVE) == EXP_OTHER   # body 优先于 query


def test_target_body_blank_falls_back_to_query():
    assert resolve_stop_target("   ", EXP_ACTIVE) == EXP_ACTIVE     # 空白 body 视为未提供


def test_target_body_none_query_used():
    assert resolve_stop_target(None, EXP_ACTIVE) == EXP_ACTIVE


def test_target_none_when_both_missing():
    assert resolve_stop_target("", None) is None                    # 均缺 -> None(回退 latest)


# ===== 路由黑盒:mock 模块级 mgr,断言传给它(经 to_thread)的目标 exp =====

def _capture_target(req, exp):
    fake = MagicMock()
    fake.stop.return_value = {"status": "stopped", "exp": EXP_ACTIVE}
    with patch.object(routes, "mgr", fake):
        asyncio.run(routes.stop_train(req, exp))
    return fake.stop.call_args.args[0]


def test_stop_body_exp_is_respected():
    # 上层规范路径:body 指定 EXP_OTHER,必须精确停它,而非误杀 latest(EXP_ACTIVE)
    assert _capture_target(TrainStopRequest(exp=EXP_OTHER), None) == EXP_OTHER


def test_stop_falls_back_to_query_exp():
    assert _capture_target(TrainStopRequest(), EXP_ACTIVE) == EXP_ACTIVE


def test_stop_blank_body_exp_ignored():
    assert _capture_target(TrainStopRequest(exp="   "), None) is None


def test_stop_no_target_passes_none():
    assert _capture_target(TrainStopRequest(), None) is None