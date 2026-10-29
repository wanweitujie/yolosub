# app/routes.py
import asyncio
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query
from .manager import TrainingManager
from .schemas import TrainStartRequest, TrainStopRequest, resolve_stop_target
from .metrics import parse_since_epoch
from modal import toStandardResponseData
from logging import Logger

router = APIRouter()
mgr = TrainingManager()
logger: Optional[Logger] = None   # 由 init_train 注入;默认 None,异常分支 `if logger is not None` 才安全
def init_train(l):
    global logger
    logger=l

@router.post("/train/start", response_model=dict)
async def start_train(req: TrainStartRequest):
    try:
        return toStandardResponseData(await mgr.start(req))
    except Exception as e:
        if logger is not None: logger.exception(e)
        msg=str(e)
        return toStandardResponseData(None,msg)

@router.get("/train/status")
async def get_status(exp: str = None):
    try:
        return toStandardResponseData(mgr.status(exp))
    except Exception as e:
        msg=str(e)
        return toStandardResponseData(None,msg)

@router.get("/train/status/{exp}")
async def get_status_exp(exp: str):
    try:
        return toStandardResponseData(mgr.status(exp))
    except Exception as e:
        msg=str(e)
        return toStandardResponseData(None,msg)

@router.get("/train/log", response_model=str)
async def get_log(exp: str = None):
    try:
        return mgr.log(exp)
    except Exception as e:
        raise HTTPException(400, detail=str(e))

@router.get("/train/log/{exp}")
async def get_log_exp(exp: str):
    try:
        return mgr.log(exp)
    except Exception as e:
        raise HTTPException(400, detail=str(e))


@router.post("/train/stop/{exp}", response_model=dict)
async def stop_train_exp(exp: str):
    """停止指定 exp 的训练并释放其占用的卡。

    to_thread:stop 内含 killpg 后同步 wait(最长 15s,目标卡死在 CUDA C 扩展时),
    放线程池执行避免阻塞事件循环(等待期间其他路由/状态轮询仍可服务)。"""
    try:
        return toStandardResponseData(await asyncio.to_thread(mgr.stop, exp))
    except Exception as e:
        msg=str(e)
        if logger is not None: logger.exception(e)
        return toStandardResponseData(None,msg)

@router.post("/train/stop", response_model=dict)
async def stop_train(req: Optional[TrainStopRequest] = Body(None),
                     exp: Optional[str] = Query(None)):
    """停止训练。停止目标优先级:body.exp(上层规范,恒 POST /train/stop + JSON {exp})
    -> query exp -> 都缺省时停最后启动的活跃训练(存量无参 API 兼容)。

    修复:旧实现签名无 body 参数,JSON body 里的 {exp} 被 FastAPI 忽略,
    导致恒落 mgr.stop()(= _latest_active())错杀"最后启动的训练",与上层想停的 exp 可能不一致。
    现在 body/query 显式消费 exp 解析出 target,交由 mgr.stop(target) 精确停止(无参回退 latest)。
    同 stop_train_exp,经 to_thread 避免同步 wait 阻塞事件循环。"""
    target = resolve_stop_target(req.exp if req else None, exp)
    try:
        return toStandardResponseData(await asyncio.to_thread(mgr.stop, target))
    except Exception as e:
        msg = str(e)
        if logger is not None: logger.exception(e)
        return toStandardResponseData(None, msg)


@router.get("/train/metrics/{exp}", response_model=dict)
async def get_metrics_exp(exp: str, since_epoch: Optional[str] = Query(None)):
    """逐 epoch 训练指标(mAP50/mAP50-95/Precision/Recall,1-based)与终态结果摘要。

    since_epoch 为 str 声明 + handler 归一:FastAPI int 声明的 422 响应体是
    {"detail": [...]} 不套项目统一信封;非法值(非整数)统一信封 success=false 报错,
    不静默降级全量——增量游标静默回全量会让按增量语义 append 的客户端出现重复数据点。
    合法负值按 -1(全量),0 为合法游标(返回全部 epoch>0 的行,即全量)。"""
    try:
        se, err = parse_since_epoch(since_epoch)
        if err:
            return toStandardResponseData(None, err)
        return toStandardResponseData(mgr.metrics_query(exp, se))
    except Exception as e:
        msg = str(e)
        if logger is not None: logger.exception(e)
        return toStandardResponseData(None, msg)
