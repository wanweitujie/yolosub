# app/schemas.py
from pydantic import BaseModel
from typing import Dict, Any, Literal, Optional

class TrainStartRequest(BaseModel):
    version: Literal["v5", "v8", "v11"]
    data: str
    epochs: int
    model: str = "yolov8n.pt"
    params: Dict[str, Any] = {}
    # 客户端任务标识:拼接进 exp 名 (exp{job_id}_{ddHHmmss});可为空(兼容未传的调用方)
    job_id: str = ""

class TrainStopRequest(BaseModel):
    """POST /train/stop 的停止目标。

    上层约定"恒用 POST /train/stop + JSON body {exp}"(不带路径参数)。
    exp 为 start 返回的训练 exp 名(如 exp{yyyyMMddHHmmss}_{job_id})。
    空/缺省时由 resolve_stop_target 回退停最后启动的活跃训练(存量无参 API 兼容)。"""
    exp: str = ""


def resolve_stop_target(body_exp, query_exp):
    """解析 POST /train/stop(无路径参版)的停止目标 exp。

    优先级:body.exp(上层规范,恒 POST /train/stop + JSON body {exp})
    -> query.exp -> 都缺失返回 None(由 manager.stop(None) 停最后启动的活跃训练)。

    修 bug:此前 /train/stop 签名无 body 参数,JSON body 里的 {exp} 被 FastAPI 忽略,
    恒落 mgr.stop()(= _latest_active())错杀"最后启动的训练"。抽为纯函数便于单测。
    空串/纯空白均视为未提供。"""
    if body_exp is not None and str(body_exp).strip():
        return str(body_exp).strip()
    if query_exp is not None and str(query_exp).strip():
        return str(query_exp).strip()
    return None


class TrainStatusResponse(BaseModel):
    """训练状态响应(对外 /train/status 的字段声明)。

    注:曾因插入 resolve_stop_target 时误删下方类声明行,致字段悬空为函数内
    不可达死代码(code review 发现后恢复);全仓库当前无直接引用,保留作声明。"""
    status: str
    message:Optional[str] = None
    pid: Optional[int] = None
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    progress: Optional[float] = None
    exp: Optional[str] = None
    version: Optional[str] = None
    # python: Optional[str] = None
    output:Optional[str] = None
    last:Optional[str]= None
    best:Optional[str]= None
    # plots:Optional[bool]= None
    # results:Optional[Any]= None