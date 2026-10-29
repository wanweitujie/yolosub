import asyncio
import json
from logging import Logger
from multiprocessing.shared_memory import SharedMemory
import threading
import time
from typing import Any, Callable
import cv2
from fastapi import APIRouter, HTTPException, Request

from detect_v.process.gpu_manager import GPUManager
from detect_v.process.gpu_manager import LOCK_HELD_TIMEOUT
from modal import ModelConfig, toStandardResponseData


router = APIRouter()
logger:Logger
gpu_manager:GPUManager
def init_detect_process(l:Logger,use_v5:bool,detector_func: Callable[
    [Any, cv2.typing.MatLike, dict[str, Any], int],   # 参数列表（依次对应 model, contents, config, decimals）
    Any                                  # 返回值类型
]):
    global logger,gpu_manager
    logger=l
    # 初始化GPU管理器
    gpu_manager = GPUManager(logger,use_v5,detector_func)
    # 在线程中异步执行预热，不阻塞当前方法
    thread = threading.Thread(target=gpu_manager.preinit_model, daemon=True)
    thread.start()
    
    
@router.post("/create_model")
async def create_model(config: ModelConfig):
    try:
        if config.model_name in gpu_manager.models:
            raise HTTPException(status_code=400, detail="Model name already exists")
        logger.info("create_model.start")
        model_info = await gpu_manager.load_model(config)
        if model_info is None:
            raise HTTPException(status_code=500, detail="Failed to load model")
        
        return toStandardResponseData(model_info.clone_to_main())
    except Exception as e:
        logger.exception(e)
        return toStandardResponseData(None,str(e))

@router.post("/detect/{model_name}")
async def detect(
    model_name: str,
    request: Request
):
    try:
        data:dict[str,any] = await request.json()
        shm_name=data["shm_name"]
        shm_size=data.get("shm_size",0)
        decimals=data.get("decimals",5)
        options=data.get("options")
        if not isinstance(decimals,int) or decimals<0 or decimals>20:
            decimals = 5
        config: dict[str,any] = None
        if options:
            # 主进程通过 JSON body 转发 options, request.json() 已将其解析为 dict;
            # 此处兼容 dict(主进程转发路径) 与 str(直接传 JSON 字符串) 两种情况
            if isinstance(options, dict):
                config = options
            elif isinstance(options, str):
                try:
                    config = json.loads(options)
                except json.JSONDecodeError:
                    raise HTTPException(status_code=422, detail="Invalid JSON in 'options'")
                except Exception as e:
                    raise HTTPException(status_code=422, detail=f"Invalid options format: {str(e)}")
            else:
                raise HTTPException(status_code=422, detail=f"Invalid options format: expected str or dict, got {type(options).__name__}")

        # 不再锁外调 get_model;detect 内部锁内取引用并校验,杜绝与 unload 的 TOCTOU
        shm: SharedMemory = None
        try:
            shm = SharedMemory(name=shm_name, create=False, size=shm_size)
            buf = shm.buf
            img_bytes = bytes(buf[:shm_size]) if buf else b""
            return toStandardResponseData(
                await asyncio.to_thread(gpu_manager.detect, model_name, img_bytes, config, decimals)
            )
        finally:
            # update_model_memory_usage 改由主进程统一周期触发(Task 7),此处删除避免每次推理重复触发
            if shm is not None:
                shm.close()
                shm = None

    except Exception as e:
        logger.exception(e)
        return toStandardResponseData(None,f"Error in detection: {str(e)}")

@router.get("/update_model_memory_usage/{model_name}")
async def update_model_memory_usage(model_name: str, reset: bool = False):
    try:
        return toStandardResponseData(
            await asyncio.to_thread(gpu_manager.update_model_memory_usage, model_name, reset_peak=reset)
        )
    except Exception as e:
        msg = str(e)
        logger.error(f"Error update_model_memory_usage: {msg}")
        return toStandardResponseData(None, msg)

@router.delete("/close_model/{model_name}")
async def close_model(model_name: str):
    try:
        success = await asyncio.to_thread(gpu_manager.unload_model, model_name)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to unload model")
        
        return toStandardResponseData({
            "name": model_name
        })
    except Exception as e:
        msg=str(e)
        logger.error(f"Error closing model: {msg}")
        return toStandardResponseData(None,msg)

@router.get("/health")
async def health():
    """健康探针。不持锁(只原子读 _lock_acquired_at),事件循环假死时不响应;
    forward hang 持 _gpu_lock 超阈值时返回 503,供主进程 /health watchdog 判死。"""
    acquired_at = gpu_manager._lock_acquired_at
    if acquired_at is not None and (time.monotonic() - acquired_at) > LOCK_HELD_TIMEOUT:
        raise HTTPException(status_code=503, detail="gpu lock held too long")
    return {"ok": True}