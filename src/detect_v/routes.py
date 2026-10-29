from fastapi import APIRouter, HTTPException,UploadFile, File,Form
from modal import ModelConfig,InferenceConfig,toStandardResponseData
from logging import Logger
from detect_v.model_manager import ModelManager
from typing import Optional
import json


router = APIRouter()
logger:Logger
gpu_manager:ModelManager
def init_detect(l:Logger, app):
    global logger,gpu_manager
    logger=l
    # 初始化GPU管理器
    gpu_manager = ModelManager(logger)
    # 注册 startup 事件，启动 24 小时闲置模型清理任务
    @app.on_event("startup")
    async def _start_cleanup():
        gpu_manager.start_cleanup_task()

@router.post("/create_model")
async def create_model(config: ModelConfig):
    try:
        if config.model_name in gpu_manager.models:
            raise HTTPException(status_code=400, detail="Model name already exists")
        logger.info("create_model.start")        
        model_info = await gpu_manager.load_model(config)
        if model_info is None:
            raise HTTPException(status_code=500, detail="Failed to load model")
        # if config.preload:
        #     gpu_manager.preload_img(model_info.model)
        
        gpu_status = await gpu_manager.get_gpu_status()
        return toStandardResponseData({
            "name": config.model_name,
            "gpu_id": model_info.gpu_id,
            "gpu_status": gpu_status,
            "memory": {
                "memory_allocated": model_info.memory_allocated,
                "memory_reserved": model_info.memory_reserved,
                "max_memory_allocated": model_info.max_memory_allocated,
            },
        })
    except Exception as e:
        msg=str(e)
        logger.error(f"Error creating model: {msg}")
        return toStandardResponseData(None,msg)

@router.get("/model_memory/{model_name}")
async def model_memory(model_name: str, reset: bool = False):
    """查询模型 allocator 显存口径(加载后/推理峰值);reset=True 重置峰值窗口(供显存测算脚本)。"""
    try:
        if model_name not in gpu_manager.models:
            raise HTTPException(status_code=404, detail="Model not found")
        m = await gpu_manager.update_model_memory_usage(model_name, reset)
        return toStandardResponseData({
            "name": model_name,
            "memory_allocated": m.memory_allocated if m else None,
            "memory_reserved": m.memory_reserved if m else None,
            "max_memory_allocated": m.max_memory_allocated if m else None,
            "speed_ms": m.last_speed_ms if m else None,
        })
    except Exception as e:
        msg = str(e)
        logger.error(f"Error model_memory: {msg}")
        return toStandardResponseData(None, msg)

@router.post("/detect/{model_name}")
async def detect(
    model_name: str,
    file: UploadFile = File(...),
    decimals : int = Form(5),
    options:Optional[str] = Form(None)
):
    try:
        if not isinstance(decimals,int) or decimals<0 or decimals>20:
            decimals = 5
        if model_name not in gpu_manager.models:
            raise HTTPException(status_code=404, detail="Model not found (if after subprocess restart, please recreate_model)")
        config: Optional[InferenceConfig] = None
        if options:
            try:
                config_data = json.loads(options)
                config = InferenceConfig(**config_data)
            except json.JSONDecodeError:
                raise HTTPException(status_code=422, detail="Invalid JSON in 'options'")
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Invalid options format: {str(e)}")
        contents = await file.read()
        if not contents:
            # 走统一响应(非 raise HTTPException):后者被外层 except 吞成 200+前缀污染 errMsg。
            # 统一 toStandardResponseData 保证 errMsg 干净,与 404/422 既有模式一致
            return toStandardResponseData(None, "上传文件为空,请检查 file 字段")
        model = gpu_manager.get_model(model_name)
        if model is None:
            raise Exception("Model not found")
        return toStandardResponseData(await gpu_manager.detect_async(model_name, contents, config, decimals))

    except Exception as e:
        logger.exception(e)
        return toStandardResponseData(None,f"Error in detection: {str(e)}")

@router.delete("/close_model/{model_name}")
async def close_model(model_name: str):
    try:
        success = await gpu_manager.unload_model(model_name)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to unload model")
        
        gpu_status = await gpu_manager.get_gpu_status()
        return toStandardResponseData({
            "name": model_name,
            "gpu_status":gpu_status,
        })
    except Exception as e:
        msg=str(e)
        logger.error(f"Error closing model: {msg}")        
        return toStandardResponseData(None,msg)
    

@router.get("/gpu_status")
async def get_gpu_status():
    """获取GPU状态"""
    try:
        status = await gpu_manager.get_gpu_status()
        return toStandardResponseData({
            "gpu_status": status,
            "loaded_models": [
                {
                    "name": name,
                    "path":info.path,
                    "gpu_id": info.gpu_id,
                    "last_used": info.last_used,
                    "memory_usage":info.memory_usage
                }
                for name, info in gpu_manager.models.items()
            ],
            "subprocess": gpu_manager.get_subprocess_status()
        })
    except Exception as e:
        msg=str(e)
        logger.error(f"Error getting GPU status: {msg}")   
        return toStandardResponseData(None,msg)
