import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0,str(Path(__file__).parent.parent.parent))


from utils_v.log import init_logger, HealthAccessLogFilter
import logging
import uvicorn
from utils_v.index import init_root_folder,set_yolo_env
set_yolo_env(False)
init_root_folder(False)


from fastapi import FastAPI
from modal import CustomJSONResponse

from detect_v.process.gpu_manager import GPUManager
global_gpu_manager : GPUManager

def detect_model(model: YOLO,img:cv2.typing.MatLike, config:dict[str,any],decimals : int = 5):
    # 返回 (框列表, speed_ms) 二元组:speed 取 results[0].speed({'preprocess','inference','postprocess'}, 单位 ms),
    # 由 gpu_manager.detect 拆包存入 model_info.last_speed_ms,detect 对外响应结构不变。
    results = model(img,**(config if config else {}))
    del img

    if results and len(results) > 0:
        return results[0].summary(normalize=False, decimals=decimals), results[0].speed

    return [], {"preprocess": 0, "inference": 0, "postprocess": 0}
app = FastAPI(
    #lifespan=lifespan,
    default_response_class=CustomJSONResponse)

if __name__ == "__main__":        
    # 配置日志
    logger = init_logger("detect_ultralytics_service")
    # executor 每 10s 探活 /health,滤掉其 200 access 记录防刷屏(非 200 保留,是假死线索);
    # 挂在 logger 层且先于 uvicorn.run:uvicorn 的 dictConfig 不会清掉 logger 级 filter
    logging.getLogger("uvicorn.access").addFilter(HealthAccessLogFilter())
    
    from detect_v.process.routes import router as router_detect_process,init_detect_process
    init_detect_process(logger,False,detect_model)
    app.include_router(router_detect_process) 
    
    uvicorn.run(app, host="0.0.0.0", port=29006) 