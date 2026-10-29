import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent.parent.parent))


from utils_v.log import init_logger, HealthAccessLogFilter
import logging
import uvicorn
from utils_v.index import init_root_folder,set_yolo_env
set_yolo_env(True)
init_root_folder(True)


from fastapi import FastAPI
from modal import CustomJSONResponse

from detect_v.process.gpu_manager import GPUManager
from detect_v.process.detect_v5 import detect_v5_model
global_gpu_manager : GPUManager

app = FastAPI(
    #lifespan=lifespan,
    default_response_class=CustomJSONResponse)

if __name__ == "__main__":
    # 配置日志
    logger = init_logger("detect_yolov5_service")
    # executor 每 10s 探活 /health,滤掉其 200 access 记录防刷屏(非 200 保留,是假死线索);
    # 挂在 logger 层且先于 uvicorn.run:uvicorn 的 dictConfig 不会清掉 logger 级 filter
    logging.getLogger("uvicorn.access").addFilter(HealthAccessLogFilter())

    from detect_v.process.routes import router as router_detect_process,init_detect_process
    init_detect_process(logger,True,detect_v5_model)
    app.include_router(router_detect_process) 
    # raise Exception("test error")
    uvicorn.run(app, host="0.0.0.0", port=29005) 