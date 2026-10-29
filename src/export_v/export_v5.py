
import json
from logging import Logger
from pathlib import Path
from typing import Optional
from modal import ModelExportConfig
import os
from modal import ModelExportConfig
import subprocess

from utils_v.index import YOLOV5_DIR, YOLOV5_PYTHON_PATH, ULTRALYTICS_PYTHON_PATH


def export_v5(logger:Logger,config: ModelExportConfig,gpu_id:Optional[int]):
    format = str(config.format)
    if format == "coreml":
        return export_model_mlmodel(config,gpu_id)
    if format == "onnx":
        # v5 onnx 走主环境:onnx 栈(numpy2/onnx1.22/ml-dtypes0.6)只在主环境自洽;
        # v5 venv 的 numpy 1.23.1(coremltools 6.0 锁)与之互斥。wrapper 不前置 venv site-packages。
        path_python = ULTRALYTICS_PYTHON_PATH
        wrapper = "process/v5_wrap_main.py"
    else:
        path_python = YOLOV5_PYTHON_PATH
        wrapper = "process/v5_wrap.py"
    path_yolov5 = YOLOV5_DIR
    export_kwargs = {
        k: v for k, v in config.model_dump(exclude_none=True).items()
        if k not in ("model_path", "version", "format", "v5_data")
    }
    args = [path_python, str((Path(__file__).parent / wrapper).resolve()), "--weights", config.model_path]
    for field, value in export_kwargs.items():
        if isinstance(value, bool):
            if value==True:
                args.append(f"--{field}")            
            continue
        elif isinstance(value, tuple):
            args.append(f"--{field}")
            args.extend(str(x) for x in value)
            continue
        
        args.extend([f"--{field}",  str(value)])
    if gpu_id is not None:
        args.extend(["--device",str(gpu_id)])
        
    # args.extend(["--include","onnx" if config.format=="ncnn" else config.format])
    args.extend(["--include", config.format])
    if logger:
        logger.info(f"export_yolov5.start:{args}")
    result = subprocess.check_output(args,
                                    text=True,
                                    timeout=360,
                                    cwd=Path(__file__).parent.parent,
                                    start_new_session=True
                                    )
    if logger:
        logger.info(f"export_yolov5.result:{result}")
    start_marker = "[yolov5.export.start$$$]"
    end_marker= "[$$$yolov5.export.end]"
    start_pos = result.find(start_marker)
    if start_pos == -1:
        raise Exception(result)
    content_start = start_pos + len(start_marker)
    end_pos = result.find(end_marker, content_start)
    if end_pos == -1:
        raise Exception(result)
    filepath = json.loads(result[content_start:end_pos].strip())[0]
    if os.path.exists(filepath):
        return filepath
    raise Exception(result)
    
    
    

def export_model_mlmodel(config: ModelExportConfig,gpu_id:Optional[int])->str:

    path_python = YOLOV5_PYTHON_PATH
    # 第三方 yolov5_convert_weight_to_coreml 目录保持 upstream 原样不改;
    # 经 wrapper 以 runpy 执行原版 convert.py,在外层把 int 类别名规范化为 str
    # (data yaml 的 names 写成数字时 model.names 为 int,CoreML NMS stringClassLabels 只收 str)
    path_wrapper = str((Path(__file__).parent / 'process' / 'v5_coreml_wrap.py').resolve())
    path_yolov5 =YOLOV5_DIR
    arg_yolov5 = '--yolov5-repo'

    args = [path_python, path_wrapper, arg_yolov5,path_yolov5]
    args.append("--weight")
    args.append(config.model_path)
    if gpu_id is not None:
        args.extend(["--device",str(gpu_id)])
    # "--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    # "--conf-thres", type=float, default=0.25, help="confidence threshold"
    # "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    if config.imgsz is not None:
        args.append("--img-size")
        args.append(f"{config.imgsz[0]}")
    if config.half == True:
        args.append("--fp16")
    elif config.int8  == True:
        args.append("--int8")
    
    print(f"mlmodel-process.start:{args}")
    try:
        result = subprocess.check_output(args,
                                    text=True,
                                    timeout=360,
                                    stderr=subprocess.PIPE,
                                    cwd=Path(__file__).parent.parent,
                                    start_new_session=True
                                    )
    except Exception as e:
         # 子进程真实报错在 stderr,不捕获的话异常消息里只剩命令行,无法定位根因。
         # 注意:不能改写 e.args——CalledProcessError 重写了 __str__(按 cmd/returncode 字段拼消息,
         # 不读 args),改写不生效,须直接拼进 msg
         stderr = getattr(e, 'stderr', None) or ''
         msg=str(e)
         if stderr:
             msg = f"{msg}\n[stderr]\n{stderr}"
         msg = msg.replace(path_python,'python')
         msg = msg.replace(path_wrapper,'-c-')
         msg = msg.replace(arg_yolov5,'-av-')
         msg = msg.replace(path_yolov5,'-v-')
         raise Exception(msg)
    print(f"mlmodel-process.result:{result}")
    
    start_marker = "CoreML export success, saved as "
    end_marker = "\n"
    start_pos = result.find(start_marker)
    if start_pos == -1:
        raise Exception(result)
    content_start = start_pos + len(start_marker)
    end_pos = result.find(end_marker, content_start)
    if end_pos == -1:
        raise Exception(result)
    filepath = result[content_start:end_pos].strip()
    if os.path.exists(filepath):
        return filepath
    raise Exception(result)