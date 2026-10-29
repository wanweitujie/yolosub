
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from export_v.export_ncnn import transform_torchscript_ncnn, _write_ncnn_classes, _read_class_names
from export_v.export_v5 import export_v5
from modal import toStandardResponseData,ModelExportConfig
import shutil
import re
from utils_v.index import get_available_gpu
from utils_v.zip import zip_with_max_compression
from datetime import datetime
import json
from logging import Logger
from export_v.export import export_ultralytics

router = APIRouter()
logger:Logger
def init_export(l:Logger):
    global logger
    logger=l

#导出的文件是否也压缩成zip
compress_output_file_to_zip = True

@router.post("/export_model")
def export_model(config: ModelExportConfig):
    result:str = None
    try:        
        if not config.model_path or not os.path.exists(config.model_path):
            raise Exception(f"模型文件不存在: {config.model_path}")
        format = str(config.format)
        # v5 简化版 ncnn 走 pnnx 链路(与 ncnn-standard 一致),绕开坏掉的 yolov5-onnx 转换。
        # format 为函数开头保存的用户原始值,不等同于会/被改写的 config.format。
        is_v5_ncnn = (config.version == "v5" and format == "ncnn")
        zip_extra_contents = None   # onnx 非 v5 时携带 classes.txt 内容(见下方 onnx 分支)
        gpu_id = get_available_gpu(100)
        low_ultralytics=False
        if format=="mlmodel":
            config.format="coreml"
            config.version = "v5"
        elif format =="mlpackage":
            config.format="coreml"
            if config.version=="v5" : 
                config.version="v8"
        elif format =="mlpackage-ios14":
            low_ultralytics=True
            config.format="mlmodel"  # 由 "coreml" 改:走 neuralnetwork(coremltools6)(ios14 spec≤5)
            config.nms = True        # F1:显式 NMS pipeline,对齐实测形态与接口文档 L363
            if config.version=="v5" :
                config.version="v8"            
        elif format=="ncnn-standard" or format=="ncnn":
            config.nms=False
            config.int8=False
            if config.optimize:
                gpu_id = None
            # ncnn-standard 的 v8/v11 动态形状替换正则按 ultralytics 8.3.x 输出编写;
            # 升级 8.4.121 后 8.3→8.4 头布局变化(分支 reshape 间插入 conv 行),正则失配致转换失败。
            # 修复:复用 ios14 低版本 venv(ultralytics 8.3.77)导出 torchscript,并强制 CPU 导出
            # (已实测:8.3.77 仅 CPU 导出时 replace_dynamic_shape_in_file 命中=True,GPU 导出仍失配)。
            if format=="ncnn-standard" and config.version in ("v8", "v11"):
                low_ultralytics = True
                gpu_id = None     
                                           
        if format=="coreml":
            config.nms=True

        # ===== 仅对特定 format 生效的参数过滤 =====
        # 与接口文档「参数互斥与自动覆盖规则 - 2. 仅对特定 format 生效的参数」一致：
        # 不生效的参数统一重置为默认（None）。下游 export_v5 / ultralytics_wrap 均通过
        # config.model_dump(exclude_none=True) 透传，None 会被自动剔除，等价于"调用方未传该参数"，
        # 避免无关参数被透传给导出脚本后被误用或触发异常。
        # 注意：此处按「用户原始 format」判断（format 在第 30 行已保存，不会被上方改写影响）。
        if format != "onnx":
            # simplify / opset / dynamic 仅对 onnx 生效
            config.simplify = None
            config.opset = None
            config.dynamic = None
        if format not in ("torchscript", "ncnn-standard", "ncnn"):
            # optimize 仅对 TorchScript 系列生效：ncnn-standard 内部走 torchscript 导出，
            # ncnn / ncnn-standard 另有「optimize=true 强制 CPU」逻辑（见上方 ncnn 分支），故一并保留
            config.optimize = None
        if not config.int8:
            # data / fraction 仅在 int8=True 时作为校准数据集被消费
            # （ncnn 系列已在上方将 int8 强制为 False，因此会一并被过滤）
            config.data = None
            config.fraction = None

        # ncnn-standard 的 fp16 由 pnnx(fp16=1)在 CPU 侧完成,无需 GPU,故豁免该守卫。
        if gpu_id is None and config.half and format!="coreml" and format!="ncnn-standard" and not is_v5_ncnn:
            raise Exception("必须运行在GPU设备上才能配置half=True")
                
                
        if format=="ncnn-standard" or is_v5_ncnn:
            # ncnn-standard 依赖 version 走 v5 简化分支或 v8/v11 动态形状替换正则,
            # version 为 None 时 replace_dynamic_shape_in_file 会报错, 此处提前校验给出更清晰的提示。
            if config.version not in ("v5", "v8", "v11"):
                raise Exception(f"ncnn-standard 转换需要明确的 version (v5/v8/v11), 当前 version={config.version!r}")
            config.format="torchscript"
            # fp16 由 pnnx(fp16=1)在 CPU 侧完成;此处仅导出 torchscript(保持 FP32,可 CPU 导出)。
            # 故先将 half 剥离(config=None 使 export_kwargs 不含 half),transform 前恢复到 config.half。
            # (v8/v11 已在上方 ncnn 分支强制 CPU;v5 走 export_v5 独立处理。)
            _ncnn_saved_half = config.half
            config.half = None
            
        if config.version == "v5":
            result = export_v5(logger,config,gpu_id)
        else:          
            result = export_ultralytics(logger,config,gpu_id,low_ultralytics=low_ultralytics)
            
        if format=="ncnn-standard" or is_v5_ncnn:
            config.format=format
            config.half = _ncnn_saved_half   # 恢复:fp16 交 pnnx 处理
            result = transform_torchscript_ncnn(config,gpu_id,result,logger)  
        elif format=="ncnn":
            torchscript_path = Path(result).with_suffix(".torchscript")
            if torchscript_path.exists():
                torchscript_path.unlink(True)
            shutil.rmtree(str(Path(result) / "__pycache__"),ignore_errors=True)
            _write_ncnn_classes(Path(result), config.model_path)   # 简化版也写 classes.txt
        elif format == "onnx" and config.version != "v5":
            # 与 ncnn 产物形态对齐:onnx 产物 zip 内附带 classes.txt(每行一个标签名)。
            # 判定用用户原始 format(第 32 行保存,未被改写);version 已由 modal 归一化为小写。
            # 读失败仅告警不阻断(onnx 文件内已内嵌 names,信息不丢失)。
            try:
                names = _read_class_names(config.model_path)
                zip_extra_contents = {"classes.txt": "\n".join(names)}
            except Exception as e:
                logger.warning(f"onnx 读类别名失败,产物将不含 classes.txt: {e}")

        if os.path.exists(result):
            isfile = os.path.isfile(result)
            if not isfile or compress_output_file_to_zip :
                zippath = zip_with_max_compression(result,result + ".zip", extra_contents=zip_extra_contents)
                if isfile:
                    os.remove(result)
                else:
                    shutil.rmtree(result)
                result = zippath
            elif zip_extra_contents:
                logger.warning("未压缩产物,classes.txt 未随包提供")

        return toStandardResponseData(result)
    except Exception as e:
        msg=str(e)
        logger.exception(e)
        if result is not None and os.path.exists(result):
            if os.path.isfile(result):
                 os.remove(result)
            else:
                shutil.rmtree(result)
        return toStandardResponseData(None,msg)


@router.post("/export_model_upload")
def export_model_upload(
        config: str = Form(default=None),
        file:UploadFile|None=File(default=None)
    ):
    try:
        if file is None:
            raise Exception("请提交pt文件")
        filename = file.filename
        if not filename.endswith(".pt"):
            raise Exception("请提交pt文件")
        if config is None:
            raise Exception("丢失config配置")
        config_obj:ModelExportConfig
        try:
            config_data = json.loads(config)
            config_obj = ModelExportConfig(**config_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="Invalid JSON format in config")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Config validation error: {str(e)}")
        current_time = datetime.now()
        year_month = current_time.strftime("%Y/%m")
        filename = current_time.strftime("%d%H%M%S%f")[:-3] + ".pt"
        save_path =str(( Path(__file__).parent.parent.parent / f"wwwroot/uploads/{year_month}/{filename}").resolve())
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        config_obj.model_path = save_path
        result = export_model(config_obj)
        
        if result["success"] and isinstance(result["data"],str):
            result["data"]=result["data"].replace("/docker-data-yolo/wwwroot/","/static/")        
        return result
    except Exception as e:
        msg=str(e)
        logger.error(f"Error export_model_upload: {msg}")
        return toStandardResponseData(None,msg)