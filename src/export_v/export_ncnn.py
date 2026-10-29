from logging import Logger
from pathlib import Path
import shutil
import subprocess
from typing import Optional, List
from modal import ModelExportConfig
import re
import logging


def _resolve_pnnx_path() -> Optional[str]:
    """动态查找 pnnx 可执行文件。

    优先级：
    1. ultralytics 包内 pnnx
    2. PATH 中的 pnnx
    """
    candidates = []
    try:
        import ultralytics
        candidates.append(str(Path(ultralytics.__file__).parent / "pnnx"))
    except ImportError:
        pass
    which = shutil.which("pnnx")
    if which:
        candidates.append(which)
    for p in candidates:
        if p and Path(p).exists():
            return p
    logging.getLogger(__name__).warning(
        "PNNX 未找到，ncnn-standard 导出将失败。"
        "请确认 ultralytics 包内含 pnnx 可执行文件，或 PATH 中存在 pnnx。"
    )
    return None

PNNX_PATH = _resolve_pnnx_path()


def _read_class_names(model_path: str) -> List[str]:
    """读模型内嵌 names,返回按 id 升序的标签名列表。

    异常不吞:加载失败等异常直接上抛,由调用方统一 try/except 兜底,
    以保证与调用方既有的告警文案一致。"""
    from ultralytics import YOLO
    names = YOLO(model_path).names
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def _write_ncnn_classes(out_dir: Path, model_path: str) -> None:
    """写类别清单 classes.txt(每个标签名一行,无序号)到 ncnn 产物目录。

    来源为模型内嵌 names(与训练 yaml 的 names 一致,无需调用方传 yaml);
    供部署侧直接读取类别顺序。读取失败仅告警,不阻断转换。"""
    try:
        ordered = _read_class_names(model_path)
        (out_dir / "classes.txt").write_text("\n".join(ordered), encoding="utf-8")
    except Exception as e:
        logging.getLogger(__name__).warning(f"写入 classes.txt 失败(不影响转换): {e}")

def transform_torchscript_ncnn(config: ModelExportConfig,gpu_id:Optional[int],torchscript_path:str,logger:Logger):
    # touchscript = re.sub(r'_ncnn_model$', '.torchscript', result)
    # if os.path.exists(touchscript) and os.path.isfile(touchscript):
    #     os.remove(touchscript)
    # return result
    p_t = Path(torchscript_path)
    f = p_t.with_suffix("")
    str_f=str(f)
    try:
        use_v5 = config.version=="v5"
        use_simplify=config.format=="ncnn"
        prex = "" if use_v5 or use_simplify else "_"
        if not f.exists():
            f.mkdir(parents=True, exist_ok=True) 
        
        ncnn_args = [
                f"ncnnparam={f / f'{prex}model.ncnn.param'}",
                f"ncnnbin={f / f'{prex}model.ncnn.bin'}",
                f"ncnnpy={f / f'{prex}model_ncnn.py'}",
            ]
        model_pnnx_py_filename="_model_pnnx"
        model_pnnx_py_path = str(f / f'{model_pnnx_py_filename}.py')
        pnnx_args = [
            f"pnnxparam={f / '_model.pnnx.param'}",
            f"pnnxbin={f / '_model.pnnx.bin'}",
            f"pnnxpy={model_pnnx_py_path}",
            f"pnnxonnx={f / '_model.pnnx.onnx'}",
        ]

        cmd = [
            PNNX_PATH,
            torchscript_path,
            *ncnn_args,
            *pnnx_args,
            f"fp16={int(config.half or False)}",
            "device=cpu",
            # "optlevel=2",
            # f"device={'cpu' if gpu_id is None else 'gpu'}",
            # f'inputshape="{[1, 3, *config.imgsz]}"',
        ]
        if use_v5 or use_simplify:
            cmd.append(f'inputshape="{[1, 3, *config.imgsz]}"')
            
        f.mkdir(exist_ok=True)  # make ncnn_model directory
        cmd_result = subprocess.run(cmd, check=True,
                                    text=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    cwd=Path(__file__).parent.parent,
                                    start_new_session=True
            )
        logger.info("命令执行成功: %s\n%s", " ".join(cmd), cmd_result.stdout.strip())
        
        
        if use_v5 or use_simplify:
            ncnn_args_new=[]
            pnnx_args_new=[]
        elif replace_dynamic_shape_in_file(config.version,model_pnnx_py_path,logger):
            cmd = ["python","-c",f"import {model_pnnx_py_filename}; {model_pnnx_py_filename}.export_torchscript()"]
            cmd_result=subprocess.run(
                cmd,
                check=True,
                cwd = str_f,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True
                )
            logger.info("命令执行成功: %s\n%s", " ".join(cmd), cmd_result.stdout.strip())
            ncnn_args_new = [
                f"ncnnparam={f / 'model.ncnn.param'}",
                f"ncnnbin={f / 'model.ncnn.bin'}",
                f"ncnnpy={f / 'model_ncnn.py'}",
            ]
            pnnx_args_new = [
                f"pnnxparam={f / 'model.pnnx.param'}",
                f"pnnxbin={f / 'model.pnnx.bin'}",
                f"pnnxpy={f / 'model_pnnx.py'}",
                f"pnnxonnx={f / 'model.pnnx.onnx'}",
            ]
            cmd = [
                PNNX_PATH,
                f"{model_pnnx_py_path}.pt",
                 *ncnn_args_new,
                 *pnnx_args_new,
                f"fp16={int(config.half or False)}",
                "device=cpu",
                # "optlevel=2",
                # f"device={'cpu' if gpu_id is None else 'gpu'}",
                f'inputshape="{[1, 3, *config.imgsz]}"',
                f'inputshape2="{[1, 3, *[x // 2 for x in config.imgsz]]}"',
            ]
            cmd_result=subprocess.run(cmd, check=True,
                                      text=True,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT,
                                    cwd=Path(__file__).parent.parent,
                                    start_new_session=True)
            logger.info("命令执行成功: %s\n%s", " ".join(cmd), cmd_result.stdout.strip())
        else:
            # replace_dynamic_shape_in_file 未命中任何动态形状代码块:
            # 该 v8/v11 模型 pnnx 生成的 _model_pnnx.py 不匹配当前正则,
            # 无法进行二次 pnnx 转换来生成固定形状的"标准" ncnn。
            # 未命中的文件内容已由 replace_dynamic_shape_in_file 通过 logger.info 打印,
            # 可据此更新正则或确认该模型是否支持 ncnn-standard。
            raise RuntimeError(
                "ncnn-standard 转换失败:未在 pnnx 生成的模型代码中匹配到动态形状替换块"
                f"(version={config.version}),无法生成固定形状的标准 ncnn。"
                "该模型可能不支持 ncnn-standard,或需更新 replace_dynamic_shape_in_file 的正则。"
            )
        # 写类别清单 classes.txt(每个标签名一行,无序号)到产物目录;标准版/简化版均提供,失败不阻断导出。
        if not use_v5:
            _write_ncnn_classes(f, config.model_path)
        # Remove debug files        
        pnnx_files = [x.split("=")[-1] for x in pnnx_args]
        ncnn_files =[] if use_v5 or use_simplify else [x.split("=")[-1] for x in ncnn_args]
        pnnx_new_files = [x.split("=")[-1] for x in pnnx_args_new]
        for f_debug in ("debug.bin", "debug.param", "debug2.bin", "debug2.param",f"{model_pnnx_py_path}.pt",str(f / 'model_ncnn.py'), *pnnx_files,*ncnn_files ,*pnnx_new_files):
            Path(f_debug).unlink(missing_ok=True)
        shutil.rmtree(str(f / "__pycache__"),ignore_errors=True)
    except:
        shutil.rmtree(str_f,ignore_errors=True)
        raise
    finally:
       p_t.unlink(missing_ok=True)
    

    # yaml_save(f / "metadata.yaml", self.metadata)  # add metadata.yaml
    return str_f
# , None

def replace_dynamic_shape_in_file(version:str,file_path:str,logger:Logger):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    total_count = 0
    cache_content = content
    if version not in ("v8", "v11"):
        # 仅 v8/v11 的 detect head 需要动态形状替换。
        # 走到这里说明 version 为 None 或非标准值 —— 上游 /export_model 的 version 字段未正确传递
        # (v5 走 transform_torchscript_ncnn 的 use_v5 分支, 不会调用本函数)。
        # 若静默返回 False 会让下游第 140 行 pnnx_args_new 未赋值而 UnboundLocalError, 故此处直接报错。
        raise RuntimeError(
            f"replace_dynamic_shape_in_file 不支持的 version={version!r}, 仅支持 v8/v11。"
            "请检查 /export_model 请求的 version 字段是否正确传递为 'v8' 或 'v11'。"
        )
    if version=="v8" or version=="v11":
        #arg3
        content,count=re.subn(
            r"(?:(\s*v_\d+\s*=\s*v_\d+\.(?:reshape|view)\(\s*\d+\s*,\s*\d+\s*,\s*)\d+(\s*\)\s*\r?\n))(?:(\s*v_\d+\s*=\s*v_\d+\.(?:reshape|view)\(\s*\d+\s*,\s*\d+\s*,\s*)\d+(\s*\)\s*\r?\n))(?:(\s*v_\d+\s*=\s*v_\d+\.(?:reshape|view)\(\s*\d+\s*,\s*\d+\s*,\s*)\d+(\s*\)\s*\r?\n))(\s*)(v_\d+)(\s*=\s*torch\.cat\(\s*\(\s*v_\d+\s*,\s*v_\d+\s*,\s*v_\d+\s*\)\s*,\s*dim\s*=\s*)(?:2|-1)(\s*\)\s*\r?\n\s*)[\s\S]*?return\s*v_\d+",
            
            r"\1-1).transpose(1, 2\2\3-1).transpose(1, 2\4\5-1).transpose(1, 2\6\7\8\g<9>1\10return \8",
            
            content)         
        total_count+=count
        if count==0:
            #arg4 yolov8x6.pt
            content,count=re.subn(
                r"(?:(\s*v_\d+\s*=\s*v_\d+\.(?:reshape|view)\(\s*\d+\s*,\s*\d+\s*,\s*)\d+(\s*\)\s*\r?\n))(?:(\s*v_\d+\s*=\s*v_\d+\.(?:reshape|view)\(\s*\d+\s*,\s*\d+\s*,\s*)\d+(\s*\)\s*\r?\n))(?:(\s*v_\d+\s*=\s*v_\d+\.(?:reshape|view)\(\s*\d+\s*,\s*\d+\s*,\s*)\d+(\s*\)\s*\r?\n))(?:(\s*v_\d+\s*=\s*v_\d+\.(?:reshape|view)\(\s*\d+\s*,\s*\d+\s*,\s*)\d+(\s*\)\s*\r?\n))(\s*)(v_\d+)(\s*=\s*torch\.cat\(\s*\(\s*v_\d+\s*,\s*v_\d+\s*,\s*v_\d+\s*,\s*v_\d+\s*\)\s*,\s*dim\s*=\s*)(?:2|-1)(\s*\)\s*\r?\n\s*)[\s\S]*?return\s*v_\d+",
                
                r"\1-1).transpose(1, 2\2\3-1).transpose(1, 2\4\5-1).transpose(1, 2\6\7-1).transpose(1, 2\8\9\10\g<11>1\12return \10",
                
                content)         
            total_count+=count
         
    if version=="v11":
        content,count = re.subn(
            r"([^\r\n]*)(v_\d+)(\s*=\s*self\.)((?:model_)+[^(]+_attn_)(qkv_conv\(\s*v_\d+\s*\)\s*\r?\n\s*v_\d+\s*=\s*v_\d+\.(?:reshape|view)\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*)\d+(\s*\)[\s\S]*?)(?:(\s*v_\d+\s*=\s*v_\d+\.(?:reshape|view)\(\s*\d+\s*,\s*\d+\s*,\s*)\d+(\s*,\s*)\d+(\s*\)\s*\r?\n))(?:(\s*v_\d+\s*=\s*v_\d+\.reshape\(\s*\d+\s*,\s*\d+\s*,\s*)\d+(\s*,\s*)\d+(\s*\)\s*\r?\n))(\s*v_\d+\s*=\s*self\.\4pe_conv\(\s*v_\d+\s*\))",
            r"\1\2\3\4\5-1\6\7\2.size(2)\8\2.size(3)\9\10\2.size(2)\11\2.size(3)\12\13",
            content)    
        total_count+=count    
        
        

    if total_count == 0:        
        logger.info(f"\r\n---------------未找到需要替换的代码块，文件没有被修改---start------>>>\r\n{cache_content}\r\n<<<-------end------")
        return False
    else:
        print(f"成功替换 {total_count} 处代码块")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    