
import json
from logging import Logger
import os
from pathlib import Path
import subprocess
from typing import Optional
from modal import ModelExportConfig
from utils_v.index import EXPORT_MLPACKAGE_IOS14_PYTHON_PATH, ULTRALYTICS_PYTHON_PATH


def export_ultralytics(logger:Logger,config: ModelExportConfig,gpu_id:Optional[int],low_ultralytics:bool=False):
    # 常规 ultralytics 导出用主环境 python(与 detect 子服务同源);iOS14 低版本用独立 venv。
    # 原裸 "python" 靠 PATH 解析到系统主环境,行为等价但属隐式依赖,统一为显式常量。
    path_python = EXPORT_MLPACKAGE_IOS14_PYTHON_PATH if low_ultralytics else ULTRALYTICS_PYTHON_PATH

    args = [path_python,str(Path(__file__).parent / "process/ultralytics_wrap.py"), json.dumps(config.__dict__),str(-1 if gpu_id is None else gpu_id),"1" if low_ultralytics else "0"]
    
    
    logger.info(f"export_ultralytics_process.start:{args}")
    # 用 capture_output=True 捕获 stdout/stderr：check_output 默认不捕获 stderr，
    # 子进程异常退出时真正的 traceback 会被吞掉，上层只能看到 "returned non-zero exit status"。
    proc = subprocess.run(args,
                        text=True,
                        timeout=360,
                        cwd=Path(__file__).parent.parent,
                        start_new_session=True,
                        capture_output=True
                        )
    if proc.returncode != 0:
        raise Exception(
            f"export_ultralytics 子进程异常退出(exit={proc.returncode})"
            f"\n---- stdout ----\n{proc.stdout}"
            f"\n---- stderr ----\n{proc.stderr}"
        )
    result = proc.stdout
    logger.info(f"export_ultralytics_process.result:{result}")
    
    start_marker = "[coreml.export.start$$$]"
    end_marker = "[$$$coreml.export.end]"
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