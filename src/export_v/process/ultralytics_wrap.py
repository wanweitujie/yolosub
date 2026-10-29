import sys
from pathlib import Path
from typing import Optional
sys.path.insert(0,str(Path(__file__).parent.parent.parent))

from modal import ModelExportConfig
from utils_v.index import init_root_folder,set_yolo_env
import json

def parse_device(arg_str: str):
    """
    把从命令行收到的 device 参数转成 ultralytics 能识别的格式
    -1     → None     (自动选择 / CPU)
    "0"    → 0
    "1"    → 1
    ...
    """
    if arg_str.strip() == "-1":
        return None
    else:
        return int(arg_str)
    
def export_ultralytics(config: ModelExportConfig,gpu_id:Optional[int]):    
    from ultralytics import YOLO,__version__
    model = YOLO(config.model_path, task='detect') # load a pretrained
    model.to('cpu' if gpu_id is None else f'cuda:{gpu_id}')
    
    print(f"ultralytics.__version__={__version__},gpu_id={gpu_id}")
    export_kwargs = {
        k: v for k, v in config.model_dump(exclude_none=True).items()
        if k != "model_path" and k != "version"
    }
    # model.to() 只搬权重,不会更新 exporter 的 args.device;必须把 device 显式传给 model.export(),
    # 否则 half=True 时 exporter 会因 select_device 落到 CPU 而报 "FP16 ... only supported on GPU"。
    # 仅 GPU 时注入;CPU 导出(upload 校验已拒 half)保持不注入,以免误导 exporter 自动选卡。
    if gpu_id is not None:
        export_kwargs["device"] = gpu_id
    result = model.export(**export_kwargs)
    del model
    return result

if __name__ == "__main__":
    prev = "coreml.export"
    try:
        if len(sys.argv) != 4:
            raise Exception("Usage: ultralytics_wrap.py <json_config> <device> <low_ultralytics>")
        low = (sys.argv[3] == "1")
        set_yolo_env(False, low)
        init_root_folder(False)
        config = ModelExportConfig(**json.loads(sys.argv[1]))
        device = parse_device(sys.argv[2])
        result = export_ultralytics(config, device)
        print(f"[{prev}.start$$$]{result}[$$${prev}.end]")
    except json.JSONDecodeError as e:
        print(f"export error in process:{e}")
        
    
    