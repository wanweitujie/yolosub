

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent.parent))

from utils_v.index import init_root_folder,set_yolo_env
import json


if __name__ == "__main__":
    
    set_yolo_env(True)
    init_root_folder(True)
    
    from yolov5.export import parse_opt,run
    try:
        opt = parse_opt()
        opt.weights = opt.weights[0] if isinstance(opt.weights, list) else opt.weights
        
        is_ncnn = opt.include[0]=="ncnn"
        if is_ncnn:
            opt.include[0]="onnx"
        
        
        result = run(**vars(opt))
        prev="yolov5.export"
        print(f"[{prev}.start$$$]{json.dumps(result)}[$$${prev}.end]")
    except Exception as e:        
        print(f"export error in process:{e}")