
from typing import Any, Callable, Dict, List, Optional
import GPUtil
from datetime import datetime
from logging import Logger
import torch
import cv2
import os
from modal import ModelConfig,ModelInfo
import asyncio
import gc
import numpy as np
import threading
import contextlib
import time
from utils_v.index import YOLOV5_DIR,format_yolo_modelpath, get_available_gpu

# 模块级常量(Phase 2 task 10 与 executor 常量统一收口)
LOCK_HELD_TIMEOUT = 60.0  # _gpu_lock 持有时长 watchdog 阈值(s),/health 据此判 503

class GPUManager:
    use_v5:bool
    detector_func: Callable[
    [Any, cv2.typing.MatLike, dict[str, Any], int],   # 参数列表（依次对应 model, contents, config, decimals）
    Any                                  # 返回值类型
]
    def __init__(self,logger:Logger,use_v5:bool,detector_func: Callable[
    [Any, cv2.typing.MatLike, dict[str, Any], int],   # 参数列表（依次对应 model, contents, config, decimals）
    Any                                  # 返回值类型
]):
        self.logger=logger
        self.use_v5=use_v5
        self.detector_func=detector_func
        self.models: Dict[str, ModelInfo] = {}
        self.gpu_memory_threshold = 0.85  # GPU内存使用阈值（85%）
        # self.model_memory_requirements = {
        #     "yolov8n": 500,  # MB
        #     "yolov8s": 800,
        #     "yolov8m": 1500,
        #     "yolov8l": 2500,
        #     "yolov8x": 3500,
        # }
        # GPU 操作统一锁:保护"模型使用 + 生命周期增删",避免 to_thread 并发后的数据竞争。
        # _lock_acquired_at 供子进程 /health watchdog 原子读,检测 forward hang。
        self._gpu_lock = threading.Lock()
        self._lock_acquired_at: float | None = None

    @contextlib.contextmanager
    def _gpu_locked(self):
        """GPU 操作统一临界区。acquire 时记时间戳(供 /health watchdog),release 清零。
        持锁超 LOCK_HELD_TIMEOUT(60s)打 WARNING,供上线后用真实分布验证阈值。"""
        self._gpu_lock.acquire()
        self._lock_acquired_at = time.monotonic()
        _entered = time.monotonic()
        try:
            yield
        finally:
            _held = time.monotonic() - _entered
            self._lock_acquired_at = None
            self._gpu_lock.release()
            if _held > LOCK_HELD_TIMEOUT:
                self.logger.warning(f"[lock] _gpu_lock held {_held:.1f}s (> {LOCK_HELD_TIMEOUT}s)")
            else:
                self.logger.debug(f"[lock] _gpu_lock held {_held:.3f}s")

    def _get_model_memory_usage_unlocked(self, model: any) -> int:
        """显存计算内核(不持锁)。调用方必须已持 _gpu_locked(或接受无锁并发)。
        抽出此 helper 是为避免 _load_locked / update_model_memory_usage 持锁后调
        get_model_memory_usage 的二次 acquire 死锁(_gpu_lock 非 RLock)。"""
        try:
            if next(model.parameters()).is_cuda:
                model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
                buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 / 1024
                return int(model_memory + buffer_memory)
            return 0
        except Exception as e:
            self.logger.error(f"Error getting model memory usage: {str(e)}")
            return 0

    def get_model_memory_usage(self, model: any) -> int:
        """获取模型实际占用的显存(MB)。遍历 parameters 与 forward 同级,持锁;
        供独立调用安全。锁内调本方法的场景必须改调 _get_model_memory_usage_unlocked。"""
        with self._gpu_locked():
            return self._get_model_memory_usage_unlocked(model)

    def _allocator_mb(self, model) -> tuple[Optional[int], Optional[int]]:
        """进程内缓存分配器读数(MB):(memory_allocated, memory_reserved)。CPU 模型 → (None, None)。
        与 nvidia-smi 无关,不受跨模型池复用影响,是每模型张量的真实占用。"""
        try:
            device = next(model.parameters()).device
            if device.type != "cuda":
                return None, None
            mb = 1024 * 1024
            return (int(torch.cuda.memory_allocated(device) / mb),
                    int(torch.cuda.memory_reserved(device) / mb))
        except Exception as e:
            self.logger.error(f"Error getting allocator stats: {str(e)}")
            return None, None

    def estimate_model_memory(self, model_path: str) -> int:
        return 300
        # """估算模型所需的GPU内存"""
        # for model_type, memory in self.model_memory_requirements.items():
        #     if model_type in model_path:
        #         return memory
        # return 1000  # 默认内存需求（MB）
    
    def preload_img(self,model):
        """预处理图片"""
        img = cv2.imread(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../assets/preload.webp")), cv2.IMREAD_COLOR)      
        if self.use_v5:
            self.preload_img_v5(model,img)
        else:
            self.preload_img_ultralytics(model,img)
            
    def preload_img_ultralytics(self,model,img):
        model(img)         

    def preload_img_v5(self,model,img):
        """预处理图片"""  
        from detect_v.process.detect_v5 import detect
        detect(model,img,5)
    
    def init_model(self,gpu_id:Optional[int],weights:str,v5_data:Optional[str],v5_fp16:Optional[bool]): 
        device =  'cpu' if gpu_id is None else f'cuda:{gpu_id}'
        if self.use_v5:
            return self.init_model_v5(device,weights,v5_data,v5_fp16)
        else:
            return self.init_model_ultralytics(device,weights)
    
    def detect(self, model_name: str, contents: bytes, config: dict[str, any], decimals: int = 5):
        """推理。model_name 在锁内从 self.models 取引用并校验,杜绝与 unload 的 TOCTOU。
        签名从 (model, ...) 改为 (model_name, ...):model 对象不再由路由层锁外传入。"""
        from fastapi import HTTPException
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        with self._gpu_locked():
            model_info = self.models.get(model_name)
            if model_info is None or model_info.model is None:
                raise HTTPException(status_code=404, detail="Model not found")
            model_info.last_used = datetime.now()
            # detector_func 返回 (框列表, speed_ms) 二元组:速度存入 last_speed_ms(经 /model_memory 带出),
            # 只返回框列表给路由层 → detect 对外响应结构不变。
            detections, speed_ms = self.detector_func(model_info.model, img, config, decimals)
            model_info.last_speed_ms = speed_ms
            return detections
        
    def init_model_ultralytics(self,device:str,weights:str):
        from ultralytics import YOLO
        
        model = YOLO(weights,task = 'detect')
        model.to(device)
        model.eval()
        return model
                
    def init_model_v5(self,device:str,weights:str,v5_data:str|None,v5_fp16:bool|None):        
        from yolov5.models.common import DetectMultiBackend

        device = torch.device(device)
        _, ext = os.path.splitext(weights)
        dnn = (ext.lower() == '.onnx')
        model = DetectMultiBackend(weights, device=device,data=v5_data,dnn=dnn,fp16=v5_fp16==True)
        model.eval()
        return model

    def preinit_model(self):
        gpu_id = get_available_gpu(500)
        if self.use_v5:
            return self.preinit_model_v5(gpu_id)
        else:
            return self.preinit_model_ultralytics(gpu_id)
        
    def preinit_model_ultralytics(self,gpu_id:int|None):
        try:
            with self._gpu_locked():
                model = self.init_model(gpu_id,f"/docker-data-yolo/models/ultralytics/yolo11n.pt",None,None)
                self.preload_img(model)
                del model
                # 与 unload_model 一致的清理:del 只把显存块归还缓存池,empty_cache 才还驱动。
                # 缺这两行时预热小模型权重滞留缓存池,初始化后 nvidia-smi 显存虚高。
                torch.cuda.empty_cache()
                gc.collect()
            self.logger.info(f"preinit_model_ultralytics completed")
        except Exception as e:
            # 预热失败属于可降级错误,不影响后续真实加载,降为 warning 避免喷 stack trace
            self.logger.warning(f"preinit_model_ultralytics failed: {e}")

    def preinit_model_v5(self,gpu_id:int|None):
        try:
            with self._gpu_locked():
                model = self.init_model(gpu_id,f"/docker-data-yolo/models/v5/yolov5n.pt",v5_data=f"{YOLOV5_DIR}/data/coco.yaml",v5_fp16=False)
                self.preload_img(model)
                del model
                # 与 unload_model 一致的清理(同上):empty_cache 归还预热残留给驱动。
                torch.cuda.empty_cache()
                gc.collect()
            self.logger.info(f"preinit_model_v5 completed")
        except Exception as e:
            # 预热失败属于可降级错误,不影响后续真实加载,降为 warning 避免喷 stack trace
            self.logger.warning(f"preinit_model_v5 failed: {e}")
            
    async def load_model(self, config: ModelConfig) -> ModelInfo:
        """加载模型到 GPU。重活(init + preload + 显存计算 + 字典写入)经 _load_locked 在
        线程池跑,事件循环不阻塞;重名重验与写入在同一 _gpu_locked 临界区,堵 create_model TOCTOU。"""
        self.logger.info("load_model.current.start")
        required_memory = self.estimate_model_memory(config.model_path)
        gpu_id = await asyncio.to_thread(get_available_gpu, required_memory)
        config.model_path = format_yolo_modelpath(config.version, config.model_path, 'model_path')
        model_info = await asyncio.to_thread(self._load_locked, config, gpu_id)
        self.logger.info(f"Model {config.model_name} actual GPU memory usage: {model_info.memory_usage}MB")
        self.logger.info(f"Model {config.model_name} loaded on GPU {gpu_id}")
        return model_info

    def _load_locked(self, config: ModelConfig, gpu_id):
        """重名重验 + init + preload + 显存 + 字典写入,同一 _gpu_locked 临界区。
        重验在 init 之前:避免重名时已 to(device) 的模型引用丢失导致显存泄漏。"""
        from fastapi import HTTPException
        with self._gpu_locked():
            if config.model_name in self.models:
                raise HTTPException(status_code=400, detail="Model name already exists")
            model = self.init_model(gpu_id, weights=config.model_path,
                                    v5_data=config.v5_data, v5_fp16=config.v5_fp16)
            if config.preload:
                self.preload_img(model)
            actual_memory = self._get_model_memory_usage_unlocked(model)
            alloc_mb, reserved_mb = self._allocator_mb(model)
            model_info = ModelInfo(model, gpu_id, actual_memory, datetime.now(),
                                   config.model_path, self.use_v5, None,
                                   alloc_mb, reserved_mb, None)
            self.models[config.model_name] = model_info
            return model_info


    def update_model_memory_usage(self, model_name: str, reset_peak: bool = False) -> Optional[ModelInfo]:
        '''更新模型占用显存(allocator 口径)。reset_peak=True 先重置进程内峰值统计(reset_peak_memory_stats),
        供显存测算脚本建立"本次 detect 窗口"的峰值基准。整体入锁。'''
        with self._gpu_locked():
            if model_name in self.models:
                model_info = self.models[model_name]
                model_info.memory_usage = self._get_model_memory_usage_unlocked(model_info.model)
                alloc_mb, reserved_mb = self._allocator_mb(model_info.model)
                model_info.memory_allocated = alloc_mb
                model_info.memory_reserved = reserved_mb
                peak = None
                if alloc_mb is not None:
                    device = next(model_info.model.parameters()).device
                    if reset_peak:
                        torch.cuda.reset_peak_memory_stats(device)
                    peak = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
                model_info.max_memory_allocated = peak
                return model_info.clone_to_main()
        return None

    def unload_model(self, model_name: str) -> bool:
        """卸载模型。del + empty_cache + gc + 字典删 在同一锁内,与 detect 互斥。"""
        try:
            with self._gpu_locked():
                if model_name not in self.models:
                    return False
                model_info = self.models[model_name]
                del model_info.model
                torch.cuda.empty_cache()
                gc.collect()
                del self.models[model_name]
            self.logger.info(f"Model {model_name} unloaded from GPU {model_info.gpu_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error unloading model: {str(e)}")
        return False

    def get_gpu_status(self) -> List[dict]:
        """获取GPU状态信息"""
        try:
            gpus = GPUtil.getGPUs()
            return [{
                'id': gpu.id,
                'memory_total': gpu.memoryTotal,
                'memory_used': gpu.memoryUsed,
                'memory_free': gpu.memoryFree,
                'gpu_load': gpu.load,
                'temperature': gpu.temperature
            } for gpu in gpus]
        except Exception as e:
            self.logger.error(f"Error getting GPU status: {str(e)}")
            return []