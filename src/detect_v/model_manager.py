
from typing import Dict, List, Optional
import GPUtil
from datetime import datetime
from logging import Logger
from modal import ModelConfig,InferenceConfig,ModelInfo
import asyncio
from detect_v.process.executor import YoloExecutorManager


    
class ModelManager:
    executor_v5:YoloExecutorManager
    executor_ultralytics:YoloExecutorManager
    _cleanup_task: Optional[asyncio.Task]
    def __init__(self,logger:Logger):
        self.logger=logger
        self.executor_v5=YoloExecutorManager(logger,True,on_dead=self._on_subprocess_dead)
        self.executor_ultralytics=YoloExecutorManager(logger,False,on_dead=self._on_subprocess_dead)
        self.models: Dict[str, ModelInfo] = {}
        self._cleanup_task = None
        self._health_tasks: List[asyncio.Task] = []

    def _on_subprocess_dead(self, use_v5: bool):
        """子进程判死/崩溃回调:清空主进程对应版本的 models。
        主进程 ModelInfo.model 本就是 None(clone_to_main 产物),纯字典操作,不碰 GPU。"""
        victims = [name for name, info in self.models.items() if info.use_v5 == use_v5]
        for name in victims:
            del self.models[name]
        self.logger.warning(
            f"[on_dead] 子进程(use_v5={use_v5})故障,已清空 {len(victims)} 个模型: {victims}")

    def start_cleanup_task(self) -> None:
        """启动闲置清理协程 + 绑定 _loop + 启动两 executor 健康协程。
        必须在 startup 事件(running loop 内)调用 —— 此处绑定 _loop,避免 executor.__init__
        早于 uvicorn.run 拿到错误 loop(意外退出自愈会静默失效,见 spec 4.2.4)。"""
        loop = asyncio.get_running_loop()
        self.executor_v5._loop = loop
        self.executor_ultralytics._loop = loop
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self.cleanup())
        self._health_tasks = [
            asyncio.create_task(self.executor_v5._health_loop()),
            asyncio.create_task(self.executor_ultralytics._health_loop()),
        ]
        # self.gpu_memory_threshold = 0.85  # GPU内存使用阈值（85%）
        # self.model_memory_requirements = {
        #     "yolov8n": 500,  # MB
        #     "yolov8s": 800,
        #     "yolov8m": 1500,
        #     "yolov8l": 2500,
        #     "yolov8x": 3500,
        # }

    # def get_model_memory_usage(self, model: YOLO) -> int:
    #     """获取模型实际占用的显存（MB）"""
    #     try:
    #         # 确保模型在GPU上
    #         if next(model.parameters()).is_cuda:
    #             # 获取模型参数占用的显存
    #             model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
                
    #             # 获取模型缓冲区占用的显存
    #             buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 / 1024
                
    #             # # 获取当前GPU的显存使用情况
    #             # torch.cuda.synchronize()  # 确保所有CUDA操作完成
    #             # current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # 转换为MB
                
    #             return int(model_memory + buffer_memory)
    #         return 0
    #     except Exception as e:
    #         self.logger.error(f"Error getting model memory usage: {str(e)}")
    #         return 0        

    # def estimate_model_memory(self, model_path: str) -> int:
    #     return 300
    #     # """估算模型所需的GPU内存"""
    #     # for model_type, memory in self.model_memory_requirements.items():
    #     #     if model_type in model_path:
    #     #         return memory
    #     # return 1000  # 默认内存需求（MB）
    
    # def preload_img(self,model):
    #     """预处理图片"""
    #     img = cv2.imread(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../assets/preload.webp")), cv2.IMREAD_COLOR)
    #     model(img)

    # def init_model(self,gpu_id:int|None,weights:str):
    #     device =  'cpu' if gpu_id is None else f'cuda:{gpu_id}'
    #     model = YOLO(weights,task = 'detect')
    #     model.to(device)
    #     return model

    # def preinit_model(self):
    #     try:
    #         gpu_id = get_available_gpu(500)
    #         model = self.init_model(gpu_id,f"/docker-data-yolo/models/ultralytics/yolo11n.pt")

    #         self.preload_img(model)
    #         del model
    #         self.logger.info(f"preinit_model completed")
    #     except Exception as e:
    #         self.logger.error(f"Error preinit model: {str(e)}")
            
    async def load_model(self,config:ModelConfig) -> ModelInfo:
        """加载模型到GPU"""
        #try:
        use_v5 = config.version=="v5"
        if use_v5:            
            model_info =  await self.executor_v5.create_model_async(config)
        else:            
            model_info = await self.executor_ultralytics.create_model_async(config)
            
        self.models[config.model_name] = model_info
        device="cpu" if model_info.gpu_id is None else f"GPU {model_info.gpu_id}"
        self.logger.info(f"{'V5' if use_v5 else 'Ultralytics'} Model {config.model_name} loaded on  {device}")
        return model_info
        
        # self.logger.info("load_model.current.start")
        # required_memory = self.estimate_model_memory(config.model_path)
        # gpu_id = get_available_gpu(required_memory)
        
        # if gpu_id is None:
        #     raise Exception("No GPU with sufficient memory available")
        # config.model_path = format_yolo_modelpath(config.version,config.model_path,'model_path')
        
        # model = self.init_model(gpu_id,weights= config.model_path)
        # if config.preload:
        #     self.preload_img(model)
        # # 获取实际显存占用
        # actual_memory = self.get_model_memory_usage(model)
        # self.logger.info(f"Model {config.model_name} actual GPU memory usage: {actual_memory}MB")
        
        # model_info = self.models[config.model_name] = ModelInfo(
        #     model=model,
        #     gpu_id=gpu_id,
        #     memory_usage=actual_memory,  # 使用实际显存占用
        #     last_used=datetime.now(),
        #     path=config.model_path,
        #     use_v5=use_v5,
        #     v5_executor= None
        # )
        
        # self.logger.info(f"Model {config.model_name} loaded on GPU {gpu_id}")
        # return model_info
        # # except Exception as e:
        # #     logger.error(f"Error loading model: {str(e)}")
        # #     return None


    def get_model(self, model_name: str) -> Optional[ModelInfo]:
        """获取模型实例"""
        if model_name in self.models:
            model_info = self.models[model_name]
            model_info.last_used = datetime.now()
            return model_info
        return None
    
    async def detect_async(self, model_name: str,contents:any, options:Optional[InferenceConfig],decimals : int = 5):
        model_info = self.get_model(model_name)
        if model_info is None:
            raise Exception("Model not found")
        if model_info.use_v5:
            result = await self.executor_v5.detect_async(model_name,contents,options,decimals)
        else:
            result = await self.executor_ultralytics.detect_async(model_name,contents,options,decimals)
        return result
        
        # model=model_info.model
        
        # nparr = np.frombuffer(contents, np.uint8)
        # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   
        
        # results = model(img,
        #                 **(options.dict() if options else {})
        #             )
        # del img
        
        # if results and len(results) > 0:
        #     return results[0].summary(normalize=False, decimals=decimals)
                
        # return []
    
    async def update_model_memory_usage(self, model_name: str, reset: bool = False) -> Optional[ModelInfo]:
        '''更新模型占用显存(allocator 口径);reset=True 重置峰值窗口(供显存测算脚本)。'''
        if model_name in self.models:
            model_info = self.models[model_name]
            if model_info.use_v5:
                m = await self.executor_v5.update_model_memory_usage_async(model_name, reset)
            else:
                m = await self.executor_ultralytics.update_model_memory_usage_async(model_name, reset)
            model_info.last_used = m.last_used
            model_info.gpu_id = m.gpu_id
            model_info.memory_usage = m.memory_usage
            model_info.memory_allocated = m.memory_allocated
            model_info.memory_reserved = m.memory_reserved
            model_info.max_memory_allocated = m.max_memory_allocated
            model_info.last_speed_ms = m.last_speed_ms
            return m
        return None

    async def unload_model(self, model_name: str) -> bool:
        """卸载模型。无论子进程 close 是否成功,主进程 dict 都移除 ——
        释放意图明确,若子进程不可达/已释放,保留条目会造成"列表有但实际没有"的残留。
        返回 True/False 仍反映子进程 close 成败。"""
        try:
            if model_name not in self.models:
                return False
            model_info = self.models[model_name]
            try:
                if model_info.use_v5:
                    await self.executor_v5.close_model_async(model_name)
                else:
                    await self.executor_ultralytics.close_model_async(model_name)
            finally:
                # 无论 close 成败都移除,保持列表=子进程真实存在的模型
                del self.models[model_name]
            return True
        except Exception as e:
            self.logger.error(f"Error unloading model: {str(e)}")
            return False

    async def get_gpu_status(self) -> List[dict]:
        """获取GPU状态信息。

        GPUtil.getGPUs() 调 nvidia-smi(同步阻塞),在 async 路由里直接调用会阻塞主进程事件循环,
        高并发(如并发 detect)下拖慢 /gpu_status 等轻路由。故经 to_thread 移出事件循环。
        """
        try:
            gpus = await asyncio.to_thread(GPUtil.getGPUs)
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

    def get_subprocess_status(self) -> dict:
        """聚合两 executor 的健康态快照(供 /gpu_status 新增 subprocess 字段)。
        按版本粒度(v5/ultralytics 独立),不汇总。向后兼容:仅新增字段。"""
        return {
            "v5": self.executor_v5.status_snapshot(),
            "ultralytics": self.executor_ultralytics.status_snapshot(),
        }
        
    async def cleanup(self):
        while True:
            try:
                current_time = datetime.now()
                
                # 清理长时间未使用的模型
                idle_threshold = 86400  # 24小时
                for model_name, model_info in list(self.models.items()):
                    if (current_time - model_info.last_used).total_seconds() > idle_threshold:
                        await self.unload_model(model_name)
                
                self.logger.info("Cleanup completed")
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {str(e)}")
            await asyncio.sleep(300)  # 每5分钟执行一次清理