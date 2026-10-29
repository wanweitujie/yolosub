from typing import  Optional,Union,Tuple,List,Any
from pydantic import BaseModel, field_validator,Field
import json
from fastapi.responses import JSONResponse
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

class CustomJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        if isinstance(content,str):
            return content.encode('utf-8')
        return json.dumps(content,separators=(",", ":"),ensure_ascii=False).encode('utf-8')
    
def toStandardResponseData(content: Any = None,errMsg:Union[str,None] = None) :
    """如果errMsg不为None则success=False"""
    success = errMsg is None
    return {
        "success" : success,
        "data" : content,
        "errMsg" : errMsg,        
    }

@dataclass
class ModelInfo:
    model: any
    gpu_id: Optional[int]
    memory_usage: int  # 预估的GPU内存使用量（MB）
    last_used: datetime
    path: str = None
    use_v5: bool = False
    v5_executor: ProcessPoolExecutor = None
    memory_allocated: Optional[int] = None     # 模型张量真实占用(MB,torch.cuda.memory_allocated)
    memory_reserved: Optional[int] = None      # 缓存分配器保留(MB,torch.cuda.memory_reserved)
    max_memory_allocated: Optional[int] = None # 峰值(MB,torch.cuda.max_memory_allocated,reset 窗口)
    last_speed_ms: Optional[dict] = None       # 最近一次推理耗时(ms):{"preprocess","inference","postprocess"},v5 第三段为 NMS
    def __post_init__(self):
        # 跨进程(httpx JSON)返回时,last_used(datetime)会被 FastAPI 序列化为 ISO 字符串,
        # dataclass 不做运行时类型校验,主进程若直接使用会触发 cleanup 中 datetime - str 报错。
        # 此处统一把字符串解析回 datetime;子进程内 last_used 本就是 datetime,不受影响。
        if isinstance(self.last_used, str):
            self.last_used = datetime.fromisoformat(self.last_used)
    def clone_to_main(self):
        return ModelInfo(None, self.gpu_id, self.memory_usage, self.last_used, self.path,
                         self.use_v5, None, self.memory_allocated, self.memory_reserved,
                         self.max_memory_allocated, self.last_speed_ms)

class InferenceConfig(BaseModel):
    conf: Optional[float] = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="置信度阈值 (0-1)"
    )
    """推理参数配置"""
    iou:Optional[float] =  Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="NMS 的 IoU 阈值 (0-1)"
    )
    imgsz: Union[int, Tuple[int, int], List[int]] = 640    
    @field_validator('imgsz')
    def validate_imgsz(cls, v):
        if isinstance(v, (int, float)):
            # 如果是单个数字，转换为元组
            return (int(v), int(v))
        elif isinstance(v, (list, tuple)):
            # 如果是列表或元组，确保长度为2
            if len(v) != 2:
                raise ValueError('imgsz必须是长度为2的列表或元组')
            return (int(v[0]), int(v[1]))
        raise ValueError('imgsz格式不正确')
    
    # 硬件和性能参数
    half: Optional[bool] = Field(
        False,
        description="是否使用FP16推理"
    )
    max_det: Optional[int] = Field(
        300,
        ge=1,
        description="每幅图像允许的最大检测次数。限制模型在单次推理中可以检测到的物体总数，防止在密集场景中产生过多的输出。"
    )    
    # vid_stride: Optional[int] = Field(
    #     1,
    #     ge=1,
    #     description="视频帧提取步长"
    # )
    # visualize: Optional[bool] = Field(
    #     False,
    #     description="在推理过程中激活模型特征的可视化，从而深入了解模型 '看到'了什么。这对调试和模型解释非常有用。"
    # )
    augment: Optional[bool] = Field(
        False,
        description="可对预测进行测试时间增强（TTA），从而在牺牲推理速度的情况下提高检测的鲁棒性。"
    )
    agnostic_nms: Optional[bool] = Field(
        False,
        description="启用与类别无关的非最大抑制 (NMS)，可合并不同类别的重叠方框。这在多类检测场景中非常有用，因为在这种场景中，类的重叠很常见。"
    )
    
    classes: Optional[List[int]] = Field(
        None,
        description="根据一组类别 ID 过滤预测结果。只有属于指定类别的检测结果才会返回。这对于在多类检测任务中集中检测相关对象非常有用。 (可选 list[int])"
    )
    @field_validator('classes')
    def validate_classes(cls, v):
        if not isinstance(v, list):
            raise ValueError('classes 必须是一个整数数组')
        # 检查列表中的每个元素是否为整数
        if not all(isinstance(item, int) for item in v):
            raise ValueError('classes 数组中的每个元素必须是整数')
        return v
    
    retina_masks: Optional[bool] = Field(
        False,
        description="返回高分辨率分割掩码。返回的掩码 (masks.data) 如果启用，将与原始图像大小相匹配。如果禁用，它们将与推理过程中使用的图像大小一致。"
    )
    embed: Optional[List[int]] = Field(
        None,
        description="指定从中提取特征向量或嵌入的层。这对聚类或相似性搜索等下游任务非常有用。"
    )
    # 可视化参数
    
    # # 视频/流相关参数
    # stream: Optional[bool] = Field(
    #     None,
    #     description="流模式"
    # )


class ModelConfig(BaseModel):
    model_path: str
    """pt/onnx模型路径"""
    model_name: str
    """创建的模型名称"""
    preload:Optional[bool] = False
    """是否预检测一张图片来保证后续首次检测速度,这会增加create_model的时间"""
    """任务类型: 'detect', 'segment', 'classify', 'pose', None"""
    task:Optional[str] = None
    version:Optional[str] = None
    '''模型版本 v5 v8 v11'''
    v5_fp16:Optional[bool] = False
    '''对于v5模型必填,是否开启fp16精度,非v5无需该参数'''
    v5_data:Optional[str] = None
    '''对于v5模型必填,传入定义类别文件路径 如data/mydata.yaml'''


class ModelExportConfig(BaseModel):
    version:Optional[str] = Field(
        None,
        description="yolo的版本 v5 v8 v11"
    )
    @field_validator('version')
    def validate_version(cls, v):
        # 规范化大小写并校验白名单, 避免调用方漏传/误传 (如 None/'V8'/'8') 在 ncnn-standard 转换时
        # 才报错甚至静默失败。None 允许 (其他 format 可由 ultralytics 自动推断版本)。
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError('version必须是一个字符串')
        v = v.lower()
        if v in ('v5', 'v8', 'v11'):
            return v
        raise ValueError(f"不受支持的version: {v}, 仅支持 v5/v8/v11")
    model_path: str = Field(
        None,
        description="要转换的模型路径"
    )
    format: str = Field(
        None,
        description="默认值 'torchscript' ，导出模型的目标格式，例如 'onnx', 'torchscript', ''）或其他格式。每种格式都能兼容不同的 部署环境."
    ) 
    @field_validator('format')
    def validate_format(cls, v):
        if v is None or not isinstance(v,str):
            raise ValueError('format格式不正确')
        v = v.lower()
        if v in ["onnx","ncnn","ncnn-standard","torchscript","mlmodel","coreml","mlpackage","mlpackage-ios14"]:
            # return "coreml" if v=="mlpackage" else v
            return v
        raise ValueError(f"不受支持的format: {v}")

    imgsz: Optional[Union[int, Tuple[int, int], List[int]]] = Field(
        None,
        description="默认值 640，模型输入所需的图像尺寸。对于正方形图像，可以是整数（例如、 640 为 640×640）或一个元组 (height, width) "
    )    
    @field_validator('imgsz')
    def validate_imgsz(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            # 如果是单个数字，转换为元组
            return (int(v), int(v))
        elif isinstance(v, (list, tuple)):
            # 如果是列表或元组，确保长度为2
            if len(v) != 2:
                raise ValueError('imgsz必须是长度为2的列表或元组')
            return (int(v[0]), int(v[1]))
        raise ValueError('imgsz格式不正确')
    # keras: Optional[bool] = Field(
    #     None,
    #     description="默认值 false ，可导出为 Keras 格式 TensorFlowSavedModel的 Keras 格式，提供与TensorFlow serving 和 API 的兼容性。"
    # )
    optimize: Optional[bool] = Field(
        None,
        description="默认值 false ，在导出到TorchScript 时针对移动设备进行优化，从而缩小模型大小并提高推理性能。与NCNN 格式或CUDA 设备不兼容。"
    )
    half: Optional[bool] = Field(
        None,
        description="默认值 false ，启用 FP16（半精度）量化，在支持的硬件上减小模型大小并可能加快推理速度。与 INT8 量化或ONNX 的CPU 输出不兼容。"
    )
    int8: Optional[bool] = Field(
        None,
        description="默认值 false ，激活 INT8 量化，进一步压缩模型并加快推理速度，同时将精度损失降到最低，主要用于边缘设备。与TensorRT 一起使用时，会执行训练后量化 (PTQ)。"
    )
    dynamic: Optional[bool] = Field(
        None,
        description="默认值 false ，允许ONNX、TensorRT 和OpenVINO 输出动态输入尺寸，提高了处理不同图像尺寸的灵活性。自动设置为 True 使用 INT8 的TensorRT 时。"
    )
    simplify: Optional[bool] = Field(
        None,
        description="默认值 true ，简化了ONNX 输出的模型图。 onnxslim这可能会提高推理引擎的性能和兼容性。"
    )
    opset: Optional[int] = Field(
        None,
        description="默认值 None ，指定ONNX 操作集版本，以便与不同的 ONNX解析器和运行时的兼容性。如果未设置，则使用最新支持的版本。"
    )
    # workspace: Optional[float] = Field(
    #     None,
    #     description="默认值 None ，设置工作区的最大大小（单位：GiB TensorRT 优化，平衡内存使用和性能。使用方法 None TensorRT 进行自动分配，最高可达设备最大值。"
    # )
    nms: Optional[bool] = Field(
        None,
        description="默认值 false ，如果支持（请参阅导出格式），则在导出模型中添加非最大值抑制 (NMS)，以提高检测后处理效率。不适用于端对端模型。"
    )
    batch: Optional[int] = Field(
        1,
        description="默认值 1 ，指定导出模型的批量推理大小，或导出模型同时处理的最大图像数。 predict 模式。对于边缘TPU 输出，该值自动设为 1。"
    )
    data: Optional[str] = Field(
        None,
        description="通往 数据集 配置文件（默认： coco8.yaml)，这对 INT8 量化校准至关重要。如果在启用 INT8 时没有指定，将分配一个默认数据集。"
    )
    fraction: Optional[float] = Field(
        None,
        description="默认值 1.0 ，指定用于 INT8 量化校准的数据集分数。允许在完整数据集的子集上进行校准，这对实验或资源有限的情况非常有用。如果在启用 INT8 时没有指定，则将使用整个数据集。"
    )			