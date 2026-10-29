from pathlib import Path
import os
import re
import traceback
import sys

from typing import Optional
import GPUtil


VENV_YOLOV5_PATH="/venv_yolov5"
YOLOV5_PACKAGE_DIR = f"{VENV_YOLOV5_PATH}/mlmodels"
YOLOV5_DIR = f"{YOLOV5_PACKAGE_DIR}/yolov5"
YOLOV5_PYTHON_PATH = f"{VENV_YOLOV5_PATH}/bin/python"
YOLOV5_SITE_PACKAGES = f"{VENV_YOLOV5_PATH}/lib/python3.10/site-packages"

# Ultralytics(v8/v11)子服务解释器:跟随主服务解释器(sys.executable)。
# GPU 容器为 /usr/bin/python(`ln -sf $(which python3)`,系统 3.10,装 ultralytics 8.4.121)。
# 历史:子服务曾用 /venv_yolov5/bin/python(--system-site-packages venv)经系统链接复用主环境包,属隐式耦合;
# 曾硬编码 /usr/bin/python 仅镜像存在该路径,缺路径致子服务 Popen FileNotFoundError(启动崩溃回归),
# 故与 train_v/manager.py 的 v8 训练同款用 sys.executable,消除"路径在不同镜像不一致"。
ULTRALYTICS_PYTHON_PATH = sys.executable


VENV_PATH_EXPORT_MLPACKAGE_IOS14="/venv_export_mlpackage_ios14"
EXPORT_MLPACKAGE_IOS14_PYTHON_PATH = f"{VENV_PATH_EXPORT_MLPACKAGE_IOS14}/bin/python"
EXPORT_MLPACKAGE_IOS14_SITE_PACKAGES = f"{VENV_PATH_EXPORT_MLPACKAGE_IOS14}/lib/python3.10/site-packages"


# 对外日志脱敏的容器内根路径(见 redact_container_paths)。venv 前缀由上方常量派生,
# 新增 venv 自动纳入;数据根无现成常量(该前缀在 gpu_manager/export 等处仍是字面量),
# 此处以字面量列出——不新造 DATA_ROOT:单为脱敏造一个只在此处使用的常量,会造成
# 「常量已存在却非事实源」的半迁移,反而误导后来者。
REDACT_PATH_PREFIXES = ("/docker-data-yolo", VENV_YOLOV5_PATH, VENV_PATH_EXPORT_MLPACKAGE_IOS14)

# 前缀后必须紧跟 / 或空白/行尾才算命中,防 /docker-data-yolo-backup 这类近似名误伤。
_REDACT_RE = re.compile(
    r"/(" + "|".join(re.escape(p.lstrip("/")) for p in REDACT_PATH_PREFIXES) + r")(/|(?=\s|$))"
)


def init_root_folder(initV5:bool=False):
    # 获取当前脚本的绝对路径
    current_file = Path(__file__)
    root_dir = current_file.parent.parent.parent.absolute()
    print(f"Root directory: {root_dir}")
    os.chdir(root_dir)
    src = str(root_dir / "src")
    if src not in sys.path:
        sys.path.insert(0,src)
    if initV5:
        if YOLOV5_SITE_PACKAGES not in sys.path: sys.path.insert(0,YOLOV5_SITE_PACKAGES)
        if YOLOV5_PACKAGE_DIR not in sys.path: sys.path.insert(0,YOLOV5_PACKAGE_DIR)
        if YOLOV5_DIR not in sys.path: sys.path.insert(0,YOLOV5_DIR)        

def compare_versions(version1, version2):    
    from packaging import version
    """
    比较两个版本号
    
    返回:
        -1: version1 < version2
         0: version1 == version2
         1: version1 > version2
    """
    v1 = version.parse(version1)
    v2 = version.parse(version2)
    
    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    else:
        return 0
    
def format_exception(e:Exception):
    return ''.join(traceback.format_exception(type(e), e, e.__traceback__))


def format_exception_safe(e:Exception):
    """返回脱敏的 traceback 用于对外展示:去掉源码片段行,绝对路径替换为文件名,
    避免泄露服务器目录结构与安装路径。完整 traceback 仍由 write_exception 写入本地 exception.log。"""
    msg = format_exception(e)
    msg = re.sub(r'\n    [^\n]+', '', msg)  # 去掉 File 行后的源码片段行(4 空格缩进)
    msg = re.sub(r'File "([^"]+)"', lambda m: f'File "{os.path.basename(m.group(1))}"', msg)  # 绝对路径 -> 文件名
    return msg


# ANSI 转义序列(CSI):实测日志出现 \x1b[1m(加粗)/\x1b[0m(重置)/\x1b[34m(蓝色)/\x1b[K(擦行,
# 无数字参数)。终止字节取 @-~ 全区间,故 [K、[3~ 一类也一并命中。
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    """剥掉 ANSI 转义序列,供面向用户层的文本出口调用。

    ultralytics 用 colorstr() 给日志染色、进度行用 \\x1b[K 擦行,前端直接渲染会看到
    残留乱码。剥掉后为纯文本(代价:丢失框架的彩色分级,如 WARNING 的黄色)。

    不处理 \\r(回车符):它不是转义序列,且在有擦行语义的进度行里删掉会改变行结构。
    """
    return _ANSI_RE.sub("", text)


def redact_container_paths(text: str) -> str:
    """把容器内根路径脱敏为中性绝对路径,供**面向用户层**的日志接口返回前调用。

    规则(前缀取自 REDACT_PATH_PREFIXES):
      - `/前缀/` → `/`(保留后续路径,结果仍是合法绝对路径)
      - 裸前缀(后跟空白或行尾)→ 整体删除(目录名本身不得残留)
    例:
        /docker-data-yolo/runs/train/x/output → /runs/train/x/output
        /venv_yolov5/mlmodels/yolov5/train.py → /mlmodels/yolov5/train.py
    通用 Linux 路径(/usr/lib/...、/usr/local/lib/dist-packages/...)不脱:不暴露容器
    身份,且大面积替换会损伤 traceback 可读性。

    **适用范围:仅日志出口。** 勿用于 `/train/status` 的 output/last/best —— 中间件依赖
    这三个字段定位真实产物,脱敏会直接击穿中间件;也勿下沉到公共响应层。详见
    docs/superpowers/specs/2026-10-15-训练日志接口脱敏-design.md 第一节。
    """
    return _REDACT_RE.sub(r"\2", text)


# 框架 select_device 打印的显卡型号括注,形如:
#   ... torch-2.11.0+cu128 CUDA:0 (Quadro RTX 5000, 15929MiB)
# 由 ultralytics/v5 的 select_device 按 `CUDA:{d} ({p.name}, {MiB}MiB)` 拼出(v5 见
# mlmodels/yolov5/utils/torch_utils.py:129),型号属硬件资产信息,不在日志接口对外暴露。
# 只匹配紧跟 CUDA:n 的、以 `NNNNMiB` 结尾的括注,避免误伤日志里其它括号内容。
_GPU_MODEL_RE = re.compile(r"(CUDA:\d+)[ \t]*\([^()\n]*?\d+[ \t]*MiB\)")


def redact_gpu_model(text: str) -> str:
    """删掉 `CUDA:n (显卡型号, NNNNNMiB)` 里的括注,只保留 `CUDA:n`,供**面向用户层**的
    日志接口返回前调用。

    例:
        CUDA:0 (Quadro RTX 5000, 15929MiB) → CUDA:0
        CUDA:1 (NVIDIA A30, 24576MiB)      → CUDA:1

    多卡时框架每张卡各打一行 `CUDA:{d} (...)`(前导空格为对齐),逐行命中、全部替换。
    只脱型号与显存总量,不动 `CUDA:n` 本身——设备序号对排查仍有意义。

    **与 redact_container_paths 同属日志出口专用**:磁盘上的 train.log 保持原样,
    仅读时替换;勿下沉到公共响应层。
    """
    return _GPU_MODEL_RE.sub(r"\1", text)


def set_yolo_env(is_v5:bool,low_ultralytics:bool=False):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if is_v5:        
        os.environ["YOLOv5_OFFLINE"]="true"
        os.environ['YOLOv5_AUTOINSTALL'] = 'false'        
        os.environ["YOLOV5_CONFIG_DIR"] = "/root/.config/yolov5/"        
    else:
        os.environ["YOLO_OFFLINE"]="true"
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        if low_ultralytics:            
            os.environ["YOLO_CONFIG_DIR"] = "/root/.config/Ultralytics_low/"

def format_yolo_modelpath(yolo_ver:str,weights:str,ptname:str):
    model_basepath:str
    if yolo_ver=="v5" :
        model_basepath="v5"
    else:
        model_basepath="ultralytics"
    
    #检验model是否存在
    if not weights or not isinstance(weights, str):
        raise RuntimeError(f"请传入{ptname}")
    elif weights.lower().startswith(("http://","https://")):
        raise RuntimeError(f"不支持http协议的{ptname}")
    
    model:str
    if '/' in weights or '\\' in weights:
        model = weights
    else:
        model = f'/docker-data-yolo/models/{model_basepath}/{weights}'
    path=Path(model).resolve()
    if not path.is_file():
        raise RuntimeError(f"传入的{ptname}文件不存在")
    return str(path)


def _gpu_free_memory_mb(gpu_id: int) -> Optional[int]:
    """返回指定 id GPU 的剩余显存(MB)。找不到或异常 -> None。

    GPU 显存查询的唯一底层实现,get_available_gpu(选卡)与训练预检(校验指定卡)均复用,
    避免两处 GPUtil 遍历分叉(阈值/异常处理语义漂移)。"""
    try:
        for g in GPUtil.getGPUs():
            if g.id == gpu_id:
                return g.memoryFree
    except Exception:
        pass
    return None


def get_available_gpu(required_memory: int) -> Optional[int]:
    """查找剩余显存最多且 > required_memory 的 GPU。无 GPU 或都不够 -> None。"""
    try:
        gpus = GPUtil.getGPUs()
        # 过滤出满足内存要求的GPU
        available_gpus = [gpu for gpu in gpus if gpu.memoryFree > required_memory]

        if available_gpus:
            # 找出剩余内存最多的GPU
            best_gpu = max(available_gpus, key=lambda gpu: gpu.memoryFree)
            return best_gpu.id
    except Exception:
        pass
       # self.logger.error(f"Error getting GPU info: {str(e)}")
    return None


def check_gpu_memory(gpu_id: int, required_memory: int) -> bool:
    """校验指定 id GPU 剩余显存是否 > required_memory(MB)。找不到或异常 -> False。

    复用 _gpu_free_memory_mb,与 get_available_gpu 共享同一查询语义(阈值/异常处理一致),
    避免预检与选卡逻辑分叉。"""
    free = _gpu_free_memory_mb(gpu_id)
    return free is not None and free > required_memory


def _normalize_amp(value) -> bool:
    """把客户端可能传的各种 amp 值归一为严格 bool。

    背景(code-review P0):train_v8 amp 透传后,客户端可能传 json null、字符串
    "False"/""、数字等,原样透给 yolo.train(amp=...) 会在 ultralytics
    torch.tensor(amp) 抛 RuntimeError/TypeError(改动前 amp 被硬编码覆盖不可达)。
    "true"/"1"/"yes"/"on"→True,"false"/"0"/""/null→False,其余非法值回退 False
    (与"不传 amp 默认 False"的历史行为一致)。"""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


def _estimate_from_weights(weights_bytes: int, imgsz: int, batch: int, amp: bool = False) -> int:
    """从权重字节 + imgsz/batch 估算训练显存预算(MB)。

    公式(审计修正:激活项独立,不乘权重):
      budget = CONTEXT(200) + weights_mb × 3.5[梯度1+Adam2+master0.5]
             + ACT_BASE(71) × (imgsz/320)² × (batch/4)
    ACT_BASE=71 由 yolo11n@imgsz320batch4 实测 304MB 反解(304 = 200 + 33 + 71)。
    激活随 imgsz²×batch×scale 缩放,scale=max(1,weights_mb/W_REF)按模型规模放大。
    amp=True(FP16 autocast):激活显存约减半,权重/梯度/优化器主精度仍 FP32 不变——
    选卡/预检不按 FP32 虚高一倍拒掉本可装下的 amp 训练(见 code-review P1)。
    无 torch 依赖,可被主进程(manager 选卡)与训练子进程(precheck)共用。
    口径注:调用方若传文件字节数(os.path.getsize)约比 state_dict 纯权重大 3-8%(含元数据)。"""
    CONTEXT_MB = 200
    WEIGHTS_COEF = 3.5
    ACT_BASE = 71
    W_REF_MB = 5.0
    MARGIN = 1.2
    weights_mb = weights_bytes / (1024 * 1024)
    # int() 防御:train_v8 手动解析参数时 "-1" 因负号未转 int,直接 batch/4 会 TypeError;
    # imgsz 防御:ultralytics 合法 list 型 imgsz=[640,480] 时 int(list) 抛 TypeError,取 max 估算
    try:
        imgsz = int(imgsz)
    except (TypeError, ValueError):
        if isinstance(imgsz, (list, tuple)) and imgsz:
            imgsz = max(int(x) for x in imgsz if isinstance(x, (int, float)))
        else:
            imgsz = 640
    try:
        batch = int(batch)
    except (TypeError, ValueError):
        batch = 16
    # scale:模型规模因子。激活随特征图通道(∝width)×层数(∝depth)增长,与权重强相关
    scale = max(1.0, weights_mb / W_REF_MB)
    activation = ACT_BASE * (imgsz / 320) ** 2 * (batch / 4) * scale
    if amp:
        activation *= 0.5   # AMP(FP16 autocast):激活显存约减半
    return int((CONTEXT_MB + weights_mb * WEIGHTS_COEF + activation) * MARGIN)

