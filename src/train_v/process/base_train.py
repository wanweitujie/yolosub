# train/base_train.py
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
import argparse
from utils_v.index import format_exception,format_exception_safe,get_available_gpu,check_gpu_memory,_estimate_from_weights

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
STATUS_FILE:Path
LOG_FILE:Path

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--status_file', type=str, default= ROOT / "status.json", help='status_file')
    parser.add_argument('--log_file', type=str, default=ROOT  / "train.log", help='log_file')
    parser.add_argument('--device', type=str, default=None, help='训练设备(manager 分配后传入)')

    return parser.parse_known_args()[0]

opt = parse_opt()
STATUS_FILE = Path(opt.status_file)
LOG_FILE = Path(opt.log_file)
DEVICE = opt.device
# LOGTEST_FILE = Path(opt.log_file + ".test.log", help='log_test_file')


# ===== v8 argv 解析(重构自 train_v8.py 手写循环,提为可单测函数)=====
# code-review 发现 7:原循环对"非布尔参数后跟 '--' 开头 token"处理错误——
# `--amp --x` 把 amp 值吞为 True + 把 --x 再注入为新标志(yolo.train(x=True) 抛 TypeError)。
# 修复:显式布尔白名单(当前仅 exist-ok),非布尔参数一律消费一个值(即使以 '--' 开头),杜绝注入。
_V8_BOOL_FLAGS = {"exist-ok"}


def parse_v8_args(argv: list) -> dict:
    """解析 v8 训练 argv 为参数字典(跳过 status_file/log_file 内部参数)。

    值与 train_v8 原有手写循环同语义:字符串先尝试 json.loads(bool/数字/字符串),
    失败再做 int/float 兜底(负数 "-1" 因负号保持字符串,由 precheck int() 兜底)。"""
    args = {}
    i = 1  # 跳过脚本名
    while i < len(argv):
        arg = argv[i]
        if arg.startswith('--'):
            key = arg[2:]
            if key in _V8_BOOL_FLAGS:
                args[key.replace('-', '_')] = True
                i += 1
                continue
            if i + 1 >= len(argv):
                value = True  # 非布尔参数末尾缺值(越界用法):兜底裸标志,不崩
                i += 1
            else:
                value = argv[i + 1]
                i += 2
            # 跳过 status_file 和 log_file 参数
            if key not in ('status_file', 'log_file'):
                # 将连字符参数名转换为下划线格式(YOLOv8 要求)
                key = key.replace('-', '_')
                # 尝试解析 JSON 值
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except Exception:
                        # 如果是数字字符串,转换为数字
                        if value.isdigit():
                            value = int(value)
                        elif value.replace('.', '').isdigit():
                            value = float(value)
                args[key] = value
        else:
            i += 1  # 孤立非 -- token(越界用法)跳过
    return args

# def write_log_test(**kwargs):
#     LOGTEST_FILE.write_text(f"{json.dumps(kwargs, ensure_ascii=False, indent=2)}")

def write_status(**kwargs):
    STATUS_FILE.write_text(json.dumps(kwargs, ensure_ascii=False, indent=2),"utf-8")


def resolve_output_dir(status_file, name: str) -> str:
    """训练产物目录的绝对路径。

    status.json 与产物目录同级(都在 exp_dir 下),故取 status_file 的**父目录**拼 name。
    原 train_v5 写成 `STATUS_FILE / name`,少了一层 parent,得到
    `{exp_dir}/status.json/output` 这种不存在的路径(2026-10-15 实测 status.json 确认);
    v8 侧用 trainer.save_dir 本就正确,两版 output 字段此前不一致。
    该字段与 last/best 并列,是中间件定位产物的契约,路径不存在会击穿下游。

    name 由 manager 固定传 "output"(--name output),与 v5 save_dir=project/name 约定一致。"""
    return str((Path(status_file).parent / name).absolute())


# error 态要与 running/finished 对齐的进度字段(客户端无需判分支取 progress)
_PROGRESS_FIELDS = ("pid", "epoch", "total_epochs", "progress")


def write_error(message: str, **fields):
    """写 status=error,字段集恒为 status/message/pid/epoch/total_epochs/progress。

    背景(2026-10-15 核实):原先 6 个 error 写入点里 4 个(显存预检三处 + train_v5/v8
    顶层 except)只写 `{status, message}`,另 2 个(看门狗)带进度字段——同一个任务换条
    失败路径,响应字段数就不同,客户端无条件读 data["progress"] 会在预检失败时 KeyError。
    统一走本函数。

    取值优先级:调用方显式传入(**fields) > status.json 里最后一次的进度 > 零值兜底。
    零值兜底覆盖「训练尚未开始就失败」(显存预检)场景。"""
    data = {"status": "error", "message": message, "pid": os.getpid(),
            "epoch": 0, "total_epochs": 0, "progress": 0.0}
    try:
        cur = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        for k in _PROGRESS_FIELDS:
            if k in cur:
                data[k] = cur[k]
    except Exception:
        pass          # 无/坏 status.json(含非对象 JSON)按零值兜底,不阻断报错写入
    data.update(fields)
    write_status(**data)


def setup_signal():
    # 注册时捕获训练主进程 pid:fork 出的 dataloader worker 继承本 handler 与该值,
    # 但其 os.getpid() 不同 —— 据此区分"主进程收到外部停止信号"与"worker 被 torch 正常清理"。
    # 曾发生 v5 训练假 stopped 事故(2026-09),根因与处置见内部档案。
    _main_pid = os.getpid()

    def handler(sig, frame):
        if os.getpid() != _main_pid:
            # fork 出的 dataloader worker(v5 autobatch 等场景 torch terminate 清理时触发):
            # 绝不写共享 status.json —— 否则会把主进程的 running 污染成假 stopped(事故根因,
            # 主进程从未收到信号、继续训练到 finished,上层却被假 stopped 误导)。
            # 硬退出保持 worker 可被 terminate 杀死,不走 Python 清理(可能卡 CUDA)。
            os._exit(0)
        try:
            cur = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
        # 仅当未进入终态才写 stopped:晚到 SIGTERM(如 stop 与训练收尾竞态)不得覆盖已写好的
        # finished/error/stopped,否则上层轮询会抓到"先 stopped 后 finished"的错位中间态。
        # 与 manager._write_stopped_if_running(running 态才写 stopped)对齐。
        if cur.get("status") not in ("finished", "error", "stopped"):
            write_status(status="stopped", message="收到中止信号")
        sys.exit(0)
    signal.signal(signal.SIGTERM, handler)

def _device_gpu_id():
    """从 --device 解析 gpu_id(排查参考用,非权威);cpu/未传/多卡 -> 首卡或 None。"""
    try:
        d = str(DEVICE or "").strip()
        if not d or d.lower() == "cpu":
            return None
        return int(d.split(",")[0])
    except Exception:
        return None

def init_status(pid, start_time, total_epochs=0):
    write_status(
        status="running",
        pid=pid,
        epoch=0,
        total_epochs=total_epochs,
        progress=0.0,
        gpu_id=_device_gpu_id(),   # 排查参考;后续回调整文件覆盖会冲掉,响应以 manager 记录补挂为准
        # start_time=start_time
    )
    start_watchdog()


# === 训练假死/致命错误看门狗 ===
# 背景:dataloader worker 的 queue feeder 线程在 /dev/shm 不足等场景会静默崩溃
# (异常仅打印到 stderr,不传播到主进程),主进程在取首个 batch 时永久挂起,
# yolo.train() 既不返回也不抛异常 -> 外层 try/except 捕获不到,status 永久卡 running。
# 本看门狗在训练子进程内旁路检测:命中已知致命日志关键字(且 status 已一周期无更新)
# 或 status.json 长时间无进度更新 -> 写 status=error 并终止进程组,把静默失败经 status 通道传导出去。
WATCHDOG_INTERVAL = 10           # 看门狗轮询间隔(秒)
# 假死阈值 30 分钟:须大于"每轮 on_train_epoch_end -> validate -> save_model -> 下一轮首个 batch"的无写窗口
# (大模型+大 val 集+慢盘时该窗口可达数分钟;若用 600s 会在正常校验/保存阶段误杀健康训练)。
# shm 挂起等已知致命错误由 FATAL_LOG_PATTERNS 路径在 ~20s 内捕获,不依赖本阈值。
STALE_THRESHOLD_SECONDS = 1800   # status.json 超过该时长无更新判为假死(30 分钟)
FATAL_LOG_PATTERNS = ("unable to allocate shared memory",)  # dataloader /dev/shm 不足等已知致命错误

_watchdog_done = threading.Event()  # 训练正常结束(含 except)时置位,停止看门狗
_log_tail_offset = 0                # 增量 tail train.log 的读取偏移

def stop_watchdog():
    """训练正常结束(on_train_end/except)后调用,防止看门狗误覆盖 finished/error 终态。"""
    _watchdog_done.set()


def _reset_watchdog_state():
    """测试隔离用:复位模块级看门狗状态(事件 + 日志 tail 偏移)。生产进程每次全新,无需调用。"""
    global _log_tail_offset
    _log_tail_offset = 0
    _watchdog_done.clear()


def _kill_group_and_exit():
    """终止自身进程组(含 dataloader worker 子进程)后退出,立即释放 CUDA context。

    os._exit 只杀主进程,worker 需等 parent_sentinel 轮询才退出,期间残留 CUDA context/显存;
    训练进程由 manager 以 start_new_session=True 拉起,是进程组组长,同组仅含自身+workers,安全。
    Windows 无 getpgid/killpg -> 兜底 os._exit。"""
    try:
        sigkill = getattr(signal, "SIGKILL", 9)  # Windows 无 SIGKILL,兜底 9
        os.killpg(os.getpgid(0), sigkill)
    except Exception:
        pass
    os._exit(1)


def _write_error_and_exit(message: str):
    """写 status=error(字段集见 write_error)后终止整个进程组。os._exit 绕过挂起的主线程。"""
    try:
        write_error(message)
    except Exception:
        pass
    _kill_group_and_exit()


def _check_fatal_patterns() -> bool:
    """增量 tail LOG_FILE 是否命中已知致命错误关键字(如 dataloader shm 分配失败)。

    只读上次偏移之后的新增内容,避免对长日志每次全量 read_text 的 O(文件大小) 重复 I/O。"""
    global _log_tail_offset
    try:
        if LOG_FILE is None or not LOG_FILE.exists():
            return False
        size = LOG_FILE.stat().st_size
        if size < _log_tail_offset:
            _log_tail_offset = 0  # 文件被截断/重建,从头读
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            f.seek(_log_tail_offset)
            chunk = f.read()
            _log_tail_offset = f.tell()
        return any(p in chunk for p in FATAL_LOG_PATTERNS)
    except Exception:
        return False


def _status_stale(now: float, stale_after: float) -> bool:
    """status.json 距上次写入超过 stale_after 秒 -> 训练无进度,判定假死。"""
    try:
        return (now - STATUS_FILE.stat().st_mtime) > stale_after
    except OSError:
        return False


def watchdog_loop():
    while not _watchdog_done.is_set():
        # 分批等待,及时响应 done 信号
        _watchdog_done.wait(WATCHDOG_INTERVAL)
        if _watchdog_done.is_set():
            break
        now = time.time()
        try:
            # 1) 已知致命错误(如 dataloader shm 分配失败)精确报错。
            #    进度门控:仅当命中且 status 已一个周期无更新才杀 —— shm 挂起时 status 必然停止刷新,
            #    (主进程阻塞在首个 batch,on_train_batch_end 不再触发);健康训练每批写 status,不会误杀。
            if _check_fatal_patterns() and _status_stale(now, WATCHDOG_INTERVAL):
                _write_error_and_exit(
                    "训练失败:检测到 dataloader 共享内存(shm)分配失败,通常因容器 /dev/shm 过小;"
                    "请在 docker run 增加 --shm-size=8g 后重试")
                return
            # 2) 通用假死兜底:status.json 长时间未更新
            if _status_stale(now, STALE_THRESHOLD_SECONDS):
                _write_error_and_exit(
                    f"训练失败:超过 {STALE_THRESHOLD_SECONDS} 秒无进度更新,判定训练进程假死"
                    "(可能 dataloader worker 静默失败/挂起)")
                return
        except Exception:
            continue


def start_watchdog():
    t = threading.Thread(target=watchdog_loop, name="train-watchdog", daemon=True)
    t.start()


# 与 STATUS_FILE 同目录,避免依赖 cwd 导致日志写到不可预测的位置
ERROR_FILE = STATUS_FILE.parent / "exception.log"
def write_exception(e:Exception):
    ERROR_FILE.write_text(format_exception(e),encoding="utf-8")


def _check_specific_gpu(gpu_id: int, required_memory: int) -> bool:
    """校验指定 id 的 GPU 剩余显存是否 > required_memory(MB)。找不到或异常 -> False。

    复用 utils_v.index.check_gpu_memory,与 get_available_gpu 共享同一 GPUtil 查询语义,
    避免阈值/异常处理分叉(原内联 GPUtil 遍历与 get_available_gpu 重复)。"""
    return check_gpu_memory(gpu_id, required_memory)


def precheck_memory(model_path: str, imgsz: int = 640, batch: int = 16, device=None, amp: bool = False,
                    total_epochs: int = 0):
    """训练前显存预检:读 pt 权重 + 估算预算 + 校验 GPU。
    不足 → write_error(...) + sys.exit(1);够 → 正常返回。

    total_epochs 仅用于失败时写进 status.json(此时训练尚未开始,无历史进度可合并),
    使 error 响应字段集与 running/finished 一致。默认 0 兼容既有调用与测试。

    amp=True 时 _estimate_from_weights 按 FP16 激活减半估预算,防 FP32 虚高误拒。
    默认 imgsz=640/batch=16 对齐 ultralytics 默认。
    torch 懒加载:仅在此函数内 import,避免 base_train 导入即依赖 torch。
    SystemExit 不被 except Exception 捕获:precheck 自己 write_status 落状态。

    v5/v8 各自环境天然就位(v5 经 init_root_folder(True) + --system-site-packages)。"""
    import torch  # 懒加载

    # autobatch(batch<=0):跳过预检,ultralytics autobatch 自探测显存选 batch,不会 OOM
    try:
        batch_i = int(batch)
    except (TypeError, ValueError):
        batch_i = 0
    if batch_i <= 0:
        return None
    # device=cpu:跳过 GPU 校验,原样返回让训练走 CPU
    if isinstance(device, str) and device.lower() == "cpu":
        return device

    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model = ckpt.get("model", ckpt)
        sd = model.state_dict() if hasattr(model, "state_dict") else model
        wb = sum(v.numel() * v.element_size() for v in sd.values())
    except Exception as e:
        write_error(f"显存预检:读取模型权重失败 {format_exception_safe(e)}", total_epochs=total_epochs)
        sys.exit(1)
        return

    budget = _estimate_from_weights(wb, imgsz, batch, amp=amp)
    # device 指定判断用 is not None + 非空串,而非 truthiness:int 0(指定 GPU 0)是合法指定,
    # 原 `if device:` 对 0 为 falsy 会跳过校验并改路由到别的卡(用户显式 GPU 0 被静默覆盖)。
    if device is not None and str(device).strip() != "":
        # 用户指定 GPU:校验主卡(逗号分隔的第一个 id)剩余是否够
        primary = str(device).split(",")[0].strip()
        try:
            primary_id = int(primary)
        except ValueError:
            # 非数字且非 cpu(上面已处理)-> 无法校验,放行原样 device
            return device
        if _check_specific_gpu(primary_id, budget):
            return device
        write_error(f"训练显存预算 {budget} MB 超过指定 GPU {primary_id} 剩余,请调小 imgsz/batch 或释放已加载模型",
                    total_epochs=total_epochs)
        sys.exit(1)
        return

    gpu_id = get_available_gpu(budget)
    if gpu_id is None:
        write_error(f"训练显存预算 {budget} MB 超过可用 GPU 剩余,请调小 imgsz/batch 或释放已加载模型",
                    total_epochs=total_epochs)
        sys.exit(1)
    return str(gpu_id)