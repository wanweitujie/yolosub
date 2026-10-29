# train/train_v8.py
import os
import sys
from pathlib import Path
#添加根目录
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils_v.index import set_yolo_env, _normalize_amp
#添加yolo配置
set_yolo_env(False)


import time
from ultralytics import YOLO


from base_train import STATUS_FILE, write_status, write_error, setup_signal, init_status, format_exception_safe, precheck_memory, stop_watchdog, parse_v8_args

setup_signal()

# 手动解析所有参数(跳过 status_file/log_file;逻辑见 base_train.parse_v8_args。
# code-review 发现 7:非布尔参数后跟 '--' 开头 token 须作值消费,防注入新标志)
args = parse_v8_args(sys.argv)

# 提取必需的参数
data = args.pop('data')
epochs = args.pop('epochs')
model = args.pop('model', 'yolov8n.pt')
amp = _normalize_amp(args.pop('amp', True))
# amp 从 params 透传:历史曾在解析后强制覆盖为 False(硬编码纯 FP32 训练),
# 覆盖用户 amp:true,致 Ada 卡 Tensor Core(FP16)算力浪费——同配置训练慢 2 倍的根因
# (根因见内部分析)。
# 现移除硬编码,默认 True 对齐 Ultralytics 官方(default.yaml `amp: True`,AMP 是官方
# 默认训练行为);显式 amp:false 仍可关闭(排查 FP32 或硬件不支持时)。
# 值经 _normalize_amp 归一为严格 bool(null/字符串 "False"/"" 原样透传会在
# ultralytics torch.tensor(amp) 抛 RuntimeError/TypeError,见 code-review P0)。

pid = os.getpid()
batch_i = 0
batches_per_epoch  = 1
def on_train_epoch_end(trainer):
    epoch=trainer.epoch+1
    total_epochs = trainer.epochs
    write_status(
        status="running",
        pid=pid,
        epoch=epoch,
        total_epochs=total_epochs,
        progress=epoch / total_epochs,
        # start_time=STATUS_FILE.stat().st_mtime
    )
    # 最后一轮训练结束:此后进入 validate/save_model/final_eval 无 status 写入阶段
    # (on_train_end 在其后触发),此时停看门狗,防阈值窗口内误杀健康训练
    if epoch >= total_epochs:
        stop_watchdog()
    
def on_train_epoch_start(trainer):
    global batch_i,batches_per_epoch
    batch_i=0
    batches_per_epoch = len(trainer.train_loader)
def on_train_batch_end(trainer):
    """
    每个 batch 结束后触发
    """
    global batch_i
    batch_i+=1
    epoch_progress_pct = batch_i / batches_per_epoch 

    # 当前整体进度（更细粒度）
    current_epoch = trainer.epoch
    total_epochs = trainer.epochs
    overall_progress_pct = current_epoch  / total_epochs + epoch_progress_pct / total_epochs
    write_status(
        status="running",
        pid=pid,
        epoch=current_epoch+1,
        total_epochs=total_epochs,
        progress=overall_progress_pct,
        # start_time=STATUS_FILE.stat().st_mtime
    )
    
def on_train_end(trainer):
    epoch=trainer.epoch+1
    total_epochs = trainer.epochs
    stop_watchdog()  # 训练正常结束,停止看门狗,防 final_eval/save 阶段误判假死
    write_status(
        status="finished",
        pid=pid,
        epoch=epoch,
        total_epochs=total_epochs,
        progress=epoch / total_epochs,
        # start_time=STATUS_FILE.stat().st_mtime,
        message="YOLO 训练完成",
        output = str(trainer.save_dir.absolute()),
        last = str(trainer.last.absolute()),
        best= str(trainer.best.absolute()),
        # plots=plots,
        # results=results
    )

try:
    # 显存预检(不足 write_status=error + sys.exit;默认 640/16 对齐 ultralytics)
    # device 闭环:用户未传 device 时,precheck 选剩余最多的卡并写回 args 透传给 yolo.train
    _user_device = args.get("device")
    _picked = precheck_memory(model, imgsz=args.get("imgsz", 640), batch=args.get("batch", 16), device=_user_device, amp=amp,
                              total_epochs=epochs)
    # 仅用户未传 device(None)时写回 precheck 选的卡;用 is None 而非 truthiness,
    # 否则 int 0(用户显式指定 GPU 0)被当未指定而覆盖(与 base_train.py 的 device 判断一致)
    if _user_device is None and _picked:
        args["device"] = _picked

    yolo = YOLO(model, verbose=True)
    init_status(pid, time.time(), epochs)
    yolo.add_callback("on_train_epoch_end", on_train_epoch_end)    
    yolo.add_callback("on_train_batch_end",on_train_batch_end)
    yolo.add_callback("on_train_epoch_start",on_train_epoch_start)    
    yolo.add_callback("on_train_end", on_train_end)
    
    yolo.train(
        data=data,
        epochs=epochs,
        amp=amp,
        **args
    )
except Exception as e:
    stop_watchdog()
    write_error(format_exception_safe(e))