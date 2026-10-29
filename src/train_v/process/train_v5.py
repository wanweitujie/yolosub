# train/train_v5.py
import sys
import os
import time
from pathlib import Path
# 添加 yolov5 到路径
sys.path.insert(0,str(Path(__file__).parent.parent.parent))

from utils_v.index import set_yolo_env,init_root_folder
#添加yolo配置
set_yolo_env(True)
init_root_folder(True)

from base_train import write_status, setup_signal, init_status,STATUS_FILE,write_exception,format_exception_safe,precheck_memory,stop_watchdog,resolve_output_dir,write_error

setup_signal()
current_epoch:int = 0
epochs : int
pid = os.getpid()
output :str


# 导入 YOLOv5 train 函数
from yolov5.train import main as yolov5_train,parse_opt
from yolov5.utils.callbacks import Callbacks

# 兜底:PyTorch 2.6+ 将 torch.load 默认 weights_only 改为 True,而 v5 检查点含 models.yolo.Model 等
# 自定义类会被安全反序列化拦截(上游 attempt_load 早已补 weights_only=False,训练路径的裸 torch.load 未补)。
# 本补丁仅作用于本训练子进程,统一兜底为 weights_only=False(受信自身训练检查点),
# 一次性覆盖 初始加载/断点续训/strip_optimizer/wandb,避免侵入性修改 yolov5 子模块源码。
import torch
_original_torch_load = torch.load
def _load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _load_weights_only_false

def on_train_epoch_end(epoch,progress = None):
    # epoch = trainer.epoch
    global current_epoch
    if epoch is None:
        epoch=current_epoch
    else:
        current_epoch=epoch
    write_status(
        status="running",
        pid=pid,
        epoch=epoch+1,
        total_epochs=epochs,
        progress=(epoch + 1) / epochs if progress is None else progress,
        # start_time=STATUS_FILE.stat().st_mtime
    )
    # 最后一轮训练结束:此后进入 strip_optimizer/val 等无 status 写入阶段(on_train_end 在其后触发),
    # 此时停看门狗,防阈值窗口内误杀健康训练。epoch 为 None 时是 batch 级进度回调,跳过判断。
    if epoch is not None and epoch + 1 >= epochs:
        stop_watchdog()
def on_train_batch_end(ni, model, imgs, targets, paths, plots, nis):
    on_train_epoch_end(None,ni/nis)

def on_train_end(last, best, plots, epoch, results):
    global current_epoch
    current_epoch=epoch
    stop_watchdog()  # 训练正常结束,停止看门狗,防收尾阶段误判假死
    write_status(
        status="finished",
        pid=pid,
        epoch=epoch+1,
        total_epochs=epochs,
        progress=(epoch + 1) / epochs,
        # start_time=STATUS_FILE.stat().st_mtime,
        message="YOLOv5 训练完成",
        output = output,
        last = str(last.absolute()),
        best= str(best.absolute()),
        # plots=plots,
        # results=results
    )

callbacks = Callbacks()
callbacks.register_action("on_train_epoch_end", callback=on_train_epoch_end)
callbacks.register_action("on_train_batch_end", callback=on_train_batch_end)
callbacks.register_action("on_train_end", callback=on_train_end)

# 'on_pretrain_routine_start': [], #无参数
#             'on_pretrain_routine_end': [], #无参数
#             'on_train_start': [], #无参数
#             'on_train_epoch_start': [], #无参数
#             'on_train_batch_start': [], #无参数
#             'optimizer_step': [],
#             'on_before_zero_grad': [],
#             'on_train_batch_end': [], # ni, model, imgs, targets, paths, plots
#             'on_train_epoch_end': [], #epoch
#             'on_val_start': [],#无参数
#             'on_val_batch_start': [], #无参数
#             'on_val_image_end': [],  #无参数
#             'on_val_batch_end': [], #无参数
#             'on_val_end': [], #无参数
#             'on_fit_epoch_end': [],  # fit = train + val    #log_vals, epoch, best_fitness, fi
#             'on_model_save': [],    #last, epoch, final_epoch, best_fitness, fi
#             'on_train_end': [],    #last, best, plots, epoch, results
#             'on_params_update': [],
#             'teardown'


try:
    opt = parse_opt(True)
    epochs = opt.epochs
    output =  resolve_output_dir(STATUS_FILE, opt.name)

    # 显存预检(不足 write_status=error + sys.exit)
    # device 闭环:用户未传 device 时,precheck 选剩余最多的卡并写回 opt 透传给 yolov5_train
    _user_device = opt.device
    _picked = precheck_memory(opt.weights, imgsz=opt.imgsz, batch=opt.batch_size, device=_user_device,
                              total_epochs=opt.epochs)
    # v5 opt.device 是 argparse str(默认 ''),用 truthiness 区分:''(未传)写回,"0"(指定)不写回
    # (与 v8 的 is None 不同:v8 经 json.loads 得 int 0/None,v5 经 argparse 得 str "0"/'')
    if not _user_device and _picked:
        opt.device = _picked

    init_status(pid, time.time(), epochs)
    # opt = types.SimpleNamespace(
    #     data=data,
    #     epochs=epochs,
    #     weights=weights,
    #     project="runs/train",
    #     name="exp",
    #     exist_ok=True,
    #     **(args.__dict__ if hasattr(args, '__dict__') else {})
    #     )
    yolov5_train(opt,callbacks=callbacks)
    # write_status(status="finished", message="YOLOv5 训练完成")
except Exception as e:
    stop_watchdog()
    write_error(format_exception_safe(e))
    write_exception(e)