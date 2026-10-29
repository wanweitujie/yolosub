# app/manager.py
import asyncio
import atexit
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import GPUtil
import yaml

from .schemas import TrainStartRequest
from .param_spec import split_params, build_v5_cli_args, build_v8_cli_args, build_hyp_data
from .metrics import parse_results_csv, summarize, detect_csv_generation
from utils_v.index import format_yolo_modelpath, YOLOV5_PYTHON_PATH, _estimate_from_weights, _normalize_amp, redact_container_paths, redact_gpu_model, strip_ansi

# 主进程未确认子进程已终止的状态;stop() 仅在这些状态下写入 stopped,避免覆盖 finished/error 终态
_RUNNING_STATUSES = frozenset({None, "starting", "running"})

# 终态集合常量化,与 _RUNNING_STATUSES 对称;status() 单查与 metrics_query() 共用防漂移
_TERMINAL_STATUSES = frozenset({"finished", "stopped", "error"})

# shutdown 全量停止的统一等待预算(秒):不逐个 wait(15),防 N 个训练挂 N×15s(docker stop 10s 后即 SIGKILL)
_SHUTDOWN_BUDGET_SECONDS = 5

# 终态记录保留上限(活跃记录全保留):防长期运行 trainings 只增不删,
# /train/status 每次调用对全部记录 O(N) 读 status.json 无界增长(前端活跃期每 2s 轮询一次)
_MAX_TERMINAL_RECORDS = 20

# === 虚拟环境路径 ===
run_dir_str = str(((Path(__file__).parent.parent.parent) / "runs/train").resolve())
if not Path(YOLOV5_PYTHON_PATH).exists():
    raise RuntimeError(f"YOLOv5 虚拟环境未找到: {YOLOV5_PYTHON_PATH}")


@dataclass
class TrainingRecord:
    """一次训练任务的运行记录。gpu_id=None 表示 CPU 槽位。"""
    exp: str
    process: subprocess.Popen
    gpu_id: Optional[int]
    version: str
    python_bin: str
    status_file: Path
    log_file: Path
    transport_yaml_files: list = field(default_factory=list)


def _unlink_all(paths):
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


class TrainingManager:
    _instance = None
    def __new__(cls):
        # 注册信号处理器：监听 SIGINT (Ctrl+C) 和 SIGTERM
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.run_dir = Path(run_dir_str)
            cls._instance.run_dir.mkdir(parents=True, exist_ok=True)
            cls._instance.trainings: dict = {}           # exp -> TrainingRecord,插入序即启动序
            cls._instance._start_lock = asyncio.Lock()   # start 分配锁;stop 故意不持锁(见 stop docstring)
            cls._instance.transport_yaml_files: list = []  # start 期间暂存,启动成功后转入 record
            signal.signal(signal.SIGINT, cls._instance.signal_handler)
            signal.signal(signal.SIGTERM, cls._instance.signal_handler)
            # uvicorn.run 启动后会用自己的退出 handler 替换上述信号注册(Unix 走 loop.add_signal_handler),
            # signal_handler 在生产主进程实际不可达;atexit 在 uvicorn 优雅退出(run() 返回→解释器退出)时必经,
            # 是清理训练子进程组的可靠路径(stop.sh 的 PATTERN 不匹配训练脚本,训练进程又以独立会话运行,
            # 不经此处清理就会孤儿化继续占卡)。SIGKILL 强杀无可执行清理,任何机制均无解。
            # _shutdown_all 幂等(按 poll 判活),与信号路径重复触发无害。
            atexit.register(cls._instance._shutdown_all)
        return cls._instance

    # ===== 信号 / shutdown =====

    def signal_handler(self, sig, frame):
        """父进程收到退出信号:终止所有活跃训练的进程组(不逐个 wait,见 _shutdown_all)。"""
        print(f"收到信号 {sig}，终止全部训练子进程组...")
        try:
            self._shutdown_all()
        except Exception:
            pass

    def _shutdown_all(self):
        """shutdown 全量停止:逐个 killpg 后统一短等待(总预算 _SHUTDOWN_BUDGET_SECONDS)。

        不逐个 wait(15):N 个训练串行 wait 最多 N×15s;docker stop 默认 10s grace 后 SIGKILL,
        wait 形同虚设且 stopped 写入可能没执行;此处 stopped 写入尽力而为。"""
        deadline = time.time() + _SHUTDOWN_BUDGET_SECONDS
        for rec in list(self.trainings.values()):
            if rec.process.poll() is None:
                try:
                    os.killpg(os.getpgid(rec.process.pid), signal.SIGKILL)
                except Exception:
                    try:
                        rec.process.kill()
                    except Exception:
                        pass
        for rec in list(self.trainings.values()):
            try:
                rec.process.wait(timeout=max(0.1, deadline - time.time()))
            except Exception:
                pass
        for rec in self.trainings.values():
            self._write_stopped_if_running(rec)

    # ===== 多槽基础 =====

    def _reap_dead(self) -> list:
        """遍历记录 poll()(顺带 reap 僵尸),返回活跃记录。

        已退出记录:持久化 version/gpu_id(_persist_record_fields)+ 清理 transport yaml,
        并按卡去重:一张卡(含 CPU 槽)最多保留一条记录——该卡有活跃训练时只留活跃
        (其历史终态清理),否则只留最近结束的一条(旧终态清理)。去重仅动内存列表,
        磁盘 exp 产物保留(/train/status/{exp} 仍可单查);全局仍受
        _MAX_TERMINAL_RECORDS 封顶兜底(每卡去重后总量约 2×(GPU+1),一般不再触发)。"""
        active, terminal = [], []
        for exp, rec in self.trainings.items():
            if rec.process.poll() is None:
                active.append(rec)
                continue
            terminal.append(exp)
            self._persist_record_fields(rec)
            if rec.transport_yaml_files:
                # 自然结束的训练也会残留 _transport.yaml(原仅 stop() 清理),退出时顺手清理
                _unlink_all(rec.transport_yaml_files)
                rec.transport_yaml_files = []
        # 每卡最多一条:活跃记录恒保留,其所在卡的终态记录清理(卡上仅展示活跃那条);
        # 空闲卡倒序遇首次 = 最新启动的终态,保留,同卡更旧终态清理。
        active_keys = {str(r.gpu_id) if r.gpu_id is not None else "cpu" for r in active}
        kept_terminal = set()
        for exp in reversed(terminal):
            rec = self.trainings[exp]
            key = str(rec.gpu_id) if rec.gpu_id is not None else "cpu"
            if key in active_keys or key in kept_terminal:
                self.trainings.pop(exp, None)
            else:
                kept_terminal.add(key)
        for exp in terminal[:-_MAX_TERMINAL_RECORDS]:
            self.trainings.pop(exp, None)
        return active

    def _persist_record_fields(self, rec: TrainingRecord):
        """进程退出后把 version/gpu_id 持久化进 status.json。

        子进程回调整文件覆盖写不携带这两字段,主进程内存记录是唯一来源;
        不落盘则服务重启后(内存记录丢失)历史 exp 的响应永久缺失两字段
        (前端会把缺 gpu_id 的 GPU 训练误渲染为 CPU)。进程已退出无并发写者,
        读改写安全;字段已一致则跳过写入(避免轮询期无谓 IO)。"""
        try:
            data = json.loads(rec.status_file.read_text(encoding="utf-8"))
            if data.get("version") == rec.version and data.get("gpu_id") == rec.gpu_id:
                return
            data["version"] = rec.version
            data["gpu_id"] = rec.gpu_id
            rec.status_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _latest_active(self) -> Optional[TrainingRecord]:
        """最后启动的活跃记录(dict 插入序即启动序,倒序找第一个活跃)。"""
        for rec in reversed(list(self.trainings.values())):
            if rec.process.poll() is None:
                return rec
        return None

    def _exp_dir(self, exp_name: str, allow_flat_fallback: bool = False) -> Path:
        """训练产物根目录(读写共用)。allow_flat_fallback 仅供读路径(status/log/metrics):
        True 且分桶路径不存在且平铺路径存在 → 回退平铺——2026-01-05~2026-09-07 窗口产物
        「时间戳命名却平铺落盘」(分桶改造 e101315 之前的一代),正则判进分桶后 status/metrics
        查不到(既有缺陷,本次顺手修复)。写路径(start 的 mkdir、_unique_exp_name 判重)
        不得传 True:新 exp 名两路径都不存在,无守卫回退会让所有新训练落平铺、分桶约定失效。
        非时间戳命名(旧版 exp_.../测试夹具)保持平铺,不受此参数影响。"""
        m = re.match(r"^exp(\d{14})", exp_name)
        if not m:
            return self.run_dir / exp_name
        bucketed = self.run_dir / m.group(1)[:6] / exp_name
        if allow_flat_fallback and not bucketed.exists() and (self.run_dir / exp_name).exists():
            return self.run_dir / exp_name
        return bucketed

    def _unique_exp_name(self, job_id: str) -> str:
        """exp 名唯一化:exp{yyyyMMddHHmmss}_{job_id},同秒/同名(job_id)重名追加序号。

        时间戳取创建时刻当前时间并烙入 exp 名;yyyyMM 由名内时间戳(name[3:9])定位桶目录;
        防 trainings[exp] 键覆盖:被覆盖训练失去记录与占用推导 → 卡被视空闲再分配 → 真双占。"""
        base = f"exp{datetime.now().strftime('%Y%m%d%H%M%S')}_{job_id}"
        name, i = base, 2
        while name in self.trainings or self._exp_dir(name).exists():
            name = f"{base}_{i}"
            i += 1
        return name

    # ===== GPU 分配(start 锁内调用) =====

    async def _pick_free_gpu(self, budget_mb: Optional[int]):
        """选一张未被活跃训练占用(且显存够)的最空卡。返回 (gpu_id, errMsg),errMsg 为 None 即成功。

        GPU 查询经 to_thread(遵守 model_manager/gpu_manager 的 to_thread 约定,不阻塞事件循环);
        三种失败返回不同 errMsg;GPUtil 异常吞掉转 errMsg(start 不变 500)。"""
        in_use = {r.gpu_id for r in self._reap_dead() if r.gpu_id is not None}
        try:
            gpus = await asyncio.to_thread(GPUtil.getGPUs)
        except Exception as e:
            return None, f"查询 GPU 失败: {e}"
        if not gpus:
            return None, "无可用 GPU"
        free = [g for g in gpus if g.id not in in_use]
        if not free:
            return None, "所有 GPU 都在训练，无空闲卡"
        if budget_mb is not None:
            free = [g for g in free if g.memoryFree > budget_mb]
            if not free:
                return None, f"无空闲 GPU 满足显存预算 {budget_mb} MB"
        return max(free, key=lambda g: g.memoryFree).id, None

    def _estimate_budget(self, model_path: str, params: dict) -> Optional[int]:
        """按权重文件字节 + imgsz/batch 估算选卡预算(MB);autobatch(batch<=0)/读不到权重 -> None。

        口径注:文件字节含元数据约比 state_dict 纯权重大 3-8%,仅用于选卡过滤,
        权威校验由子进程 precheck(state_dict) 复核。"""
        batch = params.get("batch", 16)
        try:
            if int(batch) <= 0:
                return None
        except (TypeError, ValueError):
            pass
        try:
            size = os.path.getsize(model_path)
        except OSError:
            return None
        # amp 归一为 bool:默认 True 与 train_v8 实际默认一致(无 amp 键时按 FP16 激活减半估
        # 预算,防 FP32 虚高一倍误拒);显式 amp:false 仍按 FP32 估
        amp = _normalize_amp(params.get("amp", True))
        return _estimate_from_weights(size, params.get("imgsz", 640), batch, amp=amp)

    # ===== 状态读取 / 兜底 =====

    def _read_status_file(self, exp_name: str) -> Optional[dict]:
        status_file = self._exp_dir(exp_name, allow_flat_fallback=True) / "status.json"
        if not status_file.exists():
            return None
        try:
            data = json.loads(status_file.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict):
            # 撕裂/手改出的合法非对象 JSON(如 [])按坏文件处理走 unknown 空态,
            # 不在此抛 TypeError 破坏统一信封
            return None
        data["exp"] = exp_name
        return data

    def _attach_record_fields(self, exp_name: str, data: dict) -> dict:
        """仿 version 先例:响应中的 version/gpu_id 从训练记录补挂。

        status.json 内这两个字段会被子进程回调的整文件覆盖写冲掉(回调不携带),
        文件内的 gpu_id 仅作进程崩溃后的排查参考,响应一律以记录为准。"""
        rec = self.trainings.get(exp_name)
        if rec is not None:
            data["version"] = rec.version
            data["gpu_id"] = rec.gpu_id
        return data

    def _attach_results(self, exp: str, data: dict) -> dict:
        """终态任务附 results 摘要(读时计算,不动训练回调——已终止/异常/历史任务天然支持)。

        status 单查与总览(status() 两个分支)共用,保证两处字段与条件恒一致。

        调用时机有两处硬约束:
          ① 必须在 _reconcile_stale_status 之后——残留 running 被兜底改判 error 的任务也该挂;
          ② 必须在 _attach_record_fields 之后——version 从记录补挂后才能分派 fitness 公式。

        刻意不并入 _attach_record_fields:后者是纯内存操作(拷 version/gpu_id),本方法是文件 I/O;
        合并会让每个调用方(含 metrics_query)无声多出一次 csv 读。
        """
        if data.get("status") not in _TERMINAL_STATUSES:
            return data
        csv_path = self._exp_dir(exp, allow_flat_fallback=True) / "output" / "results.csv"
        rows = parse_results_csv(csv_path)
        if rows:
            # version 回退链:status.json 缺 version(平铺窗口历史任务)时按 csv 表头
            # 判代际,仍判不出按 v8——静默错按 v8 公式算 v5 任务 best 会错行
            data["results"] = summarize(
                rows, data.get("version") or detect_csv_generation(csv_path) or "v8")
        return data

    def _reconcile_stale_status(self, exp: str, data: dict) -> dict:
        """按记录粒度兜底:记录存在且进程已退出且状态停在 starting/running -> 补写 error。

        无记录的 exp(跨主进程重启的历史)不兜底,原样返回。"""
        rec = self.trainings.get(exp)
        if rec is None or rec.process.poll() is None:
            return data
        if data.get("status") not in _RUNNING_STATUSES:
            return data
        data["status"] = "error"
        msg = data.get("message") or ""
        data["message"] = (msg + " | " if msg else "") + "训练进程异常退出,未写入终态(可能被系统 OOM 终止或崩溃)"
        try:
            # exp 是响应附加字段不落盘;version/gpu_id 已由 _persist_record_fields 持久化,写回须保留
            write_data = {k: v for k, v in data.items() if k != "exp"}
            rec.status_file.write_text(json.dumps(write_data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return data

    def _reconcile_all_stale(self):
        """start 前兜底:遍历全部记录,进程已死但状态残留 starting/running -> 补写 error。"""
        for rec in self.trainings.values():
            if rec.process.poll() is None:
                continue
            data = self._read_status_file(rec.exp)
            if data is not None:
                self._reconcile_stale_status(rec.exp, data)

    # ===== 启停 =====

    async def start(self, req: TrainStartRequest):
        """启动训练(多槽)。全程持 _start_lock:占用快照→to_thread 查 GPU→决策→Popen→记录。

        device 语义:未传=自动分配空闲卡;"cpu"=CPU 槽(全局 1 个);"N"=指定卡(校验未被占用);
        多卡值("0,1")直接拒绝。分配失败 raise RuntimeError,由路由转 success=false。"""
        async with self._start_lock:
            self._reconcile_all_stale()
            req.model = format_yolo_modelpath(req.version, req.model, '预训练权重(model)')
            params = dict(req.params or {})
            device = params.pop("device", None)
            # 版本参数校验(split_params 内静默过滤保留键;非法键一次列全 raise,锚定在
            # _unique_exp_name/mkdir 之前——校验失败零残留)。v5+amp 旧特判随之下线(报错取代)
            try:
                cli_params, hyp_params = split_params(req.version, params)
            except ValueError as e:
                # 纯函数层契约是 ValueError(test_param_spec 锚定);manager 对 routes 的既有
                # 错误信封是 RuntimeError(GPU 分配失败同款)——归一后抛,消息原文保留
                raise RuntimeError(str(e)) from e
            dev_str = str(device).strip() if device is not None else ""
            cpu = dev_str.lower() == "cpu"
            gpu_id: Optional[int] = None
            if dev_str and not cpu:
                if "," in dev_str:
                    raise RuntimeError("暂不支持单训练多卡")
                try:
                    gpu_id = int(dev_str)
                except ValueError:
                    raise RuntimeError(f"不支持的 device 值: {device}")
                busy = next((r for r in self._reap_dead() if r.gpu_id == gpu_id), None)
                if busy is not None:
                    raise RuntimeError(f"GPU {gpu_id} 正在训练中（exp={busy.exp}）")
            elif cpu:
                if any(r.gpu_id is None for r in self._reap_dead()):
                    raise RuntimeError("CPU 训练已占用")
            else:
                budget = self._estimate_budget(req.model, params)
                gpu_id, err = await self._pick_free_gpu(budget)
                if err:
                    raise RuntimeError(err)

            exp_name = self._unique_exp_name(req.job_id)
            exp_dir = self._exp_dir(exp_name)
            status_file = exp_dir / "status.json"
            log_file = exp_dir / "train.log"
            if req.version == "v5":
                python_bin = YOLOV5_PYTHON_PATH
                script_name = "train_v5.py"
            else:
                python_bin = sys.executable
                script_name = "train_v8.py"
            try:
                exp_dir.mkdir(parents=True, exist_ok=True)
                hyp_arg = []
                if hyp_params:
                    # 仅 params 含 hyp 键才生成/注入(无键保持 v5 原生默认 hyp 路径,防内嵌
                    # 默认与 vendored 漂移时悄悄改掉默认超参)。写在 transport 之前:
                    # transport 成功后 hyp 失败会残留 transport 文件,反之零残留(spec §六.2)
                    hyp_file = exp_dir / "_hyp.yaml"
                    hyp_file.write_text(yaml.safe_dump(build_hyp_data(hyp_params),
                                                        default_flow_style=False), encoding="utf-8")
                    self.transport_yaml_files.append(str(hyp_file))
                    hyp_arg = ["--hyp", str(hyp_file)]
                # transport yaml 写入各自 exp_dir:文件名仅由源路径决定,同数据集并行训练若共用一份
                # _transport.yaml,先退出者的清理(_reap_dead/stop)会删掉后来者尚未读取的文件
                yaml_path = self.transport_yaml_path(req.data, exp_dir)
            except Exception:
                # 失败清理三步(spec §六.5):① 无条件 unlink 本次已知路径(hyp 自身写一半失败时
                # 不在全局列表,靠已知路径删,_unlink_all 内部 missing_ok 安全)② 清全局列表
                # (防残留路径被下一次成功 start 收编)③ 删本次 mkdir 的空 exp_dir(空目录同样
                # 让同名重试被 _unique_exp_name 判重改名)
                # transport 半成品同样按已知路径补删:dump 中途失败时 append 尚未执行,
                # 半截 *_transport.yaml 不在全局列表(spec §六.5① 已知路径推导删除;
                # 文件名与 transport_yaml_path 内部表达式一致:stem + "_transport.yaml",
                # dict/非 dict 两分支共用同一 new_datapath)
                _unlink_all([str(exp_dir / "_hyp.yaml"),
                             str(exp_dir / (Path(req.data).stem + "_transport.yaml"))]
                            + list(self.transport_yaml_files))
                self.transport_yaml_files = []
                try:
                    exp_dir.rmdir()
                except OSError:
                    pass
                # 分桶父目录(YYYYMM)由同一次 mkdir(parents=True) 一并新建,空桶同属本次残留,
                # 一并删(rmdir 仅空目录可删,桶内尚有其他训练产物时自然失败跳过;平铺名
                # parent 即 run_dir,守卫跳过防误删训练根目录)
                if exp_dir.parent != self.run_dir:
                    try:
                        exp_dir.parent.rmdir()
                    except OSError:
                        pass
                raise

            cmd = [python_bin, str((Path(__file__).parent / f"process/{script_name}").resolve()),
                   "--status_file", str(status_file), "--log_file", str(log_file)]
            cmd += ["--data", yaml_path, "--epochs", str(req.epochs),
                    "--project", str(exp_dir), "--name", "output", "--exist-ok"]
            model_key = "weights" if req.version == "v5" else "model"
            cmd += [f"--{model_key}", req.model]
            cmd += ["--device", "cpu" if cpu else str(gpu_id)]
            if req.version == "v5":
                cmd += build_v5_cli_args(cli_params)          # v5:旗帜映射/假值/cache/freeze 规则
            else:
                cmd += build_v8_cli_args(cli_params)          # v8/v11:键已白名单化,同名透传不变
            cmd += hyp_arg                                     # 仅 v5 有 hyp 键时非空
            print(f"[启动] {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                cwd=Path(__file__).parent.parent,
                stdout=log_file.open("w", buffering=1),
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
            pid = process.pid
            # starting 状态直接带 gpu_id:前端启动即显示分配卡;子进程回调会覆盖,响应以记录补挂为准
            status_file.write_text(json.dumps({
                "status": "starting", "pid": pid, "exp": exp_name,
                "version": req.version, "gpu_id": gpu_id,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            self.trainings[exp_name] = TrainingRecord(
                exp=exp_name, process=process, gpu_id=(None if cpu else gpu_id),
                version=req.version, python_bin=python_bin,
                status_file=status_file, log_file=log_file,
                transport_yaml_files=list(self.transport_yaml_files))
            self.transport_yaml_files = []
            return {"exp": exp_name, "pid": pid, "version": req.version,
                    "gpu_id": (None if cpu else gpu_id)}

    def _write_stopped_if_running(self, rec: TrainingRecord):
        """仅当状态仍处 running 态时写 stopped,避免覆盖 finished/error 终态。"""
        try:
            data = json.loads(rec.status_file.read_text(encoding="utf-8"))
        except Exception:
            return
        if data.get("status") in _RUNNING_STATUSES:
            prev = data.get("message") or ""
            data["status"] = "stopped"
            data["message"] = (prev + " | " if prev else "") + "主进程主动停止"
            try:
                rec.status_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

    def stop(self, exp: str = None) -> dict:
        """停止指定 exp 的训练;无参 = 停最后启动的活跃训练(存量 API 兼容)。

        刻意不持 _start_lock:signal_handler 是同步上下文不能 await;与持锁 start 交错无害
        (stop 只减少占用,最坏保守拒绝刚释放的卡,不会双占)。"""
        if exp is not None:
            rec = self.trainings.get(exp)
            if rec is None or rec.process.poll() is not None:
                raise RuntimeError(f"训练 {exp} 不存在或已结束")
        else:
            rec = self._latest_active()
            if rec is None:
                raise RuntimeError("没有正在运行的任务")
        try:
            os.killpg(os.getpgid(rec.process.pid), signal.SIGTERM)
            rec.process.wait(timeout=15)
        except Exception:
            # 兜底同样按进程组杀:wait 超时多因主线程卡在 CUDA C 扩展(SIGTERM handler 无法执行),
            # 只 kill 组长会残留 dataloader worker 持有 CUDA context/显存(与 _kill_group_and_exit 同理)
            try:
                os.killpg(os.getpgid(rec.process.pid), signal.SIGKILL)
            except Exception:
                try:
                    rec.process.kill()
                except Exception:
                    pass
        finally:
            self._write_stopped_if_running(rec)
            _unlink_all(rec.transport_yaml_files)
            rec.transport_yaml_files = []
        return {"status": "stopped", "exp": rec.exp}

    # ===== 查询 =====

    def status(self, exp: str = None):
        """指定 exp:单训练状态(记录补挂 version/gpu_id,无记录返回 unknown/原样)。

        无 exp:多训练总览 {trainings, gpu_occupancy, cpu_busy}。gpu_occupancy 仅含被占用项,
        不查 GPUtil 枚举全卡(高频轮询避免每次 nvidia-smi 的开销与事件循环阻塞),
        完整卡列表由前端 /gpu_status 取后按 gpu_id join。"""
        if exp is not None:
            data = self._read_status_file(exp)
            if data is None:
                return {"status": "unknown"}
            data = self._reconcile_stale_status(exp, data)
            data = self._attach_record_fields(exp, data)
            return self._attach_results(exp, data)
        active = self._reap_dead()
        trainings_list = []
        for rec in reversed(list(self.trainings.values())):
            data = self._read_status_file(rec.exp)
            if data is None:
                continue
            data = self._reconcile_stale_status(rec.exp, data)
            data = self._attach_record_fields(rec.exp, data)
            trainings_list.append(self._attach_results(rec.exp, data))
        return {
            "trainings": trainings_list,
            "gpu_occupancy": {str(r.gpu_id): r.exp for r in active if r.gpu_id is not None},
            "cpu_busy": any(r.gpu_id is None for r in active),
        }

    def log(self, exp: str = None):
        """日志读接口(/train/log、/train/log/{exp})的数据源。

        返回前统一净化(日志面向用户层;磁盘文件保持原样,容器内排查仍需真实内容):
          1. 剥 ANSI 转义序列(框架染色/进度擦行,前端渲染会看到残留乱码)
          2. 容器内根路径脱敏(不得露出运行在容器内)
          3. 显卡型号脱敏(`CUDA:0 (Quadro RTX 5000, 15929MiB)` → `CUDA:0`,不暴露硬件资产)
        两路由都走本方法,单点覆盖。"""
        if exp is None:
            rec = self._latest_active()
            if rec is None:
                return ""
            log_file = rec.log_file
        else:
            log_file = self._exp_dir(exp, allow_flat_fallback=True) / "train.log"
        if not log_file.exists():
            return ""
        return redact_gpu_model(redact_container_paths(strip_ansi(log_file.read_text())))

    def metrics_query(self, exp: str, since_epoch: int = -1) -> dict:
        """GET /train/metrics/{exp} 数据源:逐 epoch 指标序列(增量)+ 终态 results 摘要。

        status 走与 status() 相同的 stale 兜底(进程已死残留 running 时两接口报态
        必须一致,防调度层终态判定被矛盾误导);csv 经 _exp_dir 读路径回退推导,
        不依赖 status.json 的 output 字段(running 态本就无该字段,且历史产物里
        train_v5 曾把该字段拼成 `{exp_dir}/status.json/output`——2026-10-15 已修,
        但读路径走 _exp_dir 能兼容修前落盘的历史产物,不回退到依赖它)。"""
        data = self._read_status_file(exp)
        if data is None:
            return {"exp": exp, "status": "unknown", "since_epoch": since_epoch,
                    "epochs": [], "last_epoch": -1}
        data = self._reconcile_stale_status(exp, data)
        data = self._attach_record_fields(exp, data)
        csv_path = self._exp_dir(exp, allow_flat_fallback=True) / "output" / "results.csv"
        rows = parse_results_csv(csv_path)
        visible = [r for r in rows if r["epoch"] > since_epoch] if since_epoch >= 0 else rows
        payload = {"exp": exp, "status": data.get("status"),
                   "since_epoch": since_epoch, "epochs": visible,
                   "last_epoch": rows[-1]["epoch"] if rows else -1}
        if data.get("status") in _TERMINAL_STATUSES and rows:
            # version 回退链同 status():缺 version 的历史任务按 csv 表头判代际分派 fitness 公式
            payload["results"] = summarize(
                rows, data.get("version") or detect_csv_generation(csv_path) or "v8")
        return payload

    def is_running(self):
        """是否存在活跃训练。保留作公共便捷查询(内部经 _latest_active 判定,当前无内部调用方)。"""
        return self._latest_active() is not None

    # ===== data yaml 路径改写(原逻辑不变) =====
    def transport_yaml_path(self, datapath, out_dir=None):
        """改写 data yaml 的数据集路径。

        out_dir:改写产物输出目录,不传则与源文件同目录(旧行为)。start 传各自 exp_dir,
        保证同数据集并行训练各持一份,互不因清理竞态删掉对方的文件。"""
        path = Path(datapath)
        if not path.exists():
            raise FileNotFoundError(f"data yaml 不存在: {datapath}")
        if out_dir is not None:
            new_datapath = str(Path(out_dir) / (path.stem + "_transport.yaml"))
        else:
            new_datapath = re.sub(r'\.yaml$', "_transport.yaml", datapath)
        with open(datapath, 'r') as f:
            data = yaml.safe_load(f)
        base_dir = os.path.dirname(os.path.abspath(datapath))

        if not isinstance(data, dict):
            # 空 yaml 或非字典结构,无法改写路径,原样写出
            with open(new_datapath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            self.transport_yaml_files.append(new_datapath)
            return new_datapath

        # 删除 download 字段:禁止框架在数据缺失时联网下载数据集,容器内应使用本地数据
        data.pop('download', None)

        # 确定 data root:有 path 且 (yaml 目录/path) 目录存在 -> 用该 path 子目录;否则用 yaml 目录
        root = base_dir
        yolo_path = data.get('path')
        if isinstance(yolo_path, str) and yolo_path:
            candidate = os.path.abspath(os.path.join(base_dir, yolo_path))
            if os.path.isdir(candidate):
                root = candidate
        data['path'] = root

        def _to_abs(p):
            # 相对 root 转绝对;已是绝对路径则保留;非字符串原样返回
            if not isinstance(p, str):
                return p
            return os.path.abspath(os.path.join(root, p))

        def _to_abs_list(v):
            if isinstance(v, list):
                return [_to_abs_list(x) for x in v]
            return _to_abs(v)

        # train/val/test 绝地化相对 root,框架直接用绝对路径
        for key in ['train', 'val', 'test']:
            v = data.get(key)
            if v is None:
                continue
            data[key] = _to_abs_list(v)

        with open(new_datapath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        self.transport_yaml_files.append(new_datapath)
        return new_datapath
