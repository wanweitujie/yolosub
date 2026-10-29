
import asyncio
import enum
import logging
import sys
import threading
import time

from typing import Optional
from modal import ModelConfig,InferenceConfig,ModelInfo
from utils_v.index import YOLOV5_PYTHON_PATH, ULTRALYTICS_PYTHON_PATH
from pathlib import Path
import subprocess
import signal
import os
import httpx
from urllib.parse import quote
from multiprocessing.shared_memory import SharedMemory
import psutil
from utils_v.shared_memory import make_shm_name, cleanup_stale_shared_memory

_executors: list["YoloExecutorManager"] = []

# ===== 假死自愈参数(spec 第 5 节) =====
HEALTH_PING_INTERVAL = 10.0
HEALTH_PING_TIMEOUT = 3.0
FAIL_THRESHOLD = 3
LOCK_HELD_TIMEOUT = 60.0        # 与 gpu_manager.LOCK_HELD_TIMEOUT 一致
READY_TIMEOUT = 60.0
READY_POLL_INTERVAL = 2.0
RESTART_MAX_RETRY = 3
RESTART_BACKOFF = [10.0, 20.0, 40.0]
DEAD_RETRY_INTERVAL = 180.0
CRASH_WINDOW = 300.0
CRASH_WINDOW_MAX = 3


class ExecutorState(enum.Enum):
    HEALTHY = "HEALTHY"
    RESTARTING = "RESTARTING"
    DEAD = "DEAD"


class ExecutorUnavailableError(Exception):
    """RESTARTING/DEAD 态转发请求被拒。"""
    pass


def _global_signal_handler(sig, frame):
    """主进程收到 SIGINT/SIGTERM 时，并行调用所有 executor 的 stop()。

    Python signal 模块每个 signal 只允许一个 handler；若每个实例各自注册，
    后注册的会覆盖前者，导致部分子进程不被优雅停止。
    并行 stop() 避免串行等待多个子进程 wait(timeout=15) 累计超时。
    """
    threads = []
    for ex in _executors:
        def _stop(_ex=ex):
            try:
                _ex.stop()
            except Exception as e:
                try:
                    _ex.logger.warning(f"signal handler 中 stop 失败: {e}")
                except Exception:
                    pass
        t = threading.Thread(target=_stop, daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=20)
    sys.exit(0)


class YoloExecutorManager:
    process:subprocess.Popen
    out_self:bool=False
    client:httpx.AsyncClient = None
    logger:logging.Logger
    use_v5:bool
    def __init__(self,
                 logger:logging.Logger,
                 use_v5:bool,
                 on_dead=None,
                 auto_spawn:bool=True
                 ):
        self.logger=logger
        self.use_v5=use_v5
        self.on_dead=on_dead
        # ===== Phase 2 Task 10:状态机基础设施 =====
        self._state = ExecutorState.HEALTHY
        self._fail_count = 0
        self._restarting = False
        self._loop = None   # 在 startup 事件(model_manager.start_cleanup_task)用 get_running_loop() 绑定
        self._restart_lock = asyncio.Lock()
        # #2 final review:asyncio 对 create_task 只持弱引用,_dead_recovery_loop 是 24/7 无限循环,
        # 若被 GC,DEAD 态将永久静默失效(health loop 在 DEAD 空转)。此处持强引用,完成时自动清理。
        self._recovery_tasks: set[asyncio.Task] = set()
        # #3 final review:_restart_epoch 幂等。排队协程持锁后凭 epoch 变化识别"对方已恢复",
        # 放弃本次重启,防并发双 spawn(锁只串行化,状态复查无法区分初始 HEALTHY vs 对方已恢复)。
        self._restart_epoch = 0
        self._crash_timestamps: list[float] = []
        self._dead_since_ts: float | None = None   # 进入 DEAD 的 epoch 时刻(供 /gpu_status)
        self._next_retry_at: float | None = None   # monotonic:下一次 DEAD 恢复尝试时刻
        self._base_url = f"http://127.0.0.1:{29005 if self.use_v5 else 29006}"
        self._timeout = httpx.Timeout(connect=5.0, read=75.0, write=10.0, pool=5.0)
        self.client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        # ===== Task 13:auto_spawn 直接走 _spawn(原 _spawn_legacy_in_init 已删除) =====
        if auto_spawn:
            self._spawn()
        # 仅第一个实例注册全局 handler,避免后注册者覆盖前者
        if not _executors:
            signal.signal(signal.SIGINT, _global_signal_handler)
            signal.signal(signal.SIGTERM, _global_signal_handler)
        _executors.append(self)

    def _spawn(self):
        """拉起子进程 + 启动 monitor 线程。供 __init__(auto_spawn)与 _restart_async/_dead_recovery_loop 复用。"""
        self.kill_yolo_process_service()
        cleaned = cleanup_stale_shared_memory()
        if cleaned:
            self.logger.info(f"启动时清理了 {cleaned} 个残留 SharedMemory 段")
        service_name = self.get_service_filename()
        script_path = str((Path(__file__).parent / service_name).resolve())
        self.logger.info(f"run {service_name} detece process:{script_path}")

        # 子进程 stdout/stderr 重定向到文件:捕获崩溃 traceback(含 uvicorn EADDRINUSE 等经 stderr 输出,
        # 原 PIPE 无人读取导致丢失成诊断盲区),并避免 PIPE 缓冲(~64KB)写满时子进程 write 阻塞 ->
        # 事件循环假死。append 模式保留跨重启历史;句柄由实例持有,避免 GC 关闭 fd 使子进程写失败。
        self._close_child_log_handles()
        log_dir = Path("log")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._child_stdout_f = open(log_dir / f"{service_name}.stdout.log", "ab")
        self._child_stderr_f = open(log_dir / f"{service_name}.stderr.log", "ab")

        # 构建命令:两个子服务用各自解释器。v5 用独立 venv(/venv_yolov5,含 yolov5 包);
        # ultralytics 用系统主环境 python(/usr/bin/python,含 ultralytics 8.4.121)。
        # 历史上两者共用 YOLOV5_PYTHON_PATH,ultralytics 靠 venv 的 --system-site-packages
        # 隐式链接主环境包,属隐式耦合;现显式分支,与 v5 隔离(venv 内 numpy 1.23.1/yolov5
        # 等包不再进入 ultralytics 子进程的 sys.path)。
        python_path = YOLOV5_PYTHON_PATH if self.use_v5 else ULTRALYTICS_PYTHON_PATH
        cmd = [python_path, script_path]
        self.process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent.parent.parent,
            stdout=self._child_stdout_f,
            stderr=self._child_stderr_f,
            start_new_session=True
        )

        monitor_thread = threading.Thread(target=self.monitor_process, daemon=True)
        monitor_thread.start()

    def _close_child_log_handles(self):
        """关闭上次的子进程 stdout/stderr 日志句柄(重启场景下 _spawn 重开前调用)。"""
        for attr in ("_child_stdout_f", "_child_stderr_f"):
            f = getattr(self, attr, None)
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def _record_failure(self):
        """记录一次转发失败,达 FAIL_THRESHOLD 触发 _restart_async(在 running loop 内 create_task)。"""
        self._fail_count += 1
        if self._state == ExecutorState.HEALTHY and self._fail_count >= FAIL_THRESHOLD:
            _t = asyncio.create_task(self._restart_async())
            self._recovery_tasks.add(_t)
            _t.add_done_callback(self._recovery_tasks.discard)

    async def _restart_async(self):
        """判死/崩溃后重启:kill 旧 → 重建 client → 重 Popen → 轮询就绪。
        3 次退避重试(10/20/40s);全失败 → DEAD + _dead_recovery_loop。"""
        trigger_epoch = self._restart_epoch   # 进锁前记"我触发时的 epoch"
        async with self._restart_lock:
            if self._state != ExecutorState.HEALTHY:
                return
            if self._restart_epoch != trigger_epoch:
                return  # 排队期间已有其它协程/恢复路径成功回到 HEALTHY(epoch 变了),放弃,防双 spawn
            self._state = ExecutorState.RESTARTING
            self._restarting = True
            self.logger.error(f"[restart] {self.get_service_filename()} 判定故障,开始重启")
            try:
                if self.on_dead is not None:
                    self.on_dead(self.use_v5)
            except Exception:
                self.logger.exception("[restart] on_dead 回调失败")

            success = False
            for attempt in range(1, RESTART_MAX_RETRY + 1):
                try:
                    await asyncio.to_thread(self._kill_old)
                    self.client = await self._rebuild_client()
                    await asyncio.to_thread(self._spawn)
                    await self._wait_ready(READY_TIMEOUT)
                    success = True
                    break
                except Exception:
                    self.logger.exception(f"[restart] 第 {attempt} 次重启失败")
                    if attempt < RESTART_MAX_RETRY:
                        await asyncio.sleep(RESTART_BACKOFF[attempt - 1])
            self._restarting = False
            if success:
                self._state = ExecutorState.HEALTHY
                self._restart_epoch += 1   # 恢复成功推进 epoch,让排队中的 pending 协程放弃
                self._fail_count = 0
                self.logger.info(f"[restart] {self.get_service_filename()} 恢复成功")
            else:
                self._state = ExecutorState.DEAD
                self._dead_since_ts = time.time()
                self.logger.error(f"[restart] {self.get_service_filename()} 进入 DEAD,将每 {DEAD_RETRY_INTERVAL}s 尝试恢复")
                _t = asyncio.create_task(self._dead_recovery_loop())
                self._recovery_tasks.add(_t)
                _t.add_done_callback(self._recovery_tasks.discard)

    async def _dead_recovery_loop(self):
        """DEAD 态慢恢复:每 DEAD_RETRY_INTERVAL 在 _restart_lock 内尝试 kill+spawn+ready。
        成功转 HEALTHY + 清 _crash_timestamps/_dead_since_ts/_next_retry_at,既有 _health_loop 自动接管。"""
        while self._state == ExecutorState.DEAD:
            self._next_retry_at = time.monotonic() + DEAD_RETRY_INTERVAL
            await asyncio.sleep(DEAD_RETRY_INTERVAL)
            if self._state != ExecutorState.DEAD:
                return
            async with self._restart_lock:
                try:
                    await asyncio.to_thread(self._kill_old)
                    self.client = await self._rebuild_client()
                    await asyncio.to_thread(self._spawn)
                    await self._wait_ready(READY_TIMEOUT)
                except Exception:
                    self.logger.exception("[recover] DEAD 恢复尝试失败,等下个周期")
                    continue
                self._state = ExecutorState.HEALTHY
                self._restart_epoch += 1
                self._fail_count = 0
                self._crash_timestamps.clear()
                self._dead_since_ts = None
                self._next_retry_at = None
                self.logger.info(f"[recover] {self.get_service_filename()} 从 DEAD 恢复")
                return

    async def _handle_unexpected_exit(self):
        """意外退出统一入口:崩溃风暴保护 + 决定重启或转 DEAD。
        在 event loop 线程执行(monitor_process 经 run_coroutine_threadsafe 调度),状态变更无并发。
        顶层 try/except 兜底,防 Future 静默吞异常。"""
        try:
            if self._state == ExecutorState.DEAD:
                return
            now = time.monotonic()
            self._crash_timestamps = [t for t in self._crash_timestamps if now - t < CRASH_WINDOW]
            self._crash_timestamps.append(now)
            if len(self._crash_timestamps) > CRASH_WINDOW_MAX:
                self.logger.error(
                    f"[restart] {self.get_service_filename()} {CRASH_WINDOW}s 内意外退出 "
                    f"{len(self._crash_timestamps)} 次(> {CRASH_WINDOW_MAX}),转 DEAD 交慢恢复")
                self._state = ExecutorState.DEAD
                self._dead_since_ts = time.time()
                try:
                    self.on_dead(self.use_v5)
                except Exception:
                    self.logger.exception("[restart] on_dead 回调失败")
                _t = asyncio.create_task(self._dead_recovery_loop())
                self._recovery_tasks.add(_t)
                _t.add_done_callback(self._recovery_tasks.discard)
                return
            await self._restart_async()
        except Exception:
            self.logger.exception("[monitor] _handle_unexpected_exit 处理失败")

    async def _health_loop(self):
        """周期 ping /health;RESTARTING/DEAD 态空转(由重启/恢复协程负责)。
        ping 失败(含 503 锁超时)计失败,达阈值触发 _restart_async。"""
        while True:
            await asyncio.sleep(HEALTH_PING_INTERVAL)
            if self._state != ExecutorState.HEALTHY:
                continue
            try:
                resp = await self.client.get("/health", timeout=HEALTH_PING_TIMEOUT)
                resp.raise_for_status()
                self._fail_count = 0
            except Exception:
                self._record_failure()
                self.logger.warning(
                    f"[health] {self.get_service_filename()} ping failed, count={self._fail_count}")

    def get_service_filename(self):
        return "v5_service.py" if self.use_v5 else "ultralytics_service.py"

    def status_snapshot(self) -> dict:
        """子进程健康态快照(供 /gpu_status 聚合)。DEAD 态附恢复 ETA 与进入时刻;RESTARTING/HEALTHY 无 ETA。
        next_retry_in 是"预计下次尝试",非确定性承诺——恢复尝试可能失败并周期重置。"""
        snap = {"state": self._state.value, "next_retry_in": None, "dead_since_ts": None}
        if self._state == ExecutorState.DEAD:
            snap["dead_since_ts"] = self._dead_since_ts
            if self._next_retry_at is not None:
                snap["next_retry_in"] = max(0.0, self._next_retry_at - time.monotonic())
        return snap
        
    def kill_yolo_process_service(self):
        count = 0
        service_name = self.get_service_filename()
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.cmdline()
                if len(cmdline) > 1 and service_name in ' '.join(cmdline):
                    print(f"发现进程: PID={proc.pid}, 命令={' '.join(cmdline)}")
                    # 先发温和的 SIGTERM（让进程有机会优雅退出）
                    proc.terminate()      
                    # 或者强制杀 proc.kill() → 等价于 kill -9
                    count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass        
        if count>0: print(f"已发送终止信号给 {count} 个 {service_name} 进程")
        else : print(f"未发现残留的 {service_name} 进程")
        
    

    def monitor_process(self):
        """子进程退出处理,三类:
        1) out_self:整体退出(stop/SIGTERM),不动作(原行为)。
        2) _restarting:主动重启(_restart_async 拉新进程),不动作。
        3) 意外退出(OOM -9 / 段错误 / 致命异常):经 _handle_unexpected_exit 自愈,不杀主进程。"""
        try:
            self.process.wait()
            if self.out_self:
                return
            if self._restarting:
                return
            self.logger.error(
                f"[monitor] {self.get_service_filename()} 意外退出(rc={self.process.returncode})")
            # _loop 在 startup 事件(start_cleanup_task)绑定。子进程在 startup 前退出时
            # (如启动初期 crash),延迟重试等 startup 绑定,而非直接放弃(否则子进程永久缺失)。
            # 双主进程场景第二主进程绑端口失败崩溃不进 startup,30s 后放弃(由 start.sh 防重复启动根治)。
            # 用 monotonic(非 time.time):避免 NTP 时钟回拨拉长/缩短等待窗口
            deadline = time.monotonic() + 30
            while self._loop is None and time.monotonic() < deadline:
                time.sleep(0.5)
            if self._loop is None:
                self.logger.error("[monitor] _loop 30s 内未绑定,放弃自愈(主进程未进入 startup)")
                return
            asyncio.run_coroutine_threadsafe(self._handle_unexpected_exit(), self._loop)
        except Exception as e:
            self.logger.exception(e)


    def stop(self):
        if not self.is_running():
            raise RuntimeError("没有正在运行的任务")
        try:
            self.out_self=True
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=15)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        finally:
            self.process = None

    def _kill_old(self):
        """重启专用:kill 旧子进程组并等待退出。与 stop() 区别:不重建 client(_restart_async 负责)、
        不设 out_self。旧 monitor 线程经 _restarting=True 分支自然退出(daemon,无需 join)。"""
        if self.process is None:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=15)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        self.process = None

    def is_running(self):
        return self.process is not None and self.process.poll() is None
    
    async def create_model_async(self,config:ModelConfig):    
        return ModelInfo(**await self.send_to_process_async("/create_model","post",config.model_dump( mode="json",exclude_none=True, exclude_unset=True,))) 

    async def detect_async(self,model_name: str,contents:bytes, options:Optional[InferenceConfig],decimals : int = 5):
        shm: SharedMemory  = None
        try:
            shm_size = len(contents)
            shm_name = make_shm_name()
            shm = SharedMemory(name=shm_name, create=True, size=shm_size)
            shm.buf[:shm_size] = contents
            body = {
                "shm_name": shm_name,
                "shm_size":shm_size,
                "decimals":decimals,
            }
            if options is not None:
                body["options"]=options.model_dump( mode="json",exclude_none=True, exclude_unset=True,)

            # model_name 可能含中文/%/#/? 等对 URL 有特殊含义的字符, 必须 percent-encode 成单一 path 段,
            # 否则主→子这一跳(子进程 uvicorn 会 unquote 回来)会与 create(body 存原名)错位。
            # safe='' 为命门: 默认 safe='/' 不编码斜杠, 会让名字里的 / 变成路径分隔符导致路由失效。
            return await self.send_to_process_async(f"/detect/{quote(model_name, safe='')}","post",body)
        finally:
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass
                shm = None
            
    async def update_model_memory_usage_async(self, model_name: str, reset: bool = False):
        qs = "?reset=1" if reset else ""
        return ModelInfo(**await self.send_to_process_async(
            f"/update_model_memory_usage/{quote(model_name, safe='')}{qs}", "get"))

    async def close_model_async(self,model_name: str):
        return await self.send_to_process_async(f"/close_model/{quote(model_name, safe='')}","delete")

    async def _teardown_client(self):
        """关闭 httpx client,清理 keep-alive 连接池。供重启/恢复时调用。"""
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def _rebuild_client(self):
        """重启/恢复时重建 client,清旧 keep-alive 连接池,避免污染失败计数。"""
        await self._teardown_client()
        self.client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self.client

    async def _wait_ready(self, timeout: float):
        """轮询 GET /health,确认新子进程就绪。uvicorn 启动期 ConnectError 容忍,持续重试到 timeout。"""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await self.client.get("/health", timeout=HEALTH_PING_TIMEOUT)
                resp.raise_for_status()
                return True
            except Exception:
                await asyncio.sleep(READY_POLL_INTERVAL)
        raise RuntimeError(f"subprocess not ready within {timeout}s")

    async def send_to_process_async(self, url: str, method: str, content: Optional[dict[str, any]] = None):
        if self._state == ExecutorState.RESTARTING:
            raise ExecutorUnavailableError(f"subprocess {self._state.value}")
        if self._state == ExecutorState.DEAD:
            raise ExecutorUnavailableError(f"subprocess {self._state.value}")
        resp: httpx.Response = None
        method = method.lower()
        client = self.client
        try:
            if method == "post":
                resp = await client.post(url, json=content)
            elif method == "delete":
                resp = await client.delete(url)
            elif method == "get":
                resp = await client.get(url)
            elif method == "put":
                resp = await client.put(url, json=content)
            else:
                raise Exception(f"not support method:{method}")
            resp.raise_for_status()
            result = resp.json()
            if not result["success"]:
                self._fail_count = 0       # 业务 errMsg:子进程健康
                raise Exception(result["errMsg"])
            self._fail_count = 0           # 成功
            return result["data"]
        except httpx.HTTPStatusError as e:
            # 防御性分支:子进程路由现状恒 200(宽 try/except),此分支暂不触发;
            # 未来若放宽子进程 try/except 让 HTTPException 冒泡,4xx 清零/5xx 计失败才生效
            if e.response.status_code < 500:
                self._fail_count = 0
            else:
                self._record_failure()
            raise
        except (httpx.TransportError, httpx.TimeoutException):
            self._record_failure()
            raise
        