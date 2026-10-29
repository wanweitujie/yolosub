"""多卡并行训练 manager 多槽单测(方案1:每卡1训练 + 1个CPU槽)。

替代原 test_train_manager_status.py:单槽契约(current_exp/全局 is_running)已废弃,
语义等价用例(残留 running 兜底/status(exp) 指定查询)迁移为按 TrainingRecord 粒度。
"""
import json
import os
import pathlib
import signal
import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

import yaml as _yaml

# 绕过 manager.py 模块级 venv 存在性检查(非容器环境无 /venv_yolov5;容器内该检查本就通过,patch 无害)
with patch.object(pathlib.Path, "exists", return_value=True):
    from train_v.manager import TrainingManager, TrainingRecord, _MAX_TERMINAL_RECORDS
    from train_v.schemas import TrainStartRequest


def _fresh_mgr(tmp_path):
    m = TrainingManager()
    m.run_dir = tmp_path / "runs" / "train"
    m.run_dir.mkdir(parents=True, exist_ok=True)
    m.trainings = {}
    m.transport_yaml_files = []
    return m


def _rec(mgr, exp, gpu_id=None, alive=True, version="v8", yamls=None):
    p = MagicMock()
    p.poll.return_value = None if alive else 0
    p.pid = 123
    rec = TrainingRecord(exp=exp, process=p, gpu_id=gpu_id, version=version,
                         python_bin="/usr/bin/python",
                         status_file=mgr.run_dir / exp / "status.json",
                         log_file=mgr.run_dir / exp / "train.log",
                         transport_yaml_files=list(yamls or []))
    mgr.trainings[exp] = rec
    return rec


def _write_status(mgr, exp, **kwargs):
    exp_dir = mgr.run_dir / exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "status.json").write_text(json.dumps(kwargs, ensure_ascii=False), encoding="utf-8")


def _gpu(gpu_id, free):
    g = MagicMock()
    g.id = gpu_id
    g.memoryFree = free
    return g


# ===== 多槽基础 =====

def test_reap_dead_returns_active_and_cleans_yaml(tmp_path):
    """活跃返回、已退出记录一次性清理 transport yaml 且幂等。"""
    mgr = _fresh_mgr(tmp_path)
    y1 = tmp_path / "a_transport.yaml"; y1.write_text("", encoding="utf-8")
    y2 = tmp_path / "b_transport.yaml"; y2.write_text("", encoding="utf-8")
    _rec(mgr, "exp_live", gpu_id=0, alive=True)
    _rec(mgr, "exp_dead", gpu_id=1, alive=False, yamls=[str(y1), str(y2)])
    active = mgr._reap_dead()
    assert [r.exp for r in active] == ["exp_live"]
    assert not y1.exists() and not y2.exists()
    mgr._reap_dead()   # 幂等:二次调用不抛
    assert mgr.trainings["exp_dead"].transport_yaml_files == []


def test_latest_active_returns_last_started(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_a", gpu_id=0, alive=True)
    _rec(mgr, "exp_b", gpu_id=1, alive=True)
    _rec(mgr, "exp_c", gpu_id=None, alive=False)   # dict 插入序即启动序,倒序找活跃
    assert mgr._latest_active().exp == "exp_b"
    mgr.trainings["exp_b"].process.poll.return_value = 0
    assert mgr._latest_active().exp == "exp_a"


def test_unique_exp_name_suffix_on_collision(tmp_path):
    """同秒/同名(job_id)重名(记录在/桶内磁盘在)追加序号,防 trainings 键覆盖致真双占。

    exp 名 = exp{yyyyMMddHHmmss}_{job_id};年月桶内检测磁盘残留(_2)。"""
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp20260825000000_jobA", gpu_id=0, alive=True)
    (mgr.run_dir / "202608" / "exp20260825000000_jobA_2").mkdir(parents=True)
    with patch("train_v.manager.datetime") as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "20260825000000"   # strftime('%Y%m%d%H%M%S')
        assert mgr._unique_exp_name("jobA") == "exp20260825000000_jobA_3"
        # 无冲突分支也须在 mock 内验证(patch 块外用真实 datetime 会对硬编码值误断言)
        assert mgr._unique_exp_name("jobB") == "exp20260825000000_jobB"


def test_singleton_registers_atexit():
    """首建单例注册 atexit(_shutdown_all):uvicorn 启动后会用自己的退出 handler 替换
    SIGINT/SIGTERM 注册,signal_handler 生产不可达;atexit 在 uvicorn 优雅退出
    (run() 返回→解释器退出)时必经,是清理训练子进程组的可靠路径。"""
    saved = TrainingManager._instance
    TrainingManager._instance = None
    try:
        with patch("train_v.manager.atexit.register") as mock_reg:
            m = TrainingManager()
        mock_reg.assert_called_once()
        assert mock_reg.call_args.args[0] == m._shutdown_all   # 注册的是本实例的兜底
    finally:
        TrainingManager._instance = saved                      # 恢复原单例,不污染后续用例


def test_reap_persists_version_gpu_id(tmp_path):
    """进程退出后 reap 把 version/gpu_id 落盘:子进程回调覆盖写不携带两字段,
    不落盘则服务重启后(内存记录丢失)历史 exp 响应永久缺失。"""
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_a", gpu_id=2, alive=False, version="v5")
    _write_status(mgr, "exp_a", status="finished")            # 终态文件不含 version/gpu_id
    mgr._reap_dead()
    data = json.loads((mgr.run_dir / "exp_a" / "status.json").read_text(encoding="utf-8"))
    assert data["version"] == "v5" and data["gpu_id"] == 2
    assert data["status"] == "finished"                       # 只补字段,不改状态
    mgr._reap_dead()                                          # 幂等:二次调用不抛不破坏
    assert json.loads((mgr.run_dir / "exp_a" / "status.json").read_text(encoding="utf-8"))["gpu_id"] == 2


def test_status_list_reconcile_keeps_persisted_fields(tmp_path):
    """总览路径:reap 先 persist 落盘,随后 reconcile 补写 error 时两字段须保留
    (status.json 是服务重启后历史 exp 的唯一字段来源)。"""
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_dead", gpu_id=1, alive=False, version="v11")
    _write_status(mgr, "exp_dead", status="running")
    result = mgr.status()
    persisted = json.loads((mgr.run_dir / "exp_dead" / "status.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "error"
    assert persisted["version"] == "v11" and persisted["gpu_id"] == 1


def test_reap_prunes_terminal_records_beyond_cap(tmp_path):
    """全局封顶兜底:终态分布在互不相同的卡上(每卡各 1 条,不触发按卡去重)时,
    超出 _MAX_TERMINAL_RECORDS 的最旧终态仍出列,活跃全留;
    出列 exp 的指定查询仍走磁盘文件(仅总览列表不再展示)。"""
    mgr = _fresh_mgr(tmp_path)
    for i in range(_MAX_TERMINAL_RECORDS + 5):
        gpu_id = i + 1                                        # 各卡不同,避开 exp_live 的卡 0
        _rec(mgr, f"exp_dead_{i:02d}", gpu_id=gpu_id, alive=False)
        _write_status(mgr, f"exp_dead_{i:02d}", status="finished")
    _rec(mgr, "exp_live", gpu_id=0, alive=True)
    _write_status(mgr, "exp_live", status="running")
    active = mgr._reap_dead()
    assert [r.exp for r in active] == ["exp_live"]
    assert len(mgr.trainings) == _MAX_TERMINAL_RECORDS + 1     # 活跃1 + 终态20
    for i in range(5):                                         # 最旧的 5 条出列
        assert f"exp_dead_{i:02d}" not in mgr.trainings
    assert mgr.status("exp_dead_00")["status"] == "finished"   # 指定查询不受影响


# ===== GPU 分配 =====

async def test_pick_no_gpus(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    with patch("train_v.manager.GPUtil.getGPUs", return_value=[]):
        gpu_id, err = await mgr._pick_free_gpu(1000)
    assert gpu_id is None and err == "无可用 GPU"


async def test_pick_all_busy(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_a", gpu_id=0, alive=True)
    with patch("train_v.manager.GPUtil.getGPUs", return_value=[_gpu(0, 8000)]):
        gpu_id, err = await mgr._pick_free_gpu(1000)
    assert gpu_id is None and err == "所有 GPU 都在训练，无空闲卡"


async def test_pick_budget_insufficient(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    with patch("train_v.manager.GPUtil.getGPUs", return_value=[_gpu(0, 100)]):
        gpu_id, err = await mgr._pick_free_gpu(1000)
    assert gpu_id is None and "无空闲 GPU 满足显存预算 1000 MB" == err


async def test_pick_ok_picks_freest_ignores_cpu_record(tmp_path):
    """CPU 记录(gpu_id=None)不占卡;选剩余最大的空闲卡。"""
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_cpu", gpu_id=None, alive=True)
    _rec(mgr, "exp_a", gpu_id=1, alive=True)
    with patch("train_v.manager.GPUtil.getGPUs", return_value=[_gpu(0, 4000), _gpu(1, 8000), _gpu(2, 200)]):
        gpu_id, err = await mgr._pick_free_gpu(1000)
    assert (gpu_id, err) == (0, None)


async def test_pick_gputil_exception_to_errmsg(tmp_path):
    """GPUtil 异常 -> errMsg,不抛(start 不变 500)。"""
    mgr = _fresh_mgr(tmp_path)
    with patch("train_v.manager.GPUtil.getGPUs", side_effect=RuntimeError("boom")):
        gpu_id, err = await mgr._pick_free_gpu(1000)
    assert gpu_id is None and "查询 GPU 失败" in err


async def test_pick_autobatch_none_budget_skips_filter(tmp_path):
    """budget=None(autobatch)只看占用,不按显存过滤。"""
    mgr = _fresh_mgr(tmp_path)
    with patch("train_v.manager.GPUtil.getGPUs", return_value=[_gpu(0, 100)]):
        gpu_id, err = await mgr._pick_free_gpu(None)
    assert (gpu_id, err) == (0, None)


# ===== start(多槽) =====

def _start_patches(mgr, gpus, getsize=5_000_000):
    """start 公共打桩:模型路径直通/权重大小/GPU 列表/Popen/yaml 改写。"""
    popen = MagicMock()
    popen.return_value.pid = 999
    popen.return_value.poll.return_value = None
    return popen, [
        patch("train_v.manager.format_yolo_modelpath", side_effect=lambda v, m, n: m),
        patch("train_v.manager.os.path.getsize", return_value=getsize),
        patch("train_v.manager.GPUtil.getGPUs", return_value=gpus),
        patch("train_v.manager.subprocess.Popen", popen),
        patch.object(TrainingManager, "transport_yaml_path", return_value="/fake/_transport.yaml"),
    ]


async def test_start_auto_assigns_gpu_and_records(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    popen, patches = _start_patches(mgr, [_gpu(0, 8000), _gpu(1, 4000)])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await mgr.start(TrainStartRequest(version="v8", data="/fake.yaml",
                                                   epochs=10, model="m.pt",
                                                   params={"batch": 16, "imgsz": 640}))
    assert result["gpu_id"] == 0                      # 选剩余最大的空闲卡
    rec = mgr.trainings[result["exp"]]
    assert rec.gpu_id == 0 and rec.version == "v8"
    cmd = popen.call_args.args[0]
    assert cmd[cmd.index("--device") + 1] == "0"       # device 下传子进程
    starting = json.loads(rec.status_file.read_text(encoding="utf-8"))
    assert starting["status"] == "starting" and starting["gpu_id"] == 0   # starting 即带 gpu_id


async def test_start_rejects_when_all_busy(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_a", gpu_id=0, alive=True)
    popen, patches = _start_patches(mgr, [_gpu(0, 8000)])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        try:
            await mgr.start(TrainStartRequest(version="v8", data="/d.yaml", epochs=1, model="m.pt"))
            assert False, "应拒绝"
        except RuntimeError as e:
            assert "所有 GPU 都在训练" in str(e)
        popen.assert_not_called()                     # 拒绝时不启子进程


async def test_start_budget_insufficient_rejected(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    popen, patches = _start_patches(mgr, [_gpu(0, 100)], getsize=50_000_000)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        try:
            await mgr.start(TrainStartRequest(version="v8", data="/d.yaml", epochs=1, model="m.pt"))
            assert False, "应拒绝"
        except RuntimeError as e:
            assert "显存预算" in str(e)


async def test_start_no_gpu_machine(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    popen, patches = _start_patches(mgr, [])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        try:
            await mgr.start(TrainStartRequest(version="v8", data="/d.yaml", epochs=1, model="m.pt"))
            assert False, "应拒绝"
        except RuntimeError as e:
            assert str(e) == "无可用 GPU"


async def test_start_explicit_gpu_busy_rejected(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_a", gpu_id=0, alive=True)
    popen, patches = _start_patches(mgr, [_gpu(0, 8000)])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        try:
            await mgr.start(TrainStartRequest(version="v8", data="/d.yaml", epochs=1,
                                              model="m.pt", params={"device": 0}))
            assert False, "应拒绝"
        except RuntimeError as e:
            assert "GPU 0 正在训练中" in str(e) and "exp_a" in str(e)


async def test_start_explicit_gpu_ok(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_a", gpu_id=1, alive=True)
    popen, patches = _start_patches(mgr, [_gpu(0, 8000), _gpu(1, 8000)])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await mgr.start(TrainStartRequest(version="v8", data="/d.yaml", epochs=1,
                                                   model="m.pt", params={"device": "0"}))
    assert result["gpu_id"] == 0


async def test_start_cpu_slot_occupied_and_ok(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_cpu", gpu_id=None, alive=True)
    popen, patches = _start_patches(mgr, [_gpu(0, 8000)])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        try:
            await mgr.start(TrainStartRequest(version="v8", data="/d.yaml", epochs=1,
                                              model="m.pt", params={"device": "cpu"}))
            assert False, "应拒绝"
        except RuntimeError as e:
            assert str(e) == "CPU 训练已占用"
    mgr.trainings["exp_cpu"].process.poll.return_value = 0   # CPU 槽空出
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await mgr.start(TrainStartRequest(version="v8", data="/d.yaml", epochs=1,
                                                   model="m.pt", params={"device": "cpu"}))
    assert result["gpu_id"] is None
    cmd = popen.call_args.args[0]
    assert cmd[cmd.index("--device") + 1] == "cpu"


def test_estimate_budget_default_amp_true(tmp_path):
    """_estimate_budget 无 amp 时默认按 amp=True(FP16 激活减半)估预算,与 train_v8
    实际默认一致(amp 默认开后,选卡预算若仍按 FP32 会虚高一倍误拒本可装下的训练)。"""
    mgr = _fresh_mgr(tmp_path)
    f = tmp_path / "m.pt"; f.write_bytes(b"\0" * 5_000_000)
    with patch("train_v.manager.os.path.getsize", return_value=5_000_000):
        b_default = mgr._estimate_budget(str(f), {})
        b_true = mgr._estimate_budget(str(f), {"amp": True})
        b_false = mgr._estimate_budget(str(f), {"amp": False})
    assert b_default == b_true
    assert b_true < b_false, f"amp 默认 True 预算应低于显式 False: {b_true} vs {b_false}"


async def test_start_v5_amp_rejected(tmp_path):
    """行为变化②(spec §九):v5+amp 从静默跳过(旧 manager 特判)改为提交侧报错——
    v5 无该参数(框架 check_amp 自动探测),静默忽略会造成"设置了却不生效"的虚假预期。"""
    mgr = _fresh_mgr(tmp_path)
    popen, patches = _start_patches(mgr, [_gpu(0, 8000)])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        with pytest.raises(RuntimeError, match="amp"):
            await mgr.start(TrainStartRequest(version="v5", data="/d.yaml", epochs=1,
                                              model="m.pt", params={"amp": True}))
    assert popen.call_count == 0                    # 校验在 Popen 前,零残留零启动
    assert mgr.trainings == {}


async def test_start_v5_batch_maps_to_batch_size(tmp_path):
    """行为变化④配套(spec §5.2):batch→--batch-size 显式映射(现状靠 argparse 前缀匹配
    碰巧生效,显式化防缩写匹配脆弱性——非 bug 修复,行为不变)。"""
    mgr = _fresh_mgr(tmp_path)
    popen, patches = _start_patches(mgr, [_gpu(0, 8000)])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        await mgr.start(TrainStartRequest(version="v5", data="/d.yaml", epochs=1,
                                          model="m.pt", params={"batch": 8}))
    cmd = popen.call_args.args[0]
    assert cmd[cmd.index("--batch-size") + 1] == "8"
    assert "--batch" not in cmd                     # 旧形态不得残留(cmd.index("--batch") 会 ValueError)
    assert cmd[cmd.index("--weights") + 1] == "m.pt"


async def test_start_rejects_multi_gpu_device(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    popen, patches = _start_patches(mgr, [_gpu(0, 8000), _gpu(1, 8000)])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        try:
            await mgr.start(TrainStartRequest(version="v8", data="/d.yaml", epochs=1,
                                              model="m.pt", params={"device": "0,1"}))
            assert False, "应拒绝"
        except RuntimeError as e:
            assert str(e) == "暂不支持单训练多卡"


async def test_start_same_second_unique_exp(tmp_path):
    """同秒两次 start(job_id 相同):exp 名 exp{ts}_{job_id},第二个追加序号,trainings 两条不覆盖。"""
    mgr = _fresh_mgr(tmp_path)
    popen, patches = _start_patches(mgr, [_gpu(0, 8000), _gpu(1, 8000)])
    frozen = patch("train_v.manager.datetime")
    with patches[0], patches[1], patches[2], patches[3], patches[4], frozen as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "20260825000000"
        r1 = await mgr.start(TrainStartRequest(version="v8", data="/d.yaml", epochs=1, model="m.pt", job_id="jobA"))
        r2 = await mgr.start(TrainStartRequest(version="v8", data="/d.yaml", epochs=1, model="m.pt", job_id="jobA"))
    assert r1["exp"] == "exp20260825000000_jobA"
    assert r2["exp"] == "exp20260825000000_jobA_2"
    assert len(mgr.trainings) == 2


async def test_start_autobatch_skips_budget(tmp_path):
    """batch=-1(autobatch):不按显存过滤,空闲小显存卡也可选。"""
    mgr = _fresh_mgr(tmp_path)
    popen, patches = _start_patches(mgr, [_gpu(0, 100)], getsize=50_000_000)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await mgr.start(TrainStartRequest(version="v8", data="/d.yaml", epochs=1,
                                                   model="m.pt", params={"batch": -1}))
    assert result["gpu_id"] == 0


# ===== stop / shutdown =====

def test_stop_by_exp(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    y = tmp_path / "t.yaml"; y.write_text("", encoding="utf-8")
    rec = _rec(mgr, "exp_a", gpu_id=0, alive=True, yamls=[str(y)])
    _write_status(mgr, "exp_a", status="running")
    # getpgid 必须打桩:record 的 pid 是假值 123,Linux 真调用 os.getpgid(123) 会
    # ProcessLookupError/PermissionError 被 except 吞掉走 kill() 分支,断言随机器上是否恰有 pid 123 随机挂
    # create=True:Windows 上 os 为冻结模块、无 getpgid/killpg,须允许打桩(生产走 Linux 真实存在)
    with patch("train_v.manager.os.getpgid", return_value=123, create=True), \
            patch("train_v.manager.os.killpg", create=True) as killpg:
        result = mgr.stop("exp_a")
    assert result == {"status": "stopped", "exp": "exp_a"}
    killpg.assert_called_once()
    rec.process.wait.assert_called_once_with(timeout=15)
    persisted = json.loads(rec.status_file.read_text(encoding="utf-8"))
    assert persisted["status"] == "stopped"
    assert not y.exists()


def test_stop_unknown_or_finished_rejected(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    for exp in ("exp_none", "exp_done"):
        try:
            mgr.stop(exp)
            assert False, "应拒绝"
        except RuntimeError as e:
            assert "不存在或已结束" in str(e)
    _rec(mgr, "exp_done", gpu_id=0, alive=False)
    try:
        mgr.stop("exp_done")
        assert False, "应拒绝"
    except RuntimeError:
        pass


def test_stop_no_arg_stops_latest_active(tmp_path):
    """无参 = 停最后启动的活跃训练(存量 API 兼容)。"""
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_a", gpu_id=0, alive=True)
    _rec(mgr, "exp_b", gpu_id=1, alive=True)
    with patch("train_v.manager.os.getpgid", return_value=123, create=True), \
            patch("train_v.manager.os.killpg", create=True):
        result = mgr.stop()
    assert result["exp"] == "exp_b"


def test_stop_no_active_uses_existing_errmsg(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    try:
        mgr.stop()
        assert False, "应拒绝"
    except RuntimeError as e:
        assert str(e) == "没有正在运行的任务"      # 沿用现有文案,不得新造


def test_stop_fallback_killpg_on_wait_timeout(tmp_path):
    """wait 超时(主线程卡在 CUDA C 扩展,SIGTERM handler 无法执行)-> 兜底
    killpg(SIGKILL) 整组击杀:只 kill 组长会残留 dataloader worker 持有 CUDA context。"""
    mgr = _fresh_mgr(tmp_path)
    rec = _rec(mgr, "exp_a", gpu_id=0, alive=True)
    _write_status(mgr, "exp_a", status="running")
    rec.process.wait.side_effect = subprocess.TimeoutExpired("cmd", 15)
    with patch("train_v.manager.os.getpgid", return_value=123, create=True), \
            patch("train_v.manager.os.killpg", create=True) as killpg, \
            patch("train_v.manager.signal.SIGKILL", 9, create=True):
        result = mgr.stop("exp_a")
    assert result["status"] == "stopped"
    assert killpg.call_args_list == [call(123, signal.SIGTERM), call(123, 9)]  # SIGTERM 后 SIGKILL 兜底


def test_shutdown_all_kills_each_without_serial_wait(tmp_path):
    """shutdown:逐个 killpg + 统一短等待,不逐个 wait(15)。

    覆盖等价性注:signal_handler 仅 try/except 包装 _shutdown_all,直接测 _shutdown_all 等价;
    僵尸回收经 _reap_dead 行为断言(遍历即调 poll)间接覆盖。"""
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_a", gpu_id=0, alive=True)
    _rec(mgr, "exp_b", gpu_id=1, alive=True)
    _rec(mgr, "exp_c", gpu_id=None, alive=False)   # 已死的不处理
    # signal.SIGKILL=9:Windows 上无 SIGKILL 常量,打桩补齐(Linux 真值即 9,语义无差)
    with patch("train_v.manager.signal.SIGKILL", 9, create=True), \
            patch("train_v.manager.os.getpgid", return_value=123, create=True), \
            patch("train_v.manager.os.killpg", create=True) as killpg:
        mgr._shutdown_all()
    assert killpg.call_count == 2
    for rec in (mgr.trainings["exp_a"], mgr.trainings["exp_b"]):
        for call in rec.process.wait.call_args_list:
            assert call.kwargs.get("timeout", 0) <= 5   # 无 15s 串行等待


# ===== status / log =====

def test_status_exp_attaches_record_fields(tmp_path):
    """响应中的 version/gpu_id 由记录补挂(status.json 内可能被回调覆盖丢失)。"""
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_a", gpu_id=1, alive=True)
    _write_status(mgr, "exp_a", status="running", epoch=3)   # 文件内无 version/gpu_id
    result = mgr.status("exp_a")
    assert result["status"] == "running" and result["version"] == "v8" and result["gpu_id"] == 1


def test_status_exp_unknown(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    assert mgr.status("exp_none") == {"status": "unknown"}


def test_status_exp_reconcile_dead_residual_running(tmp_path):
    """记录存在+进程死+残留 running -> 补写 error 并持久化(按记录粒度)。"""
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_dead", gpu_id=0, alive=False)
    _write_status(mgr, "exp_dead", status="running")
    result = mgr.status("exp_dead")
    assert result["status"] == "error"
    persisted = json.loads((mgr.run_dir / "exp_dead" / "status.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "error"


def test_status_exp_no_record_history_passthrough(tmp_path):
    """无记录的历史 exp(跨主进程重启)不兜底,原样返回。"""
    mgr = _fresh_mgr(tmp_path)
    _write_status(mgr, "exp_hist", status="running")
    result = mgr.status("exp_hist")
    assert result["status"] == "running"
    assert "version" not in result and "gpu_id" not in result


def test_status_list_contract(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_old", gpu_id=None, alive=False); _write_status(mgr, "exp_old", status="finished")
    _rec(mgr, "exp_new", gpu_id=0, alive=True); _write_status(mgr, "exp_new", status="running", epoch=1)
    result = mgr.status()
    assert result["trainings"][0]["exp"] == "exp_new"          # 最近优先
    assert result["trainings"][1]["exp"] == "exp_old"
    assert result["gpu_occupancy"] == {"0": "exp_new"}          # 仅占用项,cpu 死记录不算
    assert result["cpu_busy"] is False


def test_status_list_empty(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    assert mgr.status() == {"trainings": [], "gpu_occupancy": {}, "cpu_busy": False}


def test_status_list_cpu_busy(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_cpu", gpu_id=None, alive=True)
    _write_status(mgr, "exp_cpu", status="running")
    assert mgr.status()["cpu_busy"] is True


def test_log_none_uses_latest_active(tmp_path):
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_a", gpu_id=0, alive=True)
    (mgr.run_dir / "exp_a").mkdir(parents=True, exist_ok=True)
    (mgr.run_dir / "exp_a" / "train.log").write_text("LOG_A", encoding="utf-8")
    _rec(mgr, "exp_b", gpu_id=1, alive=False)
    assert mgr.log() == "LOG_A"
    assert mgr.log("exp_b") == ""


# ===== start 前全量兜底 =====

async def test_start_reconciles_all_stale_before_alloc(tmp_path):
    """start 前:多条进程已死+残留 running 的记录全部补写 error。"""
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_d1", gpu_id=0, alive=False); _write_status(mgr, "exp_d1", status="running")
    _rec(mgr, "exp_d2", gpu_id=1, alive=False); _write_status(mgr, "exp_d2", status="starting")
    popen, patches = _start_patches(mgr, [])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        try:
            await mgr.start(TrainStartRequest(version="v8", data="/d.yaml", epochs=1, model="m.pt"))
        except RuntimeError:
            pass   # 0 卡被拒也无妨,兜底已执行
    for exp in ("exp_d1", "exp_d2"):
        persisted = json.loads((mgr.run_dir / exp / "status.json").read_text(encoding="utf-8"))
        assert persisted["status"] == "error"


# ===== transport yaml 按 exp 隔离 =====

def test_transport_yaml_written_into_out_dir(tmp_path):
    """out_dir:transport yaml 写入各自 exp_dir(文件名仅由源路径决定,同数据集并行训练
    若共用一份,先退出者的清理会删掉后来者尚未读取的文件)。"""
    mgr = _fresh_mgr(tmp_path)
    src = tmp_path / "data.yaml"
    src.write_text(_yaml.dump({"train": "images/train", "val": "images/val", "names": {0: "a"}}),
                   encoding="utf-8")
    exp_a = mgr.run_dir / "expA"; exp_a.mkdir(parents=True)
    exp_b = mgr.run_dir / "expB"; exp_b.mkdir(parents=True)
    try:
        out_a = mgr.transport_yaml_path(str(src), exp_a)
        mgr.transport_yaml_files = []                       # start 成功后会转入 record,这里手动清
        out_b = mgr.transport_yaml_path(str(src), exp_b)
        mgr.transport_yaml_files = []
        assert pathlib.Path(out_a) == exp_a / "data_transport.yaml"
        assert pathlib.Path(out_b) == exp_b / "data_transport.yaml"   # 各自一份,不共享
        assert pathlib.Path(out_a).exists() and pathlib.Path(out_b).exists()
        data = _yaml.safe_load(pathlib.Path(out_a).read_text(encoding="utf-8"))
        assert os.path.isabs(data["train"])                 # 路径改写仍生效(绝对化)
    finally:
        mgr.transport_yaml_files = []


# ===== 每卡最多保留一条记录(终态按卡去重) =====

def test_reap_dead_gpu_with_active_keeps_active_only(tmp_path):
    """卡 0 当前有活跃训练:该卡的历史终态记录全部清理,列表只留活跃那条。"""
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_a", gpu_id=0, alive=False)
    _rec(mgr, "exp_b", gpu_id=0, alive=False)
    _rec(mgr, "exp_c", gpu_id=0, alive=True)   # 卡 0 当前活跃
    _rec(mgr, "exp_d", gpu_id=1, alive=False)
    active = mgr._reap_dead()
    assert [r.exp for r in active] == ["exp_c"]
    assert "exp_a" not in mgr.trainings and "exp_b" not in mgr.trainings
    assert "exp_c" in mgr.trainings                        # 活跃保留
    assert "exp_d" in mgr.trainings                        # 空闲卡保留最新终态


def test_reap_dead_idle_gpu_keeps_latest_terminal_only(tmp_path):
    """卡空闲:同卡多条终态记录,只保留最新启动的一条(旧终态清理)。"""
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_old", gpu_id=2, alive=False)
    _rec(mgr, "exp_mid", gpu_id=2, alive=False)
    _rec(mgr, "exp_new", gpu_id=2, alive=False)
    mgr._reap_dead()
    assert list(mgr.trainings.keys()) == ["exp_new"]


def test_reap_dead_cpu_slot_keeps_one(tmp_path):
    """CPU 槽(gpu_id=None)同样最多保留一条记录(与 GPU 卡一致)。"""
    mgr = _fresh_mgr(tmp_path)
    _rec(mgr, "exp_cpu_old", gpu_id=None, alive=False)
    _rec(mgr, "exp_cpu_new", gpu_id=None, alive=False)
    mgr._reap_dead()
    assert list(mgr.trainings.keys()) == ["exp_cpu_new"]


# ===== exp 名格式 & 年月分桶(exp{yyyyMMddHHmmss}_{job_id}) =====

def test_exp_dir_buckets_timestamped(tmp_path):
    """新格式 exp 名按名内时间戳前 6 位年月分桶 runs/train/YYYYMM/exp,防单层膨胀。"""
    mgr = _fresh_mgr(tmp_path)
    assert mgr._exp_dir("exp20260907164512_jobA") == mgr.run_dir / "202609" / "exp20260907164512_jobA"
    assert mgr._exp_dir("exp20260907164512_jobA_2") == mgr.run_dir / "202609" / "exp20260907164512_jobA_2"


def test_exp_dir_flat_for_non_timestamped(tmp_path):
    """非时间戳命名(旧版 exp_... / 测试夹具)保持平铺,不被误算进桶目录。"""
    mgr = _fresh_mgr(tmp_path)
    assert mgr._exp_dir("exp_live") == mgr.run_dir / "exp_live"
    assert mgr._exp_dir("exp_20251106_034929") == mgr.run_dir / "exp_20251106_034929"


def test_unique_exp_name_embeds_job_id(tmp_path):
    """exp 名 = exp{ts}_{job_id};yyyyMM 来自名内时间戳。"""
    mgr = _fresh_mgr(tmp_path)
    with patch("train_v.manager.datetime") as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "20260907164512"
        name = mgr._unique_exp_name("abc123")
    assert name == "exp20260907164512_abc123"
    assert mgr._exp_dir(name) == mgr.run_dir / "202609" / name


def test_status_log_read_from_bucketed(tmp_path):
    """新训练落桶后,status/log 按 exp 名内年月读取(不依赖查询当月)。"""
    mgr = _fresh_mgr(tmp_path)
    exp = "exp20260907164512_jobA"
    bucket = mgr.run_dir / "202609" / exp
    bucket.mkdir(parents=True)
    (bucket / "status.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    (bucket / "train.log").write_text("epoch progress", encoding="utf-8")
    assert mgr._read_status_file(exp)["status"] == "running"
    assert mgr.log(exp) == "epoch progress"


async def test_start_v5_exp_dir_bucketed(tmp_path):
    """v5 start 同样走年月分桶:exp 名 exp{ts}_{job_id},--project 指向 run_dir/YYYYMM/exp,start 实际建桶目录。"""
    mgr = _fresh_mgr(tmp_path)
    popen, patches = _start_patches(mgr, [_gpu(0, 8000)])
    frozen = patch("train_v.manager.datetime")
    with patches[0], patches[1], patches[2], patches[3], patches[4], frozen as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "20260825000000"
        result = await mgr.start(TrainStartRequest(version="v5", data="/d.yaml", epochs=1,
                                                   model="m.pt", job_id="jobV5"))
    assert result["exp"] == "exp20260825000000_jobV5"
    cmd = popen.call_args.args[0]
    assert cmd[cmd.index("--project") + 1] == str(mgr.run_dir / "202608" / "exp20260825000000_jobV5")
    assert (mgr.run_dir / "202608" / "exp20260825000000_jobV5").is_dir()
