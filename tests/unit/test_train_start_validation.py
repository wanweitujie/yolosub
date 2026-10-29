# tests/unit/test_train_start_validation.py
"""manager 侧新能力单测:_exp_dir 平铺回退守卫、start 参数校验与 v5 分流、
metrics_query 与 status 终态摘要。依赖 manager 的用例须在文件头部打桩(见下)。"""
import json
import pathlib
import pytest
from unittest.mock import MagicMock, patch

# 打桩前置:manager 模块级 Path(YOLOV5_PYTHON_PATH).exists() 在 Windows 直接 RuntimeError,
# 不打桩会中断整轮 pytest 收集(照抄 test_train_manager_multigpu.py:15-18 既有约定)
with patch.object(pathlib.Path, "exists", return_value=True):
    from train_v.manager import TrainingManager

from train_v.schemas import TrainStartRequest


def _fresh_mgr(tmp_path):
    """与 multigpu._start_patches 文件内的 _fresh_mgr 同款:单例重置目录与状态。"""
    mgr = TrainingManager()
    mgr.run_dir = tmp_path / "runs" / "train"
    mgr.run_dir.mkdir(parents=True, exist_ok=True)
    mgr.trainings = {}
    mgr.transport_yaml_files = []
    return mgr


class TestExpDirFlatFallback:
    """守卫三条(spec §七):平铺存在才回退 / 读写分离 / 新名仍返回分桶。"""

    def test_new_exp_name_returns_bucketed_when_both_missing(self, tmp_path):
        """新 exp 名(两路径都不存在)→ 仍返回分桶路径(无守卫的字面回退会让所有新训练落平铺)。"""
        mgr = _fresh_mgr(tmp_path)
        assert mgr._exp_dir("exp20260928120000_jobA", allow_flat_fallback=True) \
            == mgr.run_dir / "202609" / "exp20260928120000_jobA"

    def test_fallback_hits_flat_when_bucket_missing_flat_exists(self, tmp_path):
        """时间戳命名且平铺目录存在(2026-01-05~09-07 窗口产物)→ 回退命中平铺。"""
        mgr = _fresh_mgr(tmp_path)
        exp = "exp20260301120000_old"
        (mgr.run_dir / exp).mkdir(parents=True)
        assert mgr._exp_dir(exp, allow_flat_fallback=True) == mgr.run_dir / exp

    def test_write_path_default_no_fallback(self, tmp_path):
        """读写分离:默认参数(写路径:start mkdir/_unique_exp_name 判重)不走回退——
        即使平铺路径已存在,仍返回分桶,防判重/落盘错位。"""
        mgr = _fresh_mgr(tmp_path)
        exp = "exp20260301120000_old"
        (mgr.run_dir / exp).mkdir(parents=True)
        assert mgr._exp_dir(exp) == mgr.run_dir / "202603" / exp


def _gpu_env(mgr, free_mb=8000):
    """start 公共打桩:与 multigpu._start_patches 同款 5 桩(本文件不 import multigpu,独立复制)。"""
    popen = MagicMock(); popen.return_value.pid = 999; popen.return_value.poll.return_value = None
    gpu = type("G", (), {"id": 0, "memoryFree": free_mb})()
    patches = [
        patch("train_v.manager.format_yolo_modelpath", side_effect=lambda v, m, n: m),
        patch("train_v.manager.os.path.getsize", return_value=5_000_000),
        patch("train_v.manager.GPUtil.getGPUs", return_value=[gpu]),
        patch("train_v.manager.subprocess.Popen", popen),
        patch.object(TrainingManager, "transport_yaml_path", return_value="/fake/_transport.yaml"),
    ]
    return popen, patches


class TestStartValidation:
    """start 提交侧校验(锚定在 pop device 之后、_unique_exp_name 之前,校验失败零残留)。"""

    @pytest.mark.asyncio
    async def test_v8_rejects_unknown_keys_no_exp_dir_leftover(self, tmp_path):
        mgr = _fresh_mgr(tmp_path)
        before = set(mgr.run_dir.rglob("*"))
        # format_yolo_modelpath 须打桩直通:校验锚点在 format 之后(device pop 行后),
        # 真实实现会对 "m.pt" 抛"文件不存在",永远到不了校验分支(简报漏此桩,实跑验证后补)
        with patch("train_v.manager.format_yolo_modelpath", side_effect=lambda v, m, n: m):
            with pytest.raises(RuntimeError, match="obj"):
                await mgr.start(TrainStartRequest(version="v8", data="/d.yaml", epochs=1,
                                                  model="m.pt", params={"obj": 1.0}))
        assert set(mgr.run_dir.rglob("*")) == before          # 无空 exp_dir 残留
        assert mgr.trainings == {}

    @pytest.mark.asyncio
    async def test_v8_rejects_params_model_data_epochs(self, tmp_path):
        """报错键9:model/data/epochs 防静默覆盖顶层字段(data 覆盖会绕过 transport yaml)。"""
        mgr = _fresh_mgr(tmp_path)
        # 同上:format 直通桩是到达校验分支的前置(见 test_v8_rejects_unknown_keys 注)
        with patch("train_v.manager.format_yolo_modelpath", side_effect=lambda v, m, n: m):
            for k in ("model", "data", "epochs", "resume", "exist_ok", "task", "mode", "cfg", "time"):
                with pytest.raises(RuntimeError, match=k):
                    await mgr.start(TrainStartRequest(version="v8", data="/d.yaml", epochs=1,
                                                      model="m.pt", params={k: 1}))

    @pytest.mark.asyncio
    async def test_v8_legacy_params_still_pass(self, tmp_path):
        """既有调用方零破坏:publisher 现行 params {batch,imgsz}(device 已 pop)不触发报错。
        amp=True 锚定 v8 bool argv 形态:非 str 值经 json.dumps 落 "true"(防改回 str() 出 "True")。"""
        mgr = _fresh_mgr(tmp_path)
        popen, patches = _gpu_env(mgr)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            r = await mgr.start(TrainStartRequest(version="v8", data="/d.yaml", epochs=1,
                                                   model="m.pt",
                                                   params={"batch": -1, "imgsz": 640, "amp": True}))
        cmd = popen.call_args.args[0]
        assert cmd[cmd.index("--batch") + 1] == "-1"          # v8 同名透传不变
        assert cmd[cmd.index("--amp") + 1] == "true"          # bool → "true"(json.dumps 形态)
        assert r["exp"] in mgr.trainings

    @pytest.mark.asyncio
    async def test_v5_bad_cache_no_residue(self, tmp_path):
        """cache 坏值前置拦截:零残留零启动(修 Important——拼装段在清理 try 外的缝隙)。"""
        mgr = _fresh_mgr(tmp_path)
        before = set(mgr.run_dir.rglob("*"))
        # format_yolo_modelpath 须打桩直通:校验锚点在 format 之后(device pop 行后),
        # 真实实现会对 "m.pt" 抛"文件不存在",永远到不了校验分支(与本类其余 reject 用例同款)
        with patch("train_v.manager.format_yolo_modelpath", side_effect=lambda v, m, n: m):
            with pytest.raises(RuntimeError, match="cache"):
                await mgr.start(TrainStartRequest(version="v5", data="/d.yaml", epochs=1,
                                                  model="m.pt", params={"cache": "ssd"}))
        assert set(mgr.run_dir.rglob("*")) == before
        assert mgr.trainings == {}


class TestV5Hyp:
    """v5 hyp yaml 生成/注入/清理(仅 params 含 hyp 键才生成;写在 transport 之前)。"""

    @pytest.mark.asyncio
    async def test_v5_hyp_keys_generate_yaml_and_inject(self, tmp_path):
        import yaml
        mgr = _fresh_mgr(tmp_path)
        popen, patches = _gpu_env(mgr)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            r = await mgr.start(TrainStartRequest(version="v5", data="/d.yaml", epochs=1,
                                                  model="m.pt", params={"lr0": 0.02, "batch": 4}))
        exp = r["exp"]
        cmd = popen.call_args.args[0]
        i = cmd.index("--hyp")
        hyp_file = pathlib.Path(cmd[i + 1])
        assert hyp_file.name == "_hyp.yaml"
        data = yaml.safe_load(hyp_file.read_text(encoding="utf-8"))
        assert data["lr0"] == 0.02                       # 用户键覆盖
        assert data["lrf"] == 0.01                        # scratch-low 默认为底
        assert cmd[cmd.index("--batch-size") + 1] == "4"
        assert str(hyp_file) in [str(p) for p in mgr.trainings[exp].transport_yaml_files]

    @pytest.mark.asyncio
    async def test_v5_no_hyp_keys_no_yaml_no_flag(self, tmp_path):
        """无 hyp 键不生成 _hyp.yaml、不传 --hyp(保持 v5 原生默认路径,spec §六.2)。"""
        mgr = _fresh_mgr(tmp_path)
        popen, patches = _gpu_env(mgr)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            await mgr.start(TrainStartRequest(version="v5", data="/d.yaml", epochs=1,
                                               model="m.pt", params={"batch": 4}))
        cmd = popen.call_args.args[0]
        assert "--hyp" not in cmd
        assert not list(mgr.run_dir.rglob("_hyp.yaml"))

    @pytest.mark.asyncio
    async def test_transport_failure_cleans_up(self, tmp_path):
        """写文件失败清理三步:已知路径 unlink + 清全局列表 + 删空 exp_dir(spec §六.5)。"""
        mgr = _fresh_mgr(tmp_path)
        popen, patches = _gpu_env(mgr)
        with patches[0], patches[1], patches[2], patches[3], \
             patch.object(TrainingManager, "transport_yaml_path", side_effect=FileNotFoundError("boom")):
            with pytest.raises(FileNotFoundError):
                await mgr.start(TrainStartRequest(version="v5", data="/d.yaml", epochs=1,
                                                  model="m.pt", params={"lr0": 0.02}))
        assert mgr.transport_yaml_files == []            # ② 全局列表已清
        assert not list(mgr.run_dir.rglob("*"))          # ① _hyp.yaml 已删 + ③ 空目录已删
        assert mgr.trainings == {}

    @pytest.mark.asyncio
    async def test_transport_half_written_file_cleaned(self, tmp_path):
        """dump 中途失败:已写出的半截 _transport.yaml 不在全局列表(append 在成功 return 前),
        按已知路径推导补删(spec §六.5①)。桩内真实 write_text 半截内容再 raise。"""
        mgr = _fresh_mgr(tmp_path)
        popen, patches = _gpu_env(mgr)

        def _half_write_then_raise(datapath, out_dir=None):
            # 与 transport_yaml_path 内部文件名表达式严格一致:Path(out_dir)/(stem + "_transport.yaml")
            p = pathlib.Path(out_dir) / (pathlib.Path(datapath).stem + "_transport.yaml")
            p.write_text("path: /data\ntrain: /data/images/train\n", encoding="utf-8")
            raise RuntimeError("yaml dump 中途失败")

        with patches[0], patches[1], patches[2], patches[3], \
             patch.object(TrainingManager, "transport_yaml_path", side_effect=_half_write_then_raise):
            with pytest.raises(RuntimeError, match="dump 中途失败"):
                await mgr.start(TrainStartRequest(version="v5", data="/d.yaml", epochs=1,
                                                  model="m.pt", params={"lr0": 0.02}))
        assert mgr.transport_yaml_files == []            # 半截文件没进过全局列表
        assert not list(mgr.run_dir.rglob("*"))          # 半截 transport + _hyp.yaml + 空目录全清
        assert mgr.trainings == {}


def _mk_exp(mgr, exp, status="finished", version="v8", csv_text=None, flat=False):
    """造一个已落盘的 exp(默认分桶;flat=True 造平铺窗口产物),返回 (exp, csv_path)。
    version=None 时不写 version 字段(仿平铺窗口历史产物:子进程回调覆盖写不带该字段)。"""
    d = mgr._exp_dir(exp, allow_flat_fallback=False) if not flat else mgr.run_dir / exp
    d.mkdir(parents=True, exist_ok=True)
    payload = {"status": status}
    if version is not None:
        payload["version"] = version
    (d / "status.json").write_text(json.dumps(payload), encoding="utf-8")
    out = d / "output"; out.mkdir(exist_ok=True)
    csvp = out / "results.csv"
    if csv_text is not None:
        csvp.write_text(csv_text, encoding="utf-8")
    return exp, csvp


V8_TWO_EPOCHS = ("epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
                 "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),"
                 "val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2\n"
                 "1,1.3,2.4,5.2,1.4,0,0,0,0,2.1,5.1,1.4,9e-05,9e-05,9e-05\n"
                 "2,3.4,2.4,5.1,1.6,0.5,0.4,0.6,0.35,2.1,5.1,1.4,9e-05,9e-05,9e-05\n")


class TestMetricsQuery:
    def test_unknown_exp_returns_exp_and_empty(self, tmp_path):
        mgr = _fresh_mgr(tmp_path)
        payload = mgr.metrics_query("exp_no_such")
        assert payload["status"] == "unknown" and payload["epochs"] == []
        assert payload["last_epoch"] == -1 and payload["exp"] == "exp_no_such"  # unknown 分支回填 exp

    def test_running_full_series_and_since_filter(self, tmp_path):
        mgr = _fresh_mgr(tmp_path)
        _mk_exp(mgr, "exp20260928120000_a", status="running", csv_text=V8_TWO_EPOCHS)
        full = mgr.metrics_query("exp20260928120000_a")
        assert [r["epoch"] for r in full["epochs"]] == [1, 2]
        assert full["last_epoch"] == 2 and "results" not in full   # running 无终态摘要
        inc = mgr.metrics_query("exp20260928120000_a", since_epoch=1)
        assert [r["epoch"] for r in inc["epochs"]] == [2]
        assert inc["last_epoch"] == 2                               # last_epoch 不受过滤影响

    def test_terminal_status_attaches_results(self, tmp_path):
        mgr = _fresh_mgr(tmp_path)
        _mk_exp(mgr, "exp20260928120000_b", status="finished", csv_text=V8_TWO_EPOCHS)
        payload = mgr.metrics_query("exp20260928120000_b")
        assert payload["results"]["best_epoch"] == 2                 # v8 fitness=map50_95,第2轮大
        assert payload["results"]["final_precision"] == 0.5

    def test_stopped_status_attaches_results(self, tmp_path):
        """stopped 终态同样挂 results(对称缺口补测:与 finished/error 同走 _TERMINAL_STATUSES)。"""
        mgr = _fresh_mgr(tmp_path)
        _mk_exp(mgr, "exp20260928120000_i", status="stopped", csv_text=V8_TWO_EPOCHS)
        assert "results" in mgr.metrics_query("exp20260928120000_i")

    def test_non_dict_status_json_returns_unknown(self, tmp_path):
        """status.json 为合法 JSON 但非对象(撕裂/手改,如 [])→ 按坏文件处理走 unknown 空态,
        不在 data["exp"] 补挂处抛 TypeError 破坏统一信封。"""
        mgr = _fresh_mgr(tmp_path)
        exp = "exp20260928120000_j"
        d = mgr._exp_dir(exp)
        d.mkdir(parents=True)
        (d / "status.json").write_text("[]", encoding="utf-8")
        assert mgr.status(exp) == {"status": "unknown"}
        payload = mgr.metrics_query(exp)
        assert payload["status"] == "unknown" and payload["epochs"] == []
        assert payload["last_epoch"] == -1 and payload["exp"] == exp

    def test_missing_version_falls_back_to_csv_generation(self, tmp_path):
        """version 回退链(修 Important):status.json 缺 version(平铺窗口历史产物)时按 csv
        表头判代际分派 fitness 公式——静默按 v8 算 v5 任务 best 会错行。数据取值让两公式
        结论不同:epoch1 map50=0.50/map50_95=0.50(v8 best);epoch2 map50=0.90/map50_95=0.48
        (v5 fitness=0.522>0.50,v5 best)。csv 内联(本文件无 v5 夹具构造器,不跨文件 import)。"""
        v5_csv = ("epoch,train/box_loss,train/obj_loss,train/cls_loss,metrics/precision,"
                  "metrics/recall,metrics/mAP_0.5,metrics/mAP_0.5:0.95,"
                  "val/box_loss,val/obj_loss,val/cls_loss,x/lr0,x/lr1,x/lr2\n"
                  "0,0.1,0.1,0.1,0.6,0.6,0.50,0.50,2.0,3.0,1.0,1e-4,1e-4,1e-4\n"
                  "1,0.1,0.1,0.1,0.6,0.6,0.90,0.48,2.0,3.0,1.0,1e-4,1e-4,1e-4\n")
        mgr = _fresh_mgr(tmp_path)
        # 场景1:无 version → detect_csv_generation 判 v5 → best=epoch2(v5 加权公式)
        _mk_exp(mgr, "exp20260928120000_g", status="finished", version=None, csv_text=v5_csv)
        results = mgr.metrics_query("exp20260928120000_g")["results"]
        assert results["best_epoch"] == 2 and results["best_map50"] == 0.9
        assert mgr.status("exp20260928120000_g")["results"]["best_epoch"] == 2  # status 单查同回退
        # 场景2(对照):显式 version="v8" 优先于表头判定 → best=epoch1(map50_95 argmax)
        _mk_exp(mgr, "exp20260928120000_h", status="finished", version="v8", csv_text=v5_csv)
        results = mgr.metrics_query("exp20260928120000_h")["results"]
        assert results["best_epoch"] == 1 and results["best_map50"] == 0.5

    def test_stale_running_reconciled_to_error_in_metrics(self, tmp_path):
        """S2 兜底:进程已死但 status.json 残留 running → metrics 与 status 一致报 error
        (防两接口报态矛盾误导调度层终态判定,spec §4.2)。"""
        mgr = _fresh_mgr(tmp_path)
        exp = "exp20260928120000_c"
        _mk_exp(mgr, exp, status="running", csv_text=V8_TWO_EPOCHS)
        popen = MagicMock(); popen.poll.return_value = 0              # 进程已退出
        from train_v.manager import TrainingRecord
        mgr.trainings[exp] = TrainingRecord(
            exp=exp, process=popen, gpu_id=0, version="v8", python_bin="py",
            status_file=mgr._exp_dir(exp) / "status.json",
            log_file=mgr._exp_dir(exp) / "train.log")
        payload = mgr.metrics_query(exp)
        assert payload["status"] == "error"
        assert "results" in payload                                  # 改判终态后也挂摘要

    def test_flat_window_exp_queryable(self, tmp_path):
        """平铺窗口产物(时间戳命名却平铺)metrics 可查(spec §九⑧ 回归)。"""
        mgr = _fresh_mgr(tmp_path)
        _mk_exp(mgr, "exp20260301120000_old", status="finished",
                csv_text=V8_TWO_EPOCHS, flat=True)
        payload = mgr.metrics_query("exp20260301120000_old")
        assert payload["status"] == "finished" and payload["last_epoch"] == 2

    def test_terminal_without_csv_no_results_key(self, tmp_path):
        mgr = _fresh_mgr(tmp_path)
        _mk_exp(mgr, "exp20260928120000_d", status="error", csv_text=None)   # 秒挂无 csv
        assert "results" not in mgr.metrics_query("exp20260928120000_d")


class TestStatusSummary:
    def test_status_single_exp_attaches_results(self, tmp_path):
        mgr = _fresh_mgr(tmp_path)
        _mk_exp(mgr, "exp20260928120000_e", status="finished", csv_text=V8_TWO_EPOCHS)
        data = mgr.status("exp20260928120000_e")
        assert data["results"]["best_epoch"] == 2

    def _mk_overview_rec(self, mgr, exp, alive=True, version="v8"):
        """造一条在 trainings 里的记录(总览遍历 trainings,不塞则断言平凡通过)。"""
        popen = MagicMock(); popen.poll.return_value = None if alive else 0
        from train_v.manager import TrainingRecord
        mgr.trainings[exp] = TrainingRecord(
            exp=exp, process=popen, gpu_id=0, version=version, python_bin="py",
            status_file=mgr._exp_dir(exp) / "status.json",
            log_file=mgr._exp_dir(exp) / "train.log")

    def test_status_overview_attaches_results(self, tmp_path):
        """总览元素也挂 results(与单查同条件:终态 + csv 有数据),字段取值与单查一致。"""
        mgr = _fresh_mgr(tmp_path)
        exp = "exp20260928120000_f"
        _mk_exp(mgr, exp, status="finished", csv_text=V8_TWO_EPOCHS)
        self._mk_overview_rec(mgr, exp)
        items = mgr.status()["trainings"]
        assert len(items) == 1                                    # 遍历非空,断言非平凡
        assert items[0]["results"] == mgr.status(exp)["results"]  # 与单查逐字段一致

    def test_status_overview_running_has_no_results(self, tmp_path):
        """门控守卫:总览里的 running 训练不挂 results(与单查/契约一致,勿随总览一起放开)。"""
        mgr = _fresh_mgr(tmp_path)
        exp = "exp20260928120000_k"
        _mk_exp(mgr, exp, status="running", csv_text=V8_TWO_EPOCHS)
        self._mk_overview_rec(mgr, exp)
        items = mgr.status()["trainings"]
        assert len(items) == 1
        assert "results" not in items[0]
