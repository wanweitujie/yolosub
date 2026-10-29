# -*- coding: utf-8 -*-
"""CUDA 11.8 → 12.8 升级的容器内运行时验证(slow 标记,需运行中的 YoloSub 容器)。

与单元侧静态版本串断言互补:
本文件在**容器内**验证实际安装版本与 cu128 对齐,并让 onnxruntime-gpu 在 CUDA
上真实跑一次导出推理。

- 探针命令通过 docker exec 在容器内执行(容器内才有 torch/onnxruntime),主机侧
  不 import 任何容器内依赖(仅 pytest/httpx/conftest)。
- 本环境无运行容器时,container fixture 自动 skip —— 预期 2 skipped,不是失败。
- 版本期望值须与 Dockerfile 中的版本串保持同步。
- export 产物压缩:onnx 是单文件,经 routes.py:102-110 的 compress_output_file_to_zip
  压缩成 .zip 后源 .onnx 被删 → /export_model 返回 data 是 .zip 文件路径;
  zip.py:54 单文件归档名为 "model.onnx"。测试 2 需先解压再定位 .onnx。
  解压用容器内 python zipfile(容器未装 unzip,见 Dockerfile apt 安装清单)。
"""
import pytest

from conftest import M_COCO_V11N, _shq, docker_exec


@pytest.fixture
def export_cleanup(container):
    """每个 export 测试后清理产物(目录),避免容器磁盘膨胀。参照 test_export.py 同款模式。"""
    produced = []
    yield produced.append
    for path in produced:
        docker_exec(container, f"rm -rf {path}", timeout=10)


@pytest.mark.slow
def test_container_runtime_versions(container):
    """容器内实际安装版本与 cu128 对齐,且 CUDAExecutionProvider 可用。"""
    cmd = "python -c " + _shq(
        "import torch,torchvision,onnxruntime,ultralytics;"
        "print(torch.__version__);print(torchvision.__version__);"
        "print(torch.version.cuda);print(torch.cuda.is_available());"
        "print(onnxruntime.__version__);"
        "print(onnxruntime.get_available_providers());"
        "print(ultralytics.__version__)")
    out = docker_exec(container, cmd)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    assert len(lines) >= 7, f"探针输出行数不足:{out}"
    assert lines[0].startswith("2.11.0")             # torch(官方 cu128 wheel 带 +cu128 本地版本后缀,如 2.11.0+cu128)
    assert lines[1].startswith("0.26.0")             # torchvision(官方 cu128 wheel 同样带 +cu128 后缀)
    assert lines[2].startswith("12.8")               # torch.version.cuda
    assert lines[3] == "True"                        # torch.cuda.is_available()
    assert lines[4] == "1.23.2"                      # onnxruntime-gpu
    assert "CUDAExecutionProvider" in lines[5]       # providers
    assert lines[6] == "8.4.121"                     # ultralytics


@pytest.mark.slow
def test_onnxruntime_cuda_execution(container, client, export_cleanup):
    """导出 yolo11n 为 onnx,用 onnxruntime CUDAExecutionProvider 真实跑一次推理。"""
    r = client.post("/export_model",
                    json={"model_path": M_COCO_V11N, "format": "onnx"}, timeout=300)
    assert r.json()["success"], r.text
    prod = r.json()["data"]          # onnx 单文件经 zip 压缩后源被删 → 实际是 .zip 文件路径
    export_cleanup(prod)
    # 解压 .zip 到固定临时目录,再从解压产物定位 .onnx(直接对 .zip find 找不到)。
    # 容器未装 unzip,用容器内 python zipfile 解压(zip.py:54 单文件归档名为 model.onnx)。
    extract_dir = "/tmp/cuda128_onnx_probe"
    unzip_cmd = "python -c " + _shq(
        "import zipfile;"
        f"zipfile.ZipFile({prod!r}).extractall({extract_dir!r})")
    docker_exec(container, f"rm -rf {extract_dir} && mkdir -p {extract_dir} && {unzip_cmd}")
    export_cleanup(extract_dir)
    onnx_path = docker_exec(container, f"find {extract_dir} -name '*.onnx' | head -1").strip()
    assert onnx_path.endswith(".onnx"), f"解压后未找到 onnx 产物:{prod}"
    probe = (
        "import onnxruntime as ort;"
        "import numpy as np;"
        f"s=ort.InferenceSession({onnx_path!r},"
        "providers=['CUDAExecutionProvider','CPUExecutionProvider']);"
        "print(s.get_providers());"
        "x=np.random.rand(*s.get_inputs()[0].shape).astype(np.float32);"
        "y=s.run(None,{s.get_inputs()[0].name:x});"
        "print([t.shape for t in y])")
    out = docker_exec(container, "python -c " + _shq(probe))
    assert "CUDAExecutionProvider" in out, f"onnxruntime 未启用 CUDA:{out}"
