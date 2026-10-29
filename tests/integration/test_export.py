"""export 全格式 parametrize(易错点:mlmodel/mlpackage/ncnn-standard 映射)。slow 标记。

⚠️ 按「格式→合适源权重」分组:
- mlmodel 强制 version=v5(export_v/routes.py:35-37)→ 走 export_v5/yolov5_convert_weight_to_coreml
  (面向 v5 结构权重),必须用 v5 源权重 M_COCO_V5N;用 v11 权重会给 v5 转换器 → 解析失败。
- ncnn-standard 需明确 version(routes.py:84-85 校验 version in v5/v8/v11)→ 必须带 "version":"v11"。
- 其余 6 格式(onnx/torchscript/ncnn/coreml/mlpackage/mlpackage-ios14)用 M_COCO_V11N 成立。
"""
import pytest

from conftest import M_COCO_V11N, M_COCO_V5N, docker_exec


def _assert_file_nonempty(container, path):
    """stat -c %s > 0(非仅存在)。"""
    size = docker_exec(container, f"stat -c %s {path} 2>/dev/null").strip()
    assert size.isdigit() and int(size) > 0, f"产物 {path} 不存在或为空"


def _assert_mlpackage_ios14_spec(container, zip_path):
    """解压产物 zip,容器内用 coremltools(ios14 venv)解析产物 .mlmodel(源 basename,
    不再重打包为 .mlpackage),断言 spec<=5 且为 pipeline(neuralnetwork NMS),
    防'导出成功但内容错'回归。
    注:docker_exec 走 `bash -c '<cmd>'`,不支持 stdin → 用 heredoc 落盘再执行。"""
    py = (
        "import sys, zipfile, glob, shutil\n"
        "src = sys.argv[1]\n"
        "dest = '/tmp/mlp_spec_' + ''.join(c for c in src if c.isalnum())\n"
        "shutil.rmtree(dest, ignore_errors=True)\n"
        "zipfile.ZipFile(src).extractall(dest)\n"
        "probe = glob.glob(dest + '/**/*.mlmodel', recursive=True)\n"
        "assert probe, 'no *.mlmodel in ' + src\n"
        "import coremltools as ct\n"
        "m = ct.models.MLModel(probe[0])\n"
        "spec = m._spec.specificationVersion\n"
        "typ = m._spec.WhichOneof('Type')\n"
        "assert spec <= 5, 'spec %d > 5' % spec\n"
        "assert typ == 'pipeline', 'type %s != pipeline' % typ\n"
        "print('MLP_SPEC_OK', spec, typ)\n"
    )
    cmd = (
        "cat > /tmp/mlp_spec_check.py << 'PYEOF'\n"
        + py
        + "PYEOF\n"
        + "/venv_export_mlpackage_ios14/bin/python /tmp/mlp_spec_check.py "
        + zip_path
    )
    out = docker_exec(container, cmd + " 2>&1", timeout=120)
    assert "MLP_SPEC_OK" in out, f"mlpackage-ios14 spec check failed:\n{out}"


def _assert_onnx_classes_txt(container, zip_path):
    """解压产物 zip,断言含 classes.txt:UTF-8 可解码、80 行、首行为 person。

    注意:文件无尾换行,80 个标签只有 79 个 '\\n',故不可用 `wc -l` 数行(会得 79)。
    沿用 _assert_mlpackage_ios14_spec 的 heredoc 落盘再执行模式(docker_exec 走 bash -c)。"""
    py = (
        "import sys, zipfile\n"
        "z = zipfile.ZipFile(sys.argv[1])\n"
        "assert 'classes.txt' in z.namelist(), z.namelist()\n"
        "t = z.read('classes.txt').decode('utf-8')   # 非 UTF-8 会在此抛错\n"
        "lines = t.split('\\n')\n"
        "assert len(lines) == 80, len(lines)\n"
        "assert lines[0] == 'person', lines[0]\n"
        "print('ONNX_CLASSES_OK', len(lines))\n"
    )
    cmd = (
        "cat > /tmp/onnx_classes_check.py << 'PYEOF'\n"
        + py
        + "PYEOF\n"
        + "python /tmp/onnx_classes_check.py "
        + zip_path
    )
    out = docker_exec(container, cmd + " 2>&1", timeout=60)
    assert "ONNX_CLASSES_OK" in out, f"onnx classes.txt check failed:\n{out}"


def resolve_product_path(resp_data):
    """export_model(JSON)响应 data 已是容器绝对路径(不走 /static/ 改写),直接返回。"""
    return resp_data


@pytest.fixture
def export_cleanup(container):
    """每个 export 测试后清理产物(.zip),避免磁盘膨胀。"""
    produced = []
    yield produced.append
    for path in produced:
        docker_exec(container, f"rm -rf {path}", timeout=10)


# (fmt, model_path, extra) —— extra 是 export 请求里除 model_path/format 外的补充字段
EXPORT_CASES = [
    ("onnx",          M_COCO_V11N, {}),
    ("torchscript",   M_COCO_V11N, {}),
    ("coreml",        M_COCO_V11N, {}),
    ("mlmodel",       M_COCO_V5N,  {}),   # 强制 v5,用 v5 源权重
    ("mlpackage",     M_COCO_V11N, {}),
    ("mlpackage-ios14", M_COCO_V11N, {}),
    ("ncnn-standard", M_COCO_V11N, {"version":"v11"}),   # 需明确 version,否则路由抛异常
    ("ncnn",          M_COCO_V11N, {}),
]


@pytest.mark.parametrize("fmt,model_path,extra", EXPORT_CASES, ids=[c[0] for c in EXPORT_CASES])
@pytest.mark.slow
def test_export_format(client, container, export_cleanup, fmt, model_path, extra):
    payload = {"model_path":model_path, "format":fmt, **extra}
    r = client.post("/export_model", json=payload, timeout=300)
    assert r.json()["success"], f"{fmt} {r.text}"
    prod = resolve_product_path(r.json()["data"])
    export_cleanup(prod)
    _assert_file_nonempty(container, prod)
    if fmt == "mlpackage-ios14":
        _assert_mlpackage_ios14_spec(container, prod)
    elif fmt == "onnx":
        # EXPORT_CASES 的 onnx 用例不带 version → 覆盖 version=None 分支(§9.2 矩阵)
        _assert_onnx_classes_txt(container, prod)
