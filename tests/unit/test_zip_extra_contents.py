"""zip_with_max_compression 的 extra_contents 参数测试。

覆盖:内存条目写入、中文 UTF-8 往返、无尾换行、不传参时的回归守卫。
"""
import zipfile

from utils_v.zip import zip_with_max_compression


def test_extra_contents_adds_entry(tmp_path):
    """传 extra_contents 时,条目被写入 zip。"""
    src = tmp_path / "m.onnx"
    src.write_bytes(b"fake onnx")
    out = str(tmp_path / "m.onnx.zip")

    zip_with_max_compression(str(src), out, extra_contents={"classes.txt": "person\nbicycle"})

    zf = zipfile.ZipFile(out)
    assert "classes.txt" in zf.namelist()
    assert zf.read("classes.txt").decode("utf-8") == "person\nbicycle"


def test_extra_contents_chinese_utf8_roundtrip(tmp_path):
    """中文类名按 UTF-8 编码写入,解码回来一致;且无尾换行。"""
    src = tmp_path / "m.onnx"
    src.write_bytes(b"fake onnx")
    out = str(tmp_path / "m.onnx.zip")
    text = "person\n安全帽\n反光衣"

    zip_with_max_compression(str(src), out, extra_contents={"classes.txt": text})

    raw = zipfile.ZipFile(out).read("classes.txt")
    assert raw.decode("utf-8") == text          # 非 UTF-8 会在此抛错
    assert not raw.startswith(b"\xef\xbb\xbf")  # 无 BOM
    assert not raw.endswith(b"\n")              # 无尾换行
    assert raw.count(b"\n") == 2                # 3 个标签只有 2 个换行


def test_extra_contents_keeps_main_file_intact(tmp_path):
    """主文件内容不受影响(sha256 一致),zip 未损坏。"""
    import hashlib
    src = tmp_path / "m.onnx"
    payload = b"fake onnx payload" * 100
    src.write_bytes(payload)
    out = str(tmp_path / "m.onnx.zip")

    zip_with_max_compression(str(src), out, extra_contents={"classes.txt": "a\nb"})

    zf = zipfile.ZipFile(out)
    assert zf.testzip() is None
    assert hashlib.sha256(zf.read("model.onnx")).hexdigest() == hashlib.sha256(payload).hexdigest()


def test_extra_contents_duplicate_name_skipped(tmp_path):
    """arcname 与既有条目重名时跳过,不产生重复条目。"""
    src = tmp_path / "m.onnx"
    src.write_bytes(b"fake onnx")
    out = str(tmp_path / "m.onnx.zip")

    zip_with_max_compression(str(src), out, extra_contents={"model.onnx": "SHOULD_BE_SKIPPED"})

    zf = zipfile.ZipFile(out)
    assert zf.namelist().count("model.onnx") == 1
    assert zf.read("model.onnx") == b"fake onnx"   # 原内容未被覆盖


def test_no_extra_contents_unchanged(tmp_path):
    """回归守卫:不传 extra_contents 时归档内容与改动前完全相同。"""
    src = tmp_path / "m.onnx"
    src.write_bytes(b"fake onnx")
    out = str(tmp_path / "m.onnx.zip")

    zip_with_max_compression(str(src), out)

    zf = zipfile.ZipFile(out)
    assert zf.namelist() == ["model.onnx"]
    assert zf.read("model.onnx") == b"fake onnx"


def test_extra_contents_on_directory_source(tmp_path):
    """源为目录时同样追加内存条目(目录路径也应支持)。"""
    d = tmp_path / "ncnn_model"
    d.mkdir()
    (d / "model.ncnn.param").write_text("param", encoding="utf-8")
    out = str(tmp_path / "out.zip")

    zip_with_max_compression(str(d), out, extra_contents={"classes.txt": "person"})

    zf = zipfile.ZipFile(out)
    assert "classes.txt" in zf.namelist()
    assert "model.ncnn.param" in zf.namelist()
