"""_read_class_names 的单元测试。

覆盖:dict/list 两种 names 形态、按 id 升序、异常不吞。
YOLO 用 mock 替代,避免单测依赖真实模型文件。
"""
from unittest.mock import MagicMock, patch

import pytest

from export_v.export_ncnn import _read_class_names, _write_ncnn_classes


def _fake_yolo(names):
    """构造一个 YOLO 替身,其 .names 返回给定值。"""
    inst = MagicMock()
    inst.names = names
    return MagicMock(return_value=inst)


def test_read_names_dict_sorted_by_id():
    """names 为 dict 时按 id 升序返回标签名列表。"""
    with patch("ultralytics.YOLO", _fake_yolo({1: "car", 0: "person", 2: "bus"})):
        assert _read_class_names("/tmp/m.pt") == ["person", "car", "bus"]


def test_read_names_list_passthrough():
    """names 为 list 时原样返回。"""
    with patch("ultralytics.YOLO", _fake_yolo(["person", "car"])):
        assert _read_class_names("/tmp/m.pt") == ["person", "car"]


def test_read_names_raises_not_swallowed():
    """加载异常必须上抛,由调用方兜底(保证告警文案与调用方一致)。"""
    with patch("ultralytics.YOLO", side_effect=RuntimeError("load failed")):
        with pytest.raises(RuntimeError, match="load failed"):
            _read_class_names("/tmp/m.pt")


def test_write_ncnn_classes_keeps_original_warning(tmp_path):
    """_write_ncnn_classes 读失败时告警文案与改造前逐字一致,且不抛异常。"""
    with patch("ultralytics.YOLO", side_effect=RuntimeError("boom")):
        with patch("export_v.export_ncnn.logging") as mock_logging:
            _write_ncnn_classes(tmp_path, "/tmp/m.pt")   # 不抛异常
            args, _ = mock_logging.getLogger.return_value.warning.call_args
            assert args[0] == "写入 classes.txt 失败(不影响转换): boom"


def test_write_ncnn_classes_writes_file(tmp_path):
    """正常路径:写出每行一个标签、无尾换行、UTF-8。"""
    with patch("ultralytics.YOLO", _fake_yolo({0: "person", 1: "安全帽"})):
        _write_ncnn_classes(tmp_path, "/tmp/m.pt")

    raw = (tmp_path / "classes.txt").read_bytes()
    # write_text 的 newline 参数为默认 None -> 换行跟随平台:
    # Linux(容器/生产)为 LF,Windows(开发机)为 CRLF。生产在容器内,故线上为 LF。
    # 这里归一化后再比对,保证断言在两种平台上都成立,且仍锁定"逐行标签、顺序正确、UTF-8"契约。
    assert raw.decode("utf-8").replace("\r\n", "\n") == "person\n安全帽"
    assert not raw.endswith(b"\n")
