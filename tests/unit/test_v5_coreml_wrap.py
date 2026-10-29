# v5 coreml wrapper 的类别标签规范化逻辑单测(纯函数,无需 torch/GPU)
from export_v.process.v5_coreml_wrap import normalize_class_labels


def test_dict_int_labels_to_str():
    # data yaml names 写成数字时,model.names 为 {0: 1, 1: 2, ...},值须转为字符串;
    # 键是类别索引不是标签,保持原样(CoreML stringClassLabels 只消费 values)
    assert normalize_class_labels({0: 1, 1: 2}) == {0: "1", 1: "2"}


def test_list_labels_to_str():
    assert normalize_class_labels([1, 2]) == ["1", "2"]


def test_tuple_labels_to_str():
    assert normalize_class_labels((0, 1)) == ["0", "1"]


def test_str_labels_unchanged():
    # 标准 coco 字符串名不受影响
    names = {0: "person", 1: "car"}
    assert normalize_class_labels(names) == names


def test_other_type_passthrough():
    # 非常规类型原样返回,不抛错
    assert normalize_class_labels(None) is None
