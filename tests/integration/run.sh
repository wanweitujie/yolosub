#!/usr/bin/env bash
# 容器集成测试(快速套件):复用运行中的 YoloSub 容器(端口 29004)
# 默认排除 slow + 不递归 self_heal/(物理隔离,需手动跑)
set -e
cd "$(dirname "$0")/../.."
pip install -r tests/requirements.txt
PYTHONPATH=src pytest tests/integration/ -v -m "not slow" --ignore=tests/integration/self_heal
