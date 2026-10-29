#!/usr/bin/env bash
# 单元测试(无 GPU,项目主环境)
set -e
pip install -r tests/requirements.txt
PYTHONPATH=src pytest tests/unit/ -v
