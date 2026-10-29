#!/bin/bash
# 重启:先 stop 再 start。避免直接 start 导致双主进程(双主进程 kill_yolo 互杀子进程)。
script_dir=$(cd "$(dirname "$0")" && pwd)
cd "$script_dir" || echo "目录切换失败: $script_dir"

echo "[restart] 停止现有服务..."
./stop.sh

# 等端口释放 + 子进程退出(stop.sh 已 sleep 25 + SIGKILL,这里再保险 2s)
sleep 2

echo "[restart] 启动服务..."
./start.sh
