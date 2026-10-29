#!/bin/bash

# 匹配主服务与两个检测子服务(uvicorn 单进程模式,无独立 worker)
PATTERN='yolo_service.py|v5_service.py|ultralytics_service.py'

# 查找所有匹配的 PID
PIDS=$(pgrep -f "$PATTERN")

# 检查是否找到了进程
if [ -z "$PIDS" ]; then
    echo "No process found matching: $PATTERN"
    exit 0
fi

echo "Found PIDs: $(echo $PIDS | tr '\n' ' ')"

# 先发 SIGTERM,触发主进程 uvicorn 优雅退出 + 子进程退出
for pid in $PIDS; do
    kill -TERM "$pid" 2>/dev/null
done

# 轮询等待退出(最多 25s,进程秒退则秒返,不硬等)
echo "已发 SIGTERM,轮询等待退出(最多 25s)..."
for i in $(seq 1 25); do
    REMAINING=$(pgrep -f "$PATTERN")
    [ -z "$REMAINING" ] && break
    sleep 1
done

# 若仍存在,强制 SIGKILL
REMAINING=$(pgrep -f "$PATTERN")
if [ -n "$REMAINING" ]; then
    echo "Some processes still alive, sending SIGKILL: $(echo $REMAINING | tr '\n' ' ')"
    for pid in $REMAINING; do
        kill -KILL "$pid" 2>/dev/null
    done
fi

echo "All processes related to YoloSub have been terminated."
