#!/bin/bash
script_dir=$(cd "$(dirname "$0")" && pwd)

cd "$script_dir" || echo "目录切换失败: $script_dir"

# 防重复启动:29004 已监听则拒绝。双主进程会各自 kill_yolo_process_service 互杀子进程
# (rc=-15 风暴),且第二主进程 uvicorn 绑端口失败崩溃,其子进程在 startup 事件前
# (_loop 未绑定)被杀,monitor 无法自愈。
check_port() {
    # python socket connect_ex 通用(不依赖 ss/lsof,精简容器可能无)。返回 0=端口有 LISTEN
    python -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1',29004))==0 else 1)" 2>/dev/null
}
if check_port; then
    echo "ERROR: 端口 29004 已被占用,主进程可能在运行。请先 ./stop.sh 或 ./restart.sh" >&2
    exit 1
fi

# 使用 nohup 运行 Python 脚本，输出重定向到 log/nohup.out.log(根目录保持整洁;log/ 已 gitignore)
# nohup.out.log 兜底 uvicorn 访问日志 + 启动早期崩溃(logging init 前的报错);业务日志见 log/main.txt,
# 子进程日志见 log/*.stdout.log / *.stderr.log
mkdir -p log
nohup python ./src/yolo_service.py >> log/nohup.out.log 2>&1 &

# 获取后台进程的 PID
PID=$!

# 输出后台进程的 PID
echo "yolo_service.py 已运行在后台, PID $PID"
