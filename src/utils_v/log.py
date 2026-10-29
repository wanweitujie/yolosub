
import logging
import re
from pathlib import Path
import sys

# uvicorn access 格式:`127.0.0.1:43290 - "GET /health HTTP/1.1" 200 OK`
# 精确匹配「"方法 /health HTTP…」+ 状态 200:避免 /detect/health_model 这类
# 路径含 /health 子串的业务请求被误滤。
_HEALTH_OK_RE = re.compile(r'"\w+ /health HTTP/[^"]*" 200 ')


class HealthAccessLogFilter(logging.Filter):
    """过滤 uvicorn access log 中 /health 探活的成功记录。

    executor 假死自愈(_health_loop)每 10s ping 一次子进程 /health,成功(200)的
    access 记录纯刷屏无诊断价值;非 200(如 503 gpu 锁超时)是假死线索,保留。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not _HEALTH_OK_RE.search(msg)


def init_logger(name:str,change_std:bool = False):
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()   # 防止重复添加（关键！）

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - PID:%(process)d - %(message)s'
    )

    # 文件 handler
    file_handler = logging.FileHandler(log_dir / f'{name}.txt', encoding='utf-8')
    file_handler.setFormatter(formatter)

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    if change_std:
        change_std_to_logger(logger)
    
    logger.info(f"=============↓↓↓↓↓↓↓↓↓↓=====================\n====================================================\n          {name} new logger \n====================================================\n")
    
    
    return logger

class LoggerWriter:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buffer = ''  # 用于缓冲不完整行

    def write(self, message):
        """
        被 print 调用，每次写入一段字符串。
        message 可能包含换行，也可能只是部分行。
        """
        if not message:
            return

        # print 默认会在消息末尾加 \n，但有时会传入空字符串或不完整行
        self.buffer += message

        # 如果缓冲区中有完整行（以换行符分隔），就逐行输出
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            stripped = line.rstrip()
            if stripped:  # 只记录非空行
                self.logger.log(self.level, stripped)

        # 如果最后没有换行符，剩余部分留在 buffer 中等待下次 write

    def flush(self):
        """
        被 print(..., flush=True) 或某些库调用。
        将缓冲区中剩余的内容立即输出（作为一条不完整行）。
        """
        if self.buffer:
            stripped = self.buffer.rstrip()
            if stripped:
                self.logger.log(self.level, stripped)
            self.buffer = ''  # 清空缓冲区

    def fileno(self):
        """
        某些工具（如 subprocess）会检查是否是终端，需实现此方法避免错误。
        """
        return sys.__stdout__.fileno() 
    
def change_std_to_logger(logger:logging.Logger):
    # 重定向stdout（print默认输出这里）
    sys.stdout = LoggerWriter(logger, logging.INFO)
    # 可选：重定向stderr（错误输出）
    sys.stderr = LoggerWriter(logger, logging.ERROR)