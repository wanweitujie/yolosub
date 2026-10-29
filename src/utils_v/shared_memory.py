"""SharedMemory 段管理工具。

本项目通过 SharedMemory 在主进程与检测子进程之间传输图像 bytes。
主进程 unlink() 写在 finally 中，但 SIGKILL / OOM 等场景下 finally 不执行，
段会在 /dev/shm 中残留。启动时主动扫描清理。
"""
import sys
import uuid
from pathlib import Path
from multiprocessing.shared_memory import SharedMemory

PREFIX = "yolosub_"


def make_shm_name() -> str:
    """生成带项目前缀的共享内存段名。"""
    return f"{PREFIX}{uuid.uuid4().hex[:8]}"


def cleanup_stale_shared_memory(prefix: str = PREFIX) -> int:
    """扫描 /dev/shm 下本项目残留段并 unlink。

    Returns:
        清理的段数量。
    """
    if not sys.platform.startswith("linux"):
        return 0
    shm_dir = Path("/dev/shm")
    if not shm_dir.exists():
        return 0
    cleaned = 0
    for p in shm_dir.glob(f"{prefix}*"):
        shm = None
        try:
            shm = SharedMemory(name=p.name, create=False)
            shm.close()
            shm.unlink()
            cleaned += 1
        except FileNotFoundError:
            pass
        except Exception:
            pass
        finally:
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass
    return cleaned
