
from utils_v.index import init_root_folder,set_yolo_env
from utils_v.log import init_logger
set_yolo_env(True)
set_yolo_env(False)
init_root_folder()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from modal import CustomJSONResponse
import uvicorn

logger = init_logger("main")



app = FastAPI(
    #lifespan=lifespan,
    default_response_class=CustomJSONResponse,
    # AGPL §13 源码入口(规格 §6.6 R2 / §8):/docs 顶部 description 展示本许可与版本锚定源码链接
    license_info={
        "name": "AGPL-3.0",
        "url": "https://www.gnu.org/licenses/agpl-3.0.html",
    },
    # 版本锚定源码入口(规格 §6.6 R2 / §8):指向发布 tag,每次发版随发布更新
    description="AGPL-3.0 开源;网络源码 offer(AGPL §13):对应版本源码见 https://github.com/wanweitujie/yolosub/tree/v1.0.0",
)

@app.get("/")
async def read_root():
    return RedirectResponse(url="/static/index.html", status_code=301)



if __name__ == "__main__":
    # 单实例锁:防止双主进程互杀子进程(rc=-15 风暴)。fcntl(Linux/Docker)进程退出自动释放;
    # Windows 无 fcntl 跳过(依赖 start.sh 端口检查)。start.sh 端口检查只挡 start.sh 一条路,
    # supervisor/systemd/直接 python 全绕过,故在进程入口加 OS 级锁根治(任何拉取方式都生效)。
    import sys as _sys
    _lock_file = None
    try:
        import fcntl as _fcntl
        _lock_file = open("/tmp/yolo_service.lock", "w")
        try:
            _fcntl.flock(_lock_file, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except BlockingIOError:
            print("ERROR: 另一个 yolo_service 实例已在运行(flock 锁定),请先停止旧实例", file=_sys.stderr)
            _sys.exit(1)
    except ImportError:
        pass   # Windows 无 fcntl,跳过(依赖 start.sh 端口检查)
    except OSError as e:
        # open 失败(/tmp 不可写或锁文件无权限)或 flock 其他错误:锁机制故障,
        # 保守退出(不启动以防双主进程互杀),需运维确认 /tmp 可写或改锁文件路径
        print(f"ERROR: 单实例锁文件故障({e}),无法启动", file=_sys.stderr)
        _sys.exit(1)

    from train_v.routes import router as router_train,init_train
    init_train(logger)
    app.include_router(router_train)

    from detect_v.routes import router as router_detect,init_detect
    init_detect(logger, app)
    app.include_router(router_detect)

    from export_v.routes import router as router_export,init_export
    init_export(logger)
    app.include_router(router_export)

    app.mount("/static",StaticFiles(directory="./wwwroot"),name="static")
    uvicorn.run(app, host="0.0.0.0", port=29004)