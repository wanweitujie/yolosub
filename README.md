# YoloSub

基于 YOLO (You Only Look Once) 目标检测模型的 RESTful API 服务，支持 YOLOv5、YOLOv8 和 YOLOv11 版本。项目使用 FastAPI 构建，提供模型加载、推理、训练和导出功能，支持 GPU 加速和多模型管理。

## 🚀 主要特性

- **多版本支持**: 同时支持 YOLOv5、v8、v11 模型
- **完整 API 服务**: 提供模型创建、检测、训练、导出全流程 RESTful API
- **GPU 管理**: 自动 GPU 分配，内存监控，闲置模型清理
- **异步处理**: 支持并发请求，YOLOv5 在独立进程中运行
- **模型导出**: 支持 ONNX、NCNN、TorchScript、CoreML、MLPackage 等多种格式
- **训练支持**: 完整的模型训练流程，状态监控和日志输出
- **跨平台部署**: 提供 Docker GPU 版本，支持 iOS 和边缘设备部署

## 📦 快速开始

### 本地运行

```bash
# 启动服务（后台运行）
./start.sh

# 停止服务
./stop.sh

# 直接运行（前台运行，调试用）
python ./src/yolo_service.py
```

服务启动后默认访问：http://localhost:29004

> 注:`./start.sh` 依赖 Linux 平台的 `pgrep`/`nohup` 命令，`.sh` 脚本已设置可执行位（`100755`）；Windows 上请使用下方 Docker 运行路径。

### Docker 运行

```bash
# 构建镜像（在仓库根目录执行）
docker build -t yolo-sub:gpu docker-create

# 大陆构建建议指定国内可达的字体镜像源（默认从 github release 拉取约 23MB 字体，可能慢或失败）
docker build --build-arg FONT_BASE_URL=<国内可达镜像源> -t yolo-sub:gpu docker-create

# 运行容器（需要 NVIDIA 容器运行时；-v 把仓库根挂载为数据卷）
# --shm-size=8g:PyTorch dataloader 靠 /dev/shm 把 batch 从 worker 传回主进程,
# Docker 默认 64MB 不够(batch16@640 单个 batch 约 78MB),过小会导致训练静默挂起(shm 分配失败)。
docker run --gpus all --shm-size=8g -p 29004:29004 -v $(pwd):/docker-data-yolo -it yolo-sub:gpu
```

## 📡 API 接口概览

项目提供 9 个主要 API 接口：

### 检测相关
- `POST /create_model` - 创建/加载模型
- `POST /detect/{model_name}` - 使用指定模型进行检测
- `DELETE /close_model/{model_name}` - 关闭指定模型
- `GET /gpu_status` - 获取 GPU 状态和已加载模型信息

### 训练相关
- `POST /train/start` - 开始模型训练
- `GET /train/status` - 获取训练状态
- `GET /train/log` - 获取训练日志
- `POST /train/stop` - 停止训练任务

### 导出相关
- `POST /export_model` - 转换模型到其他格式

**详细 API 文档**: 服务启动后访问 `/docs`(Swagger UI,含完整请求/响应示例与在线调试)

## 📁 项目结构

```
src/
├── detect_v/          # 检测模块（模型管理、GPU 分配、推理逻辑）
├── export_v/          # 模型导出模块
├── train_v/           # 模型训练模块
├── utils_v/           # 工具模块（配置、日志、路径处理）
├── modal.py           # 数据模型定义（Pydantic）
└── yolo_service.py    # 主服务入口

docker-create/         # Docker 构建配置（GPU 版本）
models/                # 预训练模型存储
wwwroot/               # 静态文件（Web 界面）
```

## 🔧 技术栈

- **Web 框架**: FastAPI
- **深度学习**: PyTorch 2.11.0 + CUDA 12.8 + cuDNN 9
- **YOLO 实现**: Ultralytics 8.4.121 (v8/v11) + YOLOv5 独立环境
- **GPU 管理**: GPUtil
- **模型导出**: ONNX, NCNN, CoreMLTools, TorchScript
- **NCNN 转换**: pnnx（Linux 二进制，约 80 MiB，见 `docker-create/files/pnnx`）
- **容器化**: Docker + NVIDIA Container Toolkit

## 🎯 项目来源

本项目基于以下开源项目构建，特别感谢这些项目的贡献：

- **[Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)** - YOLOv5 目标检测框架
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** - YOLOv8/v11 目标检测框架（Ultralytics 实现）
- **[yolov5_convert_weight_to_coreml](https://github.com/ClintRen/yolov5_convert_weight_to_coreml)** - YOLOv5 模型转 CoreML 工具

这些开源项目为 YoloSub 提供了核心的 YOLO 实现和模型转换功能。

## ⚙️ 配置说明

### 环境要求
- Python 3.10+（与镜像一致，基础镜像为 python:3.10）
- CUDA 12.8 (GPU 版本)
- Docker (容器化部署)
- NVIDIA GPU + 驱动 (GPU 加速)

### 模型存储路径
- Ultralytics 模型: `models/ultralytics/`
- YOLOv5 模型: `models/v5/`
- 训练输出: `runs/train/{exp_name}/`
- 数据目录: `docker-data-yolo/` (Docker 容器内映射)

### 虚拟环境
项目使用多个虚拟环境管理依赖：
- 主环境: YOLOv8/v11 (Ultralytics)
- `/venv_yolov5`: YOLOv5 专用环境
- `/venv_export_mlpackage_ios14`: iOS 14+ 模型导出环境

## 🐳 Docker 部署

### 数据卷映射
将宿主机的数据目录映射到容器内的 `/docker-data-yolo`：
```bash
-v /path/to/your/data:/docker-data-yolo
```

### 环境变量
- `HTTP_PROXY`/`HTTPS_PROXY`: Docker 构建时的代理设置
- 模型路径使用容器内的绝对路径：`/docker-data-yolo/models/...`

### 性能优化
- GPU 版本使用 NVIDIA 容器运行时获得最佳性能
- 支持 FP16 推理减少显存占用
- 自动模型闲置清理（24小时未使用）

## 🧪 测试

- 单元测试：`bash tests/run.sh`（主环境，无需容器）
- 集成测试：`bash tests/integration/run.sh`（需运行中的容器与测试模型；公开仓库不含内部测试模型，整个集成套件按 `conftest.py` 的 `require_test_models` fixture **整体 skip**，业务标识已脱敏，不会逐用例报错）

## 📚 详细文档

- **[CHANGELOG.md](CHANGELOG.md)** - 版本更新日志

## 🔄 开发说明

### 代码规范
- 使用 Pydantic 模型进行数据验证
- 统一响应格式：`{"success": boolean, "data": any, "errMsg": string|null}`
- 模块化设计，功能分离

### 扩展开发
添加新功能时：
1. 遵循现有模块结构，将相关代码放入对应目录
2. 通过 `ModelManager` 扩展模型类型支持
3. 在 `YoloExecutorManager` 中添加新的执行器类型
4. 同步更新 README 与 `/docs` 中的接口说明

### 错误处理
- 异常统一捕获和日志记录
- HTTP 状态码标准化
- 友好的错误信息返回

## 📄 许可与致谢

- 本项目本体以 **AGPL-3.0** 协议提供。
- 内置的 vendored yolov5 部分**保留 GPL-3.0**（依 GPL-3.0 §13 并入组合作品）。
- 若以网络服务形式部署本项目，依 **AGPL-3.0 第 13 条**（Remote Network Interaction）须向服务使用者提供对应版本的完整源码，且该源码 offer **同时涵盖**上述按 GPL-3.0 许可并入的组件。服务以网络形式运行期间，对应版本源码在公开仓库 [wanweitujie/yolosub](https://github.com/wanweitujie/yolosub) 以版本标签形式持续可得。

### 上游组件许可对照表

| 组件 | 许可 | 说明 |
|------|------|------|
| `ultralytics`（pip，8.4.121） | AGPL-3.0 | AGPL 义务来源之一 |
| vendored yolov5（`docker-create/files/mlmodels/yolov5/`） | GPL-3.0 | 自带 LICENSE；含本地修改，见 `LOCAL_CHANGES.md` |
| `yolov5_convert_weight_to_coreml` | GPL-3.0 | 自带 LICENSE |
| `pnnx` 二进制 | 见 `docker-create/files/pnnx.LICENSE` | Linux 二进制（约 80 MiB） |
| `models/ultralytics/yolo11n.pt`、`src/yolo26n.pt` | AGPL-3.0 | 来自 `ultralytics/assets` v8.4.0 官方发布物 |
| `models/v5/yolov5n.pt`、`src/yolov5n.pt` | GPL-3.0 | 来自 `ultralytics/yolov5` v6.1 release 官方发布物 |
| `src/assets/preload.webp` | 项目自备 | 模型预热占位图 |
| 第三方 Python 依赖 | 各自许可 | fastapi（MIT）、torch（BSD）、coremltools（BSD-3）、onnxruntime（MIT）、ncnn（BSD-3）、tqdm（MPL-2.0）等 |

> **Arial 字体声明**：`Arial.ttf` / `Arial.Unicode.ttf` 为 Monotype/Microsoft 专有字体，不随本仓库分发，改为构建期下载并校验 SHA-256；含字体的镜像（含 `docker save` 交付）再分发时一并携带该字体。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。