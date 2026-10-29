# 使用 NVIDIA 提供的带有 CUDA 和 cuDNN 的 Ubuntu 基础镜像
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY


# 设置非交互式安装，防止 apt-get 安装过程中出现要求用户输入的情况
ENV DEBIAN_FRONTEND=noninteractive

# 安装基本工具和库
RUN apt-get update && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf $(which python3) /usr/bin/python

# 版本对齐说明:
# - 基础镜像 12.8.1-cudnn-runtime 内置 cuDNN 9;torch 2.11.0+cu128 亦依赖 cuDNN 9,二者一致。
# - 历史坑:2.3<=torch<2.5 存在 CUDA lazy init bug,当前 2.11.0 已无此问题,勿回退到该区间。

RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install ultralytics==8.4.121 -i https://mirrors.aliyun.com/pypi/simple/

RUN pip install coremltools==8  -i https://mirrors.aliyun.com/pypi/simple/

RUN pip install fastapi GPUtil uvicorn python-multipart scikit-learn===1.5.1 -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install opencv-python===4.11.0.86 -i https://mirrors.aliyun.com/pypi/simple/ --no-deps
RUN pip install pnnx===20251119 ncnn===1.0.20250916 -i https://mirrors.aliyun.com/pypi/simple/
# numpy==2.2.6 版本钉扎:依赖链兼容性要求(onnx/ml-dtypes 0.6.0 需 numpy>=2)
RUN pip install numpy==2.2.6 -i https://mirrors.aliyun.com/pypi/simple/
# 防再漂:onnx/ml-dtypes 与 numpy 一并钉死(三件套依赖链配套)
RUN pip install onnx==1.22.0 ml-dtypes==0.6.0 onnxslim onnxruntime-gpu==1.23.2 -i https://mirrors.aliyun.com/pypi/simple/
# [v5 onnx 走主环境] yolov5 缺失依赖 + torch2.11 dynamo 配套(闭包,全部 --no-deps)
RUN pip install pandas==2.3.3 pytz==2026.3.post1 seaborn==0.13.2 onnxscript==0.7.1 onnx-ir==1.0.0 --no-deps -i https://mirrors.aliyun.com/pypi/simple/

#yolov5
RUN python -m venv /venv_yolov5 --system-site-packages

COPY ./files/mlmodels /venv_yolov5/mlmodels
RUN find /venv_yolov5/mlmodels -type f -exec chmod 644 {} \;
RUN find /venv_yolov5/mlmodels -type d -exec chmod 755 {} \;

RUN /venv_yolov5/bin/pip install -r /venv_yolov5/mlmodels/yolov5/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

RUN /venv_yolov5/bin/pip install numpy==1.23.1 coremltools==6.0 -i https://mirrors.aliyun.com/pypi/simple/
# v5 图像代码用 FreeTypeFont.getsize(Pillow>=10 已移除),固定 <10 使该 API 可用并仅 shadow 到 v5 venv
RUN /venv_yolov5/bin/pip install 'Pillow==9.5.0' -i https://mirrors.aliyun.com/pypi/simple/

#mlpackage ios14 support
RUN python -m venv /venv_export_mlpackage_ios14 --system-site-packages
RUN /venv_export_mlpackage_ios14/bin/pip install ultralytics==8.3.77 --no-deps -i https://mirrors.aliyun.com/pypi/simple/
RUN /venv_export_mlpackage_ios14/bin/pip install --no-deps coremltools==6.0 numpy==1.23.1 protobuf==3.20.1 -i https://mirrors.aliyun.com/pypi/simple/
COPY ./files/pnnx /venv_export_mlpackage_ios14/lib/python3.10/site-packages/ultralytics/pnnx
RUN chmod +x /venv_export_mlpackage_ios14/lib/python3.10/site-packages/ultralytics/pnnx

# RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
#     && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
#     && dpkg-reconfigure -f noninteractive locales \
#     && locale-gen en_US.UTF-8

# ENV LANG=en_US.UTF-8
# ENV LANGUAGE=en_US:en
# ENV LC_ALL=en_US.UTF-8    

RUN pip install httpx -i https://mirrors.aliyun.com/pypi/simple/

# 设置工作目录
WORKDIR /

#复制其他资源文件
# 字体:v5 与 v8/v11 配置目录分开(set_yolo_env 中 v5 用 /root/.config/yolov5/,v8/v11 用 /root/.config/Ultralytics/),
# 两处都预置 Arial.ttf 与 Arial.Unicode.ttf,避免 check_font 联网下载(容器离线时下载失败会致训练崩溃)
# 字体构建期下载(公开仓库不携带专有字体;运行期离线可用性不变,规格 §6.4)
ARG FONT_BASE_URL=https://github.com/ultralytics/assets/releases/download/v0.0.0
ARG FONT_ARIAL_SHA256=525979822591a3447cfc49d943d6f7683508e25543407871c0ed8fed05fd2bd9
ARG FONT_ARIALU_SHA256=876af2cd4854644e7f3e7feb2f688997fdb3343c6df6693611209c9dfb47ccec
RUN mkdir -p /tmp/yolo_fonts /root/.config/Ultralytics /root/.config/yolov5 \
    && curl -fsSL --retry 3 --retry-delay 5 --max-time 300 -o /tmp/yolo_fonts/Arial.ttf \
        ${FONT_BASE_URL}/Arial.ttf \
    && curl -fsSL --retry 3 --retry-delay 5 --max-time 300 -o /tmp/yolo_fonts/Arial.Unicode.ttf \
        ${FONT_BASE_URL}/Arial.Unicode.ttf \
    && echo "${FONT_ARIAL_SHA256}  /tmp/yolo_fonts/Arial.ttf" | sha256sum -c - \
    && echo "${FONT_ARIALU_SHA256}  /tmp/yolo_fonts/Arial.Unicode.ttf" | sha256sum -c - \
    && cp /tmp/yolo_fonts/Arial.ttf /root/.config/Ultralytics/ \
    && cp /tmp/yolo_fonts/Arial.Unicode.ttf /root/.config/Ultralytics/ \
    && cp /tmp/yolo_fonts/Arial.ttf /root/.config/yolov5/ \
    && cp /tmp/yolo_fonts/Arial.Unicode.ttf /root/.config/yolov5/ \
    && rm -rf /tmp/yolo_fonts
COPY ./files/pnnx /usr/local/lib/python3.10/dist-packages/ultralytics/pnnx
RUN chmod +x /usr/local/lib/python3.10/dist-packages/ultralytics/pnnx

# 复制 entrypoint.sh 到容器中
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
# 修改 entrypoint.sh 的权限，使其可执行
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN pip cache purge
RUN /venv_yolov5/bin/pip cache purge

ENV http_proxy=
ENV https_proxy=

# 设置脚本为入口点
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# 设置默认行为
CMD ["bash"]