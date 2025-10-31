FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git ninja-build cmake build-essential \
    ffmpeg libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python3 -m pip install --upgrade "pip<24" "setuptools<70" "wheel<0.43"
RUN python3 -m pip install --no-build-isolation --no-cache-dir chumpy==0.70

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

RUN python3 -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.0+cu118 torchvision==0.15.1+cu118

COPY third_party /tmp/third_party
RUN python3 -m pip install --no-cache-dir /tmp/third_party/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl

ENV XDG_CACHE_HOME=/workspace/.cache \
    TORCH_HOME=/workspace/.cache \
    HF_HOME=/workspace/.cache \
    CUDA_CACHE_PATH=/workspace/.cache \
    PYTHONPATH=/workspace

RUN mkdir -p /workspace/.cache/mplcache /workspace/.cache/output && \
    chmod -R 777 /workspace/.cache

COPY . .

CMD ["python3", "-m", "streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]