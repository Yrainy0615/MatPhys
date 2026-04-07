FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG USER_ID=1130
ARG GROUP_ID=300
ARG USER_NAME="yyang"

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST=8.6+PTX
ENV FORCE_CUDA=1
ENV PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-lc"]

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    groupadd -g "${GROUP_ID}" "${USER_NAME}" && \
    useradd -u "${USER_ID}" -m "${USER_NAME}" -g "${USER_NAME}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    freeglut3-dev \
    git \
    libassimp-dev \
    libavcodec-dev \
    libavdevice-dev \
    libboost-all-dev \
    libegl1 \
    libeigen3-dev \
    libembree-dev \
    libgl1 \
    libgl1-mesa-dev \
    libglew-dev \
    libglib2.0-0 \
    libglu1-mesa-dev \
    libglfw3-dev \
    libgtk-3-dev \
    libopencv-dev \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxfixes3 \
    libxinerama1 \
    libxkbcommon-x11-0 \
    libxrandr2 \
    libxxf86vm-dev \
    libxxf86vm1 \
    pkg-config \
    sudo \
    vim \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_25.1.1-2-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR" && \
    rm -f /tmp/miniconda.sh && \
    conda config --system --set auto_update_conda false && \
    conda create -y -n phystwin python=3.10.19 pip && \
    conda clean -afy

RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME} && \
    chmod 0440 /etc/sudoers.d/${USER_NAME}

WORKDIR /tmp/build/Phys-GS

COPY requirements.txt /tmp/requirements.txt

RUN conda run -n phystwin pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121

RUN grep -vE '^(torch|torchvision|torchaudio)==|^pyrealsense2==|^#|^$' /tmp/requirements.txt > /tmp/requirements.runtime.txt && \
    conda run --no-capture-output -n phystwin pip install -v -r /tmp/requirements.runtime.txt

COPY . .

RUN conda run -n phystwin pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html

RUN conda run --no-capture-output -n phystwin pip install --no-build-isolation ./gaussian_splatting/submodules/diff-gaussian-rasterization && \
    conda run --no-capture-output -n phystwin pip install --no-build-isolation ./gaussian_splatting/submodules/simple-knn

RUN echo "conda activate phystwin" >> /root/.bashrc && \
    echo "conda activate phystwin" >> "/home/${USER_NAME}/.bashrc"

WORKDIR /home/yyang/mnt/workspace

CMD ["/bin/bash"]
