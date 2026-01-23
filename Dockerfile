# DexGraspNet2 Docker Image
# Base: Ubuntu 20.04 + CUDA 11.8 + CUDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# CRITICAL: Disable NVIDIA driver version check to allow newer host drivers
ENV NVIDIA_DISABLE_REQUIRE=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Use faster mirrors (comment out if outside China)
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# Install system dependencies including headless rendering and GUI support
RUN apt-get update && \
    apt-get install -y \
    git \
    wget \
    unzip \
    curl \
    x11-apps \
    libopenblas-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libegl1 \
    libegl1-mesa-dev \
    libglvnd0 \
    libglvnd-dev \
    libglx0 \
    libxrender1 \
    libxext6 \
    ninja-build \
    mesa-utils \
    libxcursor1 \
    libxcursor-dev \
    libxinerama1 \
    libxinerama-dev \
    libxi6 \
    libxi-dev \
    libxrandr2 \
    libxrandr-dev \
    libglfw3 \
    libglfw3-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy third_party dependencies
COPY third_party/ /opt/third_party/

# Install Miniconda
RUN bash /opt/third_party/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm /opt/third_party/Miniconda3-latest-Linux-x86_64.sh && \
    /opt/miniconda3/bin/conda init && \
    /opt/miniconda3/bin/conda tos accept && \
    /opt/miniconda3/bin/conda create -y -n py38 python=3.8 && \
    /opt/miniconda3/bin/conda clean -ya

# Set pip mirror (comment out if outside China)
RUN /opt/miniconda3/bin/conda run -n py38 pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# Install PyTorch with CUDA 11.8
RUN /opt/miniconda3/bin/conda run -n py38 pip install \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch3D from local package
RUN /opt/miniconda3/bin/conda install -n py38 -y --use-local \
    /opt/third_party/pytorch3d-0.7.5-py38_cu118_pyt201.tar.bz2

# Install local editable packages
RUN /opt/miniconda3/bin/conda run -n py38 pip install -e /opt/third_party/TorchSDF
RUN /opt/miniconda3/bin/conda run -n py38 pip install -e /opt/third_party/torchprimitivesdf
RUN /opt/miniconda3/bin/conda run -n py38 pip install -e /opt/third_party/isaacgym/python
RUN /opt/miniconda3/bin/conda run -n py38 pip install -e /opt/third_party/nflows

# Install Python packages (pinned versions for compatibility)
RUN /opt/miniconda3/bin/conda run -n py38 pip install \
    plotly \
    transforms3d \
    open3d==0.17.0 \
    urdf_parser_py \
    tensorboard \
    coacd \
    rich \
    ikpy \
    einops \
    graspnetAPI \
    wandb \
    trimesh

# Install diffusers with pinned version (fixes torch.xpu compatibility issue)
RUN /opt/miniconda3/bin/conda run -n py38 pip install \
    'diffusers[torch]==0.21.0' \
    'huggingface_hub<0.24'

# Install MinkowskiEngine (requires CUDA)
ENV CUDA_HOME=/usr/local/cuda
WORKDIR /opt/third_party/MinkowskiEngine
RUN TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0" \
    /opt/miniconda3/bin/conda run -n py38 python setup.py install --blas=openblas --force_cuda

# Install remaining packages
RUN /opt/miniconda3/bin/conda run -n py38 pip install \
    numpy==1.23.0 \
    PyOpenGL \
    glfw \
    pyglm \
    healpy \
    rtree

# Setup conda environment activation and library paths
ENV PATH="/opt/miniconda3/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/miniconda3/envs/py38/lib:${LD_LIBRARY_PATH}"
RUN /opt/miniconda3/bin/conda init bash && \
    echo "conda activate py38" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/opt/miniconda3/envs/py38/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

# Set working directory
WORKDIR /root/DexGraspNet2

# Default command
CMD ["/bin/bash"]
