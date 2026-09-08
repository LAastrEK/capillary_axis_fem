FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Базовые утилиты, Python и графические библиотеки для gmsh
#    ВАЖНО: python3-h5py установлен через apt, чтобы совпадал с HDF5 от FEniCS
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-matplotlib \
    python3-numpy \
    python3-h5py \
    python3-h5py-serial \
    libgl1 \
    libglu1-mesa \
    libxcursor1 \
    libxrender1 \
    libxft2 \
    libxinerama1 \
    libgomp1 \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# 2. Подключаем PPA FEniCS и устанавливаем legacy FEniCS (dolfin)
RUN add-apt-repository -y ppa:fenics-packages/fenics && \
    apt-get update && \
    apt-get install -y --no-install-recommends fenics && \
    rm -rf /var/lib/apt/lists/*

# 3. Создаём venv с доступом к системным пакетам
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 4. Устанавливаем ТОЛЬКО gmsh и meshio через pip
#    h5py НЕ ставим через pip — он уже есть в системе и совместим с HDF5 от FEniCS
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gmsh meshio

WORKDIR /app
COPY . /app

CMD ["python3", "run_simulation.py", "--remesh"]