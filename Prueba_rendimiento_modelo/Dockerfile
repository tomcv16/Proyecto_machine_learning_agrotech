FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch solo CPU (sin CUDA). En la Raspberry (arm64) este mismo Dockerfile funciona.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch torchvision --index-url ${TORCH_INDEX}

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.pt analizar_foto.py benchmark.py ./

CMD ["python", "benchmark.py", "latencia"]