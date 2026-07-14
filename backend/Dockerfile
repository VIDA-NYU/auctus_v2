# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Container-friendly Python defaults: no .pyc files, unbuffered stdout/stderr,
# and no pip cache baked into image layers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies:
#   build-essential - compile Python packages that ship without a manylinux wheel
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU-only build of PyTorch FIRST, from PyTorch's dedicated CPU wheel
# index. torch is only pulled in transitively (via sentence-transformers, to embed
# search queries) and runs on CPU here, so installing the +cpu build up front stops
# the default PyPI wheel from dragging in the ~6 GB NVIDIA CUDA runtime. The later
# requirements install then sees torch already satisfied and skips it.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the remaining Python dependencies. Kept after the torch layer so both
# stay cached until their respective inputs change.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY . .

# Drop root: run the service as an unprivileged user in production.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Container-level liveness check. Kubernetes will use its own probes, but this
# keeps `docker`/Compose health reporting accurate. Hits "/" which returns 200
# without depending on OpenSearch being reachable.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/').status==200 else 1)"

# Production ASGI server (uvicorn[standard] is declared in requirements.txt).
# --proxy-headers / --forwarded-allow-ips let it sit correctly behind the
# HSRN Kubernetes ingress.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
