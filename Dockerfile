FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Install AveryML
COPY pyproject.toml README.md LICENSE ./
COPY averyml/ averyml/
COPY configs/ configs/
COPY scripts/ scripts/

RUN pip install --no-cache-dir -e ".[vllm,training,eval,dashboard]"

# HuggingFace cache
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

# Default volumes for data persistence
VOLUME ["/app/data", "/app/checkpoints", "/app/results", "/app/.cache"]

# Dashboard port
EXPOSE 7860

ENTRYPOINT ["averyml"]
CMD ["--help"]
