ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# pip network settings (overridable at build time)
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_EXTRA_INDEX_URL=
ENV PIP_DEFAULT_TIMEOUT=180 \
        PIP_RETRIES=10 \
        PIP_DISABLE_PIP_VERSION_CHECK=1 \
        PIP_NO_CACHE_DIR=1

# Install deps first for better Docker layer cache behavior
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
        if [ -n "$PIP_EXTRA_INDEX_URL" ]; then \
            pip install --prefer-binary --index-url "$PIP_INDEX_URL" --extra-index-url "$PIP_EXTRA_INDEX_URL" -r /app/requirements.txt; \
        else \
            pip install --prefer-binary --index-url "$PIP_INDEX_URL" -r /app/requirements.txt; \
        fi

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]