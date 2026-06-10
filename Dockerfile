# Use BuildKit for faster builds: DOCKER_BUILDKIT=1 docker build .
FROM python:3.12 AS builder

WORKDIR /app

# Combine apt commands to reduce layers and clean up immediately
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmupdf-dev \ 
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# FIX: Use a cache mount for pip so subsequent builds are nearly instant
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user --no-cache-dir -r requirements.txt

# Final Stage
FROM python:3.12

WORKDIR /app

# Only install necessary runtime shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libharfbuzz0b \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY main.py .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Matches your CMD port
EXPOSE 8000

# Improved Healthcheck (uses internal python to avoid installing 'requests' in final image if possible)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"] 
