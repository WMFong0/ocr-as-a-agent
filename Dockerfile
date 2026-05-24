# Multi-stage build for smaller final image.
FROM python:3.13.12-slim as builder

WORKDIR /app

# Install system dependencies for building wheels (PIL, PyMuPDF, etc.).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage - minimal production image.
FROM python:3.13.12-slim

WORKDIR /app

# Install only runtime system dependencies (for PDF rendering, image processing).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libharfbuzz0b \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage.
COPY --from=builder /root/.local /root/.local

# Copy application code.
COPY main.py .

# Ensure pip packages are on PATH.
ENV PATH=/root/.local/bin:$PATH

# Disable Python output buffering for real-time Docker logs.
ENV PYTHONUNBUFFERED=1

# Expose to 8082
EXPOSE 8002

# Health check.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Run the FastAPI app.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
