FROM python:3.10-slim

# Set environment variables for better pip behavior
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set cache directories to /app instead of home directory
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file FIRST (for better caching)
COPY requirements.txt .

# Install Python dependencies (this layer will be cached unless requirements.txt changes)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code LAST (so code changes don't invalidate pip cache)
COPY app/ ./app/

# Create necessary directories
RUN mkdir -p logs && \
    mkdir -p .cache/huggingface && \
    chmod 755 .cache && \
    chmod 755 .cache/huggingface

# Create a non-root user for security
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8080

# Health check using python instead of curl for non-root user
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"] 