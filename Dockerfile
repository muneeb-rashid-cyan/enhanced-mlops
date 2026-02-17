# Multi-stage Docker build for Enhanced MLOps API

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY pyproject.toml ./
COPY requirements.txt* ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir uv
RUN uv pip install fastapi uvicorn[standard] scikit-learn mlflow joblib numpy pandas pydantic python-multipart requests

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r mlops && useradd -r -g mlops mlops

# Copy application code
COPY app/ ./app/
COPY models/ ./models/
COPY tests/ ./tests/
COPY *.py ./

# Create necessary directories with proper permissions
RUN mkdir -p mlruns logs models/trained && \
    chown -R mlops:mlops /app

# Switch to non-root user
USER mlops

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]