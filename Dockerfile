# Production Dockerfile for TitleCraft AI
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r titlecraft && useradd -r -g titlecraft titlecraft

# Copy requirements and install Python dependencies
COPY requirements.txt .
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

# Create directories for logs and cache
RUN mkdir -p logs cache && \
    chown -R titlecraft:titlecraft /app

# Switch to non-root user
USER titlecraft

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command with Gunicorn
CMD ["gunicorn", "src.api.production_app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--access-logfile", "logs/access.log", "--error-logfile", "logs/error.log", "--log-level", "info"]