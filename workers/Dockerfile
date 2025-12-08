FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker scripts
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs

# Health check (checks if worker process is running)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD pgrep -f "python" || exit 1

# Default command (can be overridden in docker-compose)
CMD ["python", "log_consumer.py"]