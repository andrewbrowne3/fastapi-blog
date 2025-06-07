FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including PostgreSQL client libraries
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints storage

# Expose port 4000
EXPOSE 4000

# Set environment variables for production
ENV PYTHONUNBUFFERED=1

# Run the application with main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000", "--timeout-keep-alive", "7200", "--timeout-graceful-shutdown", "7200", "--workers", "1"]

