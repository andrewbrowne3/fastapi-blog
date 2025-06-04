FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Ensure .env is copied (in case .dockerignore excludes it)
COPY .env .

# Create checkpoints directory
RUN mkdir -p checkpoints

# Expose port 4000
EXPOSE 4000

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set very long timeouts to handle ReAct processing
CMD ["uvicorn", "main_with_db:app", "--host", "0.0.0.0", "--port", "4000", "--timeout-keep-alive", "7200", "--timeout-graceful-shutdown", "7200", "--workers", "1"]

