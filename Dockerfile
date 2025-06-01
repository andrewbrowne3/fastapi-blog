FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY blog_fastapi.py .
COPY .env .

# Create checkpoints directory
RUN mkdir -p checkpoints

# Expose port 4000
EXPOSE 4000

# Set environment variables for production
ENV PYTHONUNBUFFERED=1

# Set very long timeouts to handle ReAct processing
CMD ["uvicorn", "blog_fastapi:app", "--host", "0.0.0.0", "--port", "4000", "--timeout-keep-alive", "7200", "--timeout-graceful-shutdown", "7200", "--workers", "1"]

