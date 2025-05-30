FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY blog_fastapi.py .

# Create checkpoints directory
RUN mkdir -p checkpoints

EXPOSE 4000

# Set very long timeouts to handle ReAct processing
CMD ["uvicorn", "blog_fastapi:app", "--host", "0.0.0.0", "--port", "4000", "--timeout-keep-alive", "7200", "--timeout-graceful-shutdown", "7200", "--workers", "1"]

