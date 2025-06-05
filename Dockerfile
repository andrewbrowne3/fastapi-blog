FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create checkpoints directory
RUN mkdir -p checkpoints

# Expose port
EXPOSE 4000

# Set environment variables for production
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "main_with_db:app", "--host", "0.0.0.0", "--port", "4000", "--reload"]

