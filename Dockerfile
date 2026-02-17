FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

# Use the same startup command as startup.sh
CMD ["gunicorn", "--bind=0.0.0.0:8000", "--timeout=600", "--workers=2", "app.main:app", "-k", "uvicorn.workers.UvicornWorker"]
