# Behavioral Authentication Engine - Production
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch for GNN (optional)
RUN pip install torch torch-geometric --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 authuser && chown -R authuser:authuser /app
USER authuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "from verify_layers import main; main()" || exit 1

# Expose port
EXPOSE 8001

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
