FROM python:3.10-slim

WORKDIR /app

# Install dependencies needed for downloading and processing models
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY ./app/requirements.txt ./

# Add requests to the dependencies if not already present
RUN pip install -r requirements.txt && \
    pip install requests

# Set environment variables for model caching locations
ENV TORCH_HOME=/models/torch
ENV TRANSFORMERS_CACHE=/models/transformers
ENV HF_HOME=/models/huggingface
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create model directories with proper permissions
RUN mkdir -p /models/torch/hub/checkpoints /models/transformers /models/huggingface && \
    chmod -R 777 /models

# Copy the model download script
COPY ./app/download_models.py ./

# Download models at build time with retry logic
RUN python download_models.py || (sleep 5 && python download_models.py) || (sleep 10 && python download_models.py)

# Verify the model files exist and show cache structure
RUN echo "=== Model cache structure ===" && \
    find /models -name "*.pth" -o -name "*.bin" -o -name "*.json" | head -20 && \
    echo "=== Torch hub structure ===" && \
    ls -la /models/torch/hub/ || echo "No torch hub directory" && \
    echo "=== HuggingFace cache structure ===" && \
    ls -la /models/huggingface/ || echo "No HF cache directory"

# Copy the rest of the application
COPY ./app .

# Make sure we run with access to the model cache
RUN chmod -R 755 /models

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
