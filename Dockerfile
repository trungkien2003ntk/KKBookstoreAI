FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY ./app/requirements.txt ./

# Install dependencies
RUN pip install -r requirements.txt

# Copy the model download script first
COPY ./app/download_models.py ./

# Set environment variables for model caching locations
ENV TORCH_HOME=/app/models/torch
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV HF_HOME=/app/models/huggingface
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create model directories
RUN mkdir -p /app/models/torch /app/models/transformers /app/models/huggingface

# Download models at build time
RUN python download_models.py

# Copy the rest of the application
COPY ./app .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]