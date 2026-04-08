FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app/

# Install CPU-only PyTorch FIRST (avoids pulling CUDA variant)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Default: run calm-market task
ENV TASK_ID=calm-market
CMD ["python", "-m", "server.app"]
