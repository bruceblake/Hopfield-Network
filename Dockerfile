FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for applications
RUN pip install --no-cache-dir \
    opencv-python-headless \
    Pillow \
    seaborn \
    pytest \
    pytest-cov \
    pytest-benchmark

# Copy source code
COPY . .

# Make CLI executable
RUN chmod +x hopfield_cli.py

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "pytest", "tests/", "-v"]