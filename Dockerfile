# Base image: Python 3.11 slim variant for universal CPU compatibility
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install system dependencies
# - libgl1-mesa-glx and libglib2.0-0 for OpenCV
# - git for installing dependencies like SAM 2 via git
# - build-essential for any C-extension compilations
# - wget/curl for network operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    git \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (astral) for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml requirements.txt ./

# Create virtual environment and install dependencies using uv
# Using a system-level python approach in Docker is also fine, but uv venv is clean.
# We will use uv pip install --system to install globally inside the container.
RUN uv pip install --system -e . && \
    uv pip install --system -r requirements.txt

# Create directories for mounted volumes
RUN mkdir -p data outputs weights media

# Copy the rest of the application code
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/

# Download weights during the build step (optional, but saves time later)
RUN python scripts/download_weights.py

# Set the default command (can be overridden when running)
CMD ["python", "-m", "pytest", "tests/", "-v"]
