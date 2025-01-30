# Base image con versión específica
FROM python:3.9.18-slim

# Install system dependencies for GUI and OpenCV
RUN apt-get update && apt-get install -y \
    python3-tk \
    libx11-6 \
    x11-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Install the package
RUN pip install -e .

# Environment variable for display
ENV DISPLAY=:0

# Agregar healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD ps aux | grep python3 || exit 1

# Command to run the application
CMD ["python3", "-m", "src.interface.detector_neumonia"]