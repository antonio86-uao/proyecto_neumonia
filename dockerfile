
# Base image
FROM python:3.9-slim

# Install system dependencies for GUI
RUN apt-get update && apt-get install -y \
    python3-tk \
    libx11-6 \
    x11-utils \
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

# Command to run the application
CMD ["python3", "src/interface/detector_neumonia.py"]
