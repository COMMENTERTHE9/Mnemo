FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    opencv-python==4.8.1.* \
    mediapipe==0.10.* \
    numpy==1.24.* \
    Pillow==10.1.*

# Create working directory
WORKDIR /app

# Copy motion extractor script
COPY motion-extractor/motion_extractor.py .

# Create data directories
RUN mkdir -p /data /tmp/mnemo_work

# Run the motion extractor
CMD ["python", "motion_extractor.py"]