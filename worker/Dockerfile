FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install yt-dlp
RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp && \
    chmod a+rx /usr/local/bin/yt-dlp

# Install Python dependencies
RUN pip install --no-cache-dir \
    opencv-python==4.8.1.* \
    numpy==1.24.* \
    Pillow==10.1.*

# Create working directory
WORKDIR /app

# Copy worker script
COPY video_processor.py .

# Create data directory
RUN mkdir -p /data /tmp/mnemo_work

# Run the worker
CMD ["python", "video_processor.py"]