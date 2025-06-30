FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel AS builder

# Prevent timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Skip gRPC for now - can be added later
# RUN apt-get update && apt-get install -y \
#     libgrpc++-dev \
#     protobuf-compiler-grpc \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy CMakeLists and source
COPY ai-inference/CMakeLists.txt ./
COPY ai-inference/src ./src
COPY ai-inference/include ./include

# Build
RUN mkdir build && cd build && \
    cmake -DCMAKE_PREFIX_PATH=/opt/conda/lib/python3.10/site-packages/torch .. && \
    make -j$(nproc)

# Runtime stage
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Copy built executable
COPY --from=builder /app/build/ai_inference .

# Copy models
COPY models/exported /models/exported

ENV AI_DEVICE=cuda
ENV MODEL_PATH=/models/exported

CMD ["./ai_inference"]