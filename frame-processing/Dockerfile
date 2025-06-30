FROM rust:1.81 as builder

WORKDIR /app

# Copy manifests
COPY frame-processing/Cargo.toml frame-processing/Cargo.lock ./

# Build dependencies - this is the caching Docker layer!
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# Copy source code
COPY frame-processing/src ./src
# Proto files will be copied separately
COPY proto ./proto

# Build application
RUN touch src/main.rs
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/frame-processing /usr/local/bin/

CMD ["frame-processing"]