FROM golang:1.21-alpine AS builder

WORKDIR /app

# Install dependencies
RUN apk add --no-cache git gcc musl-dev sqlite-dev

# Copy go mod files
COPY go.mod ./
COPY go.sum* ./

# Download dependencies first
RUN go mod download

# Copy source code
COPY . ./

# Build the application
RUN go mod download && CGO_ENABLED=1 GOOS=linux go build -a -ldflags '-linkmode external -extldflags "-static"' -o orchestrator ./cmd/orchestrator

# Runtime stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy the binary from builder
COPY --from=builder /app/orchestrator .

EXPOSE 8080

CMD ["./orchestrator"]