package main

import (
    "log"
    "net"
    "net/http"
    "os"
    "os/signal"
    "syscall"

    "github.com/video-memory/orchestrator/internal/database"
    "google.golang.org/grpc"
)

func main() {
    log.Println("Starting Video Memory Orchestrator")

    // Get configuration from environment
    port := os.Getenv("ORCHESTRATOR_PORT")
    if port == "" {
        port = "8080"
    }

    // Initialize database connection
    db, err := database.NewDB()
    if err != nil {
        log.Fatalf("Failed to connect to database: %v", err)
    }
    defer db.Close()
    
    log.Println("Database connection established")

    // Create health check endpoint
    http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        w.Write([]byte("OK"))
    })
    
    // Start HTTP server for health checks
    go func() {
        if err := http.ListenAndServe(":8080", nil); err != nil {
            log.Printf("Health check server error: %v", err)
        }
    }()

    // Create gRPC server
    lis, err := net.Listen("tcp", ":50051")
    if err != nil {
        log.Fatalf("Failed to listen: %v", err)
    }

    grpcServer := grpc.NewServer()
    
    // TODO: Register gRPC services
    // TODO: Initialize coordinator with database
    
    // Start server in goroutine
    go func() {
        log.Printf("gRPC server listening on port 50051")
        if err := grpcServer.Serve(lis); err != nil {
            log.Fatalf("Failed to serve: %v", err)
        }
    }()

    log.Printf("Orchestrator ready - HTTP on :%s, gRPC on :50051", port)

    // Wait for interrupt signal
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
    <-sigChan

    log.Println("Shutting down orchestrator...")
    grpcServer.GracefulStop()
}