package main

import (
    "log"
    "net"
    "net/http"
    "os"
    "os/signal"
    "syscall"

    "github.com/video-memory/orchestrator/internal/database"
    "github.com/video-memory/orchestrator/internal/api"
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

    // Create API handler
    handler := api.NewHandler(db)

    // Setup HTTP routes
    http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        w.Write([]byte("OK"))
    })
    
    // CORS middleware wrapper
    withCORS := func(h http.HandlerFunc) http.HandlerFunc {
        return func(w http.ResponseWriter, r *http.Request) {
            w.Header().Set("Access-Control-Allow-Origin", "*")
            w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
            
            if r.Method == "OPTIONS" {
                w.WriteHeader(http.StatusOK)
                return
            }
            
            h(w, r)
        }
    }
    
    // API endpoints with CORS
    http.HandleFunc("/api/v1/video/process", withCORS(handler.ProcessVideo))
    http.HandleFunc("/api/v1/memory/", withCORS(handler.QueryMemory))
    http.HandleFunc("/api/v1/video/status", withCORS(handler.GetVideoStatus))
    http.HandleFunc("/api/v1/auth/cookies", withCORS(handler.UpdateCookies))
    
    // Start HTTP server
    go func() {
        log.Printf("Starting HTTP server on :8080")
        if err := http.ListenAndServe(":8080", nil); err != nil {
            log.Printf("HTTP server error: %v", err)
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