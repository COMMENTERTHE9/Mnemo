package api

import (
    "encoding/json"
    "fmt"
    "net/http"
    "time"
    "log"
    "database/sql"
    "os"
)

type ProcessVideoRequest struct {
    VideoURL        string                 `json:"video_url"`
    ProcessingLevel string                 `json:"processing_level"`
    Options         map[string]interface{} `json:"options,omitempty"`
}

type ProcessVideoResponse struct {
    VideoID   string `json:"video_id"`
    Status    string `json:"status"`
    Message   string `json:"message"`
    CreatedAt string `json:"created_at"`
}

type QueryMemoryRequest struct {
    Query       string  `json:"query"`
    DetailLevel int     `json:"detail_level,omitempty"`
    StartTime   float64 `json:"start_time,omitempty"`
    EndTime     float64 `json:"end_time,omitempty"`
}

type QueryMemoryResponse struct {
    VideoID string         `json:"video_id"`
    Results []QueryResult  `json:"results"`
}

type QueryResult struct {
    NodeID          string  `json:"node_id"`
    RelevanceScore  float32 `json:"relevance_score"`
    Timestamp       float64 `json:"timestamp"`
    Summary         string  `json:"summary"`
    Context         string  `json:"context"`
    Reconstructable bool    `json:"reconstructable"`
}

type Handler struct {
    db *sql.DB
}

func NewHandler(db *sql.DB) *Handler {
    return &Handler{db: db}
}

// ProcessVideo handles video processing requests
func (h *Handler) ProcessVideo(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var req ProcessVideoRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request body", http.StatusBadRequest)
        return
    }

    // Validate request
    if req.VideoURL == "" {
        http.Error(w, "video_url is required", http.StatusBadRequest)
        return
    }

    // Generate video ID
    videoID := fmt.Sprintf("video_%d", time.Now().UnixNano())
    
    // Insert into database
    _, err := h.db.Exec(`
        INSERT INTO video_metadata (video_id, filename, status, created_at)
        VALUES (?, ?, 'pending', ?)
    `, videoID, req.VideoURL, time.Now().Unix()*1000)
    
    if err != nil {
        log.Printf("Failed to insert video metadata: %v", err)
        http.Error(w, "Failed to create video record", http.StatusInternalServerError)
        return
    }

    // Add to processing queue
    _, err = h.db.Exec(`
        INSERT INTO processing_queue (video_id, task_type, priority, status)
        VALUES (?, 'download', 10, 'pending')
    `, videoID)
    
    if err != nil {
        log.Printf("Failed to add to queue: %v", err)
        http.Error(w, "Failed to queue video", http.StatusInternalServerError)
        return
    }

    resp := ProcessVideoResponse{
        VideoID:   videoID,
        Status:    "queued",
        Message:   "Video queued for processing",
        CreatedAt: time.Now().Format(time.RFC3339),
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(resp)
}

// QueryMemory handles memory query requests
func (h *Handler) QueryMemory(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodGet && r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    // Extract video ID from URL path
    videoID := r.URL.Path[len("/api/v1/memory/"):]
    if idx := len(videoID) - len("/query"); idx > 0 {
        videoID = videoID[:idx]
    }

    var req QueryMemoryRequest
    if r.Method == http.MethodPost {
        if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
            http.Error(w, "Invalid request body", http.StatusBadRequest)
            return
        }
    } else {
        req.Query = r.URL.Query().Get("q")
    }

    // Query database for memory nodes
    rows, err := h.db.Query(`
        SELECT node_id, summary, importance, start_time
        FROM memory_nodes
        WHERE video_id = ? AND importance > 0.3
        ORDER BY importance DESC
        LIMIT 10
    `, videoID)
    
    if err != nil {
        log.Printf("Failed to query memory nodes: %v", err)
        http.Error(w, "Failed to query memories", http.StatusInternalServerError)
        return
    }
    defer rows.Close()

    var results []QueryResult
    for rows.Next() {
        var result QueryResult
        var importance float32
        err := rows.Scan(&result.NodeID, &result.Summary, &importance, &result.Timestamp)
        if err != nil {
            continue
        }
        result.RelevanceScore = importance
        result.Reconstructable = true
        results = append(results, result)
    }

    resp := QueryMemoryResponse{
        VideoID: videoID,
        Results: results,
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(resp)
}

// GetVideoStatus returns the processing status of a video
func (h *Handler) GetVideoStatus(w http.ResponseWriter, r *http.Request) {
    videoID := r.URL.Query().Get("video_id")
    if videoID == "" {
        http.Error(w, "video_id parameter required", http.StatusBadRequest)
        return
    }

    var status string
    err := h.db.QueryRow(`
        SELECT status FROM video_metadata WHERE video_id = ?
    `, videoID).Scan(&status)
    
    if err == sql.ErrNoRows {
        http.Error(w, "Video not found", http.StatusNotFound)
        return
    } else if err != nil {
        http.Error(w, "Database error", http.StatusInternalServerError)
        return
    }

    resp := map[string]string{
        "video_id": videoID,
        "status":   status,
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(resp)
}

// UpdateCookies handles cookie updates from browser extension
func (h *Handler) UpdateCookies(w http.ResponseWriter, r *http.Request) {
    // Enable CORS for browser extension
    w.Header().Set("Access-Control-Allow-Origin", "*")
    w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
    w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
    
    // Handle preflight OPTIONS request
    if r.Method == http.MethodOptions {
        w.WriteHeader(http.StatusOK)
        return
    }
    
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    // Read cookie data
    var cookieData struct {
        Cookies string `json:"cookies"`
    }
    
    if err := json.NewDecoder(r.Body).Decode(&cookieData); err != nil {
        http.Error(w, "Invalid request body", http.StatusBadRequest)
        return
    }

    // Write cookies to file
    cookiesPath := "/data/youtube_cookies.txt"
    if err := os.WriteFile(cookiesPath, []byte(cookieData.Cookies), 0600); err != nil {
        log.Printf("Failed to write cookies: %v", err)
        http.Error(w, "Failed to save cookies", http.StatusInternalServerError)
        return
    }

    log.Printf("Updated YouTube cookies")
    
    resp := map[string]string{
        "status": "success",
        "message": "Cookies updated successfully",
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(resp)
}