package database

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// DB interface allows switching between SQLite and ScyllaDB
type DB interface {
	GetPendingTasks(limit int) ([]ProcessingTask, error)
	AssignTask(taskID int64, workerID string) error
	CompleteTask(taskID int64) error
	FailTask(taskID int64, errorMsg string) error
	InsertGapperReport(report GapperReport) error
	InsertMemoryNode(node MemoryNode) error
	Close() error
}

// SQLiteDB implements DB interface for SQLite
type SQLiteDB struct {
	conn *sql.DB
}

// ProcessingTask represents a task in the queue
type ProcessingTask struct {
	ID        int64     `json:"id"`
	VideoID   string    `json:"video_id"`
	TaskType  string    `json:"task_type"`
	Priority  int       `json:"priority"`
	Status    string    `json:"status"`
	CreatedAt time.Time `json:"created_at"`
}

// GapperReport represents a gapper analysis result
type GapperReport struct {
	VideoID    string                 `json:"video_id"`
	GapperType string                 `json:"gapper_type"`
	Timestamp  time.Time              `json:"timestamp"`
	GapperID   string                 `json:"gapper_id"`
	StartFrame int                    `json:"start_frame"`
	EndFrame   int                    `json:"end_frame"`
	Summary    string                 `json:"summary"`
	Importance float32                `json:"importance"`
	Features   map[string]interface{} `json:"features"`
}

// MemoryNode represents a node in the memory hierarchy
type MemoryNode struct {
	VideoID        string   `json:"video_id"`
	NodeLevel      int      `json:"node_level"`
	NodeID         string   `json:"node_id"`
	ParentID       string   `json:"parent_id"`
	StartTime      float64  `json:"start_time"`
	EndTime        float64  `json:"end_time"`
	Summary        string   `json:"summary"`
	Importance     float32  `json:"importance"`
	NarrativeTags  []string `json:"narrative_tags"`
	DeletedByAI    string   `json:"deleted_by_ai"`
	CompressionData string  `json:"compression_data"`
}

// NewSQLiteDB creates a new SQLite database connection
func NewSQLiteDB(dbPath string) (*SQLiteDB, error) {
	if dbPath == "" {
		dbPath = "./data/video_memory.db"
	}

	log.Printf("Connecting to SQLite database: %s", dbPath)

	// Ensure directory exists
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory: %w", err)
	}

	conn, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Test connection
	if err := conn.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	// Enable foreign keys
	if _, err := conn.Exec("PRAGMA foreign_keys = ON"); err != nil {
		return nil, fmt.Errorf("failed to enable foreign keys: %w", err)
	}

	// Initialize schema if needed
	if err := initializeSchema(conn); err != nil {
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	return &SQLiteDB{conn: conn}, nil
}

// GetPendingTasks retrieves pending tasks from the queue
func (db *SQLiteDB) GetPendingTasks(limit int) ([]ProcessingTask, error) {
	query := `
		SELECT id, video_id, task_type, priority, status, created_at
		FROM processing_queue
		WHERE status = 'pending'
		ORDER BY priority DESC, created_at ASC
		LIMIT ?
	`

	rows, err := db.conn.Query(query, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var tasks []ProcessingTask
	for rows.Next() {
		var task ProcessingTask
		var createdAtMs int64
		
		err := rows.Scan(&task.ID, &task.VideoID, &task.TaskType, 
			&task.Priority, &task.Status, &createdAtMs)
		if err != nil {
			return nil, err
		}
		
		task.CreatedAt = time.Unix(createdAtMs/1000, (createdAtMs%1000)*1e6)
		tasks = append(tasks, task)
	}

	return tasks, nil
}

// AssignTask assigns a task to a worker
func (db *SQLiteDB) AssignTask(taskID int64, workerID string) error {
	query := `
		UPDATE processing_queue 
		SET status = 'assigned', 
		    assigned_to = ?,
		    started_at = ?
		WHERE id = ? AND status = 'pending'
	`

	result, err := db.conn.Exec(query, workerID, time.Now().UnixMilli(), taskID)
	if err != nil {
		return err
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}

	if rowsAffected == 0 {
		return fmt.Errorf("task %d not found or already assigned", taskID)
	}

	return nil
}

// CompleteTask marks a task as completed
func (db *SQLiteDB) CompleteTask(taskID int64) error {
	query := `
		UPDATE processing_queue 
		SET status = 'completed',
		    completed_at = ?
		WHERE id = ?
	`

	_, err := db.conn.Exec(query, time.Now().UnixMilli(), taskID)
	return err
}

// FailTask marks a task as failed with an error message
func (db *SQLiteDB) FailTask(taskID int64, errorMsg string) error {
	query := `
		UPDATE processing_queue 
		SET status = 'failed',
		    completed_at = ?,
		    error_message = ?
		WHERE id = ?
	`

	_, err := db.conn.Exec(query, time.Now().UnixMilli(), errorMsg, taskID)
	return err
}

// InsertGapperReport inserts a new gapper report
func (db *SQLiteDB) InsertGapperReport(report GapperReport) error {
	featuresJSON, err := json.Marshal(report.Features)
	if err != nil {
		return err
	}

	query := `
		INSERT INTO gapper_reports 
		(video_id, gapper_type, timestamp, gapper_id, start_frame, 
		 end_frame, summary, importance, features)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
	`

	_, err = db.conn.Exec(query,
		report.VideoID, report.GapperType, report.Timestamp.UnixMilli(),
		report.GapperID, report.StartFrame, report.EndFrame,
		report.Summary, report.Importance, string(featuresJSON))

	return err
}

// InsertMemoryNode inserts a new memory node
func (db *SQLiteDB) InsertMemoryNode(node MemoryNode) error {
	tagsJSON, err := json.Marshal(node.NarrativeTags)
	if err != nil {
		return err
	}

	query := `
		INSERT INTO memory_nodes 
		(video_id, node_level, node_id, parent_id, start_time, 
		 end_time, summary, importance, narrative_tags, deleted_by_ai, compression_data)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`

	_, err = db.conn.Exec(query,
		node.VideoID, node.NodeLevel, node.NodeID, node.ParentID,
		node.StartTime, node.EndTime, node.Summary, node.Importance,
		string(tagsJSON), node.DeletedByAI, node.CompressionData)

	return err
}

// Close closes the database connection
func (db *SQLiteDB) Close() error {
	return db.conn.Close()
}

// NewDB creates a database connection based on environment configuration
func NewDB() (*sql.DB, error) {
	dbType := os.Getenv("DATABASE_TYPE")
	if dbType == "" {
		dbType = "sqlite"
	}

	switch dbType {
	case "sqlite":
		dbPath := os.Getenv("DATABASE_PATH")
		sqliteDB, err := NewSQLiteDB(dbPath)
		if err != nil {
			return nil, err
		}
		return sqliteDB.conn, nil
	case "scylla":
		// TODO: Implement ScyllaDB connection for production
		return nil, fmt.Errorf("ScyllaDB support not yet implemented")
	default:
		return nil, fmt.Errorf("unknown database type: %s", dbType)
	}
}

// initializeSchema creates tables if they don't exist
func initializeSchema(db *sql.DB) error {
	schema := `
	-- Enable foreign keys
	PRAGMA foreign_keys = ON;

	-- Gapper reports table
	CREATE TABLE IF NOT EXISTS gapper_reports (
		video_id TEXT NOT NULL,
		gapper_type TEXT NOT NULL,
		timestamp INTEGER NOT NULL,
		gapper_id TEXT NOT NULL,
		start_frame INTEGER NOT NULL,
		end_frame INTEGER NOT NULL,
		summary TEXT,
		importance REAL,
		features TEXT,
		PRIMARY KEY (video_id, gapper_type, timestamp, gapper_id)
	);

	-- Memory hierarchy table
	CREATE TABLE IF NOT EXISTS memory_nodes (
		video_id TEXT NOT NULL,
		node_level INTEGER NOT NULL,
		node_id TEXT NOT NULL,
		parent_id TEXT,
		start_time REAL NOT NULL,
		end_time REAL NOT NULL,
		summary TEXT,
		importance REAL,
		narrative_tags TEXT,
		deleted_by_ai TEXT,
		compression_data TEXT,
		PRIMARY KEY (video_id, node_level, node_id)
	);

	-- Peripheral detections table
	CREATE TABLE IF NOT EXISTS peripheral_detections (
		video_id TEXT NOT NULL,
		frame_number INTEGER NOT NULL,
		detection_id TEXT NOT NULL,
		box_coordinates TEXT NOT NULL,
		anomaly_type TEXT,
		confidence REAL,
		forwarded_to_gapper INTEGER DEFAULT 0,
		PRIMARY KEY (video_id, frame_number, detection_id)
	);

	-- Video metadata table
	CREATE TABLE IF NOT EXISTS video_metadata (
		video_id TEXT PRIMARY KEY,
		filename TEXT NOT NULL,
		duration_seconds REAL,
		fps REAL,
		width INTEGER,
		height INTEGER,
		created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000),
		processed_at INTEGER,
		status TEXT DEFAULT 'pending'
	);

	-- Processing queue table
	CREATE TABLE IF NOT EXISTS processing_queue (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		video_id TEXT NOT NULL,
		task_type TEXT NOT NULL,
		priority INTEGER DEFAULT 5,
		status TEXT DEFAULT 'pending',
		assigned_to TEXT,
		created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000),
		started_at INTEGER,
		completed_at INTEGER,
		error_message TEXT,
		FOREIGN KEY (video_id) REFERENCES video_metadata(video_id)
	);
	`

	_, err := db.Exec(schema)
	return err
}