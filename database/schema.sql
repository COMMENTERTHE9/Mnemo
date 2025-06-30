-- SQLite schema for Video Memory Framework
-- Converted from ScyllaDB schema for local development

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Gapper reports table
CREATE TABLE IF NOT EXISTS gapper_reports (
    video_id TEXT NOT NULL,
    gapper_type TEXT NOT NULL,
    timestamp INTEGER NOT NULL, -- Unix timestamp in milliseconds
    gapper_id TEXT NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    summary TEXT,
    importance REAL,
    features TEXT, -- JSON string for map storage
    PRIMARY KEY (video_id, gapper_type, timestamp, gapper_id)
);

-- Index for performance
CREATE INDEX IF NOT EXISTS idx_gapper_reports_importance 
ON gapper_reports(video_id, importance DESC);

CREATE INDEX IF NOT EXISTS idx_gapper_reports_timestamp 
ON gapper_reports(video_id, timestamp DESC);

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
    narrative_tags TEXT, -- JSON array for set storage
    deleted_by_ai TEXT,
    compression_data TEXT,
    PRIMARY KEY (video_id, node_level, node_id)
);

-- Index for hierarchical queries
CREATE INDEX IF NOT EXISTS idx_memory_nodes_parent 
ON memory_nodes(parent_id);

CREATE INDEX IF NOT EXISTS idx_memory_nodes_time 
ON memory_nodes(video_id, start_time, end_time);

-- Peripheral detections table
CREATE TABLE IF NOT EXISTS peripheral_detections (
    video_id TEXT NOT NULL,
    frame_number INTEGER NOT NULL,
    detection_id TEXT NOT NULL,
    box_coordinates TEXT NOT NULL, -- JSON array [x, y, width, height]
    anomaly_type TEXT,
    confidence REAL,
    forwarded_to_gapper INTEGER DEFAULT 0, -- SQLite uses 0/1 for boolean
    PRIMARY KEY (video_id, frame_number, detection_id)
);

-- Index for frame lookups
CREATE INDEX IF NOT EXISTS idx_peripheral_frame 
ON peripheral_detections(video_id, frame_number);

-- Video metadata table (additional for SQLite)
CREATE TABLE IF NOT EXISTS video_metadata (
    video_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    duration_seconds REAL,
    fps REAL,
    width INTEGER,
    height INTEGER,
    created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    processed_at INTEGER,
    status TEXT DEFAULT 'pending' -- pending, processing, completed, failed
);

-- Processing queue table (for orchestrator)
CREATE TABLE IF NOT EXISTS processing_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    task_type TEXT NOT NULL, -- frame_gap, compress, peripheral_scan
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'pending', -- pending, assigned, processing, completed, failed
    assigned_to TEXT, -- worker ID
    created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    started_at INTEGER,
    completed_at INTEGER,
    error_message TEXT,
    FOREIGN KEY (video_id) REFERENCES video_metadata(video_id)
);

-- Index for queue operations
CREATE INDEX IF NOT EXISTS idx_queue_status 
ON processing_queue(status, priority DESC, created_at);