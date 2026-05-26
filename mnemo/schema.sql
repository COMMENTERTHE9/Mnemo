PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS video_metadata (
    video_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    duration_seconds REAL,
    fps REAL,
    width INTEGER,
    height INTEGER,
    created_at INTEGER NOT NULL,
    processed_at INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',           -- pending|downloading|processing|completed|failed
    motion_status TEXT NOT NULL DEFAULT 'pending',    -- pending|processing|completed|skipped|failed
    gapper_status TEXT NOT NULL DEFAULT 'pending'     -- pending|processing|completed|failed
);

CREATE TABLE IF NOT EXISTS processing_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 5,
    status TEXT NOT NULL DEFAULT 'pending',
    assigned_to TEXT,
    created_at INTEGER NOT NULL,
    started_at INTEGER,
    completed_at INTEGER,
    error_message TEXT,
    FOREIGN KEY (video_id) REFERENCES video_metadata(video_id)
);
CREATE INDEX IF NOT EXISTS idx_queue_status
    ON processing_queue(status, priority DESC, created_at);

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
    PRIMARY KEY (video_id, gapper_type, timestamp, gapper_id),
    FOREIGN KEY (video_id) REFERENCES video_metadata(video_id)
);
CREATE INDEX IF NOT EXISTS idx_gapper_importance
    ON gapper_reports(video_id, importance DESC);

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
    PRIMARY KEY (video_id, node_level, node_id),
    FOREIGN KEY (video_id) REFERENCES video_metadata(video_id)
);
CREATE INDEX IF NOT EXISTS idx_memory_parent ON memory_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_memory_time
    ON memory_nodes(video_id, start_time, end_time);

CREATE TABLE IF NOT EXISTS peripheral_detections (
    video_id TEXT NOT NULL,
    frame_number INTEGER NOT NULL,
    detection_id TEXT NOT NULL,
    box_coordinates TEXT NOT NULL,
    anomaly_type TEXT,
    confidence REAL,
    forwarded_to_gapper INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (video_id, frame_number, detection_id),
    FOREIGN KEY (video_id) REFERENCES video_metadata(video_id)
);
