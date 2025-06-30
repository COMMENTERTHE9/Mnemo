#!/bin/bash

# Initialize SQLite database for Video Memory Framework

DB_PATH="${1:-./data/video_memory.db}"
SCHEMA_PATH="./database/schema.sql"

# Create data directory if it doesn't exist
mkdir -p "$(dirname "$DB_PATH")"

echo "Initializing SQLite database at: $DB_PATH"

# Create database and apply schema
if [ -f "$DB_PATH" ]; then
    echo "Database already exists. Backing up..."
    cp "$DB_PATH" "${DB_PATH}.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Apply schema
sqlite3 "$DB_PATH" < "$SCHEMA_PATH"

if [ $? -eq 0 ]; then
    echo "Database initialized successfully!"
    
    # Insert some test data if database is empty
    VIDEO_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM video_metadata;")
    
    if [ "$VIDEO_COUNT" -eq "0" ]; then
        echo "Inserting test data..."
        sqlite3 "$DB_PATH" <<EOF
-- Insert test video
INSERT INTO video_metadata (video_id, filename, duration_seconds, fps, width, height, status)
VALUES ('test-video-001', 'test_video.mp4', 120.5, 30.0, 1920, 1080, 'pending');

-- Insert test gapper report
INSERT INTO gapper_reports (video_id, gapper_type, timestamp, gapper_id, start_frame, end_frame, summary, importance, features)
VALUES ('test-video-001', 'scene', strftime('%s', 'now') * 1000, 'gap-001', 0, 300, 'Opening scene', 0.8, '{"scene_type": "intro"}');

-- Insert test memory node
INSERT INTO memory_nodes (video_id, node_level, node_id, parent_id, start_time, end_time, summary, importance, narrative_tags)
VALUES ('test-video-001', 1, 'node-001', NULL, 0.0, 10.0, 'Introduction segment', 0.9, '["intro", "establishing_shot"]');

-- Insert test processing task
INSERT INTO processing_queue (video_id, task_type, priority)
VALUES ('test-video-001', 'frame_gap', 10);
EOF
        echo "Test data inserted."
    fi
    
    # Show database info
    echo -e "\nDatabase info:"
    echo "Size: $(du -h "$DB_PATH" | cut -f1)"
    echo "Tables:"
    sqlite3 "$DB_PATH" ".tables"
    
else
    echo "Error: Failed to initialize database"
    exit 1
fi