CREATE KEYSPACE IF NOT EXISTS video_memory
WITH replication = {
    'class': 'NetworkTopologyStrategy',
    'datacenter1': 3
};

USE video_memory;

-- Gapper reports table (time-series optimized)
CREATE TABLE gapper_reports (
    video_id UUID,
    gapper_type TEXT,
    timestamp TIMESTAMP,
    gapper_id TEXT,
    start_frame INT,
    end_frame INT,
    summary TEXT,
    importance FLOAT,
    features FROZEN<MAP<TEXT, TEXT>>,
    PRIMARY KEY ((video_id, gapper_type), timestamp, gapper_id)
) WITH CLUSTERING ORDER BY (timestamp DESC, gapper_id ASC)
AND compaction = {
    'class': 'TimeWindowCompactionStrategy',
    'compaction_window_unit': 'HOURS',
    'compaction_window_size': 1
};

-- Memory hierarchy table
CREATE TABLE memory_nodes (
    video_id UUID,
    node_level INT,
    node_id UUID,
    parent_id UUID,
    start_time DECIMAL,
    end_time DECIMAL,
    summary TEXT,
    importance FLOAT,
    narrative_tags SET<TEXT>,
    deleted_by_ai TEXT,
    compression_data TEXT,
    PRIMARY KEY ((video_id, node_level), node_id)
);

-- Peripheral detections table
CREATE TABLE peripheral_detections (
    video_id UUID,
    frame_number INT,
    detection_id UUID,
    box_coordinates LIST<INT>,
    anomaly_type TEXT,
    confidence FLOAT,
    forwarded_to_gapper BOOLEAN,
    PRIMARY KEY ((video_id), frame_number, detection_id)
) WITH CLUSTERING ORDER BY (frame_number ASC);

-- Create materialized view for fast lookups
CREATE MATERIALIZED VIEW reports_by_importance AS
    SELECT * FROM gapper_reports
    WHERE video_id IS NOT NULL 
    AND gapper_type IS NOT NULL
    AND timestamp IS NOT NULL
    AND gapper_id IS NOT NULL
    AND importance IS NOT NULL
    PRIMARY KEY ((video_id, importance), gapper_type, timestamp, gapper_id)
    WITH CLUSTERING ORDER BY (gapper_type ASC, timestamp DESC);