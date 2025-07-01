#!/usr/bin/env python3
"""
Simple video processor worker for Mnemo
Downloads videos and extracts frames for processing
"""

import os
import sys
import time
import json
import sqlite3
import logging
import subprocess
from pathlib import Path
import cv2
import numpy as np
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, db_path="/data/video_memory.db"):
        self.db_path = db_path
        self.work_dir = Path("/tmp/mnemo_work")
        self.work_dir.mkdir(exist_ok=True)
        
    def connect_db(self):
        """Connect to SQLite database"""
        return sqlite3.connect(self.db_path)
    
    def get_next_task(self):
        """Get next download task from queue"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        # Get pending download task
        cursor.execute("""
            SELECT id, video_id FROM processing_queue 
            WHERE task_type = 'download' AND status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
        """)
        
        task = cursor.fetchone()
        if task:
            task_id, video_id = task
            # Mark as processing
            cursor.execute("""
                UPDATE processing_queue 
                SET status = 'processing', started_at = ?
                WHERE id = ?
            """, (int(time.time() * 1000), task_id))
            conn.commit()
            
            # Get video URL
            cursor.execute("SELECT filename FROM video_metadata WHERE video_id = ?", (video_id,))
            video_url = cursor.fetchone()[0]
            
            conn.close()
            return task_id, video_id, video_url
        
        conn.close()
        return None, None, None
    
    def download_video(self, video_url, video_id):
        """Download video using yt-dlp"""
        output_path = self.work_dir / f"{video_id}.mp4"
        
        logger.info(f"Downloading video: {video_url}")
        
        # Use yt-dlp to download video
        cmd = [
            "yt-dlp",
            "-f", "best[ext=mp4]/best",
            "-o", str(output_path),
            "--no-playlist",
            video_url
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Video downloaded successfully: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download video: {e.stderr}")
            raise
    
    def extract_video_metadata(self, video_path):
        """Extract metadata from video file"""
        cap = cv2.VideoCapture(str(video_path))
        
        metadata = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return metadata
    
    def extract_frames(self, video_path, video_id, sample_rate=1.0):
        """Extract frames from video at given sample rate (frames per second)"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / sample_rate)
        
        frames_dir = self.work_dir / video_id / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        saved_count = 0
        
        logger.info(f"Extracting frames at {sample_rate} fps (every {frame_interval} frames)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Save frame
                frame_path = frames_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                
                # Calculate simple importance score (variance of Laplacian for blur detection)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                variance = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # Store frame data in database
                self.store_frame_data(video_id, frame_count, frame_count/fps, variance)
                
                saved_count += 1
                
                if saved_count % 10 == 0:
                    logger.info(f"Processed {saved_count} frames...")
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {saved_count} frames from {frame_count} total frames")
        return saved_count
    
    def store_frame_data(self, video_id, frame_number, timestamp, importance):
        """Store frame data as a gapper report"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        # Create a simple gapper report for the frame
        gapper_id = f"frame_gapper_{frame_number}"
        features = {
            "blur_variance": float(importance),
            "has_content": bool(importance > 100)  # Convert numpy bool to Python bool
        }
        
        cursor.execute("""
            INSERT INTO gapper_reports 
            (video_id, gapper_type, timestamp, gapper_id, start_frame, 
             end_frame, summary, importance, features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video_id, 
            "frame", 
            int(timestamp * 1000),  # Convert to milliseconds
            gapper_id,
            frame_number,
            frame_number,
            f"Frame at {timestamp:.2f}s",
            min(importance / 1000, 1.0),  # Normalize importance
            json.dumps(features)
        ))
        
        conn.commit()
        conn.close()
    
    def create_video_summary(self, video_id, metadata, frame_count):
        """Create a simple video summary"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        # Create root memory node
        root_node_id = f"{video_id}_root"
        
        cursor.execute("""
            INSERT INTO memory_nodes 
            (video_id, node_level, node_id, parent_id, start_time, 
             end_time, summary, importance, narrative_tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video_id,
            4,  # Meta level
            root_node_id,
            None,
            0.0,
            metadata['duration'],
            f"Video with {frame_count} extracted frames, {metadata['duration']:.1f} seconds long",
            1.0,
            json.dumps(["full_video", "processed"])
        ))
        
        # Update video metadata
        cursor.execute("""
            UPDATE video_metadata 
            SET duration_seconds = ?, fps = ?, width = ?, height = ?, 
                processed_at = ?, status = 'completed'
            WHERE video_id = ?
        """, (
            metadata['duration'],
            metadata['fps'],
            metadata['width'],
            metadata['height'],
            int(time.time() * 1000),
            video_id
        ))
        
        conn.commit()
        conn.close()
    
    def complete_task(self, task_id):
        """Mark task as completed"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE processing_queue 
            SET status = 'completed', completed_at = ?
            WHERE id = ?
        """, (int(time.time() * 1000), task_id))
        
        conn.commit()
        conn.close()
    
    def fail_task(self, task_id, error_msg):
        """Mark task as failed"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE processing_queue 
            SET status = 'failed', completed_at = ?, error_message = ?
            WHERE id = ?
        """, (int(time.time() * 1000), error_msg, task_id))
        
        conn.commit()
        conn.close()
    
    def process_one(self):
        """Process one video from the queue"""
        task_id, video_id, video_url = self.get_next_task()
        
        if not task_id:
            return False
        
        logger.info(f"Processing video {video_id} from {video_url}")
        
        try:
            # Download video
            video_path = self.download_video(video_url, video_id)
            
            # Extract metadata
            metadata = self.extract_video_metadata(video_path)
            logger.info(f"Video metadata: {metadata}")
            
            # Extract frames (1 frame per second)
            frame_count = self.extract_frames(video_path, video_id, sample_rate=1.0)
            
            # Create summary
            self.create_video_summary(video_id, metadata, frame_count)
            
            # Mark task as completed
            self.complete_task(task_id)
            
            # Cleanup
            os.remove(video_path)
            
            logger.info(f"Successfully processed video {video_id}")
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to process video {video_id}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.fail_task(task_id, str(e))
            return True
    
    def run(self):
        """Run the processor in a loop"""
        logger.info("Video processor started")
        
        while True:
            try:
                # Process one video
                had_work = self.process_one()
                
                if not had_work:
                    # No work available, wait
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.run()