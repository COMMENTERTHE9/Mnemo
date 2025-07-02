#!/usr/bin/env python3
"""
Motion extraction service using MediaPipe
Processes frames to extract human pose data
"""

import os
import sys
import json
import sqlite3
import logging
import time
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MotionExtractor:
    def __init__(self, db_path="/data/video_memory.db"):
        self.db_path = db_path
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 0, 1, or 2. Higher = more accurate but slower
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # MediaPipe Holistic for full body + hands + face
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def connect_db(self):
        """Connect to SQLite database"""
        return sqlite3.connect(self.db_path)
    
    def get_next_task(self):
        """Get next motion extraction task"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        # Get videos that have been processed but not motion extracted
        cursor.execute("""
            SELECT DISTINCT vm.video_id 
            FROM video_metadata vm
            WHERE vm.status = 'completed'
            AND NOT EXISTS (
                SELECT 1 FROM gapper_reports gr 
                WHERE gr.video_id = vm.video_id 
                AND gr.gapper_type = 'motion'
            )
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def extract_pose_from_frame(self, frame):
        """Extract pose landmarks from a single frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Convert landmarks to dict
            landmarks = {}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks[f"point_{idx}"] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                }
            return landmarks
        return None
    
    def extract_holistic_from_frame(self, frame):
        """Extract full body, hands, and face landmarks"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        holistic_data = {}
        
        # Pose landmarks (33 points)
        if results.pose_landmarks:
            holistic_data["pose"] = {}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                holistic_data["pose"][self.get_pose_landmark_name(idx)] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                }
        
        # Face landmarks (468 points)
        if results.face_landmarks:
            holistic_data["face"] = {
                "landmark_count": len(results.face_landmarks.landmark),
                "detected": True
            }
        
        # Left hand (21 points)
        if results.left_hand_landmarks:
            holistic_data["left_hand"] = {}
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                holistic_data["left_hand"][f"point_{idx}"] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                }
        
        # Right hand (21 points)
        if results.right_hand_landmarks:
            holistic_data["right_hand"] = {}
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                holistic_data["right_hand"][f"point_{idx}"] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                }
        
        return holistic_data if holistic_data else None
    
    def get_pose_landmark_name(self, idx):
        """Get human-readable name for pose landmark"""
        landmark_names = {
            0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
            4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
            7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
            11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow",
            14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
            17: "left_pinky", 18: "right_pinky", 19: "left_index",
            20: "right_index", 21: "left_thumb", 22: "right_thumb",
            23: "left_hip", 24: "right_hip", 25: "left_knee",
            26: "right_knee", 27: "left_ankle", 28: "right_ankle",
            29: "left_heel", 30: "right_heel", 31: "left_foot_index",
            32: "right_foot_index"
        }
        return landmark_names.get(idx, f"point_{idx}")
    
    def calculate_motion_features(self, current_pose, previous_pose):
        """Calculate motion features between two poses"""
        if not current_pose or not previous_pose:
            return {"motion_detected": False}
        
        features = {
            "motion_detected": True,
            "joint_velocities": {},
            "total_movement": 0.0
        }
        
        # Calculate movement for each joint
        for joint_name in current_pose.get("pose", {}).keys():
            if joint_name in previous_pose.get("pose", {}):
                curr = current_pose["pose"][joint_name]
                prev = previous_pose["pose"][joint_name]
                
                # Calculate Euclidean distance
                movement = np.sqrt(
                    (curr["x"] - prev["x"])**2 + 
                    (curr["y"] - prev["y"])**2 + 
                    (curr["z"] - prev["z"])**2
                )
                
                features["joint_velocities"][joint_name] = movement
                features["total_movement"] += movement
        
        # Detect specific actions
        features["action_hints"] = self.detect_actions(current_pose, features["joint_velocities"])
        
        return features
    
    def detect_actions(self, pose, velocities):
        """Detect common actions from pose and movement"""
        actions = []
        
        if not pose or "pose" not in pose:
            return actions
        
        pose_data = pose["pose"]
        
        # Check if arms are raised
        if "left_wrist" in pose_data and "left_shoulder" in pose_data:
            if pose_data["left_wrist"]["y"] < pose_data["left_shoulder"]["y"]:
                actions.append("left_arm_raised")
        
        if "right_wrist" in pose_data and "right_shoulder" in pose_data:
            if pose_data["right_wrist"]["y"] < pose_data["right_shoulder"]["y"]:
                actions.append("right_arm_raised")
        
        # Check for jumping (high ankle movement)
        if velocities.get("left_ankle", 0) > 0.1 or velocities.get("right_ankle", 0) > 0.1:
            actions.append("possible_jump")
        
        # Check for walking/running (alternating knee movement)
        if velocities.get("left_knee", 0) > 0.05 and velocities.get("right_knee", 0) > 0.05:
            actions.append("walking_or_running")
        
        return actions
    
    def process_video_frames(self, video_id):
        """Process all frames for a video"""
        work_dir = Path("/tmp/mnemo_work") / video_id / "frames"
        
        if not work_dir.exists():
            logger.error(f"Frames directory not found: {work_dir}")
            return False
        
        frame_files = sorted(work_dir.glob("*.jpg"))
        logger.info(f"Processing {len(frame_files)} frames for motion extraction")
        
        previous_pose = None
        motion_sequence = []
        
        for idx, frame_path in enumerate(frame_files):
            # Load frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            
            # Extract pose
            pose_data = self.extract_holistic_from_frame(frame)
            
            # Calculate motion features if we have a previous frame
            motion_features = None
            if previous_pose is not None:
                motion_features = self.calculate_motion_features(pose_data, previous_pose)
            
            # Extract frame number from filename
            frame_number = int(frame_path.stem.split('_')[1])
            
            # Store motion data
            self.store_motion_data(video_id, frame_number, pose_data, motion_features)
            
            # Keep sequence for evolution detection
            motion_sequence.append({
                "frame": frame_number,
                "pose": pose_data,
                "features": motion_features
            })
            
            previous_pose = pose_data
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1} frames...")
        
        # Analyze motion sequence for patterns
        self.analyze_motion_sequence(video_id, motion_sequence)
        
        return True
    
    def analyze_motion_sequence(self, video_id, sequence):
        """Analyze the full motion sequence for patterns and variants"""
        if len(sequence) < 5:
            return
        
        # Detect motion segments (continuous movement)
        motion_segments = []
        current_segment = None
        
        for item in sequence:
            if item["features"] and item["features"].get("total_movement", 0) > 0.02:
                if current_segment is None:
                    current_segment = {
                        "start_frame": item["frame"],
                        "frames": [item]
                    }
                else:
                    current_segment["frames"].append(item)
            else:
                if current_segment and len(current_segment["frames"]) > 3:
                    current_segment["end_frame"] = current_segment["frames"][-1]["frame"]
                    motion_segments.append(current_segment)
                current_segment = None
        
        # Store motion segments for evolution
        conn = self.connect_db()
        cursor = conn.cursor()
        
        for segment in motion_segments:
            summary = f"Motion segment: {len(segment['frames'])} frames of movement"
            
            # Detect predominant action
            all_actions = []
            for frame in segment["frames"]:
                if frame["features"] and "action_hints" in frame["features"]:
                    all_actions.extend(frame["features"]["action_hints"])
            
            if all_actions:
                most_common = max(set(all_actions), key=all_actions.count)
                summary = f"Motion: {most_common} ({len(segment['frames'])} frames)"
            
            cursor.execute("""
                INSERT INTO gapper_reports 
                (video_id, gapper_type, timestamp, gapper_id, start_frame, 
                 end_frame, summary, importance, features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video_id,
                "motion_segment",
                int(time.time() * 1000),
                f"motion_seg_{segment['start_frame']}",
                segment["start_frame"],
                segment["end_frame"],
                summary,
                0.8,  # High importance for motion segments
                json.dumps({"actions": list(set(all_actions))})
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Detected {len(motion_segments)} motion segments")
    
    def store_motion_data(self, video_id, frame_number, pose_data, motion_features):
        """Store motion data in database"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        # Create motion gapper report
        gapper_id = f"motion_{frame_number}"
        
        features = {
            "has_pose": pose_data is not None,
            "pose_data": pose_data,
            "motion_features": motion_features
        }
        
        # Calculate importance based on motion
        importance = 0.3  # Base importance
        if motion_features and motion_features.get("total_movement", 0) > 0.1:
            importance = min(0.9, 0.3 + motion_features["total_movement"])
        
        summary = "No person detected"
        if pose_data:
            summary = "Person detected"
            if motion_features and motion_features.get("action_hints"):
                summary = f"Action: {', '.join(motion_features['action_hints'])}"
        
        cursor.execute("""
            INSERT INTO gapper_reports 
            (video_id, gapper_type, timestamp, gapper_id, start_frame, 
             end_frame, summary, importance, features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video_id,
            "motion",
            int(frame_number * 33.33),  # Assuming ~30fps, convert to ms
            gapper_id,
            frame_number,
            frame_number,
            summary,
            importance,
            json.dumps(features)
        ))
        
        conn.commit()
        conn.close()
    
    def process_one(self):
        """Process one video for motion extraction"""
        video_id = self.get_next_task()
        
        if not video_id:
            return False
        
        logger.info(f"Processing motion for video {video_id}")
        
        try:
            success = self.process_video_frames(video_id)
            
            if success:
                logger.info(f"Successfully extracted motion for video {video_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process video {video_id}: {e}")
            import traceback
            traceback.print_exc()
            return True
    
    def run(self):
        """Run the motion extractor in a loop"""
        logger.info("Motion extractor started")
        
        while True:
            try:
                had_work = self.process_one()
                
                if not had_work:
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    extractor = MotionExtractor()
    extractor.run()