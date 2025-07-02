#!/usr/bin/env python3
"""
Test script to visualize motion extraction
"""

import cv2
import mediapipe as mp
import sys

def test_motion_on_image(image_path):
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=True)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = pose.process(rgb_image)
    
    # Draw landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )
        print("Pose detected!")
        
        # Print some key points
        landmarks = results.pose_landmarks.landmark
        print(f"Nose position: x={landmarks[0].x:.2f}, y={landmarks[0].y:.2f}")
        print(f"Left shoulder: x={landmarks[11].x:.2f}, y={landmarks[11].y:.2f}")
        print(f"Right shoulder: x={landmarks[12].x:.2f}, y={landmarks[12].y:.2f}")
    else:
        print("No pose detected in image")
    
    # Save output
    output_path = image_path.replace('.jpg', '_pose.jpg')
    cv2.imwrite(output_path, image)
    print(f"Saved pose visualization to: {output_path}")
    
    # Show image (if display available)
    try:
        cv2.imshow('Pose Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("No display available, skipping visualization")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_motion_on_image(sys.argv[1])
    else:
        print("Usage: python test-motion.py <image_path>")
        print("Example: python test-motion.py extracted_frames/frame_000120.jpg")