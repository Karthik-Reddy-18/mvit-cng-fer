"""
face_detector.py — Face detection using OpenCV Haar Cascade
This finds face(s) in an image and crops them out.
"""
 
import cv2
import numpy as np
from pathlib import Path
 
class FaceDetector:
    """
    Detects faces using OpenCV's built-in Haar Cascade classifier.
    This is a simplified version of the paper's ImSSD detector —
    sufficient for learning purposes.
    """
    
    def __init__(self):
        # Load pre-trained Haar Cascade — this comes with OpenCV, no download needed
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✅ Face detector loaded successfully")
    
    def detect_faces(self, image_bgr):
        """
        Detect all faces in an image.
        
        Args:
            image_bgr: numpy array (BGR format from OpenCV)
        
        Returns:
            faces: list of cropped face arrays
            boxes: list of (x, y, w, h) bounding boxes
        """
        # Convert to grayscale for detection (faster and more accurate)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        # scaleFactor=1.1: checks multiple scales (1.1 = 10% smaller each step)
        # minNeighbors=5: higher = fewer false positives
        # minSize=(30,30): minimum face size to detect
        detections = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        boxes = []
        
        if len(detections) == 0:
            print("⚠️  No face detected in image")
            return faces, boxes
        
        for (x, y, w, h) in detections:
            # Add a small margin around the face (10% on each side)
            margin = int(0.1 * min(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image_bgr.shape[1], x + w + margin)
            y2 = min(image_bgr.shape[0], y + h + margin)
            
            # Crop face region
            face = image_bgr[y1:y2, x1:x2]
            faces.append(face)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        
        return faces, boxes
    def detect_and_draw(self, image_bgr, emotion_label=None, confidence=None):
        """
        Detect faces and draw bounding boxes on the image.
        Returns image with annotations.
        """
        faces, boxes = self.detect_faces(image_bgr)
        result = image_bgr.copy()
        
        for i, (x, y, w, h) in enumerate(boxes):
            # Draw rectangle (green box)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add emotion label if provided
            if emotion_label is not None:
                label = f"{emotion_label}"
                if confidence is not None:
                    label += f" ({confidence:.1f}%)"
                
                # Background for text
                cv2.rectangle(result, (x, y-30), (x+w, y), (0, 255, 0), -1)
                cv2.putText(result, label, (x+5, y-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return result, faces, boxes
