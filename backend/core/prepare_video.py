import os
import cv2
from pathlib import Path
from tqdm import tqdm
from src.config import VIDEO_RAW, VIDEO_FRAMES

# Load a pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face_frames(in_path, out_dir, every_n=30, max_frames=30):
    """
    Extracts frames from a video, detects the largest face, crops it, and saves it.
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        return
        
    count, saved = 0, 0
    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % every_n == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Process only the largest face found in the frame
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]
                
                # Add some padding around the cropped face for better context
                pad = 30
                face_crop = frame[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]

                if face_crop.size == 0:
                    continue

                out_path = os.path.join(out_dir, f"{Path(in_path).stem}_{saved}.jpg")
                cv2.imwrite(out_path, face_crop)
                saved += 1
        count += 1
    cap.release()

def process_videos(label):
    in_dir = os.path.join(VIDEO_RAW, label)
    out_dir = os.path.join(VIDEO_FRAMES, label)
    os.makedirs(out_dir, exist_ok=True)

    video_files = [f for f in os.listdir(in_dir) if f.endswith(".mp4")]
    
    for fname in tqdm(video_files, desc=f"Processing {label} videos"):
        in_path = os.path.join(in_dir, fname)
        extract_face_frames(in_path, out_dir)

if __name__ == "__main__":
    print("--- Video Frame Extractor with Face Cropping ---")
    print("This script will create a new 'frames' dataset containing ONLY cropped faces.")
    # Important: Ensure old 'frames' folder is deleted before running to avoid mixing datasets.
    for lbl in ["real", "fake"]:
        process_videos(lbl)
    print("\n✅ All face frames saved in:", VIDEO_FRAMES)

