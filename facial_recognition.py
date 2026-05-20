import cv2
import numpy as np
from insightface.app import FaceAnalysis
import time
import os
import sys

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# Camera index (pass as argument: python facial_recognition.py 2)
# Common values: 0 = built-in webcam, 1 = external camera, 2 = OBS virtual camera
CAMERA_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Path to known faces folder
KNOWN_FACES_DIR = "known_faces"

# Similarity threshold — lower = stricter matching (recommended: 0.35 to 0.45)
SIMILARITY_THRESHOLD = 0.4

# Process every Nth frame (higher = faster but less responsive)
PROCESS_EVERY_N_FRAMES = 5

# Display window size
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────

print("Loading InsightFace model...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))
print("Model loaded.")

# ──────────────────────────────────────────────
# LOAD KNOWN FACES
# ──────────────────────────────────────────────

known_embeddings = []
known_names = []

def load_known_face(path, name):
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read image: {path}")
        return
    faces = app.get(img)
    if len(faces) == 0:
        print(f"No face found in {path} — skipping")
        return
    known_embeddings.append(faces[0].embedding)
    known_names.append(name)
    print(f"Loaded: {name} ({os.path.basename(path)})")

# Automatically load all images from known_faces/
# Folder structure: known_faces/PersonName/photo1.jpg, photo2.jpg ...
# OR flat structure: known_faces/PersonName.jpg, PersonName2.jpg ...

if not os.path.exists(KNOWN_FACES_DIR):
    print(f"Warning: '{KNOWN_FACES_DIR}' folder not found. Create it and add photos.")
else:
    # Support subfolder structure: known_faces/Name/photo.jpg
    for entry in os.scandir(KNOWN_FACES_DIR):
        if entry.is_dir():
            for photo in os.scandir(entry.path):
                if photo.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    load_known_face(photo.path, entry.name)
        # Support flat structure: known_faces/Name.jpg
        elif entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(entry.name)[0].rstrip('0123456789')
            load_known_face(entry.path, name)

print(f"\nTotal known face encodings loaded: {len(known_embeddings)}")

# ──────────────────────────────────────────────
# CONNECT TO CAMERA
# ──────────────────────────────────────────────

print(f"\nConnecting to camera index {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
time.sleep(3)

# Burn through initial black frames
for i in range(30):
    cap.read()

if not cap.isOpened():
    print("Connection unsuccessful — check camera index.")
    exit()

print("Camera connected. Press Q to quit.\n")

cv2.namedWindow("Facial Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Facial Recognition", WINDOW_WIDTH, WINDOW_HEIGHT)

# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────

frame_count = 0
last_detections = []  # Persist boxes across frames

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    frame_count += 1

    # Only run detection every N frames
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        faces = app.get(frame)
        last_detections = []

        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            name = "Unknown"

            # Compare against known faces
            for known_emb, known_name in zip(known_embeddings, known_names):
                similarity = np.dot(embedding, known_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(known_emb)
                )
                if similarity > SIMILARITY_THRESHOLD:
                    name = known_name
                    break

            last_detections.append((bbox, name))

    # Draw boxes on every frame using last known detections
    for bbox, name in last_detections:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
        cv2.rectangle(frame, (bbox[0], bbox[1] - 45), (bbox[2], bbox[1]), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (bbox[0] + 6, bbox[1] - 8),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 2)

    cv2.imshow("Facial Recognition", frame)
    cv2.resizeWindow("Facial Recognition", WINDOW_WIDTH, WINDOW_HEIGHT)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
