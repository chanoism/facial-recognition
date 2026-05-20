# Real-Time Facial Recognition

Real-time facial recognition using [InsightFace](https://github.com/deepinsightai/insightface) and OpenCV. Detects and identifies faces from any camera source including webcams, external cameras, and OBS virtual camera.

---

## Features

- Real-time face detection and recognition
- Supports multiple known faces
- Works with any camera index (webcam, external camera, OBS virtual camera)
- Persistent bounding boxes across frames for smooth display
- Automatic known face loading from folder structure
- Configurable similarity threshold and performance settings

---

## Requirements

- Python 3.11 (recommended — dlib/InsightFace compatibility)
- Windows or macOS
- A camera or OBS virtual camera

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/facial-recognition
cd facial-recognition
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install insightface onnxruntime opencv-python numpy
```

---

## Adding Known Faces

Create a `known_faces/` folder in the project directory. You can use either structure:

### Option A: Subfolder per person (recommended for multiple photos)
```
known_faces/
  John/
    photo1.jpg
    photo2.jpg
  Sarah/
    photo1.jpg
```

### Option B: Flat structure
```
known_faces/
  John.jpg
  John2.jpg
  Sarah.jpg
```

**Tips for best recognition accuracy:**
- Use clear, well-lit, forward-facing photos
- One photo per person is enough to start
- Adding 3-4 photos per person improves accuracy
- Photos should be similar lighting/conditions to your camera setup

---

## Usage

### Basic (uses default camera index 0)
```bash
python facial_recognition.py
```

### Specify camera index
```bash
python facial_recognition.py 1    # external camera
python facial_recognition.py 2    # OBS virtual camera
```

### Finding your camera index
```bash
python -c "import cv2; [print(i, cv2.VideoCapture(i).isOpened()) for i in range(5)]"
```

### Press Q to quit

---

## Configuration

Edit the configuration section at the top of `facial_recognition.py`:

```python
CAMERA_INDEX = 0              # Default camera (overridden by command line arg)
SIMILARITY_THRESHOLD = 0.4    # Lower = stricter matching (0.35 to 0.45 recommended)
PROCESS_EVERY_N_FRAMES = 5    # Higher = faster but less responsive
WINDOW_WIDTH = 1280           # Display window width
WINDOW_HEIGHT = 720           # Display window height
```

---

## Using OBS Virtual Camera

If you want to use a DSLR or mirrorless camera (like a Sony A7 IV):

1. Install [OBS Studio](https://obsproject.com/)
2. Add your camera as a **Video Capture Device** source in OBS
3. Click **Start Virtual Camera** in OBS
4. Run the script with the OBS camera index (usually 2):
```bash
python facial_recognition.py 2
```

---

## .gitignore

The `known_faces/` folder is excluded from git to protect people's privacy. Never commit photos of people to a public repository.

A `.gitignore` file is included that excludes:
- `known_faces/` — your face photos
- `venv/` and `venv311/` — virtual environment
- `__pycache__/` — Python cache files
- `*.pyc` — compiled Python files
- `*.jpg`, `*.png` — image files

---

## How It Works

1. **Model loading** — InsightFace's `buffalo_l` model loads on startup
2. **Known face encoding** — each photo in `known_faces/` is converted to a 512-dimensional embedding vector
3. **Live detection** — every 5th frame is analyzed for faces
4. **Matching** — each detected face embedding is compared to known embeddings using cosine similarity
5. **Display** — matched faces get a green box and name label; unknown faces are labeled "Unknown"

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Camera window is black | Try a different camera index |
| No faces detected | Check lighting and ensure face is forward-facing |
| Wrong person recognized | Lower SIMILARITY_THRESHOLD (e.g. 0.35) |
| Too many false positives | Raise SIMILARITY_THRESHOLD (e.g. 0.45) |
| Slow performance | Increase PROCESS_EVERY_N_FRAMES |
| InsightFace install fails | Make sure you're using Python 3.11 |
| opencv-python GUI error | Run `pip uninstall opencv-python-headless -y && pip install opencv-python` |

---

## License

MIT
