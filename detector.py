from ultralytics import YOLO

# Load YOLO model once
model = YOLO("yolov8n.pt")

def detect(frame):

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        verbose=False
    )

    return results