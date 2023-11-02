from ultralytics import YOLO

# Load the model
model = YOLO("yolov8m.pt")

# Training
results = model.train(
    data = "custom_dataset.yaml",
    imgsz = 640,
    epochs = 50
    batch = 8,
    name = "yolov8m_custom"
)

