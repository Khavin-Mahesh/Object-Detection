from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="/Users/khavinm/Desktop/Object detection/coco/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8   # lets it pick best batch size
    # device="cpu", # uncomment if needed
)
