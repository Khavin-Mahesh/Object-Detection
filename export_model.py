from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx")
print("✅ Model exported to yolov8n.onnx")
