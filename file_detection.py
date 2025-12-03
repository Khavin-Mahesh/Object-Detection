from ultralytics import YOLO
import cv2
import sys

def main(video_path):
    model = YOLO("yolov8n.pt")  # local YOLO model
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("File Detection (ESC to quit)", annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Finished processing")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_detection.py <video_path>")
        sys.exit(1)

    main(sys.argv[1])
