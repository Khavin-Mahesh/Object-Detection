from ultralytics import YOLO
import cv2
from collections import Counter

# these are what I want to see
HOUSEHOLD = {
    "bottle","cup","bowl","fork","knife","spoon",
    "chair","couch","dining table","bed","tv",
    "laptop","mouse","keyboard","remote","cell phone",
    "book","clock","vase","refrigerator","microwave","oven",
    "toaster","sink","potted plant","toothbrush","scissors","teddy bear"
}

def main(cam_index=0, weights="yolov8n.pt", imgsz=640, conf=0.30):
    model = YOLO(weights)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, imgsz=imgsz, conf=conf, iou=0.45, verbose=False)
        r = results[0]
        names = r.names
        keep = []

        # filter detections to HOUSEHOLD classes
        det_counts = Counter()
        for box in r.boxes:
            cls = int(box.cls)
            label = names[cls]
            if label in HOUSEHOLD:
                keep.append(True)
                det_counts[label] += 1
            else:
                keep.append(False)

        # draw only the kept detections
        annotated = r.plot()  # draws all
        # mask out non-household by drawing filled boxes over them (quick hack)
        for k, box in zip(keep, r.boxes.xyxy.cpu().numpy()):
            if not k:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,0,0), -1)

        # show a compact count HUD
        y = 20
        for label, cnt in det_counts.most_common(6):
            cv2.putText(annotated, f"{label}: {cnt}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y += 24

        cv2.imshow("Household Object Detection (ESC to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
