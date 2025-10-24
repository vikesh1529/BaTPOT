import cv2
import time
from yolo_detector import YoloDetector
from tracker import Tracker

MODEL_PATH = "models/yolov10m.pt"
VIDEO_PATH = "assets/bus_station.mp4"  

def main():
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.25)
    tracker = Tracker()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.perf_counter()
        detections = detector.detect(frame)
        tracking_ids, boxes, labels = tracker.track(detections, frame)

        for tracking_id, bounding_box, label in zip(tracking_ids, boxes, labels):
            x1, y1, x2, y2 = map(int, bounding_box)
            color = (0, 255, 0) if label == "backpack" else (255, 0, 0) if label == "suitcase" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ID:{tracking_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Bag/Suitcase Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
