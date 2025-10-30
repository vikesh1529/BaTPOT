from ultralytics import YOLO
import cv2

def main():
    model = YOLO("yolov10n.pt")

    results = model.track(
        source="proj/assets/escalator_small.mp4",
        tracker="bytetrack.yaml",  # built-in ByteTrack config
        show=True,                 # show video with IDs
        persist=True,              # keep same IDs between frames
        conf=0.1,                  # detection confidence
        classes=[24, 26, 28]               # optional: track only 'handbag' (COCO class 26)
    )

if __name__ == "__main__":
    main()
