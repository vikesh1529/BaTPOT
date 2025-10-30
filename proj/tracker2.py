# tracker.py
from ultralytics.trackers.byte_tracker import BYTETracker
import numpy as np

class Tracker:
    def __init__(self):
        self.tracker = BYTETracker(
            track_thresh=0.5,
            match_thresh=0.8,
            track_buffer=30,
            frame_rate=30
        )

    def track(self, detections, frame):
        # detections = [(bbox, conf, cls), ...]
        dets = []
        for bbox, conf, cls in detections:
            x1, y1, x2, y2 = bbox
            dets.append([x1, y1, x2, y2, conf, cls])
        dets = np.array(dets)

        tracks = self.tracker.update(dets, frame.shape)
        boxes, ids = [], []

        for t in tracks:
            x1, y1, x2, y2 = t[:4]
            track_id = int(t[4])
            boxes.append([x1, y1, x2, y2])
            ids.append(track_id)
        return ids, boxes
