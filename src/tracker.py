from ultralytics import YOLO

from config import (
    MODEL_NAME,
    IMG_SIZE,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    TRACKER,
)


class VehicleTracker:

    def __init__(self):

        self.model = YOLO(MODEL_NAME)

    def track(self, frame):

        results = self.model.track(
            source=frame,

            imgsz=IMG_SIZE,

            conf=CONF_THRESHOLD,

            iou=IOU_THRESHOLD,

            tracker=TRACKER,

            persist=True,

            verbose=False,
        )

        return results[0]