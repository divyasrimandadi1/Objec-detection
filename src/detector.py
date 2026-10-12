import torch

from ultralytics import YOLO

from config import (
    MODEL_NAME,
    IMG_SIZE,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
)


class VehicleDetector:

    def __init__(self):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {self.device}")

        self.model = YOLO(MODEL_NAME)

        print(f"Loaded model: {MODEL_NAME}")

    def detect(self, frame):

        results = self.model.predict(
            source=frame,

            imgsz=IMG_SIZE,

            conf=CONF_THRESHOLD,

            iou=IOU_THRESHOLD,

            device=self.device,

            verbose=False,
        )

        return results[0]