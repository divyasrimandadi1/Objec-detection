import cv2
import time
import torch

from ultralytics import YOLO

from config import (
    MODEL_NAME,
    IMG_SIZE,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    TRACKER,
    VEHICLE_CLASSES,
    FRAME_SKIP,
    SHOW_WINDOW,
    SAVE_OUTPUT,
    OUTPUT_DIR,
)


class EdgeVehicleDetector:

    def __init__(self):

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        print("=" * 60)
        print("EDGE VEHICLE DETECTION")
        print("=" * 60)

        print(f"Device : {self.device}")
        print(f"Model  : {MODEL_NAME}")
        print(f"Image  : {IMG_SIZE}")
        print(f"Conf   : {CONF_THRESHOLD}")
        print(f"IoU    : {IOU_THRESHOLD}")

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        self.model = YOLO(MODEL_NAME)

        # Warm-up
        print("Warming up model...")

        dummy = torch.zeros(
            1,
            3,
            IMG_SIZE,
            IMG_SIZE
        )

        if self.device.startswith("cuda"):
            dummy = dummy.cuda()

        print("Model ready.")

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self, frame):

        results = self.model.track(

            source=frame,

            imgsz=IMG_SIZE,

            conf=CONF_THRESHOLD,

            iou=IOU_THRESHOLD,

            tracker=TRACKER,

            persist=True,

            device=self.device,

            verbose=False,
        )

        result = results[0]

        return result

    # ========================================================
    # DRAW RESULTS
    # ========================================================

    def draw_results(self, frame, result):

        if result.boxes is None:
            return frame

        boxes = result.boxes

        for i in range(len(boxes)):

            cls_id = int(boxes.cls[i])

            confidence = float(boxes.conf[i])

            # Only process vehicles
            if cls_id not in VEHICLE_CLASSES:
                continue

            class_name = VEHICLE_CLASSES[cls_id]

            # Bounding box
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Tracking ID
            track_id = None

            if boxes.id is not None:
                track_id = int(boxes.id[i])

            # Label
            if track_id is not None:

                label = (
                    f"{class_name} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )

            else:

                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # Label background
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
            )

            cv2.rectangle(
                frame,
                (x1, y1 - th - 8),
                (x1 + tw, y1),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return frame

    # ========================================================
    # RUN VIDEO
    # ========================================================

    def run(self, video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():

            raise RuntimeError(
                f"Cannot open video: {video_path}"
            )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Video resolution : {width}x{height}")
        print(f"Video FPS        : {fps}")

        writer = None

        if SAVE_OUTPUT:

            OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                OUTPUT_DIR /
                "vehicle_detection.mp4"
            )

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height),
            )

            print(
                f"Saving output to: {output_path}"
            )

        frame_number = 0

        previous_time = time.time()

        fps_counter = 0

        display_fps = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # ------------------------------------------------
            # FRAME SKIPPING
            # ------------------------------------------------

            if FRAME_SKIP > 1:

                if frame_number % FRAME_SKIP != 0:

                    continue

            # ------------------------------------------------
            # DETECTION + TRACKING
            # ------------------------------------------------

            result = self.process_frame(frame)

            # ------------------------------------------------
            # DRAW
            # ------------------------------------------------

            frame = self.draw_results(
                frame,
                result
            )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            fps_counter += 1

            current_time = time.time()

            elapsed = current_time - previous_time

            if elapsed >= 1:

                display_fps = (
                    fps_counter / elapsed
                )

                fps_counter = 0

                previous_time = current_time

            cv2.putText(
                frame,
                f"FPS: {display_fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            if SHOW_WINDOW:

                cv2.imshow(
                    "Edge Vehicle Detection",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):

                    break

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            if writer is not None:

                writer.write(frame)

        cap.release()

        if writer is not None:

            writer.release()

        cv2.destroyAllWindows()

        print("Detection completed.")