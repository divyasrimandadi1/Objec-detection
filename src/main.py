import os
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def resolve_local_video_path(video_source):
    video_path = Path(video_source).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(f"Local video file not found: {video_path}")

    return str(video_path.resolve())


# ============================================================
# CONFIGURATION
# ============================================================


def resolve_model_path():
    base_dir = Path(__file__).resolve().parent.parent
    preferred_models = (
        ("yolo26n.pt", "yolo26s.pt")
        if not torch.cuda.is_available()
        else ("yolo26s.pt", "yolo26n.pt")
    )
    candidates = [base_dir / "models" / "best.pt"]
    for model_name in preferred_models:
        candidates.extend((base_dir / "models" / model_name, base_dir / model_name))

    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_file() and path.stat().st_size > 1024:
            return str(path)

    return preferred_models[0]


MODEL_PATH = resolve_model_path()
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

CPU_THREAD_LIMIT = 2
FRAME_SKIP = 1
CPU_IMG_SIZE = 960
GPU_IMG_SIZE = 1536
IMG_SIZE = CPU_IMG_SIZE if not torch.cuda.is_available() else GPU_IMG_SIZE


def create_output_writer(video_path, fps, width, height, frame_skip=1):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_name = Path(video_path).stem or "vehicle_detection"
    output_path = OUTPUT_DIR / f"{source_name}_detections.mp4"
    source_fps = fps if fps > 0 else 30.0
    output_fps = source_fps / frame_skip
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        output_fps,
        (width, height),
    )

    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    return writer, output_path
USE_TEST_TIME_AUGMENTATION = torch.cuda.is_available()
CONF_THRESHOLD = 0.12
FALLBACK_CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5


def configure_cpu_threads(device):
    if device != "cpu":
        return None

    thread_count = min(CPU_THREAD_LIMIT, os.cpu_count() or 1)
    torch.set_num_threads(thread_count)
    return thread_count

TRACKER = "bytetrack.yaml"

VEHICLE_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def get_model_vehicle_classes(model):
    model_names = model.names
    name_items = model_names.items() if isinstance(model_names, dict) else enumerate(model_names)
    normalized_names = {
        int(class_id): str(class_name).strip().lower()
        for class_id, class_name in name_items
    }

    return {
        class_id: class_name
        for class_id, class_name in VEHICLE_CLASSES.items()
        if class_name in normalized_names.values()
    }


def detect_vehicles(model, frame, device, vehicle_classes=None):
    active_classes = vehicle_classes or get_model_vehicle_classes(model)
    if not active_classes:
        raise ValueError("The selected model has no supported vehicle classes")
    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=FALLBACK_CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=device,
        verbose=False,
        classes=list(active_classes.keys()),
        max_det=200,
        augment=USE_TEST_TIME_AUGMENTATION,
    )
    return results[0]

ZOOM_IN_ON_LARGE_BOXES = True
ZOOM_AREA_THRESHOLD = 0.0035
ZOOM_PADDING_RATIO = 0.35
AMBIGUOUS_CLASS_IDS = {2, 7}

TRUCK_AREA_RATIO = 0.11
TRUCK_ASPECT_RATIO = 1.5
BUS_AREA_RATIO = 0.08

ROI_TOP_RATIO = 0.0
ROI_BOTTOM_RATIO = 1.0
ROI_LEFT_RATIO = 0.0
ROI_RIGHT_RATIO = 1.0
MIN_VEHICLE_AREA_RATIO = 0.0001
CLASS_HISTORY_LENGTH = 7
LOW_LIGHT_BRIGHTNESS_THRESHOLD = 75
LOW_LIGHT_GAMMA = 0.6
LOW_LIGHT_GAMMA_LUT = np.array(
    [((value / 255.0) ** LOW_LIGHT_GAMMA) * 255 for value in range(256)],
    dtype=np.uint8,
)


def enhance_low_light(frame):
    """Brighten dark frames for inference while leaving well-lit frames unchanged."""
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if cv2.mean(grayscale)[0] >= LOW_LIGHT_BRIGHTNESS_THRESHOLD:
        return frame

    return cv2.LUT(frame, LOW_LIGHT_GAMMA_LUT)


def extract_roi(frame):
    h, w = frame.shape[:2]
    x1 = int(w * ROI_LEFT_RATIO)
    y1 = int(h * ROI_TOP_RATIO)
    x2 = int(w * ROI_RIGHT_RATIO)
    y2 = int(h * ROI_BOTTOM_RATIO)
    return frame[y1:y2, x1:x2], (x1, y1)


def merge_overlapping_boxes(boxes, iou_threshold=0.45):
    """Merge split detections of the same vehicle into a single box."""
    merged = []

    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        matched = False

        for existing in merged:
            ex1, ex2 = existing["x1"], existing["x2"]
            ey1, ey2 = existing["y1"], existing["y2"]
            inter_w = max(0, min(x2, ex2) - max(x1, ex1))
            inter_h = max(0, min(y2, ey2) - max(y1, ey1))
            inter = inter_w * inter_h
            union = area + max(0, ex2 - ex1) * max(0, ey2 - ey1) - inter
            iou = inter / union if union > 0 else 0.0

            if iou >= iou_threshold and box["cls_id"] != existing["cls_id"]:
                existing["x1"] = min(existing["x1"], x1)
                existing["y1"] = min(existing["y1"], y1)
                existing["x2"] = max(existing["x2"], x2)
                existing["y2"] = max(existing["y2"], y2)
                if box["cls_id"] in {5, 7} or existing["cls_id"] in {5, 7}:
                    selected_class = 7 if box["cls_id"] == 7 or existing["cls_id"] == 7 else 5
                    if box["cls_id"] == selected_class:
                        existing["confidence"] = box["confidence"]
                    existing["cls_id"] = selected_class
                elif box["confidence"] > existing["confidence"]:
                    existing["cls_id"] = box["cls_id"]
                    existing["confidence"] = box["confidence"]
                matched = True
                break

        if not matched:
            merged.append(dict(box))

    return merged


def reclassify_vehicle(cls_id, x1, y1, x2, y2, frame_shape):
    """Heuristic correction for the generic YOLO fallback where large trucks are often labeled as cars."""
    h, w = frame_shape[:2]
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    box_area = box_w * box_h
    frame_area = w * h

    if frame_area <= 0:
        return cls_id

    area_ratio = box_area / frame_area
    aspect_ratio = (box_w / max(box_h, 1)) if box_h > 0 else 0

    # Prefer truck classification for wide, long objects before generic bus/car labels.
    if cls_id in {2, 5, 7} and area_ratio > TRUCK_AREA_RATIO and aspect_ratio >= TRUCK_ASPECT_RATIO:
        return 7

    if cls_id in {2, 7} and area_ratio > BUS_AREA_RATIO and box_w > 0.38 * w and aspect_ratio < 1.7:
        return 5

    if cls_id == 2 and area_ratio > 0.09 and box_w > 0.42 * w and aspect_ratio >= 1.45:
        return 7

    return cls_id


def stabilize_vehicle_class(track_id, cls_id, class_history):
    """Smooth per-frame class changes for a tracked vehicle."""
    if track_id is None:
        return cls_id

    state = class_history.get(track_id)
    if state is None:
        state = {
            "observations": deque(maxlen=CLASS_HISTORY_LENGTH),
            "stable_class": cls_id,
        }
        class_history[track_id] = state

    observations = state["observations"]
    observations.append(cls_id)
    counts = Counter(observations)
    highest_count = max(counts.values())
    candidates = {
        candidate
        for candidate, count in counts.items()
        if count == highest_count
    }
    stable_class = state["stable_class"]

    if stable_class not in candidates:
        stable_class = max(candidates, key=lambda candidate: counts[candidate])
        state["stable_class"] = stable_class

    return stable_class


def refine_large_vehicle(model, frame, x1, y1, x2, y2, device):
    """Run a second, tighter pass on large ambiguous objects to reduce car/truck confusion."""
    h, w = frame.shape[:2]
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    box_area = box_w * box_h

    if box_area <= 0:
        return None

    if box_area < ZOOM_AREA_THRESHOLD * w * h:
        return None

    pad_x = int(box_w * ZOOM_PADDING_RATIO)
    pad_y = int(box_h * ZOOM_PADDING_RATIO)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(w, x2 + pad_x)
    crop_y2 = min(h, y2 + pad_y)

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    if crop.size == 0:
        return None

    crop_results = model.track(
        crop,
        persist=False,
        tracker=TRACKER,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=device,
        verbose=False,
        classes=list(VEHICLE_CLASSES.keys()),
        augment=USE_TEST_TIME_AUGMENTATION,
    )

    if not crop_results or len(crop_results) == 0:
        return None

    crop_result = crop_results[0]

    if crop_result.boxes is None or len(crop_result.boxes) == 0:
        return None

    best_idx = 0
    best_conf = -1.0

    for idx, conf in enumerate(crop_result.boxes.conf):
        score = float(conf)
        if score > best_conf:
            best_conf = score
            best_idx = idx

    cls_id = int(crop_result.boxes.cls[best_idx])
    if cls_id not in VEHICLE_CLASSES:
        return None

    x1c, y1c, x2c, y2c = (
        crop_result.boxes.xyxy[best_idx].cpu().numpy().astype(int)
    )

    return {
        "cls_id": cls_id,
        "class_name": VEHICLE_CLASSES[cls_id],
        "confidence": float(crop_result.boxes.conf[best_idx]),
        "x1": crop_x1 + x1c,
        "y1": crop_y1 + y1c,
        "x2": crop_x1 + x2c,
        "y2": crop_y1 + y2c,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # GET VIDEO PATH FROM TERMINAL
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print("\nUsage:")
        print(
            'python main.py "path_to_video.mp4"'
        )

        print("\nExample:")
        print(
            r'python main.py "D:\TrafficVideos\night.mp4"'
        )

        return

    try:
        video_path = resolve_local_video_path(sys.argv[1])
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        return

    print("\n========================================")
    print("      EDGE VEHICLE DETECTION")
    print("========================================")

    print(f"Video: {video_path}")

    # --------------------------------------------------------
    # SELECT DEVICE
    # --------------------------------------------------------

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    cpu_threads = configure_cpu_threads(device)
    if cpu_threads is not None:
        print(f"PyTorch CPU threads: {cpu_threads}")

    # --------------------------------------------------------
    # LOAD YOLO MODEL
    # --------------------------------------------------------

    print("Loading YOLO model...")

    try:
        model = YOLO(MODEL_PATH)
    except Exception as exc:
        fallback = "yolo26n.pt" if device == "cpu" else "yolo26s.pt"
        print(f"Warning: could not load {MODEL_PATH}: {exc}")
        print(f"Falling back to {fallback}")
        try:
            model = YOLO(fallback)
        except Exception as fallback_exc:
            fallback = "yolo11n.pt"
            print(f"Warning: could not load {fallback}: {fallback_exc}")
            print(f"Falling back to {fallback}")
            model = YOLO(fallback)

    vehicle_classes = get_model_vehicle_classes(model)
    if set(vehicle_classes.values()) != set(VEHICLE_CLASSES.values()):
        fallback = "yolo26n.pt" if device == "cpu" else "yolo26s.pt"
        print(
            "Loaded checkpoint does not contain all requested classes. "
            f"Using class-aware fallback: {fallback}"
        )
        model = YOLO(fallback)
        vehicle_classes = get_model_vehicle_classes(model)
    if set(vehicle_classes.values()) != set(VEHICLE_CLASSES.values()):
        raise RuntimeError(f"Model is missing requested classes: {VEHICLE_CLASSES.values()}")

    print(f"Model loaded successfully: {model.ckpt_path if hasattr(model, 'ckpt_path') else MODEL_PATH}")
    print(f"Vehicle classes: {', '.join(vehicle_classes.values())}")
    print(f"Inference confidence: {FALLBACK_CONF_THRESHOLD:.2f}")

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("\nERROR: Could not open video.")
        print(f"Path: {video_path}")

        return

    print("Video opened successfully.")

    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"Resolution: {width} x {height}"
    )

    print(
        f"Video FPS: {fps:.2f}"
    )

    print(f"Inference image size: {IMG_SIZE}")
    frame_skip = FRAME_SKIP
    print(f"Frame sampling: every {frame_skip} frame(s)")

    writer, output_path = create_output_writer(video_path, fps, width, height, frame_skip)
    print(f"Saving annotated video to: {output_path}")

    # --------------------------------------------------------
    # PROCESS VIDEO
    # --------------------------------------------------------

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1
        if (frame_number - 1) % frame_skip != 0:
            continue

        frame = enhance_low_light(frame)

        # ----------------------------------------------------
        # YOLO DETECTION + BYTE TRACK
        # ----------------------------------------------------

        roi_frame, roi_offset = extract_roi(frame)
        roi_x, roi_y = roi_offset

        result = detect_vehicles(model, roi_frame, device, vehicle_classes)

        # ----------------------------------------------------
        # DRAW DETECTIONS
        # ----------------------------------------------------

        if result.boxes is not None:
            valid_boxes = []

            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i])
                if cls_id not in vehicle_classes:
                    continue

                confidence = float(result.boxes.conf[i])
                x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)

                x1 += roi_x
                y1 += roi_y
                x2 += roi_x
                y2 += roi_y

                box_area = max(0, x2 - x1) * max(0, y2 - y1)
                frame_area = frame.shape[0] * frame.shape[1]
                if box_area < MIN_VEHICLE_AREA_RATIO * frame_area:
                    continue

                if ZOOM_IN_ON_LARGE_BOXES and device.startswith("cuda") and cls_id in AMBIGUOUS_CLASS_IDS:
                    refined = refine_large_vehicle(
                        model,
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        device,
                    )
                    if refined is not None:
                        cls_id = refined["cls_id"]
                        confidence = refined["confidence"]
                        x1 = int(refined["x1"])
                        y1 = int(refined["y1"])
                        x2 = int(refined["x2"])
                        y2 = int(refined["y2"])

                valid_boxes.append({
                    "cls_id": cls_id,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                })

            for box in valid_boxes:
                cls_id = box["cls_id"]
                confidence = box["confidence"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                class_name = vehicle_classes[cls_id]

                label = f"{class_name.upper()} {confidence:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )


        writer.write(frame)

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        cv2.imshow(
            "YOLO Edge Vehicle Detection",
            frame
        )

        # ----------------------------------------------------
        # PRESS Q TO EXIT
        # ----------------------------------------------------

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            break

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    cap.release()
    writer.release()

    cv2.destroyAllWindows()

    print("\nDetection completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()