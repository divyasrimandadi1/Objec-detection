import argparse
import csv
import gc
import subprocess
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .main import (
    FALLBACK_CONF_THRESHOLD,
    IMG_SIZE,
    enhance_low_light,
    get_model_vehicle_classes,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_NAMES = ("yolo26n.pt", "yolo26s.pt", "yolo11n.pt", "yolo11s.pt")


def sample_frame_indices(total_frames, sample_count):
    if total_frames < 1:
        raise ValueError("Video has no readable frames")
    if sample_count < 1:
        raise ValueError("Sample count must be positive")

    count = min(total_frames, sample_count)
    return np.linspace(0, total_frames - 1, count, dtype=np.int64).tolist()


def load_sample_frames(video_path, sample_count):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total_frames, sample_count)
    frames = []
    brightness = []

    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(float(gray.mean()))
            frames.append(enhance_low_light(frame))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"Could not read sample frames from: {video_path}")

    return frames, brightness


def load_labeled_samples(dataset_dir, sample_count):
    image_dir = dataset_dir / "images" / "val"
    label_dir = dataset_dir / "labels" / "val"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected validation images and labels under {dataset_dir}")

    indices = sample_frame_indices(len(image_paths), sample_count)
    frames = []
    brightness = []
    ground_truth = []

    for index in indices:
        image_path = image_paths[index]
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        height, width = frame.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes = []
        if label_path.is_file():
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                values = line.split()
                if len(values) != 5:
                    raise ValueError(f"Malformed YOLO label in {label_path}:{line_number}")
                class_id, center_x, center_y, box_width, box_height = map(float, values)
                if class_id != 0:
                    raise ValueError(f"Expected generic vehicle class 0 in {label_path}:{line_number}")
                half_width = box_width * width / 2
                half_height = box_height * height / 2
                center_x *= width
                center_y *= height
                boxes.append((
                    center_x - half_width,
                    center_y - half_height,
                    center_x + half_width,
                    center_y + half_height,
                ))

        brightness.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
        frames.append(enhance_low_light(frame))
        ground_truth.append(boxes)

    if not frames:
        raise RuntimeError(f"Could not read validation images under: {image_dir}")

    return frames, brightness, ground_truth


def default_model_paths():
    paths = []
    custom_checkpoint = BASE_DIR / "models" / "best.pt"
    if custom_checkpoint.is_file() and custom_checkpoint.stat().st_size > 1024:
        paths.append(custom_checkpoint.resolve())

    for name in DEFAULT_MODEL_NAMES:
        for candidate in (BASE_DIR / "models" / name, BASE_DIR / name):
            if candidate.is_file() and candidate.stat().st_size > 1024:
                paths.append(candidate.resolve())
                break
    return paths


def get_benchmark_class_ids(model):
    names = model.names
    name_items = names.items() if isinstance(names, dict) else enumerate(names)
    class_names = {int(class_id): str(name).strip().lower() for class_id, name in name_items}
    if "vehicle" in class_names.values():
        return [class_id for class_id, name in class_names.items() if name == "vehicle"]

    allowed_names = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
    return [class_id for class_id, name in class_names.items() if name in allowed_names]


def process_rss_mb():
    try:
        import psutil
    except ImportError:
        return ""

    return psutil.Process().memory_info().rss / (1024 * 1024)


def box_iou(first, second):
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def average_precision(recall_values, precision_values):
    if not recall_values:
        return 0.0

    recall_values = np.asarray([0.0, *recall_values, 1.0])
    precision_values = np.asarray([0.0, *precision_values, 0.0])
    precision_values = np.maximum.accumulate(precision_values[::-1])[::-1]
    sample_points = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(np.max(precision_values[recall_values >= recall]))
        if np.any(recall_values >= recall)
        else 0.0
        for recall in sample_points
    ]
    return float(np.mean(interpolated))


def evaluate_detections(ground_truth_by_frame, predictions_by_frame):
    if len(ground_truth_by_frame) != len(predictions_by_frame):
        raise ValueError("Ground truth and predictions must have equal frame counts")

    target_count = sum(len(boxes) for boxes in ground_truth_by_frame)
    predictions = [
        (frame_index, prediction)
        for frame_index, frame_predictions in enumerate(predictions_by_frame)
        for prediction in frame_predictions
    ]
    predictions.sort(key=lambda item: item[1]["confidence"], reverse=True)

    if target_count == 0:
        raise ValueError("Cannot calculate detection accuracy without ground-truth objects")

    aps = []
    threshold_stats = {}
    duplicate_count = 0

    for threshold in np.arange(0.5, 0.96, 0.05):
        matched_by_frame = [set() for _ in ground_truth_by_frame]
        true_positives = []
        false_positives = []
        threshold_duplicates = 0

        for frame_index, prediction in predictions:
            targets = ground_truth_by_frame[frame_index]
            overlaps = [box_iou(prediction["box"], target) for target in targets]
            best_index = int(np.argmax(overlaps)) if overlaps else -1
            best_iou = overlaps[best_index] if overlaps else 0.0

            if best_iou >= threshold and best_index not in matched_by_frame[frame_index]:
                matched_by_frame[frame_index].add(best_index)
                true_positives.append(1)
                false_positives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(1)
                if threshold == 0.5 and best_iou >= threshold:
                    threshold_duplicates += 1

        cumulative_true = np.cumsum(true_positives)
        cumulative_false = np.cumsum(false_positives)
        recalls = cumulative_true / target_count
        precisions = cumulative_true / np.maximum(cumulative_true + cumulative_false, 1)
        aps.append(average_precision(recalls.tolist(), precisions.tolist()))

        if threshold == 0.5:
            true_count = int(cumulative_true[-1]) if len(cumulative_true) else 0
            false_count = int(cumulative_false[-1]) if len(cumulative_false) else 0
            threshold_stats = {
                "precision_iou50": true_count / max(true_count + false_count, 1),
                "recall_iou50": true_count / target_count,
                "false_positives_iou50": false_count,
                "missed_ground_truth_iou50": target_count - true_count,
                "duplicate_detections_iou50": threshold_duplicates,
            }

    return {
        **threshold_stats,
        "ap50": aps[0] if aps else 0.0,
        "map50_95": float(np.mean(aps)) if aps else 0.0,
        "ground_truth_objects": target_count,
    }


def cpu_temperature_c():
    try:
        import psutil
        sensors = psutil.sensors_temperatures()
    except (ImportError, AttributeError, NotImplementedError, OSError):
        return ""

    readings = [reading.current for group in sensors.values() for reading in group if reading.current]
    return max(readings) if readings else ""


def nvidia_smi_telemetry():
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return {"gpu_power_w": "", "gpu_temperature_c": "", "vram_used_mb": ""}

    values = completed.stdout.strip().split(",")
    if len(values) != 3:
        return {"gpu_power_w": "", "gpu_temperature_c": "", "vram_used_mb": ""}
    return {
        "gpu_power_w": float(values[0].strip()),
        "gpu_temperature_c": float(values[1].strip()),
        "vram_used_mb": float(values[2].strip()),
    }


def benchmark_model(model_path, frames, device, image_size, confidence, ground_truth_by_frame=None):
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    load_started = time.perf_counter()
    model = YOLO(str(model_path))
    prediction_class_ids = get_benchmark_class_ids(model)
    if not prediction_class_ids:
        raise ValueError(f"Model has no supported benchmark classes: {model_path}")
    model_names = model.names
    model_name_items = model_names.items() if isinstance(model_names, dict) else enumerate(model_names)
    model_class_names = {int(class_id): str(name).strip().lower() for class_id, name in model_name_items}
    vehicle_classes = get_model_vehicle_classes(model)
    if not vehicle_classes and "vehicle" in model_class_names.values():
        vehicle_classes = {
            class_id: "vehicle"
            for class_id, class_name in model_class_names.items()
            if class_name == "vehicle"
        }
    load_seconds = time.perf_counter() - load_started
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    rss_after_load = process_rss_mb()

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    model.predict(
        frames[0],
        imgsz=image_size,
        conf=confidence,
        classes=prediction_class_ids,
        max_det=200,
        device=device,
        augment=False,
        verbose=False,
    )

    latencies = []
    class_counts = Counter()
    boxes_per_frame = []
    predictions_by_frame = []
    rss_peak_mb = rss_after_load

    for frame in frames:
        started = time.perf_counter()
        result = model.predict(
            frame,
            imgsz=image_size,
            conf=confidence,
            classes=prediction_class_ids,
            max_det=200,
            device=device,
            augment=False,
            verbose=False,
        )[0]
        latencies.append(time.perf_counter() - started)
        current_rss_mb = process_rss_mb()
        if current_rss_mb != "":
            rss_peak_mb = max(rss_peak_mb, current_rss_mb)

        boxes = result.boxes
        frame_count = len(boxes) if boxes is not None else 0
        boxes_per_frame.append(frame_count)
        frame_predictions = []
        if boxes is not None:
            for class_id, score, coordinates in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
                class_name = model_class_names[int(class_id)]
                class_counts[class_name] += 1
                if ground_truth_by_frame is not None and class_name != "person":
                    frame_predictions.append({
                        "box": tuple(float(value) for value in coordinates),
                        "confidence": float(score),
                    })
        predictions_by_frame.append(frame_predictions)

    peak_cuda_mb = ""
    if device.startswith("cuda"):
        peak_cuda_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    mean_latency = float(np.mean(latencies))
    accuracy_metrics = (
        evaluate_detections(ground_truth_by_frame, predictions_by_frame)
        if ground_truth_by_frame is not None
        else {}
    )
    memory_after_inference = process_rss_mb()
    gpu_telemetry = nvidia_smi_telemetry() if device.startswith("cuda") else {
        "gpu_power_w": "",
        "gpu_temperature_c": "",
        "vram_used_mb": "",
    }
    inference_telemetry = {
        "cpu_power_w": "",
        "cpu_temperature_c": cpu_temperature_c(),
        **gpu_telemetry,
    }
    del model
    gc.collect()

    row = {
        "model": model_path.name,
        "checkpoint": str(model_path),
        "checkpoint_mb": file_size_mb,
        "parameters_m": parameter_count / 1_000_000,
        "load_seconds": load_seconds,
        "mean_latency_ms": mean_latency * 1000,
        "p95_latency_ms": float(np.percentile(latencies, 95)) * 1000,
        "inference_fps": 1 / mean_latency if mean_latency else 0,
        "mean_boxes_per_frame": float(np.mean(boxes_per_frame)),
        "vehicle_classes": list(vehicle_classes.values()),
        "vehicle_class_counts": dict(class_counts),
        "rss_after_load_mb": rss_after_load,
        "rss_after_inference_mb": memory_after_inference,
        "rss_peak_mb": rss_peak_mb,
        "peak_vram_mb": peak_cuda_mb,
        **inference_telemetry,
        **accuracy_metrics,
    }
    return row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare checkpoints on shared video frames and/or a labeled validation split."
    )
    parser.add_argument("videos", nargs="*", type=Path, help="One or more local videos; every model receives identical sampled frames")
    parser.add_argument("--ground-truth", type=Path, help="Prepared YOLO dataset directory containing images/val and labels/val")
    parser.add_argument("--models", nargs="+", type=Path, help="Checkpoint paths; defaults to available bundled YOLO models")
    parser.add_argument("--samples", type=int, default=30, help="Uniformly sampled frames per video (default: 30)")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Common inference size for all models")
    parser.add_argument("--conf", type=float, default=FALLBACK_CONF_THRESHOLD, help="Common confidence threshold")
    parser.add_argument("--device", default="cpu", help="Common inference device, e.g. cpu or cuda:0")
    parser.add_argument("--output", type=Path, help="CSV output path")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.videos and not args.ground_truth:
        raise SystemExit("Provide one or more video files, --ground-truth, or both")
    video_paths = [path.expanduser().resolve() for path in args.videos]
    for video_path in video_paths:
        if not video_path.is_file():
            raise SystemExit(f"Video file not found: {video_path}")
    ground_truth_dir = args.ground_truth.expanduser().resolve() if args.ground_truth else None
    if ground_truth_dir and not ground_truth_dir.is_dir():
        raise SystemExit(f"Ground-truth directory not found: {ground_truth_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")

    if args.device == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))

    model_paths = [path.expanduser().resolve() for path in args.models] if args.models else default_model_paths()
    model_paths = [path for path in model_paths if path.is_file() and path.stat().st_size > 1024]
    if not model_paths:
        raise SystemExit("No valid model checkpoints found")

    benchmark_inputs = []
    for video_path in video_paths:
        frames, brightness = load_sample_frames(video_path, args.samples)
        benchmark_inputs.append((str(video_path), frames, brightness, None))
    if ground_truth_dir:
        frames, brightness, ground_truth = load_labeled_samples(ground_truth_dir, args.samples)
        benchmark_inputs.append((f"labeled:{ground_truth_dir}", frames, brightness, ground_truth))

    default_output_name = (
        f"{video_paths[0].stem}_benchmark.csv"
        if len(video_paths) == 1 and not ground_truth_dir
        else "model_benchmark.csv"
    )
    output_path = args.output or (BASE_DIR / "output" / default_output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}; image size: {args.imgsz}; confidence: {args.conf}")
    print("Accuracy is scored against generic vehicle boxes; this dataset cannot score truck/car subtype labels.")

    rows = []
    for input_name, frames, brightness, ground_truth in benchmark_inputs:
        print(f"Input: {input_name}; frames: {len(frames)}; mean brightness: {np.mean(brightness):.1f}/255")
        for model_path in model_paths:
            row = benchmark_model(
                model_path,
                frames,
                args.device,
                args.imgsz,
                args.conf,
                ground_truth,
            )
            row["input"] = input_name
            row["sample_frames"] = len(frames)
            row["mean_brightness"] = float(np.mean(brightness))
            rows.append(row)
            summary = (
                f"{row['model']}: {row['mean_latency_ms']:.1f} ms/frame, "
                f"{row['inference_fps']:.1f} FPS, "
                f"{row['mean_boxes_per_frame']:.2f} boxes/frame, "
                f"{row['checkpoint_mb']:.1f} MB"
            )
            if "ap50" in row:
                summary += (
                    f", AP50 {row['ap50']:.3f}, recall {row['recall_iou50']:.3f}, "
                    f"duplicates {row['duplicate_detections_iou50']}"
                )
            print(summary)

    fieldnames = list(dict.fromkeys(field for row in rows for field in row))
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("CPU power/temperature require board-specific sensors; blank telemetry means unavailable.")
    print(f"CSV results: {output_path}")


if __name__ == "__main__":
    main()
