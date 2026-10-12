import argparse
import os
import shutil
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "data" / "vehicles-nighttime"
DATASET_DIR = BASE_DIR / "data" / "night_vehicle_yolo"
PRETRAINED_MODEL = BASE_DIR / "yolo26n.pt"
FINAL_MODEL = BASE_DIR / "models" / "best.pt"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 1024
SEQUENCE_RANGES = ((0, 2006), (2007, 8247), (8248, 10913))


def parse_ground_truth_row(row, line_number=None):
    values = row.split()
    if not values:
        return None
    if len(values) < 2:
        raise ValueError(f"Invalid ground-truth row at line {line_number}")

    image_id = int(values[0])
    box_count = int(values[1])
    expected_value_count = 2 + box_count * 4
    if box_count < 0 or len(values) != expected_value_count:
        raise ValueError(f"Invalid box count or coordinates at line {line_number}")

    boxes = []
    for box_index in range(box_count):
        offset = 2 + box_index * 4
        x, y, width, height = map(float, values[offset:offset + 4])
        if width <= 0 or height <= 0:
            continue

        x1 = max(0.0, min(float(IMAGE_WIDTH), x))
        y1 = max(0.0, min(float(IMAGE_HEIGHT), y))
        x2 = max(0.0, min(float(IMAGE_WIDTH), x + width))
        y2 = max(0.0, min(float(IMAGE_HEIGHT), y + height))
        if x2 <= x1 or y2 <= y1:
            continue

        center_x = ((x1 + x2) / 2) / IMAGE_WIDTH
        center_y = ((y1 + y2) / 2) / IMAGE_HEIGHT
        normalized_width = (x2 - x1) / IMAGE_WIDTH
        normalized_height = (y2 - y1) / IMAGE_HEIGHT
        boxes.append((center_x, center_y, normalized_width, normalized_height))

    return image_id, boxes


def split_for_image(image_id):
    for start, end in SEQUENCE_RANGES:
        if start <= image_id <= end:
            fraction = (image_id - start) / max(1, end - start + 1)
            return "val" if fraction >= 0.8 else "train"

    return "train" if image_id % 5 else "val"


def source_image_path(image_dir, image_id):
    candidates = (
        image_dir / f"img_{image_id}.jpg",
        image_dir / f"img_{image_id:05d}.jpg",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def link_image(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_dataset(
    source_dir=SOURCE_DIR,
    dataset_dir=DATASET_DIR,
    max_images=0,
    allow_missing_images=False,
):
    annotation_path = source_dir / "gt10913.txt"
    image_dir = source_dir / "data"
    if not annotation_path.is_file() or not image_dir.is_dir():
        raise FileNotFoundError(
            f"Expected gt10913.txt and data/*.jpg under {source_dir}"
        )

    counts = {"train": 0, "val": 0}
    image_count = 0
    for line_number, row in enumerate(annotation_path.read_text(encoding="utf-8").splitlines(), 1):
        parsed = parse_ground_truth_row(row, line_number)
        if parsed is None:
            continue

        image_id, boxes = parsed
        if max_images and image_id >= max_images:
            continue

        source_image = source_image_path(image_dir, image_id)
        if not source_image.is_file():
            if allow_missing_images:
                continue
            raise FileNotFoundError(f"Missing image for annotation row {line_number}: {source_image}")

        image = cv2.imread(str(source_image), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not decode image: {source_image}")
        height, width = image.shape

        split = split_for_image(image_id)
        image_path = dataset_dir / "images" / split / source_image.name
        label_path = dataset_dir / "labels" / split / f"{source_image.stem}.txt"
        link_image(source_image, image_path)

        label_path.parent.mkdir(parents=True, exist_ok=True)
        with label_path.open("w", encoding="utf-8", newline="\n") as label_file:
            for center_x, center_y, normalized_width, normalized_height in boxes:
                if width != IMAGE_WIDTH or height != IMAGE_HEIGHT:
                    raise ValueError(
                        f"Unexpected image size {width}x{height} in {source_image}"
                    )
                label_file.write(
                    f"0 {center_x:.6f} {center_y:.6f} "
                    f"{normalized_width:.6f} {normalized_height:.6f}\n"
                )

        counts[split] += 1
        image_count += 1

    if not counts["train"] or not counts["val"]:
        raise ValueError("The prepared dataset needs non-empty train and validation splits")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = dataset_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8", newline="\n") as yaml_file:
        yaml_file.write(
            f"path: {dataset_dir.resolve().as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: vehicle\n"
        )

    return yaml_path, counts, image_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and train a one-class nighttime vehicle detector."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=0, help="Optional image-id limit for a quick run")
    parser.add_argument("--allow-missing-images", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not PRETRAINED_MODEL.is_file() or PRETRAINED_MODEL.stat().st_size <= 1024:
        raise SystemExit(f"Pretrained checkpoint not found or empty: {PRETRAINED_MODEL}")

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    data_yaml, split_counts, image_count = prepare_dataset(
        args.source.resolve(),
        args.dataset.resolve(),
        args.max_images,
        args.allow_missing_images,
    )
    print(f"Prepared {image_count} images: {split_counts}; dataset config: {data_yaml}")
    print(f"Training on CPU with 2 PyTorch threads, image size {args.imgsz}, batch {args.batch}")

    model = YOLO(str(PRETRAINED_MODEL))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device="cpu",
        project=str(BASE_DIR / "runs"),
        name="night_vehicle_yolo26n",
        exist_ok=True,
        patience=5,
        cache=False,
        amp=False,
        plots=False,
        verbose=True,
    )

    best_checkpoint = Path(model.trainer.best)
    if not best_checkpoint.is_file() or best_checkpoint.stat().st_size <= 1024:
        raise RuntimeError(f"Training did not produce a valid best checkpoint: {best_checkpoint}")

    trained_model = YOLO(str(best_checkpoint))
    if list(trained_model.names.values()) != ["vehicle"]:
        raise RuntimeError(f"Unexpected trained class names: {trained_model.names}")

    FINAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, FINAL_MODEL)
    print(f"Saved trained one-class vehicle model to: {FINAL_MODEL}")


if __name__ == "__main__":
    main()
