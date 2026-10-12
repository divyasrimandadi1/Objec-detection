import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import src.benchmark_models as benchmark_module
import src.main as main_module
import src.train_night_vehicle as night_training_module


class ResolveVideoSourceTests(unittest.TestCase):
    def test_night_ground_truth_row_converts_vehicle_boxes_to_normalized_yolo(self):
        parsed = night_training_module.parse_ground_truth_row(
            "12 2 100 200 200 100 1200 900 200 200"
        )

        self.assertEqual(parsed[0], 12)
        self.assertEqual(
            parsed[1],
            [
                (200 / 1280, 250 / 1024, 200 / 1280, 100 / 1024),
                (1240 / 1280, 962 / 1024, 80 / 1280, 124 / 1024),
            ],
        )

    def test_night_ground_truth_supports_empty_frames(self):
        self.assertEqual(night_training_module.parse_ground_truth_row("7 0"), (7, []))

    def test_night_ground_truth_rejects_malformed_box_rows(self):
        with self.assertRaises(ValueError):
            night_training_module.parse_ground_truth_row("7 1 20 30 40")

    def test_night_dataset_validation_uses_later_sequence_frames(self):
        self.assertEqual(night_training_module.split_for_image(100), "train")
        self.assertEqual(night_training_module.split_for_image(1900), "val")
        self.assertEqual(night_training_module.split_for_image(7000), "val")

    def test_night_dataset_finds_zero_padded_image_names(self):
        with TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            expected = image_dir / "img_02007.jpg"
            expected.touch()

            self.assertEqual(night_training_module.source_image_path(image_dir, 2007), expected)

    def test_benchmark_samples_frames_uniformly(self):
        self.assertEqual(
            benchmark_module.sample_frame_indices(100, 5),
            [0, 24, 49, 74, 99],
        )

    def test_benchmark_rejects_invalid_sample_counts(self):
        with self.assertRaises(ValueError):
            benchmark_module.sample_frame_indices(0, 5)
        with self.assertRaises(ValueError):
            benchmark_module.sample_frame_indices(100, 0)

    def test_benchmark_vehicle_classes_support_coco_and_single_class_models(self):
        class CocoModel:
            names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        class NightModel:
            names = {0: "vehicle"}

        self.assertEqual(benchmark_module.get_benchmark_class_ids(CocoModel()), [0, 1, 2, 3, 5, 7])
        self.assertEqual(benchmark_module.get_benchmark_class_ids(NightModel()), [0])

    def test_labeled_benchmark_loader_reads_yolo_validation_boxes(self):
        with TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            image_dir = dataset_dir / "images" / "val"
            label_dir = dataset_dir / "labels" / "val"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            image = np.zeros((100, 200, 3), dtype=np.uint8)
            main_module.cv2.imwrite(str(image_dir / "sample.jpg"), image)
            (label_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.25\n", encoding="utf-8")

            frames, brightness, ground_truth = benchmark_module.load_labeled_samples(dataset_dir, 1)

        self.assertEqual(len(frames), 1)
        self.assertEqual(ground_truth, [[(50.0, 37.5, 150.0, 62.5)]])
        self.assertEqual(len(brightness), 1)

    def test_benchmark_perfect_detection_has_perfect_accuracy(self):
        metrics = benchmark_module.evaluate_detections(
            [[(0, 0, 10, 10)]],
            [[{"box": (0, 0, 10, 10), "confidence": 0.9}]],
        )

        self.assertEqual(metrics["precision_iou50"], 1.0)
        self.assertEqual(metrics["recall_iou50"], 1.0)
        self.assertEqual(metrics["duplicate_detections_iou50"], 0)
        self.assertAlmostEqual(metrics["ap50"], 1.0)

    def test_benchmark_counts_repeated_box_as_duplicate(self):
        metrics = benchmark_module.evaluate_detections(
            [[(0, 0, 10, 10)]],
            [[
                {"box": (0, 0, 10, 10), "confidence": 0.9},
                {"box": (0, 0, 10, 10), "confidence": 0.8},
            ]],
        )

        self.assertEqual(metrics["duplicate_detections_iou50"], 1)
        self.assertEqual(metrics["false_positives_iou50"], 1)
        self.assertEqual(metrics["recall_iou50"], 1.0)

    def test_default_benchmark_models_are_valid_checkpoints(self):
        model_paths = benchmark_module.default_model_paths()

        self.assertTrue(model_paths)
        self.assertTrue(all(path.is_file() and path.stat().st_size > 1024 for path in model_paths))

    def test_local_video_path_is_resolved(self):
        with TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "traffic.mp4"
            video_path.touch()

            resolved = main_module.resolve_local_video_path(str(video_path))

        self.assertEqual(resolved, str(video_path.resolve()))

    def test_nonexistent_or_web_video_path_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            main_module.resolve_local_video_path("https://example.com/traffic.mp4")

    def test_output_writer_uses_input_name_and_fallback_fps(self):
        class FakeWriter:
            def isOpened(self):
                return True

        with TemporaryDirectory() as temp_dir, \
                patch.object(main_module, "OUTPUT_DIR", Path(temp_dir)), \
                patch.object(main_module.cv2, "VideoWriter_fourcc", return_value=123), \
                patch.object(main_module.cv2, "VideoWriter", return_value=FakeWriter()) as video_writer:
            writer, output_path = main_module.create_output_writer(
                "C:/videos/traffic.mp4", 0, 1280, 720
            )

        self.assertIsInstance(writer, FakeWriter)
        self.assertEqual(output_path, Path(temp_dir) / "traffic_detections.mp4")
        self.assertEqual(video_writer.call_args.args[2:], (30.0, (1280, 720)))

    def test_every_frame_is_configured_for_inference(self):
        self.assertEqual(main_module.FRAME_SKIP, 1)

    def test_cpu_image_size_uses_high_detail_setting(self):
        self.assertEqual(main_module.CPU_IMG_SIZE, 960)

    def test_roi_includes_the_full_frame(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        roi, offset = main_module.extract_roi(frame)

        self.assertTrue(np.shares_memory(roi, frame))
        self.assertEqual(roi.shape, frame.shape)
        self.assertEqual(offset, (0, 0))

    def test_output_video_fps_accounts_for_skipped_frames(self):
        class FakeWriter:
            def isOpened(self):
                return True

        with TemporaryDirectory() as temp_dir, \
                patch.object(main_module, "OUTPUT_DIR", Path(temp_dir)), \
                patch.object(main_module.cv2, "VideoWriter_fourcc", return_value=123), \
                patch.object(main_module.cv2, "VideoWriter", return_value=FakeWriter()) as video_writer:
            main_module.create_output_writer(
                "C:/videos/traffic.mp4", 30, 1280, 720, frame_skip=6
            )

        self.assertEqual(video_writer.call_args.args[2:], (5.0, (1280, 720)))

    def test_cpu_thread_count_is_capped_by_available_cores(self):
        with patch.object(main_module.os, "cpu_count", return_value=2), \
                patch.object(main_module.torch, "set_num_threads") as set_num_threads:
            thread_count = main_module.configure_cpu_threads("cpu")

        self.assertEqual(thread_count, 2)
        set_num_threads.assert_called_once_with(2)

    def test_model_prefers_yolo26n_checkpoint_on_cpu(self):
        with patch.object(main_module.torch.cuda, "is_available", return_value=False), \
                patch.object(main_module.Path, "exists", return_value=False), \
                patch.object(main_module.Path, "is_file", return_value=False):
            self.assertEqual(main_module.resolve_model_path(), "yolo26n.pt")

    def test_all_requested_vehicle_classes_are_enabled(self):
        self.assertEqual(
            main_module.VEHICLE_CLASSES,
            {
                0: "person",
                1: "bicycle",
                2: "car",
                3: "motorcycle",
                5: "bus",
                7: "truck",
            },
        )

    def test_inference_uses_raw_predictions_without_tracker_filtering(self):
        expected_result = object()

        class FakeModel:
            def predict(self, **kwargs):
                self.kwargs = kwargs
                return [expected_result]

            def track(self, *args, **kwargs):
                raise AssertionError("Detection should not pass through the tracker")

        model = FakeModel()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        result = main_module.detect_vehicles(
            model, frame, "cpu", main_module.VEHICLE_CLASSES
        )

        self.assertIs(result, expected_result)
        self.assertEqual(model.kwargs["conf"], 0.15)
        self.assertEqual(model.kwargs["classes"], [0, 1, 2, 3, 5, 7])

    def test_inference_uses_class_aware_confidence(self):
        class FakeModel:
            names = {
                0: "person",
                1: "bicycle",
                2: "car",
                3: "motorcycle",
                5: "bus",
                7: "truck",
            }

            def predict(self, **kwargs):
                self.kwargs = kwargs
                return [object()]

        model = FakeModel()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        main_module.detect_vehicles(model, frame, "cpu")

        self.assertEqual(model.kwargs["conf"], 0.15)

    def test_generic_single_class_model_is_not_class_mapped(self):
        class FakeModel:
            names = {0: "vehicle"}

        self.assertEqual(main_module.get_model_vehicle_classes(FakeModel()), {})

    def test_coco_model_only_uses_matching_vehicle_names(self):
        class FakeModel:
            names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        self.assertEqual(
            main_module.get_model_vehicle_classes(FakeModel()),
            main_module.VEHICLE_CLASSES,
        )

    def test_nearby_vehicle_boxes_are_not_merged_without_overlap(self):
        boxes = [
            {"cls_id": 2, "confidence": 0.91, "x1": 100, "y1": 80, "x2": 260, "y2": 220},
            {"cls_id": 2, "confidence": 0.89, "x1": 220, "y1": 70, "x2": 340, "y2": 230},
        ]

        merged = main_module.merge_overlapping_boxes(boxes, iou_threshold=0.45)

        self.assertEqual(len(merged), 2)

    def test_overlapping_same_class_vehicles_are_not_merged(self):
        boxes = [
            {"cls_id": 2, "confidence": 0.91, "x1": 100, "y1": 80, "x2": 260, "y2": 220},
            {"cls_id": 2, "confidence": 0.89, "x1": 110, "y1": 85, "x2": 270, "y2": 225},
        ]

        merged = main_module.merge_overlapping_boxes(boxes)

        self.assertEqual(len(merged), 2)

    def test_overlapping_car_truck_predictions_keep_truck_confidence(self):
        boxes = [
            {"cls_id": 2, "confidence": 0.64, "x1": 825, "y1": 507, "x2": 876, "y2": 539},
            {"cls_id": 7, "confidence": 0.407, "x1": 825, "y1": 507, "x2": 876, "y2": 540},
        ]

        merged = main_module.merge_overlapping_boxes(boxes)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["cls_id"], 7)
        self.assertAlmostEqual(merged[0]["confidence"], 0.407)

    def test_large_truck_like_box_is_reclassified_as_truck(self):
        h, w = 720, 1280
        box = (150, 260, 720, 520)

        cls_id = main_module.reclassify_vehicle(2, *box, (h, w, 3))

        self.assertEqual(cls_id, 7)

    def test_small_truck_detection_is_not_downgraded_to_car(self):
        cls_id = main_module.reclassify_vehicle(7, 100, 100, 150, 150, (720, 1280, 3))

        self.assertEqual(cls_id, 7)

    def test_medium_truck_box_uses_zoom_refinement(self):
        class FakeBoxes:
            conf = main_module.torch.tensor([0.144])
            cls = main_module.torch.tensor([7])
            xyxy = main_module.torch.tensor([[0, 0, 116, 76]])

            def __len__(self):
                return 1

        class FakeResult:
            boxes = FakeBoxes()

        class FakeModel:
            def track(self, *args, **kwargs):
                return [FakeResult()]

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        refined = main_module.refine_large_vehicle(
            FakeModel(), frame, 1111, 434, 1227, 510, "cpu"
        )

        self.assertIsNotNone(refined)
        self.assertEqual(refined["cls_id"], 7)

    def test_track_class_stays_stable_when_car_truck_predictions_alternate(self):
        history = {}
        track_id = 12

        classes = [
            main_module.stabilize_vehicle_class(track_id, cls_id, history)
            for cls_id in (7, 2, 7, 2, 7, 2)
        ]

        self.assertEqual(classes, [7, 7, 7, 7, 7, 7])

    def test_low_light_enhancement_brightens_dark_frames_only(self):
        dark_frame = np.full((24, 24, 3), 32, dtype=np.uint8)
        daylight_frame = np.full((24, 24, 3), 180, dtype=np.uint8)

        enhanced_dark = main_module.enhance_low_light(dark_frame)
        enhanced_daylight = main_module.enhance_low_light(daylight_frame)

        self.assertGreater(float(enhanced_dark.mean()), float(dark_frame.mean()))
        np.testing.assert_array_equal(enhanced_daylight, daylight_frame)

if __name__ == "__main__":
    unittest.main()
