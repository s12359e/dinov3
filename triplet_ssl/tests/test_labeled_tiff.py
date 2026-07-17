import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from triplet_ssl.eval.labeled_tiff import (
    evaluate_labeled_tiffs,
    parse_point_label,
    validation_selection_key,
)
from triplet_ssl.models.order_aware_fusion import OrderAwareTripletFusionHead
from triplet_ssl.train_triplet import _validate_and_update_best


class TrackingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


def touch(path: Path) -> None:
    path.touch()


class LabeledTiffValidationTest(unittest.TestCase):
    def test_best_bundle_updates_only_when_validation_key_improves(self):
        class DatasetContract:
            source_dtype = "float32"
            source_layout = "YXS"
            sample_format = 3

        teacher = nn.Linear(4, 4)
        teacher_fusion = OrderAwareTripletFusionHead(4, 6, dropout=0.0)
        student_fusion = OrderAwareTripletFusionHead(4, 6, dropout=0.0)
        data_config = {"crop_size": 128, "channel_order": [0, 1, 2]}

        def result(top5, top1, distance):
            return {
                "summary": {
                    "n_images": 10,
                    "top_k": 5,
                    "match_radius_px": 15.0,
                    "coordinate_base": 0,
                    "top_k_hit_count": int(top5 * 10),
                    "top_k_hit_rate": top5,
                    "top5_hit_count": int(top5 * 10),
                    "top5_hit_rate": top5,
                    "top1_hit_count": int(top1 * 10),
                    "top1_hit_rate": top1,
                    "mean_min_distance": distance,
                    "mean_matched_rank": 2.0,
                    "matched_images": int(top5 * 10),
                },
                "predictions": [],
            }

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch("triplet_ssl.train_triplet.evaluate_labeled_tiffs",
                       return_value=result(0.5, 0.2, 8.0)):
                summary, best_key = _validate_and_update_best(
                    teacher, teacher_fusion, student_fusion, DatasetContract(),
                    data_config, 3, 224, 16, 10, out, torch.device("cpu"),
                    {}, 5, 15.0, 0, out, None)
            best_path = out / "phase3_best.pth"
            first = torch.load(best_path, map_location="cpu")
            self.assertEqual(first["global_step"], 10)
            self.assertEqual(first["validation"]["top5_hit_rate"], 0.5)
            self.assertEqual(
                summary["selection_rule"],
                "top5_hit_rate_then_top1_hit_rate_then_lower_mean_min_distance")

            # A later but worse candidate still gets a report, but must not
            # replace the deployment bundle selected at step 10.
            with patch("triplet_ssl.train_triplet.evaluate_labeled_tiffs",
                       return_value=result(0.4, 0.3, 2.0)):
                _, unchanged_key = _validate_and_update_best(
                    teacher, teacher_fusion, student_fusion, DatasetContract(),
                    data_config, 3, 224, 16, 20, out, torch.device("cpu"),
                    {}, 5, 15.0, 0, out, best_key)
            second = torch.load(best_path, map_location="cpu")
            self.assertEqual(unchanged_key, best_key)
            self.assertEqual(second["global_step"], 10)
            self.assertTrue((out / "phase3_val_step000020.json").is_file())

    def test_parse_point_label_is_strict_and_converts_coordinate_base(self):
        self.assertEqual(parse_point_label("sample#12,34.tif"), (12, 34))
        self.assertEqual(parse_point_label("sample#12,34.TIFF", coordinate_base=1), (11, 33))
        for invalid in (
            "sample.tif",
            "sample#1,2#3,4.tif",
            "sample#1,2_extra.tif",
            "sample#1, 2.tif",
            "#1,2.tif",
            "sample#-1,2.tif",
            "sample#1,2.png",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_point_label(invalid)
        with self.assertRaises(ValueError):
            parse_point_label("sample#1,2.tif", coordinate_base=2)

    def test_top_five_matching_clipping_strict_radius_and_mode_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a#8,8.tif"
            second = root / "b#31,31.tiff"
            touch(first)
            touch(second)

            maps = {
                first.name: np.array([[9.0, 9.0], [1.0, 0.0]], np.float32),
                second.name: np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
            }
            model = TrackingModule().train()
            fusion = TrackingModule().eval()
            observed_modes = []

            def fake_score(model_arg, path, device, **kwargs):
                self.assertIs(model_arg, model)
                self.assertIs(kwargs["fusion_head"], fusion)
                self.assertEqual(kwargs["method"], "fusion")
                observed_modes.append((model.training, fusion.training))
                return maps[path.name], {}, (32, 32), None

            result = evaluate_labeled_tiffs(
                model,
                fusion,
                root,
                "cpu",
                score_kwargs={"method": "fusion"},
                score_fn=fake_score,
            )

        self.assertEqual(observed_modes, [(False, False), (False, False)])
        self.assertTrue(model.training)
        self.assertFalse(fusion.training)
        predictions = result["predictions"]
        # Equal scores use deterministic row-major order.
        self.assertEqual(
            [(p["grid_x"], p["grid_y"]) for p in predictions[0]["top_points"][:2]],
            [(0, 0), (1, 0)],
        )
        self.assertTrue(predictions[0]["top1_hit"])
        self.assertEqual(predictions[0]["matched_rank"], 1)
        # The final patch centre (24, 24) remains inside this full image.
        second_point = predictions[1]["top_points"][0]
        self.assertEqual((second_point["x"], second_point["y"]), (24.0, 24.0))
        self.assertTrue(predictions[1]["top1_hit"])
        summary = result["summary"]
        self.assertEqual(summary["top5_hit_rate"], 1.0)
        self.assertEqual(summary["top1_hit_rate"], 1.0)
        self.assertEqual(summary["mean_matched_rank"], 1.0)
        json.dumps(result, allow_nan=False)
        self.assertEqual(
            validation_selection_key(result),
            (1.0, 1.0, -summary["mean_min_distance"]),
        )

        # Exactly 15 pixels is not a match; the required comparison is strict.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strict#23,8.tif"
            touch(path)

            def strict_score(*args, **kwargs):
                scores = np.array([[1.0, 0.0], [0.0, 0.0]], np.float32)
                return scores, {}, (32, 32), None

            strict = evaluate_labeled_tiffs(
                model,
                None,
                path,
                "cpu",
                score_kwargs={},
                top_k=1,
                match_radius_px=15,
                score_fn=strict_score,
            )
        self.assertEqual(strict["predictions"][0]["top_points"][0]["distance_px"], 15.0)
        self.assertFalse(strict["predictions"][0]["top_k_hit"])
        self.assertIsNone(strict["summary"]["mean_matched_rank"])

    def test_partial_edge_patch_centre_is_clipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edge#16,16.tif"
            touch(path)

            def fake_score(*args, **kwargs):
                scores = np.array([[0.0, 0.0], [0.0, 1.0]], np.float32)
                return scores, {}, (17, 17), None

            result = evaluate_labeled_tiffs(
                TrackingModule(),
                None,
                path,
                "cpu",
                score_kwargs={},
                top_k=1,
                score_fn=fake_score,
            )
        point = result["predictions"][0]["top_points"][0]
        self.assertEqual((point["x"], point["y"]), (16.0, 16.0))
        self.assertTrue(point["matched"])

    def test_rejects_bad_dataset_and_restores_mode_after_score_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least one"):
                evaluate_labeled_tiffs(
                    TrackingModule(), None, directory, "cpu", score_kwargs={}
                )

        cases = (
            ("outside#32,0.tif", np.zeros((2, 2), np.float32), "outside"),
            ("nonfinite#0,0.tif", np.array([[np.nan, 0], [0, 0]]), "NaN or Inf"),
            ("wrong_shape#0,0.tif", np.zeros((1, 1), np.float32), "expected"),
        )
        for filename, scores, message in cases:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / filename
                touch(path)
                model = TrackingModule().train()

                def fake_score(*args, **kwargs):
                    self.assertFalse(model.training)
                    return scores, {}, (32, 32), None

                with self.assertRaisesRegex(ValueError, message):
                    evaluate_labeled_tiffs(
                        model,
                        None,
                        path,
                        "cpu",
                        score_kwargs={},
                        score_fn=fake_score,
                    )
                self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
