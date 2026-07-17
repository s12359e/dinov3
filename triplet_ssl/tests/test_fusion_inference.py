import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from PIL import Image

from triplet_ssl import IMG_MEAN, IMG_STD
from triplet_ssl.data.triplet_dataset import read_tiff3
from triplet_ssl.infer import (
    detections_from,
    load_inference_bundle,
    resolve_context_halo,
    resolve_inference_method,
    resolve_inference_tile,
    score_map,
)
from triplet_ssl.models.order_aware_fusion import OrderAwareTripletFusionHead
from triplet_ssl.train_triplet import load_checkpoint as load_training_checkpoint


class FakeBackbone(nn.Module):
    """Cheap deterministic patch tokens for exercising the real tiling path."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward_features(self, x):
        v = (F.avg_pool2d(x.mean(dim=1, keepdim=True), 16, 16)
             .flatten(2).transpose(1, 2) * self.scale)
        return {"x_norm_patchtokens": torch.cat((v, v.square(), v.sin(), torch.ones_like(v)), -1)}


class ZeroFusionHead(nn.Module):
    def forward(self, target, ref1, ref2):
        return target.new_zeros(target.shape[:2])


class LocalIndexFusionHead(nn.Module):
    def forward(self, target, ref1, ref2):
        idx = torch.arange(target.shape[1], device=target.device, dtype=target.dtype)
        return (idx / 10.0).expand(target.shape[0], -1)


def write_tiff(path, h, w):
    yy, xx = np.mgrid[:h, :w]
    arr = np.stack(((xx + yy) % 255, (2 * xx + yy) % 255, (xx + 2 * yy) % 255), -1)
    Image.fromarray(arr.astype(np.uint8)).save(path)


class FusionInferenceTest(unittest.TestCase):
    def test_float32_single_page_hwc_and_chw_layouts(self):
        with tempfile.TemporaryDirectory() as td:
            hwc_path = Path(td) / "train_hwc.tiff"
            hwc = np.empty((384, 384, 3), np.float32)
            hwc[..., 0], hwc[..., 1], hwc[..., 2] = 11.25, 22.5, 33.75
            tifffile.imwrite(
                hwc_path, hwc, photometric="rgb", planarconfig="contig",
                metadata={"axes": "YXS"})
            target, ref1, ref2, meta = read_tiff3(
                hwc_path, expected_dtype="float32", return_metadata=True)
            self.assertEqual(target.shape, (384, 384, 3))
            self.assertEqual(meta["source_layout"], "YXS")
            self.assertEqual(meta["sample_format"], 3)
            np.testing.assert_allclose(
                [target[0, 0, 0], ref1[0, 0, 0], ref2[0, 0, 0]],
                [11.25, 22.5, 33.75])

            chw_path = Path(td) / "test_chw.tiff"
            chw = np.empty((3, 448, 464), np.float32)
            chw[0], chw[1], chw[2] = 17.125, 91.5, 203.875
            tifffile.imwrite(
                chw_path, chw, photometric="rgb", planarconfig="separate",
                metadata={"axes": "SYX"})
            # Verify role mapping occurs after SYX -> YXS canonicalization.
            target, ref1, ref2, meta = read_tiff3(
                chw_path, channel_order=(2, 0, 1), expected_dtype="float32",
                return_metadata=True)
            self.assertEqual(target.shape, (448, 464, 3))
            self.assertEqual(meta["source_layout"], "SYX")
            np.testing.assert_allclose(
                [target[0, 0, 0], ref1[0, 0, 0], ref2[0, 0, 0]],
                [203.875, 17.125, 91.5])

            scores, _, hw, _ = score_map(
                FakeBackbone(), chw_path, torch.device("cpu"), tile=128,
                context_halo=32, chunk=16, method="fusion",
                fusion_head=ZeroFusionHead(), channel_order=(2, 0, 1),
                expected_dtype="float32")
            self.assertEqual(hw, (448, 464))
            self.assertEqual(scores.shape, (28, 29))
            np.testing.assert_allclose(scores, 0.5)

    def test_float_tiff_rejects_nonfinite_and_out_of_range_values(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad_float.tiff"
            bad = np.zeros((8, 8, 3), np.float32)
            bad[0, 0, 0] = np.nan
            tifffile.imwrite(path, bad, photometric="rgb", metadata={"axes": "YXS"})
            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                read_tiff3(path)

            bad[0, 0, 0] = 256.0
            tifffile.imwrite(path, bad, photometric="rgb", metadata={"axes": "YXS"})
            with self.assertRaisesRegex(ValueError, "must already be in"):
                read_tiff3(path)

    def test_fusion_score_map_crops_padding_and_sigmoids_once(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.tiff"
            write_tiff(path, 480, 480)
            scores, aux, hw, target = score_map(
                FakeBackbone(), path, torch.device("cpu"), tile=128, chunk=16,
                context_halo=32, method="fusion", fusion_head=ZeroFusionHead())
        self.assertEqual(scores.shape, (30, 30))
        self.assertEqual(hw, (480, 480))
        self.assertEqual(target.shape, (480, 480, 3))
        np.testing.assert_allclose(scores, 0.5)
        self.assertIn("fusion_prob", aux)
        self.assertTrue(all(v.shape == scores.shape for v in aux.values()))
        self.assertTrue(all(np.isfinite(v).all() for v in aux.values()))

    def test_partial_edge_grid_and_detection_box_are_clipped(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "partial.tif"
            write_tiff(path, 479, 481)
            scores, aux, hw, _ = score_map(
                FakeBackbone(), path, torch.device("cpu"), tile=128,
                method="fusion", fusion_head=ZeroFusionHead())
        self.assertEqual(scores.shape, (30, 31))
        hot = np.zeros_like(scores)
        hot[-1, -1] = 1.0
        faux = {k: np.zeros_like(hot) for k in ("d_t1", "d_t2", "d_12")}
        faux["fusion_prob"] = hot.copy()
        det = detections_from(hot, 0.5, faux, image_hw=hw)[0]
        self.assertLessEqual(det["x"] + det["w"], hw[1])
        self.assertLessEqual(det["y"] + det["h"], hw[0])
        self.assertEqual(det["presence"], "target_unique_1_0_0")

    def test_chunking_does_not_change_fusion_map(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiles.tif"
            write_tiff(path, 128, 256)
            kw = dict(tile=128, context_halo=32, method="fusion",
                      fusion_head=ZeroFusionHead())
            a, _, _, _ = score_map(
                FakeBackbone(), path, torch.device("cpu"), chunk=1, **kw)
            b, _, _, _ = score_map(
                FakeBackbone(), path, torch.device("cpu"), chunk=8, **kw)
        np.testing.assert_allclose(a, b)

    def test_method_resolution_is_explicit_for_legacy_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "requires a phase-3 checkpoint"):
            resolve_inference_method("fusion", None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(resolve_inference_method("auto", None), "residual")
        self.assertTrue(caught)
        self.assertEqual(resolve_inference_method("auto", ZeroFusionHead()), "fusion")
        with self.assertRaisesRegex(ValueError, "unknown inference method"):
            resolve_inference_method("not-a-method", None)

    def test_fusion_default_tile_comes_from_checkpoint_contract(self):
        meta = {"fusion_head_config": {"train_tile": 224}}
        self.assertEqual(resolve_inference_tile(None, "fusion", meta), 224)
        self.assertEqual(resolve_inference_tile(None, "residual", meta), 128)
        self.assertEqual(resolve_inference_tile(64, "fusion", meta), 64)
        self.assertEqual(resolve_context_halo(None, "fusion", 128), 32)
        self.assertEqual(resolve_context_halo(None, "residual", 128), 0)

    def test_context_halo_stitches_only_central_tile_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "central.tif"
            write_tiff(path, 128, 128)
            scores, _, _, _ = score_map(
                FakeBackbone(), path, torch.device("cpu"), tile=128,
                context_halo=32, method="fusion",
                fusion_head=LocalIndexFusionHead())
        expected_logits = np.arange(64, dtype=np.float32).reshape(8, 8)[2:6, 2:6] / 10
        expected = 1.0 / (1.0 + np.exp(-expected_logits))
        self.assertEqual(scores.shape, (8, 8))
        for y in (0, 4):
            for x in (0, 4):
                np.testing.assert_allclose(
                    scores[y:y + 4, x:x + 4], expected, atol=1e-6)

    def test_fusion_tile_contract_is_enforced(self):
        head = OrderAwareTripletFusionHead.from_config(
            OrderAwareTripletFusionHead(4, 6).export_config(
                patch_size=16, train_tile=128))
        with self.assertRaisesRegex(ValueError, "trained with tile=128"):
            score_map(FakeBackbone(), "unused.tif", torch.device("cpu"),
                      tile=64, method="fusion", fusion_head=head)
        # Explicit legacy diagnostics do not execute the fusion head, so its
        # context contract must not constrain their independently chosen tile.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "legacy_tile.tif"
            write_tiff(path, 64, 64)
            scores, _, _, _ = score_map(
                FakeBackbone(), path, torch.device("cpu"), tile=64,
                method="residual", fusion_head=head)
        self.assertEqual(scores.shape, (4, 4))

    def test_programmatic_auto_fallback_forces_residual_min(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "auto.tif"
            write_tiff(path, 128, 128)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auto, _, _, _ = score_map(
                    FakeBackbone(), path, torch.device("cpu"), method="auto",
                    residual_mode="mean")
            direct, _, _, _ = score_map(
                FakeBackbone(), path, torch.device("cpu"), method="residual",
                residual_mode="min")
        np.testing.assert_allclose(auto, direct)

    def test_bundle_roundtrip_loads_head_strictly(self):
        torch.manual_seed(4)
        head = OrderAwareTripletFusionHead(4, 6, dropout=0.0).eval()
        config = head.export_config(patch_size=16, train_tile=128)
        checkpoint = {
            "checkpoint_version": 2,
            "teacher_backbone": FakeBackbone().state_dict(),
            "teacher_fusion_head": head.state_dict(),
            "fusion_head_config": config,
            "preprocess": {
                "mean": [11.0, 12.0, 13.0],
                "std": [2.0, 3.0, 4.0],
                "input_scaling": "float_0_255_identity_v1",
                "source_dtype": "float32",
                "source_layout": "YXS",
                "sample_format": 3,
                "supported_tiff_layouts": ["YXS", "SYX", "3PAGE_YX"],
                "uint16_black_level": 0,
                "uint16_white_level": 4095,
                "channel_order": ["target", "ref1", "ref2"],
                "source_channel_indices": [0, 1, 2],
                "register": False,
            },
            "phase": 3,
            "global_step": 20,
            "validation": {"top5_hit_rate": 0.875, "match_radius_px": 15.0},
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "deploy.pth"
            torch.save(checkpoint, path)
            _, loaded, meta = load_inference_bundle(
                path, torch.device("cpu"), backbone_factory=FakeBackbone)
        inputs = tuple(torch.randn(2, 7, 4) for _ in range(3))
        torch.testing.assert_close(head(*inputs), loaded(*inputs))
        self.assertEqual(meta["checkpoint_version"], 2)
        self.assertEqual(meta["preprocess"]["uint16_white_level"], 4095)
        self.assertEqual(meta["preprocess"]["mean"], [11.0, 12.0, 13.0])
        self.assertEqual(meta["preprocess"]["source_dtype"], "float32")
        self.assertEqual(meta["global_step"], 20)
        self.assertEqual(meta["validation"]["top5_hit_rate"], 0.875)

    def test_uint16_tiff_uses_explicit_sensor_range(self):
        values = (np.arange(16, dtype=np.uint16).reshape(4, 4) * 273)
        pages = [Image.fromarray(values), Image.fromarray(values // 2),
                 Image.fromarray(np.zeros_like(values))]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "twelve_bit.tiff"
            pages[0].save(path, save_all=True, append_images=pages[1:])
            target, ref1, ref2 = read_tiff3(
                path, uint16_black_level=0, uint16_white_level=4095)
        self.assertEqual(target.dtype, np.float32)
        self.assertEqual(int(target[0, 0, 0]), 0)
        self.assertEqual(int(target[-1, -1, 0]), 255)
        self.assertEqual(int(ref2.max()), 0)
        self.assertGreater(int(ref1[-1, -1, 0]), 120)

    def test_uint16_sub_byte_contrast_survives_as_float(self):
        # At a 0..4095 range one 8-bit step is ~16 sensor counts. Keep a 4-count
        # difference so a low-contrast optical PSF is not rounded away at ingest.
        base = np.full((4, 4), 1000, np.uint16)
        changed = base.copy(); changed[2, 2] += 4
        pages = [Image.fromarray(changed), Image.fromarray(base), Image.fromarray(base)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sub_byte.tiff"
            pages[0].save(path, save_all=True, append_images=pages[1:])
            target, ref1, _ = read_tiff3(
                path, uint16_black_level=0, uint16_white_level=4095)
        delta = float(target[2, 2, 0] - ref1[2, 2, 0])
        self.assertGreater(delta, 0.0)
        self.assertLess(delta, 1.0)

    def test_legacy_uint16_decode_mode_reproduces_high_byte(self):
        values = np.array([[255, 256], [511, 65535]], dtype=np.uint16)
        pages = [Image.fromarray(values) for _ in range(3)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "legacy16.tiff"
            pages[0].save(path, save_all=True, append_images=pages[1:])
            target, _, _ = read_tiff3(path, uint16_decode_mode="legacy_high_byte")
        np.testing.assert_array_equal(target[:, :, 0], np.array([[0, 1], [1, 255]], np.uint8))

    def test_partial_fusion_bundle_is_rejected(self):
        checkpoint = {
            "checkpoint_version": 2,
            "teacher_backbone": FakeBackbone().state_dict(),
            "teacher_fusion_head": {},
            "preprocess": {},
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "broken.pth"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "must either both be present"):
                load_inference_bundle(path, torch.device("cpu"), backbone_factory=FakeBackbone)

    def test_backbone_only_bundle_preserves_preprocessing_contract(self):
        checkpoint = {
            "checkpoint_version": 1,
            "teacher_backbone": FakeBackbone().state_dict(),
            "preprocess": {
                "mean": [1.0, 2.0, 3.0], "std": [4.0, 5.0, 6.0],
                "input_scaling": "fixed_uint16_range_to_0_255_float_v1",
                "uint16_black_level": 64, "uint16_white_level": 4095,
                "channel_order": ["target", "ref1", "ref2"],
                "source_channel_indices": [2, 0, 1], "register": True,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "phase2.pth"
            torch.save(checkpoint, path)
            _, head, meta = load_inference_bundle(
                path, torch.device("cpu"), backbone_factory=FakeBackbone)
        self.assertIsNone(head)
        self.assertEqual(meta["preprocess"]["uint16_white_level"], 4095)
        self.assertEqual(meta["preprocess"]["source_channel_indices"], [2, 0, 1])
        self.assertEqual(meta["preprocess"]["uint16_decode_mode"], "float_linear")

    def test_legacy_backbone_wrappers_and_ddp_prefixes_load(self):
        expected = torch.tensor(3.25)
        states = [
            {"scale": expected},
            {"model": {"scale": expected}},
            {"teacher": {"module.backbone.scale": expected}},
            {"state_dict": {"backbone.scale": expected}},
        ]
        with tempfile.TemporaryDirectory() as td:
            for i, checkpoint in enumerate(states):
                path = Path(td) / f"legacy_{i}.pth"
                torch.save(checkpoint, path)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    backbone, head, _ = load_inference_bundle(
                        path, torch.device("cpu"), backbone_factory=FakeBackbone)
                self.assertIsNone(head)
                torch.testing.assert_close(backbone.scale.detach(), expected)

    def test_missing_backbone_parameter_fails_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "missing.pth"
            torch.save({"state_dict": {"not_scale": torch.tensor(1.0)}}, path)
            with self.assertRaisesRegex(ValueError, "does not fully cover"):
                load_inference_bundle(path, torch.device("cpu"), backbone_factory=FakeBackbone)

    def test_unexpected_backbone_parameter_fails_loudly(self):
        checkpoint = {"state_dict": {
            "scale": torch.tensor(1.0), "foreign.weight": torch.tensor(2.0)}}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "unexpected.pth"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "outside the canonical inference"):
                load_inference_bundle(path, torch.device("cpu"), backbone_factory=FakeBackbone)
            with self.assertRaisesRegex(ValueError, "outside the canonical training"):
                load_training_checkpoint(FakeBackbone(), path)

    def test_training_loader_strips_wrappers_and_rejects_missing_weights(self):
        with tempfile.TemporaryDirectory() as td:
            valid = Path(td) / "valid.pth"
            torch.save({"teacher": {"module.backbone.scale": torch.tensor(2.5)}}, valid)
            backbone = FakeBackbone()
            load_training_checkpoint(backbone, valid)
            torch.testing.assert_close(backbone.scale.detach(), torch.tensor(2.5))

            invalid = Path(td) / "invalid.pth"
            torch.save({"state_dict": {"not_scale": torch.tensor(1.0)}}, invalid)
            with self.assertRaisesRegex(ValueError, "does not fully initialize"):
                load_training_checkpoint(FakeBackbone(), invalid)


if __name__ == "__main__":
    unittest.main()
