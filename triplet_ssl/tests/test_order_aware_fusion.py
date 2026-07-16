import unittest

import torch

from triplet_ssl.models.order_aware_fusion import (
    ALL_PRESENCE_PATTERNS,
    OrderAwareTripletFusionHead,
    is_target_unique_presence,
    select_safe_background_negatives,
    target_unique_fusion_loss,
)


class OrderAwareFusionHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_truth_table_has_only_target_only_positive(self):
        labels = {p: is_target_unique_presence(p) for p in ALL_PRESENCE_PATTERNS}
        self.assertEqual(sum(labels.values()), 1)
        self.assertTrue(labels[(1, 0, 0)])
        self.assertFalse(labels[(0, 1, 1)])

    def test_reference_swap_is_exact_invariant(self):
        head = OrderAwareTripletFusionHead(8, 6, dropout=0.0).eval()
        target, ref1, ref2 = (torch.randn(2, 5, 8) for _ in range(3))
        torch.testing.assert_close(
            head(target, ref1, ref2), head(target, ref2, ref1), atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(
            head.fused_features(target, ref1, ref2),
            head.fused_features(target, ref2, ref1), atol=1e-7, rtol=1e-7)

    def test_target_role_is_not_exchangeable(self):
        head = OrderAwareTripletFusionHead(8, 6, dropout=0.0).eval()
        target, ref1, ref2 = (torch.randn(2, 5, 8) for _ in range(3))
        a = head.fused_features(target, ref1, ref2)
        b = head.fused_features(ref1, target, ref2)
        self.assertFalse(torch.allclose(a, b))

    def test_shape_finite_and_gradient(self):
        head = OrderAwareTripletFusionHead(8, 6, dropout=0.0)
        target, ref1, ref2 = (torch.randn(2, 7, 8, requires_grad=True) for _ in range(3))
        logits = head(target, ref1, ref2)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertTrue(torch.isfinite(logits).all())
        logits.square().mean().backward()
        grads = [p.grad for p in head.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
                            for g in grads))

    def test_loss_uses_only_known_event_sites_and_balances_classes(self):
        logits = torch.tensor([[2.0, -2.0, 10.0]], requires_grad=True)
        positive = torch.tensor([[1.0, 0.0, 0.0]])
        events = torch.tensor([[1.0, 1.0, 0.0]])
        loss, logs = target_unique_fusion_loss(logits, positive, events)
        expected = 0.5 * (torch.nn.functional.softplus(torch.tensor(-2.0))
                          + torch.nn.functional.softplus(torch.tensor(-2.0)))
        torch.testing.assert_close(loss.detach(), expected)
        self.assertEqual(logs["fusion_n_pos"], 1)
        self.assertEqual(logs["fusion_n_neg"], 1)
        loss.backward()
        self.assertEqual(float(logits.grad[0, 2]), 0.0)  # unknown background ignored

    def test_config_and_state_roundtrip(self):
        head = OrderAwareTripletFusionHead(8, 6, dropout=0.0).eval()
        config = head.export_config(patch_size=16, train_tile=128)
        clone = OrderAwareTripletFusionHead.from_config(config).eval()
        clone.load_state_dict(head.state_dict(), strict=True)
        inputs = tuple(torch.randn(2, 5, 8) for _ in range(3))
        torch.testing.assert_close(head(*inputs), clone(*inputs))

    def test_corrupt_truth_table_config_is_rejected(self):
        config = OrderAwareTripletFusionHead(8, 6).export_config()
        config["truth_table"] = "target_missing_011"
        with self.assertRaisesRegex(ValueError, "unsupported fusion truth table"):
            OrderAwareTripletFusionHead.from_config(config)

    def test_safe_background_negatives_select_only_low_residual_non_events(self):
        # Token 0/1 match a ref; token 2 is target-unique; token 3 is an event.
        target = torch.tensor([[[1., 0.], [0., 1.], [1., 0.], [0., 1.]]])
        ref1 = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [0., 1.]]])
        ref2 = torch.tensor([[[0., 1.], [0., 1.], [0., 1.], [1., 0.]]])
        events = torch.tensor([[0., 0., 0., 1.]])
        selected = select_safe_background_negatives(
            target, ref1, ref2, events, fraction=0.5)
        self.assertFalse(bool(selected[0, 2]))
        self.assertFalse(bool(selected[0, 3]))
        self.assertEqual(int(selected.sum()), 2)

    def test_tiny_truth_table_can_be_learned(self):
        # Distinct clean/PSF token fixtures repeated across the eight patterns.
        clean = torch.tensor([1.0, 0.0, 0.0, 0.0])
        psf = torch.tensor([0.0, 1.0, 0.5, -0.5])
        patterns = list(ALL_PRESENCE_PATTERNS)
        z = lambda bit: psf if bit else clean
        target = torch.stack([z(p[0]) for p in patterns])[:, None, :]
        ref1 = torch.stack([z(p[1]) for p in patterns])[:, None, :]
        ref2 = torch.stack([z(p[2]) for p in patterns])[:, None, :]
        labels = torch.tensor([is_target_unique_presence(p) for p in patterns],
                              dtype=torch.float32)[:, None]
        valid = torch.ones_like(labels)

        head = OrderAwareTripletFusionHead(4, 12, dropout=0.0, prior_prob=0.1)
        opt = torch.optim.Adam(head.parameters(), lr=0.03)
        for _ in range(220):
            logits = head(target, ref1, ref2)
            loss, _ = target_unique_fusion_loss(logits, labels, valid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        prob = head(target, ref1, ref2).sigmoid().detach().flatten()
        pos_idx = patterns.index((1, 0, 0))
        self.assertGreater(float(prob[pos_idx]), 0.90)
        self.assertLess(float(torch.cat((prob[:pos_idx], prob[pos_idx + 1:])).max()), 0.10)


if __name__ == "__main__":
    unittest.main()
