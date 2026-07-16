import unittest

import numpy as np

from triplet_ssl.data.synth_defect import (
    NUISANCE_COMBOS, TripletSyntheticPSF, presence_requires_pull_exemption)


class SyntheticSupervisionTest(unittest.TestCase):
    def test_missing_target_pattern_is_a_nuisance(self):
        self.assertIn((0, 1, 1), NUISANCE_COMBOS)
        self.assertNotIn((1, 0, 0), NUISANCE_COMBOS)

    def test_event_mask_contains_positive_mask(self):
        img = np.full((128, 128, 3), 128, np.uint8)
        synth = TripletSyntheticPSF(n_events=(3, 3), defect_prob=1.0, seed=9)
        out = synth(img, img, img, return_event_mask=True)
        self.assertEqual(len(out), 5)
        positive, events = out[3], out[4]
        self.assertIsNotNone(positive)
        self.assertGreater(int(positive.sum()), 0)
        self.assertGreater(int(events.sum()), int(positive.sum()))
        self.assertTrue(np.all((positive == 0) | (events > 0)))

    def test_directionally_unmatched_patterns_are_exempt_from_pull(self):
        self.assertTrue(presence_requires_pull_exemption((1, 0, 0)))
        self.assertTrue(presence_requires_pull_exemption((0, 1, 1)))
        for combo in ((0, 0, 0), (0, 1, 0), (0, 0, 1),
                      (1, 1, 0), (1, 0, 1), (1, 1, 1)):
            self.assertFalse(presence_requires_pull_exemption(combo))

    def test_full_supervision_masks_are_nested(self):
        img = np.full((128, 128, 3), 128, np.uint8)
        synth = TripletSyntheticPSF(n_events=(4, 4), defect_prob=1.0, seed=11)
        out = synth(img, img, img, return_supervision=True)
        self.assertEqual(len(out), 6)
        positive, events, unmatched = out[3:]
        self.assertTrue(np.all((positive == 0) | (unmatched > 0)))
        self.assertTrue(np.all((unmatched == 0) | (events > 0)))

    def test_legacy_four_value_api_remains_available(self):
        img = np.full((64, 64, 3), 128, np.uint8)
        synth = TripletSyntheticPSF(n_events=(1, 1), defect_prob=0.0, seed=1)
        self.assertEqual(len(synth(img, img, img)), 4)


if __name__ == "__main__":
    unittest.main()
