"""Velum dropout repair in extract_vtd: synthetic-mask tests.

Run from the repository root: python -m unittest discover tests
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import extract_vtd as ev  # noqa: E402

H = W = 40


def blob(cx, cy=20, r=6):
    yy, xx = np.mgrid[:H, :W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def moving_clip(T=11):
    """Velum blob drifting right by 1 px per frame."""
    return np.stack([blob(12 + t) for t in range(T)])


def centroid_x(mask):
    return np.where(mask)[1].mean()


class VelumFillTest(unittest.TestCase):
    def setUp(self):
        self.clip = moving_clip()
        self.ref_area = float(self.clip[0].sum())
        self.ref_mask = blob(17)

    def test_interior_gap_is_interpolated(self):
        bad = self.clip.copy()
        bad[3:8] = False
        bad[5, 20, 15] = True  # a stray fragment still counts as a dropout
        out, filled = ev._fill_velum(bad, self.ref_area, self.ref_mask)
        self.assertEqual(list(np.flatnonzero(filled)), [3, 4, 5, 6, 7])
        for t in (0, 1, 2, 8, 9, 10):
            np.testing.assert_array_equal(out[t], self.clip[t])
        for t in range(3, 8):
            self.assertAlmostEqual(out[t].sum() / self.ref_area, 1.0, delta=0.25)
            self.assertAlmostEqual(centroid_x(out[t]), 12 + t, delta=0.75)

    def test_edge_gaps_copy_nearest_good(self):
        bad = self.clip.copy()
        bad[:2] = False
        bad[9:] = False
        out, filled = ev._fill_velum(bad, self.ref_area, self.ref_mask)
        self.assertEqual(list(np.flatnonzero(filled)), [0, 1, 9, 10])
        for t in (0, 1):
            np.testing.assert_array_equal(out[t], self.clip[2])
        for t in (9, 10):
            np.testing.assert_array_equal(out[t], self.clip[8])

    def test_clip_without_good_frames_uses_reference(self):
        bad = np.zeros_like(self.clip)
        out, filled = ev._fill_velum(bad, self.ref_area, self.ref_mask)
        self.assertTrue(filled.all())
        for t in range(len(out)):
            np.testing.assert_array_equal(out[t], self.ref_mask)

    def test_clean_clip_and_disabled_repair_are_untouched(self):
        out, filled = ev._fill_velum(self.clip, self.ref_area, self.ref_mask)
        self.assertIs(out, self.clip)
        self.assertFalse(filled.any())
        bad = self.clip.copy()
        bad[3:8] = False
        out, filled = ev._fill_velum(bad, 0.0, None)  # what ratio 0 produces
        self.assertIs(out, bad)
        self.assertFalse(filled.any())

    def test_speaker_reference_survives_mostly_bad_frames(self):
        """70% of the speaker's frames are dropouts; reference still reflects the
        real velum, and a ratio of 0 disables the repair."""
        still = np.stack([blob(17)] * 10)
        gone = np.zeros_like(still)
        gone[:, 20, 17] = True
        with tempfile.TemporaryDirectory() as d:
            files = []
            for i, v in enumerate([still, still, still] + [gone] * 7):
                f = Path(d) / f"clip{i}.npz"
                np.savez(f, **{"velum": v, "tongue": v})
                files.append(f)
            ref_area, ref_mask = ev._velum_reference(files)
            self.assertEqual(ref_area, float(blob(17).sum()))
            np.testing.assert_array_equal(ref_mask, blob(17))
            old = ev.VELUM_MIN_AREA_RATIO
            try:
                ev.VELUM_MIN_AREA_RATIO = 0.0
                self.assertEqual(ev._velum_reference(files), (0.0, None))
            finally:
                ev.VELUM_MIN_AREA_RATIO = old


if __name__ == "__main__":
    unittest.main()
