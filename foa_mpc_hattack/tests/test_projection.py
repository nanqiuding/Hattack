import unittest
import torch


class ProjectionTests(unittest.TestCase):
    def test_linf_and_pixel_bounds(self):
        src = torch.rand(1, 3, 16, 16) * 255
        delta = torch.randn_like(src) * 50
        delta = delta.clamp(-16, 16)
        delta = (src + delta).clamp(0, 255) - src
        self.assertLessEqual(float(delta.abs().max()), 16.00001)
        self.assertGreaterEqual(float((src + delta).min()), 0)
        self.assertLessEqual(float((src + delta).max()), 255)


if __name__ == "__main__":
    unittest.main()
