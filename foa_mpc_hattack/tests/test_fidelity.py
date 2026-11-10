import unittest
import torch
from foa_mpc_hattack.fidelity import density_state, fidelity_loss, squared_fidelity


class FidelityTests(unittest.TestCase):
    def test_psd_trace_and_self_similarity(self):
        layers = torch.randn(2, 12, 64)
        q = density_state(layers)
        self.assertTrue(torch.allclose(q.diagonal(dim1=-2, dim2=-1).sum(-1), torch.ones(2), atol=1e-5))
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(q).min()), -1e-6)
        self.assertTrue(torch.allclose(squared_fidelity(q, q), torch.ones(2), atol=2e-3))

    def test_pixel_path_has_finite_nonzero_gradient(self):
        adv = torch.randn(1, 12, 32, requires_grad=True)
        target = torch.randn(1, 12, 32)
        loss = fidelity_loss(adv, target)
        grad, = torch.autograd.grad(loss, adv)
        self.assertTrue(bool(torch.isfinite(loss) and torch.isfinite(grad).all()))
        self.assertGreater(float(grad.norm()), 0)

    def test_repeated_layer_eigenvalues_are_finite(self):
        adv = torch.ones(1, 12, 32, requires_grad=True)
        target = torch.randn(1, 12, 32)
        loss = fidelity_loss(adv, target)
        grad, = torch.autograd.grad(loss, adv)
        self.assertTrue(bool(torch.isfinite(loss) and torch.isfinite(grad).all()))


if __name__ == "__main__":
    unittest.main()
