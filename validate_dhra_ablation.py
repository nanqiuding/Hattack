"""Check that the ablation full setting is algebraically identical to DHRA loss."""
import torch
from config import Config, TomlLoader
from dhra_ablation import components, total
from methods.dhra_loss import dhra_loss

torch.manual_seed(7)
a = [torch.randn(1, 4, 8, requires_grad=True) for _ in range(3)]
b = [torch.randn(1, 4, 8) for _ in range(3)]
cfg = TomlLoader('config.toml', Config)
reference = dhra_loss(a, b, cfg.optim.lambda_coh, cfg.optim.lambda_dec, cfg.optim.gamma)
ablation = total(components(a, b, 'quantum'), 'quantum', True, True, cfg)
assert torch.allclose(reference, ablation, atol=1e-6), (reference, ablation)
g1 = torch.autograd.grad(reference, a, retain_graph=True)
g2 = torch.autograd.grad(ablation, a)
assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(g1, g2))
print('FULL_LOSS_AND_GRADIENT_MATCH', float(reference))
