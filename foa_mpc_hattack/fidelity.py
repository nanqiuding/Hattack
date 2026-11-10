"""Stable squared Uhlmann fidelity for per-model layer-CLS Gram states."""
import torch
import torch.nn.functional as F


def density_state(layer_cls: torch.Tensor, eps=1e-8) -> torch.Tensor:
    if layer_cls.ndim != 3:
        raise ValueError("layer_cls must be [B,L,D]")
    z = F.normalize(layer_cls.float(), dim=-1, eps=eps)
    q = z @ z.transpose(-1, -2) / z.shape[1]
    q = (q + q.transpose(-1, -2)) * 0.5
    eye = torch.eye(q.shape[-1], device=q.device, dtype=q.dtype)
    q = q + eps * eye
    return q / q.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]


def target_sqrt(q_target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        eig, vec = torch.linalg.eigh((q_target + q_target.transpose(-1, -2)) * 0.5)
        return (vec * eig.clamp_min(0).sqrt().unsqueeze(-2)) @ vec.transpose(-1, -2)


def squared_fidelity(q_adv: torch.Tensor, q_target: torch.Tensor, eps=1e-8) -> torch.Tensor:
    root = target_sqrt(q_target)
    middle = root @ q_adv @ root
    middle = (middle + middle.transpose(-1, -2)) * 0.5
    eig = torch.linalg.eigvalsh(middle.float())
    return eig.clamp_min(eps).sqrt().sum(-1).square().clamp(0, 1)


def fidelity_loss(adv_layers: torch.Tensor, target_layers: torch.Tensor) -> torch.Tensor:
    return (1 - squared_fidelity(density_state(adv_layers),
                                 density_state(target_layers).detach())).mean()
