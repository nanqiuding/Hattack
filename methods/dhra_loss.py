"""Full-hierarchy quantum objectives from DHRA-Attack, equations (3)-(18)."""
import torch
import torch.nn.functional as F


def density_state(layer_features):
    # layer_features: [batch, layers, hidden]
    z = F.normalize(layer_features.float(), dim=-1)
    return z @ z.transpose(-1, -2) / z.shape[1]


def psd_sqrt(matrix):
    matrix = (matrix + matrix.transpose(-1, -2)) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    # The zero eigenspace makes an exact square root gradient unstable.
    values = eigenvalues.clamp_min(1e-8).sqrt()
    return (eigenvectors * values.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)


def fidelity(target_state, adversarial_state):
    root = psd_sqrt(target_state)
    middle = root @ adversarial_state @ root
    return psd_sqrt(middle).diagonal(dim1=-2, dim2=-1).sum(-1).square().clamp(max=1.0)


def coherence(state):
    layers = state.shape[-1]
    diagonal = state.diagonal(dim1=-2, dim2=-1).abs().sum(-1)
    return (state.abs().sum(dim=(-2, -1)) - diagonal) / (layers - 1)


def dhra_loss(adversarial_features, target_features, lambda_coh=1.0,
              lambda_dec=1.0, gamma=1.0, eps_c=1e-8):
    """Return scalar loss to minimize; feature lists are one tensor per surrogate."""
    base, coherent, decoherence = [], [], []
    for adversarial, target in zip(adversarial_features, target_features):
        adv_state = density_state(adversarial)
        target_state = density_state(target)
        base.append(1 - F.cosine_similarity(adversarial[:, -1], target[:, -1], dim=-1))
        coherent.append(1 - fidelity(target_state, adv_state))
        target_c = coherence(target_state)
        adv_c = coherence(adv_state)
        decoherence.append((target_c - adv_c).clamp_min(0) / (target_c + eps_c))
    dec = torch.stack(decoherence)
    return (torch.stack(base).mean() + lambda_coh * torch.stack(coherent).mean()
            + lambda_dec * (dec.mean() + gamma * dec.var(dim=0, unbiased=False).mean()))
