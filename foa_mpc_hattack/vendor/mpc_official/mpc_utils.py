"""Two MPC utility functions, copied verbatim without unrelated Hydra imports."""
import os
import torch
import torch.nn.functional as F


def info_nce_loss(feat_adv, feat_tgt, feat_src, temperature=0.2, mode='img', omega=2.0):
    feat_adv = F.normalize(feat_adv, dim=-1)
    feat_tgt = F.normalize(feat_tgt, dim=-1)
    feat_src = F.normalize(feat_src, dim=-1)
    if mode == 'text':
        sim_pos = torch.sum(feat_adv * feat_tgt, dim=-1).mean(dim=1) * omega / temperature
        sim_neg = torch.sum(feat_adv * feat_src, dim=-1).mean(dim=1) / temperature
    else:
        sim_pos = torch.sum(feat_adv * feat_tgt, dim=-1) * omega / temperature
        sim_neg = torch.sum(feat_adv * feat_src, dim=-1) / temperature
    logits = torch.stack([sim_pos, sim_neg], dim=1)
    labels = torch.zeros(feat_adv.size(0), dtype=torch.long, device=feat_adv.device)
    loss = F.cross_entropy(logits, labels)
    return - loss


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
