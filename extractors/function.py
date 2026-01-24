import torch
import os, sys
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from contextlib import contextmanager

from kmeans_pytorch import kmeans

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def get_cluster_center(embedding_img: Tensor, num_cluster=5) -> Tensor:
    with suppress_output():
        _, cluster_center = kmeans(
            X=embedding_img,
            num_clusters=num_cluster,
            distance="euclidean",
            device=embedding_img.device,
            iter_limit=50
        )
    return cluster_center

def OT(src_dis: Tensor, tgt_dis: Tensor):
    src_dis_norm = F.normalize(src_dis, dim=1)
    tgt_dis_norm = F.normalize(tgt_dis, dim=1)
    sim = torch.einsum("md,nd->mn", src_dis_norm, tgt_dis_norm).contiguous()
    wdist = 1 - sim
    xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
    yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
    with torch.no_grad():
        KK = torch.exp(-wdist / 0.1)
        T = Sinkhorn(KK, xx, yy)
    if torch.isnan(T).any():
        return None
    sim_op = torch.sum(T * sim, dim=(0, 1))
    loss = torch.sum(sim_op)
    return loss

def Sinkhorn(K: Tensor, u: Tensor, v: Tensor):
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-2
    for _ in range(100):
        r0 = r
        r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
        c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.outer(r, c) * K
    return T

def ComplexLoss(total_loss: list, prev_loss: list):
    if len(prev_loss) == 0:
        for l in total_loss:
            prev_loss.append(l.detach())

    weights = []
    for i in range(len(total_loss)):
        ratio = total_loss[i].item() / (prev_loss[i].item() + 1e-8)
        weights.append(ratio)
    
    T = 1.0
    K = len(weights)
    weights_np = np.array(weights)
    weights_softmax = np.exp(weights_np / T)
    weights_softmax /= np.sum(weights_softmax)
    weights_softmax *= K

    for i in range(len(total_loss)):
        prev_loss[i] = total_loss[i].detach()

    total_loss = sum(
        weights_softmax[i] * total_loss[i]
        for i in range(len(total_loss))
    )
    return total_loss