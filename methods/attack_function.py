import torch
from tqdm import tqdm

from config import Optim
from extractors import EnsembleExtractor, EnsembleLoss

def fgsm_attack(
    optim: Optim,
    ensemble_extractor: EnsembleExtractor,
    ensemble_loss: EnsembleLoss,
    source_crop: torch.nn.Module,
    target_crop: torch.nn.Module,
    src_img: torch.Tensor,
    tgt_img: torch.Tensor,
):
    delta = torch.zeros_like(src_img, requires_grad=True)

    for epoch in tqdm(range(optim.steps), desc="Attack progress"):

        ensemble_loss.pin_layer = False if epoch % 30 == 0 else True

        ensemble_loss.set_ground_truth(target_crop(tgt_img))

        adv_image = source_crop(src_img + delta)
        features = ensemble_extractor(adv_image)
        loss = ensemble_loss(*features)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
        delta.data = torch.clamp(
            delta + optim.alpha * torch.sign(grad),
            min = -optim.epsilon,
            max = optim.epsilon,
        )

    adv_image = src_img + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image

def mifgsm_attack(
    optim: Optim,
    ensemble_extractor: EnsembleExtractor,
    ensemble_loss: EnsembleLoss,
    source_crop: torch.nn.Module,
    target_crop: torch.nn.Module,
    src_img: torch.Tensor,
    tgt_img: torch.Tensor,
):
    delta = torch.zeros_like(src_img, requires_grad=True)
    momentum = torch.zeros_like(src_img, requires_grad=False)

    for _ in tqdm(range(optim.steps), desc="Attack progress"):
        ensemble_loss.set_ground_truth(target_crop(tgt_img))

        adv_image = source_crop(src_img + delta)
        features = ensemble_extractor(adv_image)
        loss = ensemble_loss(*features)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        momentum = momentum * 0.9 + grad
        delta.data = torch.clamp(
            delta + optim.alpha * torch.sign(momentum),
            min = -optim.epsilon,
            max = optim.epsilon,
        )

    adv_image = src_img + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image

def pgd_attack(
    optim: Optim,
    ensemble_extractor: EnsembleExtractor,
    ensemble_loss: EnsembleLoss,
    source_crop: torch.nn.Module,
    target_crop: torch.nn.Module,
    src_img: torch.Tensor,
    tgt_img: torch.Tensor,
):
    delta = torch.zeros_like(src_img, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=optim.alpha)

    for _ in tqdm(range(optim.steps), desc="Attack progress"):
        ensemble_loss.set_ground_truth(target_crop(tgt_img))

        adv_image = source_crop(src_img + delta)
        features = ensemble_extractor(adv_image)
        loss = -ensemble_loss(*features)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        delta.data = torch.clamp(
            delta,
            min = -optim.epsilon,
            max = optim.epsilon,
        )

    adv_image = src_img + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image

ATTACK_FUNC_MAP = {
    "fgsm": fgsm_attack,
    "mifgsm": mifgsm_attack,
    "pgd": pgd_attack
}