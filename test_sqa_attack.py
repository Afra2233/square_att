import os
import math
import random
import numpy as np
from dataclasses import dataclass, replace
from typing import Literal, Tuple, Dict, List

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.datasets import (
    CIFAR100,
    Food101,
    OxfordIIITPet,
    STL10,
    FGVCAircraft,
)

import clip


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_list(name, ds):
    """
    Try to get class names robustly for different torchvision datasets.
    """
    if hasattr(ds, "classes") and ds.classes is not None:
        return ds.classes

    name = name.lower()

    if name == "stl10":
        return [
            "airplane", "bird", "car", "cat", "deer",
            "dog", "horse", "monkey", "ship", "truck"
        ]

    if name in ("oxfordiiitpet", "pets"):
        if hasattr(ds, "_labels") and hasattr(ds, "_iiit_pet_categories"):
            return ds._iiit_pet_categories
        # fallback
        raise RuntimeError("[FATAL] OxfordIIITPet class names not found.")

    if name == "fgvc_aircraft":
        if hasattr(ds, "_labels") and hasattr(ds, "_image_files"):
            # torchvision versions differ; try classes first
            pass
        if hasattr(ds, "classes") and ds.classes is not None:
            return ds.classes
        raise RuntimeError("[FATAL] FGVCAircraft class names not found.")

    raise RuntimeError(f"[FATAL] Dataset {name} has no .classes; please provide class names.")


def make_subset_loader(
    ds,
    batch_size: int,
    num_workers: int,
    subset_size: int,
    seed: int,
) -> DataLoader:
    n = len(ds)
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    idx = perm[: min(subset_size, n)]
    sampler = SubsetRandomSampler(idx)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


# =========================================================
# Build CLIP Text Features
# =========================================================
@torch.inference_mode()
def build_text_features(class_names, clip_model, device, dataset_name: str):
    """
    Prompt ensemble for zero-shot CLIP.
    """
    dname = dataset_name.lower()

    if dname == "food101":
        templates = [
            "a photo of {}, a type of food",
            "a dish of {}",
            "a photo of a plate of {}",
            "a close-up photo of {}",
            "a photo of cooked {}",
        ]
    elif dname in ("fgvc_aircraft",):
        templates = [
            "a photo of a {} aircraft",
            "a photo of the {} aircraft",
            "a close-up photo of a {} aircraft",
            "a photo of an airplane of type {}",
        ]
    elif dname in ("oxfordiiitpet", "pets"):
        templates = [
            "a photo of a {} pet",
            "a photo of a {}",
            "a close-up photo of a {}",
            "a photo of the pet {}",
        ]
    else:
        templates = [
            "a photo of a {}",
            "a blurry photo of a {}",
            "a close-up photo of a {}",
            "a photo of the {}",
            "a good photo of a {}",
            "a cropped photo of a {}",
        ]

    all_text_features = []
    for c in class_names:
        name = c.replace("_", " ")
        prompts = [t.format(name) for t in templates]
        tokens = clip.tokenize(prompts).to(device)
        text_feats = clip_model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_feat = text_feats.mean(dim=0)
        text_feat = text_feat / text_feat.norm()
        all_text_features.append(text_feat)

    text_features = torch.stack(all_text_features, dim=0)
    return text_features


# =========================================================
# CLIP Zero-shot Classifier
# =========================================================
class CLIPZeroShot(nn.Module):
    def __init__(self, clip_model, text_features, device):
        super().__init__()
        self.clip_model = clip_model
        self.text_features = text_features.to(device)

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean.to(device))
        self.register_buffer("std", std.to(device))

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def encode_image_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        f = self.clip_model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return f

    def logits_from_features(self, f: torch.Tensor) -> torch.Tensor:
        return 100.0 * (f @ self.text_features.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encode_image_features(x)
        return self.logits_from_features(f)


# =========================================================
# Defense types
# =========================================================
PreprocType = Literal[
    "none",
    "multiview_vote",
]

ViewType = Literal[
    "identity",
    "horizontal_flip",
    "resize_pad_96",
    "center_crop_96",
]


# =========================================================
# Deterministic multi-view transforms
# =========================================================
@torch.inference_mode()
def apply_view_batch(x: torch.Tensor, view_t: ViewType) -> torch.Tensor:
    """
    Deterministic views.
    """
    if view_t == "identity":
        return x

    B, C, H, W = x.shape
    out = []

    for i in range(B):
        xi = x[i]

        if view_t == "horizontal_flip":
            xo = TF.hflip(xi)

        elif view_t == "resize_pad_96":
            new_h = max(1, int(round(H * 0.96)))
            new_w = max(1, int(round(W * 0.96)))
            resized = TF.resize(
                xi,
                size=[new_h, new_w],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            pad_top = (H - new_h) // 2
            pad_bottom = H - new_h - pad_top
            pad_left = (W - new_w) // 2
            pad_right = W - new_w - pad_left
            fill_value = float(xi.mean().item())
            xo = TF.pad(
                resized,
                [pad_left, pad_top, pad_right, pad_bottom],
                fill=fill_value,
            )

        elif view_t == "center_crop_96":
            crop_h = max(1, int(round(H * 0.96)))
            crop_w = max(1, int(round(W * 0.96)))
            top = (H - crop_h) // 2
            left = (W - crop_w) // 2
            cropped = xi[:, top:top + crop_h, left:left + crop_w]
            xo = TF.resize(
                cropped,
                size=[H, W],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

        else:
            raise ValueError(f"Unknown view type: {view_t}")

        out.append(xo.clamp(0.0, 1.0))

    return torch.stack(out, dim=0)


# =========================================================
# Unified defended forward
# =========================================================
@torch.inference_mode()
def defended_predict(
    model: CLIPZeroShot,
    x: torch.Tensor,
    preproc_mode: PreprocType,
    view_list: List[ViewType],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        pred:   (B,)
        logits: (B,C)
    """
    if preproc_mode == "none":
        logits = model(x)

    elif preproc_mode == "multiview_vote":
        all_logits = []
        for vt in view_list:
            xv = apply_view_batch(x, vt)
            lv = model(xv)
            all_logits.append(lv)
        logits = torch.stack(all_logits, dim=0).mean(dim=0)

    else:
        raise ValueError(f"Unknown preproc_mode: {preproc_mode}")

    pred = logits.argmax(dim=1)
    return pred, logits


# =========================================================
# Metrics
# =========================================================
@torch.inference_mode()
def top1_confidence(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    return probs.max(dim=1).values


def margin_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    tmp = logits.clone()
    tmp.scatter_(1, y.view(-1, 1), -1e9)
    other = tmp.max(dim=1).values
    return true - other


@torch.no_grad()
def defended_forward_for_attacker(
    model: CLIPZeroShot,
    x: torch.Tensor,
    preproc_mode: PreprocType,
    view_list: List[ViewType],
    eot_M: int = 1,
) -> torch.Tensor:
    logits_sum = None
    M = max(1, eot_M)

    for _ in range(M):
        _, logits = defended_predict(
            model=model,
            x=x,
            preproc_mode=preproc_mode,
            view_list=view_list,
        )
        logits_sum = logits if logits_sum is None else (logits_sum + logits)

    return logits_sum / float(M)


@torch.no_grad()
def defended_margin_loss(
    model: CLIPZeroShot,
    x: torch.Tensor,
    y: torch.Tensor,
    preproc_mode: PreprocType,
    view_list: List[ViewType],
    eot_M: int = 1,
) -> torch.Tensor:
    logits = defended_forward_for_attacker(
        model=model,
        x=x,
        preproc_mode=preproc_mode,
        view_list=view_list,
        eot_M=eot_M,
    )
    return margin_loss(logits, y)


@torch.no_grad()
def linf_project(x_adv: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
    x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv


# =========================================================
# Square Attack
# =========================================================
@dataclass
class SquareAttackConfig:
    eps: float = 8 / 255
    n_iters: int = 100
    eot_M: int = 1
    min_square: int = 1
    max_square: int = 64
    seed: int = 0


def square_size_schedule(i: int, n_iters: int, H: int, W: int, min_s: int, max_s: int) -> int:
    frac = 1.0 - (i / max(n_iters - 1, 1))
    s = int(round(min_s + (max_s - min_s) * (frac ** 2)))
    s = max(min_s, min(s, min(H, W)))
    return s


@torch.no_grad()
def square_attack(
    model: CLIPZeroShot,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: SquareAttackConfig,
    preproc_mode: PreprocType,
    view_list: List[ViewType],
) -> torch.Tensor:
    set_seed(cfg.seed)

    B, C, H, W = x.shape
    max_s = min(cfg.max_square, H, W)

    x_adv = x + cfg.eps * torch.sign(torch.randn_like(x))
    x_adv = linf_project(x_adv, x, cfg.eps)

    best = defended_margin_loss(
        model=model,
        x=x_adv,
        y=y,
        preproc_mode=preproc_mode,
        view_list=view_list,
        eot_M=cfg.eot_M,
    )

    for i in range(cfg.n_iters):
        s = square_size_schedule(i, cfg.n_iters, H, W, cfg.min_square, max_s)
        x_new = x_adv.clone()

        for b in range(B):
            top = random.randint(0, H - s) if H > s else 0
            left = random.randint(0, W - s) if W > s else 0
            patch_sign = 1.0 if random.random() < 0.5 else -1.0
            patch = (x[b, :, top:top + s, left:left + s] + patch_sign * cfg.eps).clamp(0.0, 1.0)
            x_new[b, :, top:top + s, left:left + s] = patch

        x_new = linf_project(x_new, x, cfg.eps)

        loss_new = defended_margin_loss(
            model=model,
            x=x_new,
            y=y,
            preproc_mode=preproc_mode,
            view_list=view_list,
            eot_M=cfg.eot_M,
        )

        improved = loss_new < best
        if improved.any():
            x_adv[improved] = x_new[improved]
            best[improved] = loss_new[improved]

    return x_adv


# =========================================================
# SimBA
# =========================================================
@dataclass
class SimBAConfig:
    eps: float = 8 / 255
    n_iters: int = 2000
    step_size: float = 2 / 255
    eot_M: int = 1
    seed: int = 0


@torch.no_grad()
def simba_attack(
    model: CLIPZeroShot,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: SimBAConfig,
    preproc_mode: PreprocType,
    view_list: List[ViewType],
) -> torch.Tensor:
    """
    Untargeted SimBA in pixel basis.
    """
    set_seed(cfg.seed)

    B, C, H, W = x.shape
    D = C * H * W
    x_adv = x.clone()

    best_loss = defended_margin_loss(
        model=model,
        x=x_adv,
        y=y,
        preproc_mode=preproc_mode,
        view_list=view_list,
        eot_M=cfg.eot_M,
    )

    for b in range(B):
        perm = torch.randperm(D, device=x.device)

        xb = x_adv[b:b+1].clone()
        x0 = x[b:b+1]
        yb = y[b:b+1]
        current = best_loss[b:b+1].clone()

        for i in range(min(cfg.n_iters, D)):
            idx = perm[i].item()

            delta = torch.zeros_like(xb).view(1, -1)
            delta[0, idx] = cfg.step_size
            delta = delta.view_as(xb)

            x_try_pos = linf_project(xb + delta, x0, cfg.eps)
            loss_pos = defended_margin_loss(
                model=model,
                x=x_try_pos,
                y=yb,
                preproc_mode=preproc_mode,
                view_list=view_list,
                eot_M=cfg.eot_M,
            )

            if loss_pos.item() < current.item():
                xb = x_try_pos
                current = loss_pos
                continue

            x_try_neg = linf_project(xb - delta, x0, cfg.eps)
            loss_neg = defended_margin_loss(
                model=model,
                x=x_try_neg,
                y=yb,
                preproc_mode=preproc_mode,
                view_list=view_list,
                eot_M=cfg.eot_M,
            )

            if loss_neg.item() < current.item():
                xb = x_try_neg
                current = loss_neg

        x_adv[b:b+1] = xb

    return x_adv


# =========================================================
# NES
# =========================================================
@dataclass
class NESConfig:
    eps: float = 8 / 255
    n_iters: int = 200
    step_size: float = 2 / 255
    sigma: float = 1e-3
    samples_per_iter: int = 20
    eot_M: int = 1
    seed: int = 0


@torch.no_grad()
def nes_attack(
    model: CLIPZeroShot,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: NESConfig,
    preproc_mode: PreprocType,
    view_list: List[ViewType],
) -> torch.Tensor:
    """
    Untargeted NES gradient estimate + projected sign updates.
    """
    set_seed(cfg.seed)
    assert cfg.samples_per_iter % 2 == 0, "NES samples_per_iter should be even."

    x_adv = x.clone()

    for _ in range(cfg.n_iters):
        grad_est = torch.zeros_like(x_adv)
        half = cfg.samples_per_iter // 2

        for _ in range(half):
            u = torch.randn_like(x_adv)

            x_pos = linf_project(x_adv + cfg.sigma * u, x, cfg.eps)
            x_neg = linf_project(x_adv - cfg.sigma * u, x, cfg.eps)

            loss_pos = defended_margin_loss(
                model=model,
                x=x_pos,
                y=y,
                preproc_mode=preproc_mode,
                view_list=view_list,
                eot_M=cfg.eot_M,
            )

            loss_neg = defended_margin_loss(
                model=model,
                x=x_neg,
                y=y,
                preproc_mode=preproc_mode,
                view_list=view_list,
                eot_M=cfg.eot_M,
            )

            coeff = ((loss_pos - loss_neg) / (2.0 * cfg.sigma)).view(-1, 1, 1, 1)
            grad_est += coeff * u

        grad_est /= float(half)
        x_adv = x_adv - cfg.step_size * torch.sign(grad_est)
        x_adv = linf_project(x_adv, x, cfg.eps)

    return x_adv


# =========================================================
# Bandits
# =========================================================
@dataclass
class BanditsConfig:
    eps: float = 8 / 255
    n_iters: int = 200
    step_size: float = 2 / 255
    fd_eta: float = 0.01
    prior_lr: float = 0.1
    prior_std: float = 1.0
    eot_M: int = 1
    seed: int = 0


@torch.no_grad()
def bandits_attack(
    model: CLIPZeroShot,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: BanditsConfig,
    preproc_mode: PreprocType,
    view_list: List[ViewType],
) -> torch.Tensor:
    """
    Practical prior-guided Bandits-style score-based attack.
    """
    set_seed(cfg.seed)

    x_adv = x.clone()
    prior = torch.zeros_like(x_adv)

    for _ in range(cfg.n_iters):
        noise = torch.randn_like(prior) * cfg.prior_std
        q1 = prior + noise
        q2 = prior - noise

        q1 = q1 / q1.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
        q2 = q2 / q2.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)

        x1 = linf_project(x_adv + cfg.fd_eta * q1, x, cfg.eps)
        x2 = linf_project(x_adv + cfg.fd_eta * q2, x, cfg.eps)

        loss1 = defended_margin_loss(
            model=model,
            x=x1,
            y=y,
            preproc_mode=preproc_mode,
            view_list=view_list,
            eot_M=cfg.eot_M,
        )

        loss2 = defended_margin_loss(
            model=model,
            x=x2,
            y=y,
            preproc_mode=preproc_mode,
            view_list=view_list,
            eot_M=cfg.eot_M,
        )

        est_deriv = ((loss1 - loss2) / (2.0 * cfg.fd_eta)).view(-1, 1, 1, 1)
        grad_est = est_deriv * noise

        prior = prior + cfg.prior_lr * grad_est
        step_dir = torch.sign(prior)

        x_adv = x_adv - cfg.step_size * step_dir
        x_adv = linf_project(x_adv, x, cfg.eps)

    return x_adv


# =========================================================
# Attack dispatcher
# =========================================================
def run_attack(
    attack_name: str,
    model: CLIPZeroShot,
    images: torch.Tensor,
    labels: torch.Tensor,
    attack_cfg,
    preproc_mode: PreprocType,
    view_list: List[ViewType],
):
    if attack_name.lower() == "square":
        return square_attack(
            model=model,
            x=images,
            y=labels,
            cfg=attack_cfg,
            preproc_mode=preproc_mode,
            view_list=view_list,
        )

    elif attack_name.lower() == "simba":
        return simba_attack(
            model=model,
            x=images,
            y=labels,
            cfg=attack_cfg,
            preproc_mode=preproc_mode,
            view_list=view_list,
        )

    elif attack_name.lower() == "nes":
        return nes_attack(
            model=model,
            x=images,
            y=labels,
            cfg=attack_cfg,
            preproc_mode=preproc_mode,
            view_list=view_list,
        )

    elif attack_name.lower() == "bandits":
        return bandits_attack(
            model=model,
            x=images,
            y=labels,
            cfg=attack_cfg,
            preproc_mode=preproc_mode,
            view_list=view_list,
        )

    else:
        raise ValueError(f"Unknown attack_name: {attack_name}")


# =========================================================
# Evaluation
# =========================================================
@torch.inference_mode()
def eval_attack_family(
    name: str,
    ds,
    clip_model,
    device: str,
    text_features: torch.Tensor,
    attack_name: str,
    attack_cfg,
    batch_size: int = 16,
    num_workers: int = 4,
    subset_size: int = 200,
    subset_seed: int = 0,
):
    print(f"\n===== {name} | attack={attack_name} | subset={subset_size} =====")

    loader = make_subset_loader(
        ds=ds,
        batch_size=batch_size,
        num_workers=num_workers,
        subset_size=subset_size,
        seed=subset_seed,
    )

    model = CLIPZeroShot(clip_model, text_features, device).to(device).eval().float()

    fixed_views: List[ViewType] = [
        "identity",
        "horizontal_flip",
        "resize_pad_96",
        "center_crop_96",
    ]

    experiment_configs = [
        {"name": "single_baseline", "preproc_mode": "none"},
        {"name": "multiview_vote", "preproc_mode": "multiview_vote"},
    ]

    stats: Dict[str, Dict[str, float]] = {}
    total = 0

    for cfg in experiment_configs:
        stats[cfg["name"]] = {
            "clean_correct": 0.0,
            "robust_correct": 0.0,
            "clean_ok_count": 0.0,
            "asr_num": 0.0,
        }

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"{name}-{attack_name}", ncols=120)):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        total += labels.numel()

        for cfg_exp in experiment_configs:
            exp_name = cfg_exp["name"]
            preproc_mode = cfg_exp["preproc_mode"]

            pred_clean, _ = defended_predict(
                model=model,
                x=images,
                preproc_mode=preproc_mode,
                view_list=fixed_views,
            )

            clean_ok = (pred_clean == labels)
            stats[exp_name]["clean_correct"] += clean_ok.sum().item()
            stats[exp_name]["clean_ok_count"] += clean_ok.sum().item()

            if hasattr(attack_cfg, "seed"):
                attack_cfg_batch = replace(attack_cfg, seed=int(attack_cfg.seed + batch_idx))
            else:
                attack_cfg_batch = attack_cfg

            x_adv = run_attack(
                attack_name=attack_name,
                model=model,
                images=images,
                labels=labels,
                attack_cfg=attack_cfg_batch,
                preproc_mode=preproc_mode,
                view_list=fixed_views,
            )

            pred_adv, _ = defended_predict(
                model=model,
                x=x_adv,
                preproc_mode=preproc_mode,
                view_list=fixed_views,
            )

            adv_ok = (pred_adv == labels)
            stats[exp_name]["robust_correct"] += adv_ok.sum().item()
            stats[exp_name]["asr_num"] += ((~adv_ok) & clean_ok).sum().item()

        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nRESULT: {name} | attack={attack_name}")
    print(f"Samples: {total}")

    for exp_name, val in stats.items():
        clean_acc = val["clean_correct"] / max(total, 1)
        robust_acc = val["robust_correct"] / max(total, 1)
        asr = val["asr_num"] / (val["clean_ok_count"] + 1e-12)

        print(f"\n--- {exp_name} ---")
        print(f"Clean Acc:          {clean_acc:.4f}")
        print(f"Robust Acc:         {robust_acc:.4f}")
        print(f"ASR(clean-correct): {asr:.4f}")

    print("=" * 110)


# =========================================================
# Main
# =========================================================
def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] Device: {device}")

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model = clip_model.eval().float()

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
    ])

    DATA_ROOT = "data"
    os.makedirs(DATA_ROOT, exist_ok=True)

    datasets = {
        "cifar100": CIFAR100(
            root=f"{DATA_ROOT}/cifar100",
            train=False,
            download=True,
            transform=transform,
        ),
        "food101": Food101(
            root=f"{DATA_ROOT}/food101",
            split="test",
            download=True,
            transform=transform,
        ),
        "oxfordiiitpet": OxfordIIITPet(
            root=f"{DATA_ROOT}/oxfordiiitpet",
            split="test",
            download=True,
            transform=transform,
        ),
        "stl10": STL10(
            root=f"{DATA_ROOT}/stl10",
            split="test",
            download=True,
            transform=transform,
        ),
        "fgvc_aircraft": FGVCAircraft(
            root=f"{DATA_ROOT}/fgvc_aircraft",
            split="test",
            download=True,
            transform=transform,
        ),
    }

    square_cfg = SquareAttackConfig(
        eps=8 / 255,
        n_iters=200,
        eot_M=1,
        min_square=1,
        max_square=64,
        seed=0,
    )

    simba_cfg = SimBAConfig(
        eps=8 / 255,
        n_iters=2000,
        step_size=2 / 255,
        eot_M=1,
        seed=0,
    )

    nes_cfg = NESConfig(
        eps=8 / 255,
        n_iters=200,
        step_size=2 / 255,
        sigma=1e-3,
        samples_per_iter=20,
        eot_M=1,
        seed=0,
    )

    bandits_cfg = BanditsConfig(
        eps=8 / 255,
        n_iters=200,
        step_size=2 / 255,
        fd_eta=0.01,
        prior_lr=0.1,
        prior_std=1.0,
        eot_M=1,
        seed=0,
    )

    # 建议先小一点，确认代码能跑
    batch_size = 8
    num_workers = 4
    subset_size = 100
    subset_seed = 0

    attack_list = [
        ("square", square_cfg),
        ("simba", simba_cfg),
        ("nes", nes_cfg),
        ("bandits", bandits_cfg),
    ]

    for name, ds in datasets.items():
        print(f"\nPreparing: {name}")
        class_names = get_class_list(name, ds)
        text_features = build_text_features(class_names, clip_model, device, dataset_name=name)

        for attack_name, attack_cfg in attack_list:
            eval_attack_family(
                name=name,
                ds=ds,
                clip_model=clip_model,
                device=device,
                text_features=text_features,
                attack_name=attack_name,
                attack_cfg=attack_cfg,
                batch_size=batch_size,
                num_workers=num_workers,
                subset_size=subset_size,
                subset_seed=subset_seed,
            )


if __name__ == "__main__":
    main()