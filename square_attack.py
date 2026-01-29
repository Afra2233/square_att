import os
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Tuple, List

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.datasets import CIFAR10, CIFAR100, Food101, OxfordIIITPet, STL10, FGVCAircraft

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
    # torchvision datasets differ; keep your original logic but be defensive
    if hasattr(ds, "classes"):
        return ds.classes
    raise RuntimeError(f"[FATAL] Dataset {name} has no .classes; please provide class names.")


# =========================================================
# Build CLIP Text Features
# =========================================================
@torch.no_grad()
def build_text_features(class_names, clip_model, device):
    prompts = [f"a photo of a {c.replace('_', ' ')}" for c in class_names]
    tokens = clip.tokenize(prompts).to(device)
    text_feats = clip_model.encode_text(tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats


# =========================================================
# CLIP Zero-shot Classifier (deterministic given x)
# x is in [0,1] and shape (B,3,224,224)
# =========================================================
class CLIPZeroShot(nn.Module):
    def __init__(self, clip_model, text_features, device):
        super().__init__()
        self.clip_model = clip_model
        self.text_features = text_features.to(device)

        # CLIP normalize buffer (expects input in [0,1])
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean.to(device))
        self.register_buffer("std", std.to(device))

    def forward(self, x):
        x = (x - self.mean) / self.std
        f = self.clip_model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        logits = 100.0 * (f @ self.text_features.T)
        return logits


# =========================================================
# Randomized Transform Defense (inference-time)
# Matches paper idea: random crop-resize (80%) or random rotation (±10°)
# =========================================================
TransformType = Literal["none", "crop_resize_80", "rotation_10"]


@torch.no_grad()
def apply_random_transform_batch(
    x: torch.Tensor,
    t: TransformType,
) -> torch.Tensor:
    """
    Apply one random transform instance per image in batch.
    x: (B,3,H,W) in [0,1]
    """
    if t == "none":
        return x

    B, C, H, W = x.shape
    out = []

    for i in range(B):
        xi = x[i]

        if t == "crop_resize_80":
            # paper: crop to 80% then resize back, bilinear
            # Use deterministic scale=0.8 exactly (not a range)
            scale = 0.8
            crop_h = int(round(H * scale))
            crop_w = int(round(W * scale))
            if crop_h < 1: crop_h = 1
            if crop_w < 1: crop_w = 1

            top = random.randint(0, H - crop_h) if H > crop_h else 0
            left = random.randint(0, W - crop_w) if W > crop_w else 0

            cropped = TF.crop(xi, top, left, crop_h, crop_w)
            resized = TF.resize(
                cropped,
                size=[H, W],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True
            )
            out.append(resized)

        elif t == "rotation_10":
            angle = random.uniform(-10.0, 10.0)
            rotated = TF.rotate(
                xi,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=0.0
            )
            out.append(rotated)

        else:
            raise ValueError(f"Unknown transform: {t}")

    return torch.stack(out, dim=0)


@torch.no_grad()
def predict_with_voting(
    model: nn.Module,
    x: torch.Tensor,
    defense_t: TransformType,
    vote_K: int,
) -> torch.Tensor:
    """
    Paper-style final prediction: aggregate multiple forward passes using voting.
    Returns predicted labels (B,).
    """
    B = x.size(0)
    votes = torch.zeros((B, model(x[:1]).size(1)), device=x.device, dtype=torch.int32)

    for _ in range(vote_K):
        xt = apply_random_transform_batch(x, defense_t)
        logits = model(xt)
        pred = logits.argmax(dim=1)
        votes.scatter_add_(1, pred.view(-1, 1), torch.ones((B, 1), device=x.device, dtype=torch.int32))

    # majority vote
    return votes.argmax(dim=1)


# =========================================================
# Loss for attack (margin loss like paper Eq.(1))
# f(x) = logit_true - max_{j!=y} logit_j  (minimize this)
# =========================================================
def margin_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    B, K = logits.shape
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    tmp = logits.clone()
    tmp.scatter_(1, y.view(-1, 1), -1e9)
    other = tmp.max(dim=1).values
    return true - other  # want < 0


@torch.no_grad()
def eot_logits(
    model: nn.Module,
    x: torch.Tensor,
    t: TransformType,
    eot_M: int,
) -> torch.Tensor:
    """
    Attacker EOT: average logits over M random transform instances
    (paper 4.2.1: average predictions before calculating directions).
    """
    if eot_M <= 1 or t == "none":
        return model(x)

    acc = None
    for _ in range(eot_M):
        xt = apply_random_transform_batch(x, t)
        logits = model(xt)
        acc = logits if acc is None else (acc + logits)
    return acc / float(eot_M)


# =========================================================
# Confident Square Attack (C-SQA) with EOT (paper Algorithm 1)
# - runs full N iterations (no early stop)
# - at each iteration: propose square patch update; accept if EOT margin loss decreases
# =========================================================
@dataclass
class SquareAttackConfig:
    eps: float = 8/255
    n_iters: int = 2000            # paper says 10,000 default; you likely lower for 1-day / 1 GPU
    eot_M: int = 10                # EOT-10 or EOT-50
    defense_transform_for_attacker: TransformType = "rotation_10"  # attacker sees same randomness used by defense
    p_init: float = 0.8            # initial fraction of pixels perturbed in first phase (common square-attack knob)
    min_square: int = 1
    max_square: int = 32           # will be clamped to image size
    seed: int = 0


def square_size_schedule(i: int, n_iters: int, H: int, W: int, min_s: int, max_s: int) -> int:
    """
    Simple schedule: start big then shrink.
    You can replace with the exact schedule from Andriushchenko et al. if you want.
    """
    # exponential decay-ish
    frac = 1.0 - (i / max(n_iters - 1, 1))
    s = int(round(min_s + (max_s - min_s) * (frac ** 2)))
    s = max(min_s, min(s, min(H, W)))
    return s


@torch.no_grad()
def confident_square_attack_eot(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: SquareAttackConfig,
) -> torch.Tensor:
    """
    Returns x_adv (B,3,H,W) in [0,1], within Linf eps around x.
    """
    set_seed(cfg.seed)

    device = x.device
    B, C, H, W = x.shape
    max_s = min(cfg.max_square, H, W)

    # init: add random sign noise within eps and project
    x_adv = x + cfg.eps * torch.sign(torch.randn_like(x))
    x_adv = torch.max(torch.min(x_adv, x + cfg.eps), x - cfg.eps)
    x_adv = x_adv.clamp(0.0, 1.0)

    # initial best loss (EOT)
    logits0 = eot_logits(model, x_adv, cfg.defense_transform_for_attacker, cfg.eot_M)
    best = margin_loss(logits0, y)  # (B,)
    # loop (no early stop)
    for i in range(cfg.n_iters):
        s = square_size_schedule(i, cfg.n_iters, H, W, cfg.min_square, max_s)

        # propose delta: modify a random square per sample
        x_new = x_adv.clone()
        for b in range(B):
            top = random.randint(0, H - s) if H > s else 0
            left = random.randint(0, W - s) if W > s else 0

            # square attack style: set patch to either x +/- eps (randomly)
            patch_sign = 1.0 if random.random() < 0.5 else -1.0
            # propose to push patch towards the boundary around x (not around current adv)
            patch = (x[b, :, top:top+s, left:left+s] + patch_sign * cfg.eps).clamp(0.0, 1.0)
            x_new[b, :, top:top+s, left:left+s] = patch

        # project back to Linf ball around original x
        x_new = torch.max(torch.min(x_new, x + cfg.eps), x - cfg.eps)
        x_new = x_new.clamp(0.0, 1.0)

        # EOT loss
        logits_new = eot_logits(model, x_new, cfg.defense_transform_for_attacker, cfg.eot_M)
        loss_new = margin_loss(logits_new, y)

        # accept if better (lower margin loss)
        improved = loss_new < best
        if improved.any():
            x_adv[improved] = x_new[improved]
            best[improved] = loss_new[improved]

    return x_adv


# =========================================================
# Evaluation (paper-style):
# - Attack uses EOT to craft adversarial examples
# - Success check uses defense voting over K predictions
# =========================================================
@torch.no_grad()
def eval_paper_style(
    name: str,
    ds,
    clip_model,
    device: str,
    text_features: torch.Tensor,
    defense_transform: TransformType,
    vote_K: int,
    attack_cfg: SquareAttackConfig,
    batch_size: int = 16,
    num_workers: int = 4,
    max_samples: Optional[int] = None,   # for quick subset runs
):
    print(f"\n===== {name} | Defense={defense_transform} | VoteK={vote_K} | Attack=Square(EOT-{attack_cfg.eot_M}, iters={attack_cfg.n_iters}, eps={attack_cfg.eps}) =====")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True, drop_last=False)

    model = CLIPZeroShot(clip_model, text_features, device).to(device).eval().float()

    total = 0
    clean_correct = 0
    robust_correct = 0
    attacked_success_on_clean = 0
    clean_correct_count = 0

    for images, labels in tqdm(loader, desc=f"{name}-eval", ncols=120):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        if max_samples is not None and total >= max_samples:
            break
        if max_samples is not None and total + images.size(0) > max_samples:
            images = images[: max_samples - total]
            labels = labels[: max_samples - total]

        # Clean prediction with defense voting (paper uses voting for random-transform defenses)
        pred_clean = predict_with_voting(model, images, defense_transform, vote_K)
        cc = (pred_clean == labels)

        # Craft adversarial examples with attacker EOT (attacker sees same defense randomness type)
        x_adv = confident_square_attack_eot(model, images, labels, attack_cfg)

        # Robust prediction: defense voting on adversarial examples
        pred_adv = predict_with_voting(model, x_adv, defense_transform, vote_K)
        rc = (pred_adv == labels)

        n = labels.numel()
        total += n
        clean_correct += cc.sum().item()
        robust_correct += rc.sum().item()

        clean_correct_count += cc.sum().item()
        attacked_success_on_clean += ((~rc) & cc).sum().item()

        if device == "cuda":
            torch.cuda.empty_cache()

    clean_acc = clean_correct / max(total, 1)
    robust_acc = robust_correct / max(total, 1)
    asr = attacked_success_on_clean / (clean_correct_count + 1e-12)

    print(f"\nRESULT: {name}")
    print(f"Samples:          {total}")
    print(f"Clean Accuracy:   {clean_acc:.4f}")
    print(f"Robust Accuracy:  {robust_acc:.4f}")
    print(f"ASR(on clean-correct): {asr:.4f}")
    print("=" * 70 + "\n")

    return clean_acc, robust_acc, asr


# =========================================================
# Main
# =========================================================
def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] Device: {device}")

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model = clip_model.eval().float()

    # Dataset: output [0,1], resize to 224 for CLIP
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
    ])

    DATA_ROOT = "data"
    datasets = {
        "cifar10": CIFAR10(f"{DATA_ROOT}/cifar10", train=False, download=True, transform=transform),
        "cifar100": CIFAR100(f"{DATA_ROOT}/cifar100", train=False, download=True, transform=transform),
        "food101": Food101(f"{DATA_ROOT}/food101", split="test", download=True, transform=transform),
        "pets": OxfordIIITPet(f"{DATA_ROOT}/pets", split="test", download=True, transform=transform),
        "fgvc_aircraft": FGVCAircraft(f"{DATA_ROOT}/fgvc_aircraft", split="test", download=True, transform=transform),
        "stl10": STL10(f"{DATA_ROOT}/stl10", split="test", download=True, transform=transform),
    }

    # ===== Paper-style knobs =====
    # Defense voting K: paper uses 10 for EOT-10 setting (and 50 for EOT-50 in some places)
    vote_K = 10

    # Choose ONE defense to run (paper compares crop-resize and rotation; run both if time)
    defenses = ["rotation_10", "crop_resize_80"]  # or just one

    # Attack config: "confident square attack" + EOT
    # NOTE: paper runs 10,000 iters; for feasibility, start smaller (e.g., 2000) then scale up.
    attack_cfg = SquareAttackConfig(
        eps=8/255,                 # set to what you want (paper sometimes uses 12.75/255)
        n_iters=2000,              # you can increase if you have time
        eot_M=10,                  # EOT-10
        defense_transform_for_attacker="rotation_10",  # set per-defense in loop below
        min_square=1,
        max_square=64,
        seed=0
    )

    batch_size = 16
    max_samples = None  # set e.g. 200 for quick run

    for name, ds in datasets.items():
        print(f"\nPreparing: {name}")
        class_names = get_class_list(name, ds)
        text_features = build_text_features(class_names, clip_model, device)

        for d in defenses:
            # attacker should match the defense randomness type (EOT over that transform)
            attack_cfg.defense_transform_for_attacker = d  # important

            eval_paper_style(
                name=f"{name}",
                ds=ds,
                clip_model=clip_model,
                device=device,
                text_features=text_features,
                defense_transform=d,
                vote_K=vote_K,
                attack_cfg=attack_cfg,
                batch_size=batch_size,
                num_workers=4,
                max_samples=max_samples,
            )


if __name__ == "__main__":
    main()
