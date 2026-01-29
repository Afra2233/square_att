import os
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

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
    if hasattr(ds, "classes"):
        return ds.classes
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
# We KEEP rotation_10 only (as you requested)
# =========================================================
TransformType = Literal["rotation_10"]


@torch.no_grad()
def apply_random_transform_batch(
    x: torch.Tensor,
    t: TransformType,
) -> torch.Tensor:
    """
    Apply one random transform instance per image in batch.
    x: (B,3,H,W) in [0,1]
    """
    B, C, H, W = x.shape
    out = []

    if t != "rotation_10":
        raise ValueError(f"Unknown transform: {t}")

    for i in range(B):
        xi = x[i]
        angle = random.uniform(-10.0, 10.0)
        rotated = TF.rotate(
            xi,
            angle=angle,
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=0.0
        )
        out.append(rotated)

    return torch.stack(out, dim=0)


# =========================================================
# Similarity-Margin–Aware Voting (your only defense decision rule)
# - do K_base passes
# - if avg margin < tau, do K_extra more
# - voting is margin-weighted (confidence-weighted)
# =========================================================
@torch.no_grad()
def predict_margin_aware(
    model: nn.Module,
    x: torch.Tensor,
    defense_t: TransformType,
    K_base: int = 10,
    K_extra: int = 20,
    margin_tau: float = 1.0,
    weighted: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Returns:
      pred_final: (B,) predicted labels
      avg_margin: (B,) average margin over used passes
      K_used: total passes used (K_base or K_base+K_extra)
    """
    B = x.size(0)

    logits_list = []
    margin_list = []

    for _ in range(K_base):
        xt = apply_random_transform_batch(x, defense_t)
        logits = model(xt)
        logits_list.append(logits)

        top2 = torch.topk(logits, k=2, dim=1).values  # (B,2)
        margin = top2[:, 0] - top2[:, 1]              # (B,)
        margin_list.append(margin)

    margins = torch.stack(margin_list, dim=0)         # (K_base,B)
    avg_margin = margins.mean(dim=0)                  # (B,)
    uncertain = avg_margin < margin_tau

    K_used = K_base
    if uncertain.any() and K_extra > 0:
        for _ in range(K_extra):
            xt = apply_random_transform_batch(x, defense_t)
            logits = model(xt)
            logits_list.append(logits)

            top2 = torch.topk(logits, k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
            margin_list.append(margin)

        K_used = K_base + K_extra
        avg_margin = torch.stack(margin_list, dim=0).mean(dim=0)

    num_classes = logits_list[0].size(1)
    votes = torch.zeros((B, num_classes), device=x.device, dtype=torch.float32)

    if weighted:
        cap = 10.0  # prevent a few huge margins dominating too much
        for logits, m in zip(logits_list, margin_list):
            pred = logits.argmax(dim=1)
            w = torch.clamp(m, min=0.0, max=cap).float()
            votes.scatter_add_(1, pred.view(-1, 1), w.view(-1, 1))
    else:
        for logits in logits_list:
            pred = logits.argmax(dim=1)
            votes.scatter_add_(1, pred.view(-1, 1), torch.ones((B, 1), device=x.device))

    pred_final = votes.argmax(dim=1)
    return pred_final, avg_margin, K_used


# =========================================================
# Loss for attack (margin loss like paper Eq.(1))
# f(x) = logit_true - max_{j!=y} logit_j  (minimize this)
# =========================================================
def margin_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
    Attacker EOT: average logits over M random transform instances.
    """
    if eot_M <= 1:
        return model(x)

    acc = None
    for _ in range(eot_M):
        xt = apply_random_transform_batch(x, t)
        logits = model(xt)
        acc = logits if acc is None else (acc + logits)
    return acc / float(eot_M)


# =========================================================
# Confident Square Attack (C-SQA) with EOT
# - runs full N iterations (no early stop)
# =========================================================
@dataclass
class SquareAttackConfig:
    eps: float = 8/255
    n_iters: int = 500
    eot_M: int = 5
    defense_transform_for_attacker: TransformType = "rotation_10"
    min_square: int = 1
    max_square: int = 64
    seed: int = 0


def square_size_schedule(i: int, n_iters: int, H: int, W: int, min_s: int, max_s: int) -> int:
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
    set_seed(cfg.seed)

    B, C, H, W = x.shape
    max_s = min(cfg.max_square, H, W)

    # init: random sign noise within eps
    x_adv = x + cfg.eps * torch.sign(torch.randn_like(x))
    x_adv = torch.max(torch.min(x_adv, x + cfg.eps), x - cfg.eps)
    x_adv = x_adv.clamp(0.0, 1.0)

    logits0 = eot_logits(model, x_adv, cfg.defense_transform_for_attacker, cfg.eot_M)
    best = margin_loss(logits0, y)  # (B,)

    for i in range(cfg.n_iters):
        s = square_size_schedule(i, cfg.n_iters, H, W, cfg.min_square, max_s)

        x_new = x_adv.clone()
        for b in range(B):
            top = random.randint(0, H - s) if H > s else 0
            left = random.randint(0, W - s) if W > s else 0

            patch_sign = 1.0 if random.random() < 0.5 else -1.0
            patch = (x[b, :, top:top+s, left:left+s] + patch_sign * cfg.eps).clamp(0.0, 1.0)
            x_new[b, :, top:top+s, left:left+s] = patch

        x_new = torch.max(torch.min(x_new, x + cfg.eps), x - cfg.eps)
        x_new = x_new.clamp(0.0, 1.0)

        logits_new = eot_logits(model, x_new, cfg.defense_transform_for_attacker, cfg.eot_M)
        loss_new = margin_loss(logits_new, y)

        improved = loss_new < best
        if improved.any():
            x_adv[improved] = x_new[improved]
            best[improved] = loss_new[improved]

    return x_adv


# =========================================================
# Evaluation
# - Clean/Robust prediction uses margin-aware decision rule (your defense)
# - Attack uses confident square + EOT
# - Evaluate on RANDOMLY SAMPLED 1000 samples per dataset
# =========================================================
@torch.no_grad()
def eval_smav(
    name: str,
    ds,
    clip_model,
    device: str,
    text_features: torch.Tensor,
    defense_transform: TransformType,
    attack_cfg: SquareAttackConfig,
    batch_size: int = 32,
    num_workers: int = 4,
    subset_size: int = 1000,
    subset_seed: int = 0,
    K_base: int = 10,
    K_extra: int = 20,
    margin_tau: float = 1.0,
):
    print(
        f"\n===== {name} | Defense={defense_transform} | SMAV(K_base={K_base},K_extra={K_extra},tau={margin_tau}) "
        f"| Attack=Square(EOT-{attack_cfg.eot_M}, iters={attack_cfg.n_iters}, eps={attack_cfg.eps}) | subset={subset_size} ====="
    )

    loader = make_subset_loader(
        ds=ds,
        batch_size=batch_size,
        num_workers=num_workers,
        subset_size=subset_size,
        seed=subset_seed,
    )

    model = CLIPZeroShot(clip_model, text_features, device).to(device).eval().float()

    total = 0
    clean_correct = 0
    robust_correct = 0
    attacked_success_on_clean = 0
    clean_correct_count = 0

    # for reporting efficiency/uncertainty
    sum_k_clean = 0
    sum_k_adv = 0
    sum_margin_clean = 0.0
    sum_margin_adv = 0.0

    for images, labels in tqdm(loader, desc=f"{name}-eval", ncols=120):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        # Clean prediction (SMAV)
        pred_clean, m_clean, k_used_clean = predict_margin_aware(
            model, images, defense_transform,
            K_base=K_base, K_extra=K_extra, margin_tau=margin_tau, weighted=True
        )
        cc = (pred_clean == labels)

        # Attack
        x_adv = confident_square_attack_eot(model, images, labels, attack_cfg)

        # Robust prediction (SMAV)
        pred_adv, m_adv, k_used_adv = predict_margin_aware(
            model, x_adv, defense_transform,
            K_base=K_base, K_extra=K_extra, margin_tau=margin_tau, weighted=True
        )
        rc = (pred_adv == labels)

        n = labels.numel()
        total += n
        clean_correct += cc.sum().item()
        robust_correct += rc.sum().item()

        clean_correct_count += cc.sum().item()
        attacked_success_on_clean += ((~rc) & cc).sum().item()

        sum_k_clean += int(k_used_clean) * n
        sum_k_adv += int(k_used_adv) * n
        sum_margin_clean += float(m_clean.mean().item()) * n
        sum_margin_adv += float(m_adv.mean().item()) * n

        if device == "cuda":
            torch.cuda.empty_cache()

    clean_acc = clean_correct / max(total, 1)
    robust_acc = robust_correct / max(total, 1)
    asr = attacked_success_on_clean / (clean_correct_count + 1e-12)

    avg_k_clean = sum_k_clean / max(total, 1)
    avg_k_adv = sum_k_adv / max(total, 1)
    avg_margin_clean = sum_margin_clean / max(total, 1)
    avg_margin_adv = sum_margin_adv / max(total, 1)

    print(f"\nRESULT: {name}")
    print(f"Samples (subset):      {total}")
    print(f"Clean Accuracy:        {clean_acc:.4f}")
    print(f"Robust Accuracy:       {robust_acc:.4f}")
    print(f"ASR(on clean-correct): {asr:.4f}")
    print(f"Avg K used (clean):    {avg_k_clean:.2f}")
    print(f"Avg K used (adv):      {avg_k_adv:.2f}")
    print(f"Avg margin (clean):    {avg_margin_clean:.3f}")
    print(f"Avg margin (adv):      {avg_margin_adv:.3f}")
    print("=" * 80 + "\n")

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

    # ===== Your requested setup =====
    defense_transform: TransformType = "rotation_10"

    # SMAV hyperparams (small and safe)
    K_base = 10
    K_extra = 20
    margin_tau = 1.0  # if too slow (always uncertain), try 0.5; if rarely uncertain, try 2.0

    # Attack config
    attack_cfg = SquareAttackConfig(
        eps=8/255,
        n_iters=500,
        eot_M=5,  # for time; set to 10 if you can afford it
        defense_transform_for_attacker=defense_transform,
        min_square=1,
        max_square=64,
        seed=0
    )

    batch_size = 32
    subset_size = 1000
    subset_seed = 0  # keep fixed for reproducibility

    for name, ds in datasets.items():
        print(f"\nPreparing: {name}")
        class_names = get_class_list(name, ds)
        text_features = build_text_features(class_names, clip_model, device)

        eval_smav(
            name=name,
            ds=ds,
            clip_model=clip_model,
            device=device,
            text_features=text_features,
            defense_transform=defense_transform,
            attack_cfg=attack_cfg,
            batch_size=batch_size,
            num_workers=4,
            subset_size=subset_size,
            subset_seed=subset_seed,
            K_base=K_base,
            K_extra=K_extra,
            margin_tau=margin_tau,
        )


if __name__ == "__main__":
    main()
