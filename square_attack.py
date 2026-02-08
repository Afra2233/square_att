import os
import random
import numpy as np
from dataclasses import dataclass, replace
from typing import Literal, Tuple, Dict

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
# CLIP Zero-shot Classifier
# x is in [0,1] and shape (B,3,224,224)
# =========================================================
class CLIPZeroShot(nn.Module):
    def __init__(self, clip_model, text_features, device):
        super().__init__()
        self.clip_model = clip_model
        self.text_features = text_features.to(device)

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
# We provide:
#   - rotation_10: random angle in [-10,10]
#   - crop_resize_80: random crop (scale in [0.8, 1.0]) then resize back
# =========================================================
TransformType = Literal["rotation_10", "crop_resize_80"]


@torch.no_grad()
def apply_random_transform_batch(x: torch.Tensor, t: TransformType) -> torch.Tensor:
    """
    Apply one random transform instance per image in batch.
    x: (B,3,H,W) in [0,1]
    """
    B, C, H, W = x.shape
    out = []

    for i in range(B):
        xi = x[i]

        if t == "rotation_10":
            angle = random.uniform(-10.0, 10.0)
            xo = TF.rotate(
                xi,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=0.0
            )

        elif t == "crop_resize_80":
            # random crop scale in [0.8, 1.0], aspect ratio fixed to 1.0 for simplicity & stability
            scale = random.uniform(0.8, 1.0)
            ch = max(1, int(round(H * scale)))
            cw = max(1, int(round(W * scale)))
            top = random.randint(0, H - ch) if H > ch else 0
            left = random.randint(0, W - cw) if W > cw else 0
            cropped = xi[:, top:top+ch, left:left+cw]
            xo = TF.resize(
                cropped, size=[H, W],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True
            )
        else:
            raise ValueError(f"Unknown transform: {t}")

        out.append(xo.clamp(0.0, 1.0))

    return torch.stack(out, dim=0)


# =========================================================
# Majority Vote Aggregation (paper-style)
# - sample K transformed variants
# - take argmax each time
# - final pred is mode (ties broken by summed logits over tied classes)
# =========================================================
@torch.no_grad()
def predict_majority_vote(
    model: nn.Module,
    x: torch.Tensor,
    defense_t: TransformType,
    K: int = 10,
) -> torch.Tensor:
    """
    Returns:
      pred_final: (B,)
    """
    B = x.size(0)
    num_classes = model(x[:1]).size(1)  # one forward to get C (cheap)
    votes = torch.zeros((B, num_classes), device=x.device, dtype=torch.int32)
    logits_sum = torch.zeros((B, num_classes), device=x.device, dtype=torch.float32)

    for _ in range(K):
        xt = apply_random_transform_batch(x, defense_t)
        logits = model(xt)  # (B,C)
        pred = logits.argmax(dim=1)  # (B,)
        votes.scatter_add_(1, pred.view(-1, 1), torch.ones((B, 1), device=x.device, dtype=torch.int32))
        logits_sum += logits

    # majority class (ties resolved by logits_sum among tied classes)
    max_votes = votes.max(dim=1, keepdim=True).values  # (B,1)
    tied = (votes == max_votes)  # (B,C) bool
    # set logits of non-tied classes to -inf so argmax picks among ties
    logits_tie = logits_sum.masked_fill(~tied, float("-inf"))
    pred_final = logits_tie.argmax(dim=1)
    return pred_final


# =========================================================
# Loss for attack (margin loss)
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
# - report:
#   (1) Undefended clean acc: model(images)
#   (2) Undefended robust acc: model(x_adv)  (x_adv crafted vs chosen defense transform via cfg)
#   (3) Defended clean/robust for each defense transform with majority vote
# =========================================================
@torch.no_grad()
def eval_defenses_majority_vote(
    name: str,
    ds,
    clip_model,
    device: str,
    text_features: torch.Tensor,
    defenses: Tuple[TransformType, ...],
    attack_cfg_base: SquareAttackConfig,
    batch_size: int = 32,
    num_workers: int = 4,
    subset_size: int = 1000,
    subset_seed: int = 0,
    K_vote_clean: int = 10,
    K_vote_adv: int = 10,
):
    print(
        f"\n===== {name} | Defenses={defenses} | Vote(K_clean={K_vote_clean},K_adv={K_vote_adv}) "
        f"| Attack=Conf-Square(EOT-{attack_cfg_base.eot_M}, iters={attack_cfg_base.n_iters}, eps={attack_cfg_base.eps}) | subset={subset_size} ====="
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
    undef_clean_correct = 0

    # per-defense stats
    stats: Dict[str, Dict[str, float]] = {}
    for d in defenses:
        stats[d] = {
            "def_clean_correct": 0,
            "def_robust_correct": 0,
            "asr_def_num": 0,     # among def-clean-correct, how many become wrong under defense after attack
            "def_clean_ok": 0,
        }

    # optional: undef robust (but note: x_adv is crafted vs a chosen defense transform; we report per-defense undef robust as well)
    undef_robust_correct_per_def: Dict[str, int] = {d: 0 for d in defenses}
    asr_undef_num_per_def: Dict[str, int] = {d: 0 for d in defenses}
    undef_clean_ok = 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"{name}-eval", ncols=120)):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        n = labels.numel()
        total += n

        # ---------- Undefended clean ----------
        logits_uc = model(images)
        pred_uc = logits_uc.argmax(dim=1)
        uc = (pred_uc == labels)
        undef_clean_correct += uc.sum().item()
        undef_clean_ok += uc.sum().item()

        # For each defense, we:
        #   (1) compute defended clean by majority vote
        #   (2) craft x_adv against that defense via EOT on its transform
        #   (3) measure undefended robust on that x_adv (for ASR_undef fairness per-defense)
        #   (4) measure defended robust by majority vote
        for d in defenses:
            # (1) defended clean
            pred_dc = predict_majority_vote(model, images, d, K=K_vote_clean)
            dc = (pred_dc == labels)
            stats[d]["def_clean_correct"] += dc.sum().item()
            stats[d]["def_clean_ok"] += dc.sum().item()

            # (2) craft adversarial examples adaptive to this defense transform
            cfg_b = replace(
                attack_cfg_base,
                seed=int(attack_cfg_base.seed) + int(batch_idx),
                defense_transform_for_attacker=d
            )
            x_adv = confident_square_attack_eot(model, images, labels, cfg_b)

            # (3) undefended robust on x_adv (crafted vs defense d)
            logits_ur = model(x_adv)
            pred_ur = logits_ur.argmax(dim=1)
            ur = (pred_ur == labels)
            undef_robust_correct_per_def[d] += ur.sum().item()
            asr_undef_num_per_def[d] += ((~ur) & uc).sum().item()

            # (4) defended robust on x_adv
            pred_dr = predict_majority_vote(model, x_adv, d, K=K_vote_adv)
            dr = (pred_dr == labels)
            stats[d]["def_robust_correct"] += dr.sum().item()
            stats[d]["asr_def_num"] += ((~dr) & dc).sum().item()

        if device == "cuda":
            torch.cuda.empty_cache()

    # ---------- print results ----------
    undef_clean_acc = undef_clean_correct / max(total, 1)

    print(f"\nRESULT: {name}")
    print(f"Samples (subset):           {total}")
    print(f"Undefended Clean Accuracy:  {undef_clean_acc:.4f}")

    for d in defenses:
        def_clean_acc = stats[d]["def_clean_correct"] / max(total, 1)
        def_robust_acc = stats[d]["def_robust_correct"] / max(total, 1)

        undef_robust_acc = undef_robust_correct_per_def[d] / max(total, 1)

        asr_def = stats[d]["asr_def_num"] / (stats[d]["def_clean_ok"] + 1e-12)
        asr_undef = asr_undef_num_per_def[d] / (undef_clean_ok + 1e-12)

        print(f"\n--- Defense: {d} ---")
        print(f"Defended  Clean Accuracy:   {def_clean_acc:.4f}  (MajorityVote K={K_vote_clean})")
        print(f"Undefended Robust Accuracy: {undef_robust_acc:.4f}  (on x_adv crafted vs {d})")
        print(f"Defended  Robust Accuracy:  {def_robust_acc:.4f}  (MajorityVote K={K_vote_adv} on x_adv crafted vs {d})")
        print(f"ASR_def   (on def-clean-ok): {asr_def:.4f}")
        print(f"ASR_undef (on undef-clean):  {asr_undef:.4f}")

    print("=" * 80 + "\n")


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

    defenses: Tuple[TransformType, ...] = ("crop_resize_80", "rotation_10")

    # Attack config (low-budget; you can scale up later)
    attack_cfg = SquareAttackConfig(
        eps=8/255,
        n_iters=500,
        eot_M=10,
        defense_transform_for_attacker="rotation_10",  # will be overridden per-defense
        min_square=1,
        max_square=64,
        seed=0
    )

    batch_size = 32
    subset_size = 1000
    subset_seed = 0

    # Voting K: in the paper they use 10 or 50 depending on EOT, but you can keep it small.
    K_vote_clean = 10
    K_vote_adv = 10

    for name, ds in datasets.items():
        print(f"\nPreparing: {name}")
        class_names = get_class_list(name, ds)
        text_features = build_text_features(class_names, clip_model, device)

        eval_defenses_majority_vote(
            name=name,
            ds=ds,
            clip_model=clip_model,
            device=device,
            text_features=text_features,
            defenses=defenses,
            attack_cfg_base=attack_cfg,
            batch_size=batch_size,
            num_workers=4,
            subset_size=subset_size,
            subset_seed=subset_seed,
            K_vote_clean=K_vote_clean,
            K_vote_adv=K_vote_adv,
        )


if __name__ == "__main__":
    main()

