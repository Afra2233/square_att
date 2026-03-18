import os
import random
import numpy as np
from dataclasses import dataclass, replace
from typing import Literal, Tuple, Dict, Optional

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

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def encode_image_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        f = self.clip_model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        return f

    def logits_from_features(self, f: torch.Tensor) -> torch.Tensor:
        return 100.0 * (f @ self.text_features.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encode_image_features(x)
        return self.logits_from_features(f)


# =========================================================
# Randomized Transform Defense (inference-time)
# identity: no transform
# rotation_10: random angle in [-10,10]
# crop_resize_80: random crop scale in [0.8, 1.0] then resize back
# =========================================================
TransformType = Literal["identity", "rotation_10", "crop_resize_80"]


@torch.no_grad()
def apply_random_transform_batch(x: torch.Tensor, t: TransformType) -> torch.Tensor:
    """
    Apply one random transform instance per image in batch.
    x: (B,3,H,W) in [0,1]
    """
    if t == "identity":
        return x

    B, C, H, W = x.shape
    out = []

    for i in range(B):
        xi = x[i]

        if t == "rotation_10":
            angle = random.uniform(-10.0, 10.0)
            fill = float(xi.mean().item())
            xo = TF.rotate(
                xi,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=fill
            )

        elif t == "crop_resize_80":
            scale = random.uniform(0.8, 1.0)
            ch = max(1, int(round(H * scale)))
            cw = max(1, int(round(W * scale)))
            top = random.randint(0, H - ch) if H > ch else 0
            left = random.randint(0, W - cw) if W > cw else 0
            cropped = xi[:, top:top + ch, left:left + cw]
            xo = TF.resize(
                cropped,
                size=[H, W],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True
            )
        else:
            raise ValueError(f"Unknown transform: {t}")

        out.append(xo.clamp(0.0, 1.0))

    return torch.stack(out, dim=0)


# =========================================================
# Aggregation
# =========================================================
AggregationType = Literal["single", "vote", "avg_logits", "avg_features"]


@torch.no_grad()
def predict_with_aggregation(
    model: CLIPZeroShot,
    x: torch.Tensor,
    defense_t: TransformType,
    aggregation: AggregationType,
    K: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        pred_final:  (B,)
        logits_final: (B,C)  surrogate/final logits used for reporting or attack
    """
    if aggregation == "single" or defense_t == "identity" and K <= 1:
        logits = model(x)
        pred = logits.argmax(dim=1)
        return pred, logits

    B = x.size(0)

    if aggregation == "vote":
        num_classes = model(x[:1]).size(1)
        votes = torch.zeros((B, num_classes), device=x.device, dtype=torch.int32)
        logits_sum = torch.zeros((B, num_classes), device=x.device, dtype=torch.float32)

        for _ in range(K):
            xt = apply_random_transform_batch(x, defense_t)
            logits = model(xt)
            pred = logits.argmax(dim=1)
            votes.scatter_add_(1, pred.view(-1, 1), torch.ones((B, 1), device=x.device, dtype=torch.int32))
            logits_sum += logits

        max_votes = votes.max(dim=1, keepdim=True).values
        tied = (votes == max_votes)
        logits_tie = logits_sum.masked_fill(~tied, float("-inf"))
        pred_final = logits_tie.argmax(dim=1)

        # return averaged logits as a smooth surrogate too
        logits_final = logits_sum / float(K)
        return pred_final, logits_final

    elif aggregation == "avg_logits":
        logits_sum = None
        for _ in range(K):
            xt = apply_random_transform_batch(x, defense_t)
            logits = model(xt)
            logits_sum = logits if logits_sum is None else (logits_sum + logits)
        logits_final = logits_sum / float(K)
        pred_final = logits_final.argmax(dim=1)
        return pred_final, logits_final

    elif aggregation == "avg_features":
        feat_sum = None
        for _ in range(K):
            xt = apply_random_transform_batch(x, defense_t)
            f = model.encode_image_features(xt)
            feat_sum = f if feat_sum is None else (feat_sum + f)
        f_bar = feat_sum / float(K)
        f_bar = f_bar / f_bar.norm(dim=-1, keepdim=True)
        logits_final = model.logits_from_features(f_bar)
        pred_final = logits_final.argmax(dim=1)
        return pred_final, logits_final

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


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
def eot_aggregated_logits(
    model: CLIPZeroShot,
    x: torch.Tensor,
    t: TransformType,
    eot_M: int,
    aggregation: AggregationType,
) -> torch.Tensor:
    """
    Attacker-side adaptive forward.
    Returns logits used by the attacker.

    For "vote", we use averaged logits as a smooth surrogate, while
    evaluation still uses true majority vote.
    """
    if aggregation == "single":
        return model(x)

    if t == "identity":
        return model(x)

    if aggregation == "avg_logits" or aggregation == "vote":
        logits_sum = None
        M = max(1, eot_M)
        for _ in range(M):
            xt = apply_random_transform_batch(x, t)
            logits = model(xt)
            logits_sum = logits if logits_sum is None else (logits_sum + logits)
        return logits_sum / float(M)

    elif aggregation == "avg_features":
        feat_sum = None
        M = max(1, eot_M)
        for _ in range(M):
            xt = apply_random_transform_batch(x, t)
            f = model.encode_image_features(xt)
            feat_sum = f if feat_sum is None else (feat_sum + f)
        f_bar = feat_sum / float(M)
        f_bar = f_bar / f_bar.norm(dim=-1, keepdim=True)
        return model.logits_from_features(f_bar)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


# =========================================================
# Confident Square Attack (C-SQA) with adaptive EOT forward
# - runs full N iterations (no early stop)
# =========================================================
@dataclass
class SquareAttackConfig:
    eps: float = 8 / 255
    n_iters: int = 200
    eot_M: int = 4
    defense_transform_for_attacker: TransformType = "identity"
    aggregation_for_attacker: AggregationType = "single"
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
    model: CLIPZeroShot,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: SquareAttackConfig,
) -> torch.Tensor:
    # deterministic per call
    set_seed(cfg.seed)

    B, C, H, W = x.shape
    max_s = min(cfg.max_square, H, W)

    # init random sign noise within eps
    x_adv = x + cfg.eps * torch.sign(torch.randn_like(x))
    x_adv = torch.max(torch.min(x_adv, x + cfg.eps), x - cfg.eps)
    x_adv = x_adv.clamp(0.0, 1.0)

    logits0 = eot_aggregated_logits(
        model=model,
        x=x_adv,
        t=cfg.defense_transform_for_attacker,
        eot_M=cfg.eot_M,
        aggregation=cfg.aggregation_for_attacker,
    )
    best = margin_loss(logits0, y)

    for i in range(cfg.n_iters):
        s = square_size_schedule(i, cfg.n_iters, H, W, cfg.min_square, max_s)

        x_new = x_adv.clone()
        for b in range(B):
            top = random.randint(0, H - s) if H > s else 0
            left = random.randint(0, W - s) if W > s else 0

            patch_sign = 1.0 if random.random() < 0.5 else -1.0
            patch = (x[b, :, top:top + s, left:left + s] + patch_sign * cfg.eps).clamp(0.0, 1.0)
            x_new[b, :, top:top + s, left:left + s] = patch

        x_new = torch.max(torch.min(x_new, x + cfg.eps), x - cfg.eps)
        x_new = x_new.clamp(0.0, 1.0)

        logits_new = eot_aggregated_logits(
            model=model,
            x=x_new,
            t=cfg.defense_transform_for_attacker,
            eot_M=cfg.eot_M,
            aggregation=cfg.aggregation_for_attacker,
        )
        loss_new = margin_loss(logits_new, y)

        improved = loss_new < best
        if improved.any():
            x_adv[improved] = x_new[improved]
            best[improved] = loss_new[improved]

    return x_adv


# =========================================================
# Evaluation
# We report:
#   - Undefended Clean Acc (single pass on clean)
#   - Undefended Robust Acc (adaptive attack vs identity, eval single pass)
# For each defense d and aggregation a:
#   - Defended Clean Acc
#   - Defended Robust Acc (adaptive attack matched to that defense)
# =========================================================
@torch.no_grad()
def eval_defenses_fair(
    name: str,
    ds,
    clip_model,
    device: str,
    text_features: torch.Tensor,
    defenses: Tuple[TransformType, ...],
    aggregations: Tuple[AggregationType, ...],
    attack_cfg_base: SquareAttackConfig,
    batch_size: int = 32,
    num_workers: int = 4,
    subset_size: int = 200,
    subset_seed: int = 0,
    K_clean: int = 4,
    K_adv: int = 4,
    print_cross_eval: bool = False,
):
    print(
        f"\n===== {name} | Defenses={defenses} | Aggregations={aggregations} "
        f"| K_clean={K_clean} K_adv={K_adv} | "
        f"Attack=Conf-Square(EOT-{attack_cfg_base.eot_M}, iters={attack_cfg_base.n_iters}, eps={attack_cfg_base.eps}) "
        f"| subset={subset_size} ====="
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
    undef_robust_correct_adaptive = 0

    stats: Dict[str, Dict[str, float]] = {}
    for d in defenses:
        for a in aggregations:
            key = f"{d}|{a}"
            stats[key] = {
                "def_clean_correct": 0,
                "def_robust_correct": 0,
                "asr_def_num": 0,
                "def_clean_ok": 0,
                "undef_robust_cross_correct": 0,
                "asr_undef_cross_num": 0,
            }

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"{name}-eval", ncols=120)):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        n = labels.numel()
        total += n

        # ---------- Undefended clean ----------
        pred_uc, logits_uc = predict_with_aggregation(
            model=model,
            x=images,
            defense_t="identity",
            aggregation="single",
            K=1,
        )
        uc = (pred_uc == labels)
        undef_clean_correct += uc.sum().item()

        # ---------- Undefended adaptive attack + robust ----------
        cfg_undef = replace(
            attack_cfg_base,
            seed=int(attack_cfg_base.seed) + int(batch_idx),
            defense_transform_for_attacker="identity",
            aggregation_for_attacker="single",
            eot_M=1,
        )
        x_adv_undef = confident_square_attack_eot(model, images, labels, cfg_undef)

        pred_ur_ad, _ = predict_with_aggregation(
            model=model,
            x=x_adv_undef,
            defense_t="identity",
            aggregation="single",
            K=1,
        )
        ur_ad = (pred_ur_ad == labels)
        undef_robust_correct_adaptive += ur_ad.sum().item()

        # ---------- Defenses x Aggregations ----------
        for d in defenses:
            for a in aggregations:
                key = f"{d}|{a}"

                pred_dc, _ = predict_with_aggregation(
                    model=model,
                    x=images,
                    defense_t=d,
                    aggregation=a,
                    K=K_clean,
                )
                dc = (pred_dc == labels)
                stats[key]["def_clean_correct"] += dc.sum().item()
                stats[key]["def_clean_ok"] += dc.sum().item()

                cfg_d = replace(
                    attack_cfg_base,
                    seed=int(attack_cfg_base.seed) + int(batch_idx),
                    defense_transform_for_attacker=d,
                    aggregation_for_attacker=a,
                    eot_M=int(attack_cfg_base.eot_M),
                )
                x_adv_d = confident_square_attack_eot(model, images, labels, cfg_d)

                if print_cross_eval:
                    pred_ur_cross, _ = predict_with_aggregation(
                        model=model,
                        x=x_adv_d,
                        defense_t="identity",
                        aggregation="single",
                        K=1,
                    )
                    ur_cross = (pred_ur_cross == labels)
                    stats[key]["undef_robust_cross_correct"] += ur_cross.sum().item()
                    stats[key]["asr_undef_cross_num"] += ((~ur_cross) & uc).sum().item()

                pred_dr, _ = predict_with_aggregation(
                    model=model,
                    x=x_adv_d,
                    defense_t=d,
                    aggregation=a,
                    K=K_adv,
                )
                dr = (pred_dr == labels)
                stats[key]["def_robust_correct"] += dr.sum().item()
                stats[key]["asr_def_num"] += ((~dr) & dc).sum().item()

        if device == "cuda":
            torch.cuda.empty_cache()

    # ---------- print results ----------
    undef_clean_acc = undef_clean_correct / max(total, 1)
    undef_robust_acc_ad = undef_robust_correct_adaptive / max(total, 1)

    print(f"\nRESULT: {name}")
    print(f"Samples (subset):                      {total}")
    print(f"Undefended Clean Accuracy:             {undef_clean_acc:.4f}")
    print(f"Undefended Robust Accuracy (adaptive): {undef_robust_acc_ad:.4f}")

    for d in defenses:
        for a in aggregations:
            key = f"{d}|{a}"
            def_clean_acc = stats[key]["def_clean_correct"] / max(total, 1)
            def_robust_acc = stats[key]["def_robust_correct"] / max(total, 1)
            asr_def = stats[key]["asr_def_num"] / (stats[key]["def_clean_ok"] + 1e-12)

            print(f"\n--- Defense: {d} | Aggregation: {a} ---")
            print(f"Defended Clean Accuracy:              {def_clean_acc:.4f}")
            print(f"Defended Robust Accuracy (adaptive):  {def_robust_acc:.4f}")
            print(f"ASR_def (on def-clean-ok):            {asr_def:.4f}")

            if print_cross_eval:
                undef_robust_cross = stats[key]["undef_robust_cross_correct"] / max(total, 1)
                asr_undef_cross = stats[key]["asr_undef_cross_num"] / (undef_clean_correct + 1e-12)
                print(f"Undefended Robust (cross-eval):       {undef_robust_cross:.4f}")
                print(f"ASR_undef (cross-eval):               {asr_undef_cross:.4f}")

    print("=" * 100 + "\n")


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
        # 建议先只开一个，省资源
        "cifar10": CIFAR10(f"{DATA_ROOT}/cifar10", train=False, download=True, transform=transform),
        # "cifar100": CIFAR100(f"{DATA_ROOT}/cifar100", train=False, download=True, transform=transform),
        # "food101": Food101(f"{DATA_ROOT}/food101", split="test", download=True, transform=transform),
        # "pets": OxfordIIITPet(f"{DATA_ROOT}/pets", split="test", download=True, transform=transform),
        # "fgvc_aircraft": FGVCAircraft(f"{DATA_ROOT}/fgvc_aircraft", split="test", download=True, transform=transform),
        # "stl10": STL10(f"{DATA_ROOT}/stl10", split="test", download=True, transform=transform),
    }

    defenses: Tuple[TransformType, ...] = ("crop_resize_80", "rotation_10")

    # 你最关心的比较对象
    aggregations: Tuple[AggregationType, ...] = (
        "vote",
        "avg_logits",
        "avg_features",
    )

    # 先用小预算看趋势
    attack_cfg = SquareAttackConfig(
        eps=8 / 255,
        n_iters=200,     # 原来 500，先降一点
        eot_M=4,         # 原来 10，先降一点
        defense_transform_for_attacker="identity",
        aggregation_for_attacker="single",
        min_square=1,
        max_square=64,
        seed=0
    )

    batch_size = 32
    subset_size = 200   # 原来 1000，先小跑
    subset_seed = 0

    K_clean = 4         # 原来 10，先小跑
    K_adv = 4

    for name, ds in datasets.items():
        print(f"\nPreparing: {name}")
        class_names = get_class_list(name, ds)
        text_features = build_text_features(class_names, clip_model, device)

        eval_defenses_fair(
            name=name,
            ds=ds,
            clip_model=clip_model,
            device=device,
            text_features=text_features,
            defenses=defenses,
            aggregations=aggregations,
            attack_cfg_base=attack_cfg,
            batch_size=batch_size,
            num_workers=4,
            subset_size=subset_size,
            subset_seed=subset_seed,
            K_clean=K_clean,
            K_adv=K_adv,
            print_cross_eval=False,
        )


if __name__ == "__main__":
    main()