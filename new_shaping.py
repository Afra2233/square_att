import os
import random
import numpy as np
from dataclasses import dataclass, replace
from typing import Literal, Tuple, Dict, List, Optional

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.datasets import CIFAR10, CIFAR100, Food101, STL10

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
    if hasattr(ds, "classes") and ds.classes is not None:
        return ds.classes

    name = name.lower()
    if name == "stl10":
        return [
            "airplane", "bird", "car", "cat", "deer",
            "dog", "horse", "monkey", "ship", "truck"
        ]

    raise RuntimeError(f"[FATAL] Dataset {name} has no usable class names.")


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
    dname = dataset_name.lower()

    if dname == "food101":
        templates = [
            "a photo of {}",
            
        ]
    elif dname == "stl10":
        templates = [
            "a photo of a {}",
            
        ]
    else:
        templates = [
            "a photo of a {}",
           
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
# Views (kept for compatibility, but not used in this script)
# =========================================================
ViewType = Literal[
    "identity",
    "horizontal_flip",
    "resize_pad_96",
    "center_crop_96",
    "rotation_fixed_plus8",
    "blur_fixed_sigma_0p6",
    "rotation_rand_8",
    "blur_rand_light",
]

ViewMode = Literal["deterministic", "stochastic"]


@torch.inference_mode()
def apply_view_batch(x: torch.Tensor, view_t: ViewType) -> torch.Tensor:
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
            xo = TF.pad(
                resized,
                [pad_left, pad_top, pad_right, pad_bottom],
                fill=float(xi.mean()),
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

        elif view_t == "rotation_fixed_plus8":
            xo = TF.rotate(
                xi,
                angle=8.0,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=float(xi.mean()),
            )

        elif view_t == "blur_fixed_sigma_0p6":
            kernel_size = 3 if min(H, W) < 128 else 5
            xo = TF.gaussian_blur(xi, kernel_size=[kernel_size, kernel_size], sigma=0.6)

        elif view_t == "rotation_rand_8":
            angle = random.uniform(-8.0, 8.0)
            xo = TF.rotate(
                xi,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=float(xi.mean()),
            )

        elif view_t == "blur_rand_light":
            kernel_size = 3 if min(H, W) < 128 else 5
            sigma = random.uniform(0.1, 1.0)
            xo = TF.gaussian_blur(xi, kernel_size=[kernel_size, kernel_size], sigma=sigma)

        else:
            raise ValueError(f"Unknown view type: {view_t}")

        out.append(xo.clamp(0.0, 1.0))

    return torch.stack(out, dim=0)


# =========================================================
# Non-semantic shaping family
# =========================================================
ShapingType = Literal[
    "none",
    "hinge_margin",
    "exp_repulsion",
]


@torch.inference_mode()
def competitor_hinge_delta(
    margin: torch.Tensor,
    gamma: float = 3.0,
    lam: float = 0.5,
    max_delta_ratio: float = 0.8,
) -> torch.Tensor:
    """
    margin = z_top1 - z_c
    If margin < gamma, suppress the competitor.
    """
    delta = lam * torch.clamp(gamma - margin, min=0.0)
    delta = torch.clamp(delta, max=max_delta_ratio * margin.clamp_min(1e-12))
    return delta


@torch.inference_mode()
def competitor_exp_delta(
    margin: torch.Tensor,
    lam: float = 0.5,
    tau: float = 2.0,
    max_delta_ratio: float = 0.8,
) -> torch.Tensor:
    """
    margin = z_top1 - z_c
    Smaller margin -> larger suppression.
    """
    delta = lam * torch.exp(-margin / tau)
    delta = torch.clamp(delta, max=max_delta_ratio * margin.clamp_min(1e-12))
    return delta


@torch.inference_mode()
def random_shape_logits(
    logits: torch.Tensor,
    shaping: ShapingType,
    shaping_topk: int = 2,
    gamma: float = 3.0,
    lam: float = 0.5,
    tau: float = 2.0,
    max_delta_ratio: float = 0.8,
) -> torch.Tensor:
    if shaping == "none":
        return logits

    if shaping not in {"hinge_margin", "exp_repulsion"}:
        raise ValueError(f"Unsupported shaping: {shaping}")

    B, C = logits.shape
    z = logits.clone()

    k = max(2, min(shaping_topk, C))
    topkv, topki = logits.topk(k=k, dim=1)
    top1_val = topkv[:, 0]

    # fair top2-vs-top5: same per-competitor rule, only scope changes
    for b in range(B):
        z1 = top1_val[b]
        cands = topki[b, 1:].tolist()

        for c in cands:
            zc = logits[b, c]
            margin = (z1 - zc).view(1)

            if shaping == "hinge_margin":
                delta = competitor_hinge_delta(
                    margin=margin,
                    gamma=gamma,
                    lam=lam,
                    max_delta_ratio=max_delta_ratio,
                )
            else:
                delta = competitor_exp_delta(
                    margin=margin,
                    lam=lam,
                    tau=tau,
                    max_delta_ratio=max_delta_ratio,
                )

            z[b, c] = zc - delta.item()

    return z


@torch.inference_mode()
def sample_random_shaping(
    enable_random_shaping: bool,
    family: Optional[List[ShapingType]] = None,
) -> ShapingType:
    if not enable_random_shaping:
        return "none"
    family = family or ["hinge_margin", "exp_repulsion"]
    return random.choice(family)


# =========================================================
# Unified defended prediction (single-view only here)
# =========================================================
@torch.inference_mode()
def defended_predict(
    model: CLIPZeroShot,
    x: torch.Tensor,
    use_random_shaping: bool,
    shaping_family: Optional[List[ShapingType]] = None,
    shaping_topk: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    aux = {}

    feats = model.encode_image_features(x)
    logits = model.logits_from_features(feats)
    aux["cross_view_std"] = torch.zeros(x.size(0), device=x.device)

    shaping = sample_random_shaping(use_random_shaping, shaping_family)
    logits_before_shape = logits.clone()

    logits = random_shape_logits(
        logits=logits,
        shaping=shaping,
        shaping_topk=shaping_topk,
    )

    shaping_id_map = {
        "none": 0,
        "hinge_margin": 1,
        "exp_repulsion": 2,
    }
    aux["shaping_id"] = torch.full(
        (x.size(0),), shaping_id_map[shaping], device=x.device, dtype=torch.long
    )

    probs_before = logits_before_shape.softmax(dim=-1)
    probs_after = logits.softmax(dim=-1)
    aux["score_shift_l1"] = (probs_after - probs_before).abs().sum(dim=1)

    pred = logits.argmax(dim=1)
    return pred, logits, aux


# =========================================================
# Fast attacker-side forward with EOT (single-view only)
# =========================================================
@torch.no_grad()
def defended_forward_for_attacker(
    model: CLIPZeroShot,
    x: torch.Tensor,
    use_random_shaping: bool,
    shaping_family: Optional[List[ShapingType]],
    shaping_topk: int = 2,
    eot_M: int = 1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    M = max(1, eot_M)

    logits_sum = None
    score_shift_sum = None
    cross_view_std_sum = None

    base_logits = model(x)
    base_cross_view_std = torch.zeros(x.size(0), device=x.device)

    for _ in range(M):
        shaping = sample_random_shaping(use_random_shaping, shaping_family)

        shaped_logits = random_shape_logits(
            base_logits,
            shaping=shaping,
            shaping_topk=shaping_topk,
        )

        probs_before = base_logits.softmax(dim=-1)
        probs_after = shaped_logits.softmax(dim=-1)
        score_shift = (probs_after - probs_before).abs().sum(dim=1)

        logits_sum = shaped_logits if logits_sum is None else (logits_sum + shaped_logits)
        score_shift_sum = score_shift if score_shift_sum is None else (score_shift_sum + score_shift)
        cross_view_std_sum = base_cross_view_std if cross_view_std_sum is None else (cross_view_std_sum + base_cross_view_std)

    out_aux = {
        "score_shift_l1": score_shift_sum / float(M),
        "cross_view_std": cross_view_std_sum / float(M),
    }
    return logits_sum / float(M), out_aux


# =========================================================
# Loss
# =========================================================
def margin_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    tmp = logits.clone()
    tmp.scatter_(1, y.view(-1, 1), -1e9)
    other = tmp.max(dim=1).values
    return true - other


# =========================================================
# Square Attack with trajectory logging
# =========================================================
@dataclass
class SquareAttackConfig:
    eps: float = 8 / 255
    n_iters: int = 200
    eot_M: int = 8
    min_square: int = 1
    max_square: int = 64
    seed: int = 0


@dataclass
class DefenseConfig:
    name: str
    use_random_shaping: bool
    shaping_family: Tuple[ShapingType, ...]
    shaping_topk: int = 2


def square_size_schedule(i: int, n_iters: int, H: int, W: int, min_s: int, max_s: int) -> int:
    frac = 1.0 - (i / max(n_iters - 1, 1))
    s = int(round(min_s + (max_s - min_s) * (frac ** 2)))
    s = max(min_s, min(s, min(H, W)))
    return s


@torch.no_grad()
def square_attack_with_logging(
    model: CLIPZeroShot,
    x: torch.Tensor,
    y: torch.Tensor,
    attack_cfg: SquareAttackConfig,
    defense_cfg: DefenseConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    set_seed(attack_cfg.seed)

    B, C, H, W = x.shape
    max_s = min(attack_cfg.max_square, H, W)

    x_adv = x + attack_cfg.eps * torch.sign(torch.randn_like(x))
    x_adv = torch.max(torch.min(x_adv, x + attack_cfg.eps), x - attack_cfg.eps)
    x_adv = x_adv.clamp(0.0, 1.0)

    logits0, aux0 = defended_forward_for_attacker(
        model=model,
        x=x_adv,
        use_random_shaping=defense_cfg.use_random_shaping,
        shaping_family=list(defense_cfg.shaping_family),
        shaping_topk=defense_cfg.shaping_topk,
        eot_M=attack_cfg.eot_M,
    )
    best = margin_loss(logits0, y)

    accepted_steps = 0
    total_steps = attack_cfg.n_iters * B
    improvement_sum = 0.0
    score_shift_sum = aux0["score_shift_l1"].sum().item()
    cross_view_std_sum = aux0["cross_view_std"].sum().item()

    margin_history = [best.detach().clone()]
    delta_sign_flips = 0
    prev_delta = None

    for i in range(attack_cfg.n_iters):
        s = square_size_schedule(i, attack_cfg.n_iters, H, W, attack_cfg.min_square, max_s)
        x_new = x_adv.clone()

        for b in range(B):
            top = random.randint(0, H - s) if H > s else 0
            left = random.randint(0, W - s) if W > s else 0
            patch_sign = 1.0 if random.random() < 0.5 else -1.0
            patch = (x[b, :, top:top + s, left:left + s] + patch_sign * attack_cfg.eps).clamp(0.0, 1.0)
            x_new[b, :, top:top + s, left:left + s] = patch

        x_new = torch.max(torch.min(x_new, x + attack_cfg.eps), x - attack_cfg.eps)
        x_new = x_new.clamp(0.0, 1.0)

        logits_new, aux_new = defended_forward_for_attacker(
            model=model,
            x=x_new,
            use_random_shaping=defense_cfg.use_random_shaping,
            shaping_family=list(defense_cfg.shaping_family),
            shaping_topk=defense_cfg.shaping_topk,
            eot_M=attack_cfg.eot_M,
        )
        loss_new = margin_loss(logits_new, y)

        delta = loss_new - best
        margin_history.append(loss_new.detach().clone())

        if prev_delta is not None:
            sign_flip = ((delta * prev_delta) < 0).float().sum().item()
            delta_sign_flips += sign_flip
        prev_delta = delta.detach().clone()

        improved = loss_new < best
        if improved.any():
            accepted_steps += improved.sum().item()
            improvement_sum += (best[improved] - loss_new[improved]).sum().item()
            x_adv[improved] = x_new[improved]
            best[improved] = loss_new[improved]

        score_shift_sum += aux_new["score_shift_l1"].sum().item()
        cross_view_std_sum += aux_new["cross_view_std"].sum().item()

    margin_hist = torch.stack(margin_history, dim=0)
    margin_deltas = margin_hist[1:] - margin_hist[:-1]

    log = {
        "accepted_step_ratio": accepted_steps / max(total_steps, 1),
        "mean_accepted_improvement": improvement_sum / max(accepted_steps, 1),
        "margin_trend_std": margin_deltas.std().item(),
        "margin_sign_flip_ratio": delta_sign_flips / max((attack_cfg.n_iters - 1) * B, 1),
        "avg_score_shift_l1": score_shift_sum / max((attack_cfg.n_iters + 1) * B, 1),
        "avg_cross_view_std": cross_view_std_sum / max((attack_cfg.n_iters + 1) * B, 1),
        "final_margin_mean": best.mean().item(),
    }

    return x_adv, log


# =========================================================
# Eval
# =========================================================
@torch.inference_mode()
def eval_defense_family(
    name: str,
    ds,
    clip_model,
    device: str,
    text_features: torch.Tensor,
    defense_list: List[DefenseConfig],
    attack_cfg: SquareAttackConfig,
    batch_size: int = 8,
    num_workers: int = 4,
    subset_size: int = 200,
    subset_seed: int = 0,
):
    print(f"\n===== {name} | subset={subset_size} | attack=adaptive square | single-view only =====")
    print(f"Final clean/robust accuracy: EOT-averaged defended accuracy")
    print(f"EOT includes random shaping: YES")

    loader = make_subset_loader(
        ds=ds,
        batch_size=batch_size,
        num_workers=num_workers,
        subset_size=subset_size,
        seed=subset_seed,
    )

    model = CLIPZeroShot(clip_model, text_features, device).to(device).eval().float()

    total = 0
    stats: Dict[str, Dict[str, float]] = {}

    for dcfg in defense_list:
        stats[dcfg.name] = {
            "clean_correct": 0.0,
            "robust_correct": 0.0,
            "clean_ok_count": 0.0,
            "asr_num": 0.0,
            "accepted_step_ratio_sum": 0.0,
            "mean_accepted_improvement_sum": 0.0,
            "margin_trend_std_sum": 0.0,
            "margin_sign_flip_ratio_sum": 0.0,
            "avg_score_shift_l1_sum": 0.0,
            "avg_cross_view_std_sum": 0.0,
            "final_margin_mean_sum": 0.0,
            "num_batches": 0.0,
        }

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"{name}", ncols=120)):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        total += labels.numel()

        for dcfg in defense_list:
            clean_logits_eval, _ = defended_forward_for_attacker(
                model=model,
                x=images,
                use_random_shaping=dcfg.use_random_shaping,
                shaping_family=list(dcfg.shaping_family),
                shaping_topk=dcfg.shaping_topk,
                eot_M=attack_cfg.eot_M,
            )

            pred_clean = clean_logits_eval.argmax(dim=1)
            clean_ok = (pred_clean == labels)
            stats[dcfg.name]["clean_correct"] += clean_ok.sum().item()
            stats[dcfg.name]["clean_ok_count"] += clean_ok.sum().item()

            cfg_attack = replace(attack_cfg, seed=attack_cfg.seed + batch_idx)

            x_adv, attack_log = square_attack_with_logging(
                model=model,
                x=images,
                y=labels,
                attack_cfg=cfg_attack,
                defense_cfg=dcfg,
            )

            adv_logits_eval, _ = defended_forward_for_attacker(
                model=model,
                x=x_adv,
                use_random_shaping=dcfg.use_random_shaping,
                shaping_family=list(dcfg.shaping_family),
                shaping_topk=dcfg.shaping_topk,
                eot_M=attack_cfg.eot_M,
            )
            pred_adv = adv_logits_eval.argmax(dim=1)
            adv_ok = (pred_adv == labels)
            stats[dcfg.name]["robust_correct"] += adv_ok.sum().item()
            stats[dcfg.name]["asr_num"] += ((~adv_ok) & clean_ok).sum().item()

            for k in [
                "accepted_step_ratio",
                "mean_accepted_improvement",
                "margin_trend_std",
                "margin_sign_flip_ratio",
                "avg_score_shift_l1",
                "avg_cross_view_std",
                "final_margin_mean",
            ]:
                stats[dcfg.name][f"{k}_sum"] += attack_log[k]
            stats[dcfg.name]["num_batches"] += 1.0

        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nRESULT: {name}")
    print(f"Samples: {total}")

    for dcfg in defense_list:
        val = stats[dcfg.name]
        clean_acc = val["clean_correct"] / max(total, 1)
        robust_acc = val["robust_correct"] / max(total, 1)
        asr = val["asr_num"] / (val["clean_ok_count"] + 1e-12)
        nb = max(val["num_batches"], 1.0)

        print(f"\n--- {dcfg.name} (shaping_topk={dcfg.shaping_topk}) ---")
        print(f"Clean Acc:                 {clean_acc:.4f}")
        print(f"Robust Acc:                {robust_acc:.4f}")
        print(f"ASR(clean-correct):        {asr:.4f}")
        print(f"Accepted step ratio:       {val['accepted_step_ratio_sum'] / nb:.4f}")
        print(f"Mean accepted improvement: {val['mean_accepted_improvement_sum'] / nb:.6f}")
        print(f"Margin trend std:          {val['margin_trend_std_sum'] / nb:.6f}")
        print(f"Margin sign-flip ratio:    {val['margin_sign_flip_ratio_sum'] / nb:.4f}")
        print(f"Avg score shift L1:        {val['avg_score_shift_l1_sum'] / nb:.6f}")
        print(f"Avg cross-view std:        {val['avg_cross_view_std_sum'] / nb:.6f}")
        print(f"Final margin mean:         {val['final_margin_mean_sum'] / nb:.6f}")

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
        "food101": Food101(
            root=f"{DATA_ROOT}/food101",
            split="test",
            download=True,
            transform=transform,
        ),
        "cifar100": CIFAR100(
            root=f"{DATA_ROOT}/cifar100",
            train=False,
            download=True,
            transform=transform,
        ),
        "stl10": STL10(
            root=f"{DATA_ROOT}/stl10",
            split="test",
            download=True,
            transform=transform,
        ),
        "cifar10": CIFAR10(
            root=f"{DATA_ROOT}/cifar10",
            train=False,
            download=True,
            transform=transform,
        ),
    }

    attack_cfg = SquareAttackConfig(
        eps=8 / 255,
        n_iters=200,
        eot_M=16,
        min_square=1,
        max_square=64,
        seed=0,
    )

    defense_list = [
        DefenseConfig(
            name="single",
            use_random_shaping=False,
            shaping_family=("none",),
            shaping_topk=2,
        ),

        # top2
        DefenseConfig(
            name="top2_hinge_only",
            use_random_shaping=True,
            shaping_family=("hinge_margin",),
            shaping_topk=2,
        ),
        DefenseConfig(
            name="top2_exp_only",
            use_random_shaping=True,
            shaping_family=("exp_repulsion",),
            shaping_topk=2,
        ),
        DefenseConfig(
            name="top2_random_mix",
            use_random_shaping=True,
            shaping_family=("hinge_margin", "exp_repulsion"),
            shaping_topk=2,
        ),

        # top5
        DefenseConfig(
            name="top5_hinge_only",
            use_random_shaping=True,
            shaping_family=("hinge_margin",),
            shaping_topk=5,
        ),
        DefenseConfig(
            name="top5_exp_only",
            use_random_shaping=True,
            shaping_family=("exp_repulsion",),
            shaping_topk=5,
        ),
        DefenseConfig(
            name="top5_random_mix",
            use_random_shaping=True,
            shaping_family=("hinge_margin", "exp_repulsion"),
            shaping_topk=5,
        ),
    ]

    batch_size = 64
    num_workers = 4
    subset_size = 200
    subset_seed = 0

    for name, ds in datasets.items():
        print(f"\nPreparing: {name}")
        class_names = get_class_list(name, ds)
        text_features = build_text_features(class_names, clip_model, device, dataset_name=name)

        eval_defense_family(
            name=name,
            ds=ds,
            clip_model=clip_model,
            device=device,
            text_features=text_features,
            defense_list=defense_list,
            attack_cfg=attack_cfg,
            batch_size=batch_size,
            num_workers=num_workers,
            subset_size=subset_size,
            subset_seed=subset_seed,
        )


if __name__ == "__main__":
    main()