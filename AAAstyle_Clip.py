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
@torch.inference_mode()
def build_text_features(class_names, clip_model, device, dataset_name: str):
    """
    Prompt ensemble.
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
        f = f / f.norm(dim=-1, keepdim=True)
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
    "multiview",
    "smoothing",
]

ViewType = Literal[
    "identity",
    "horizontal_flip",
    "resize_pad_96",
    "center_crop_96",
]


# =========================================================
# Smoothing
# =========================================================
@torch.inference_mode()
def apply_smoothing_batch(x: torch.Tensor, sigma: float = 0.8) -> torch.Tensor:
    """
    x: (B,3,H,W) in [0,1]
    """
    B, C, H, W = x.shape
    kernel_size = 3 if min(H, W) < 128 else 5
    out = []
    for i in range(B):
        xi = x[i]
        xo = TF.gaussian_blur(
            xi,
            kernel_size=[kernel_size, kernel_size],
            sigma=sigma,
        )
        out.append(xo.clamp(0.0, 1.0))
    return torch.stack(out, dim=0)


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

        else:
            raise ValueError(f"Unknown view type: {view_t}")

        out.append(xo.clamp(0.0, 1.0))

    return torch.stack(out, dim=0)


# =========================================================
# Multi-view feature aggregation
# =========================================================
@torch.inference_mode()
def collect_multi_view_features(
    model: CLIPZeroShot,
    x: torch.Tensor,
    view_list: List[ViewType],
) -> torch.Tensor:
    """
    Returns:
        feats_bkd: (B, K, D)
    """
    feats = []
    for vt in view_list:
        xv = apply_view_batch(x, vt)
        f = model.encode_image_features(xv)
        feats.append(f)
    return torch.stack(feats, dim=1)


@torch.inference_mode()
def trimmed_mean_features(feats_bkd: torch.Tensor, trim_ratio: float = 0.25) -> torch.Tensor:
    """
    Robust feature aggregation:
    remove feature outlier views, then average.
    """
    B, K, D = feats_bkd.shape
    if K == 1 or trim_ratio <= 0.0:
        f = feats_bkd.mean(dim=1)
        return f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    center = feats_bkd.mean(dim=1)
    center = center / center.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    sims = (feats_bkd * center.unsqueeze(1)).sum(dim=-1)  # (B,K)
    k_keep = max(1, int(round(K * (1.0 - trim_ratio))))
    topk_idx = sims.topk(k_keep, dim=1).indices

    gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, D)
    kept = feats_bkd.gather(dim=1, index=gather_idx)

    f = kept.mean(dim=1)
    f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return f


# =========================================================
# AAA-style CLIP similarity shaping
# =========================================================
@torch.inference_mode()
def clip_similarity_shaping(
    logits: torch.Tensor,
    topk_competitors: int = 5,
    alpha: float = 1.5,
    tau: float = 6.0,
    preserve_top1_gap: float = 0.05,
) -> torch.Tensor:
    """
    Deterministic CLIP-specific post-process.
    Only modifies top competitors, not top1.
    """
    B, C = logits.shape
    shaped = logits.clone()

    topv, topi = logits.topk(k=min(topk_competitors + 1, C), dim=1)
    top1_idx = topi[:, 0]
    top1_val = topv[:, 0]
    second_val = topv[:, 1] if topv.size(1) > 1 else (top1_val - 1.0)
    current_gap = (top1_val - second_val).clamp_min(1e-6)

    for b in range(B):
        c1 = top1_idx[b].item()
        s1 = top1_val[b].item()
        gap = current_gap[b].item()

        comp_indices = topi[b, 1:].tolist()

        for c in comp_indices:
            sc = logits[b, c].item()
            rel = (sc - s1) / max(tau, 1e-6)
            warp = alpha * np.tanh(rel)

            # keep top1 stable by capping shift
            max_shift = preserve_top1_gap * gap
            delta = max(-max_shift, warp)

            shaped[b, c] = logits[b, c] + float(delta)

    return shaped


# =========================================================
# Unified defended forward
# =========================================================
@torch.inference_mode()
def defended_predict(
    model: CLIPZeroShot,
    x: torch.Tensor,
    preproc_mode: PreprocType,
    use_aastyle_clip: bool,
    view_list: List[ViewType],
    trim_ratio: float = 0.25,
    smoothing_sigma: float = 0.8,
    shaping_topk: int = 5,
    shaping_alpha: float = 1.5,
    shaping_tau: float = 6.0,
    preserve_top1_gap: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        pred:   (B,)
        logits: (B,C)
    """
    if preproc_mode == "none":
        f = model.encode_image_features(x)

    elif preproc_mode == "smoothing":
        xs = apply_smoothing_batch(x, sigma=smoothing_sigma)
        f = model.encode_image_features(xs)

    elif preproc_mode == "multiview":
        feats_bkd = collect_multi_view_features(model, x, view_list=view_list)
        f = trimmed_mean_features(feats_bkd, trim_ratio=trim_ratio)

    else:
        raise ValueError(f"Unknown preproc_mode: {preproc_mode}")

    logits = model.logits_from_features(f)

    if use_aastyle_clip:
        logits = clip_similarity_shaping(
            logits=logits,
            topk_competitors=shaping_topk,
            alpha=shaping_alpha,
            tau=shaping_tau,
            preserve_top1_gap=preserve_top1_gap,
        )

    pred = logits.argmax(dim=1)
    return pred, logits


# =========================================================
# Metrics
# =========================================================
@torch.inference_mode()
def top1_confidence(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    return probs.max(dim=1).values


@torch.inference_mode()
def top1_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = logits.topk(k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


# =========================================================
# Loss for attack
# =========================================================
def margin_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    tmp = logits.clone()
    tmp.scatter_(1, y.view(-1, 1), -1e9)
    other = tmp.max(dim=1).values
    return true - other


# =========================================================
# Adaptive forward for attacker
# =========================================================
@torch.no_grad()
def defended_forward_for_attacker(
    model: CLIPZeroShot,
    x: torch.Tensor,
    preproc_mode: PreprocType,
    use_aastyle_clip: bool,
    view_list: List[ViewType],
    eot_M: int = 1,
    trim_ratio: float = 0.25,
    smoothing_sigma: float = 0.8,
    shaping_topk: int = 5,
    shaping_alpha: float = 1.5,
    shaping_tau: float = 6.0,
    preserve_top1_gap: float = 0.05,
) -> torch.Tensor:
    logits_sum = None
    M = max(1, eot_M)

    for _ in range(M):
        _, logits = defended_predict(
            model=model,
            x=x,
            preproc_mode=preproc_mode,
            use_aastyle_clip=use_aastyle_clip,
            view_list=view_list,
            trim_ratio=trim_ratio,
            smoothing_sigma=smoothing_sigma,
            shaping_topk=shaping_topk,
            shaping_alpha=shaping_alpha,
            shaping_tau=shaping_tau,
            preserve_top1_gap=preserve_top1_gap,
        )
        logits_sum = logits if logits_sum is None else (logits_sum + logits)

    return logits_sum / float(M)


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
def confident_square_attack_eot(
    model: CLIPZeroShot,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: SquareAttackConfig,
    preproc_mode: PreprocType,
    use_aastyle_clip: bool,
    view_list: List[ViewType],
    trim_ratio: float = 0.25,
    smoothing_sigma: float = 0.8,
    shaping_topk: int = 5,
    shaping_alpha: float = 1.5,
    shaping_tau: float = 6.0,
    preserve_top1_gap: float = 0.05,
) -> torch.Tensor:
    set_seed(cfg.seed)

    B, C, H, W = x.shape
    max_s = min(cfg.max_square, H, W)

    x_adv = x + cfg.eps * torch.sign(torch.randn_like(x))
    x_adv = torch.max(torch.min(x_adv, x + cfg.eps), x - cfg.eps)
    x_adv = x_adv.clamp(0.0, 1.0)

    logits0 = defended_forward_for_attacker(
        model=model,
        x=x_adv,
        preproc_mode=preproc_mode,
        use_aastyle_clip=use_aastyle_clip,
        view_list=view_list,
        eot_M=cfg.eot_M,
        trim_ratio=trim_ratio,
        smoothing_sigma=smoothing_sigma,
        shaping_topk=shaping_topk,
        shaping_alpha=shaping_alpha,
        shaping_tau=shaping_tau,
        preserve_top1_gap=preserve_top1_gap,
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

        logits_new = defended_forward_for_attacker(
            model=model,
            x=x_new,
            preproc_mode=preproc_mode,
            use_aastyle_clip=use_aastyle_clip,
            view_list=view_list,
            eot_M=cfg.eot_M,
            trim_ratio=trim_ratio,
            smoothing_sigma=smoothing_sigma,
            shaping_topk=shaping_topk,
            shaping_alpha=shaping_alpha,
            shaping_tau=shaping_tau,
            preserve_top1_gap=preserve_top1_gap,
        )
        loss_new = margin_loss(logits_new, y)

        improved = loss_new < best
        if improved.any():
            x_adv[improved] = x_new[improved]
            best[improved] = loss_new[improved]

    return x_adv


# =========================================================
# Evaluation
# =========================================================
@torch.inference_mode()
def eval_all_defenses(
    name: str,
    ds,
    clip_model,
    device: str,
    text_features: torch.Tensor,
    attack_cfg: SquareAttackConfig,
    batch_size: int = 16,
    num_workers: int = 4,
    subset_size: int = 200,
    subset_seed: int = 0,
):
    print(f"\n===== {name} | subset={subset_size} =====")

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
        {"name": "single_baseline",            "preproc_mode": "none",      "use_aastyle_clip": False},
        {"name": "multiview_only",             "preproc_mode": "multiview", "use_aastyle_clip": False},
        {"name": "multiview_plus_aastyleclip", "preproc_mode": "multiview", "use_aastyle_clip": True},
        {"name": "smoothing_only",             "preproc_mode": "smoothing", "use_aastyle_clip": False},
        {"name": "smoothing_plus_aastyleclip", "preproc_mode": "smoothing", "use_aastyle_clip": True},
    ]

    total = 0
    stats: Dict[str, Dict[str, float]] = {}

    for cfg in experiment_configs:
        stats[cfg["name"]] = {
            "clean_correct": 0.0,
            "robust_correct": 0.0,
            "clean_ok_count": 0.0,
            "asr_num": 0.0,
            "clean_conf_sum": 0.0,
            "adv_conf_sum": 0.0,
            "clean_margin_sum": 0.0,
            "adv_margin_sum": 0.0,
        }

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"{name}-eval", ncols=120)):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        n = labels.numel()
        total += n

        for cfg_exp in experiment_configs:
            exp_name = cfg_exp["name"]
            preproc_mode = cfg_exp["preproc_mode"]
            use_aastyle_clip = cfg_exp["use_aastyle_clip"]

            pred_clean, logits_clean = defended_predict(
                model=model,
                x=images,
                preproc_mode=preproc_mode,
                use_aastyle_clip=use_aastyle_clip,
                view_list=fixed_views,
                trim_ratio=0.25,
                smoothing_sigma=0.8,
                shaping_topk=5,
                shaping_alpha=1.5,
                shaping_tau=6.0,
                preserve_top1_gap=0.05,
            )

            clean_ok = (pred_clean == labels)
            stats[exp_name]["clean_correct"] += clean_ok.sum().item()
            stats[exp_name]["clean_ok_count"] += clean_ok.sum().item()
            stats[exp_name]["clean_conf_sum"] += top1_confidence(logits_clean).sum().item()
            stats[exp_name]["clean_margin_sum"] += top1_margin(logits_clean).sum().item()

            cfg_attack = replace(attack_cfg, seed=attack_cfg.seed + batch_idx)

            x_adv = confident_square_attack_eot(
                model=model,
                x=images,
                y=labels,
                cfg=cfg_attack,
                preproc_mode=preproc_mode,
                use_aastyle_clip=use_aastyle_clip,
                view_list=fixed_views,
                trim_ratio=0.25,
                smoothing_sigma=0.8,
                shaping_topk=5,
                shaping_alpha=1.5,
                shaping_tau=6.0,
                preserve_top1_gap=0.05,
            )

            pred_adv, logits_adv = defended_predict(
                model=model,
                x=x_adv,
                preproc_mode=preproc_mode,
                use_aastyle_clip=use_aastyle_clip,
                view_list=fixed_views,
                trim_ratio=0.25,
                smoothing_sigma=0.8,
                shaping_topk=5,
                shaping_alpha=1.5,
                shaping_tau=6.0,
                preserve_top1_gap=0.05,
            )

            adv_ok = (pred_adv == labels)
            stats[exp_name]["robust_correct"] += adv_ok.sum().item()
            stats[exp_name]["asr_num"] += ((~adv_ok) & clean_ok).sum().item()
            stats[exp_name]["adv_conf_sum"] += top1_confidence(logits_adv).sum().item()
            stats[exp_name]["adv_margin_sum"] += top1_margin(logits_adv).sum().item()

        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nRESULT: {name}")
    print(f"Samples: {total}")

    for exp_name, val in stats.items():
        clean_acc = val["clean_correct"] / max(total, 1)
        robust_acc = val["robust_correct"] / max(total, 1)
        asr = val["asr_num"] / (val["clean_ok_count"] + 1e-12)

        clean_conf = val["clean_conf_sum"] / max(total, 1)
        adv_conf = val["adv_conf_sum"] / max(total, 1)
        clean_margin = val["clean_margin_sum"] / max(total, 1)
        adv_margin = val["adv_margin_sum"] / max(total, 1)

        print(f"\n--- {exp_name} ---")
        print(f"Clean Acc:          {clean_acc:.4f}")
        print(f"Robust Acc:         {robust_acc:.4f}")
        print(f"ASR(clean-correct): {asr:.4f}")
        print(f"Clean Top1 Conf:    {clean_conf:.4f}")
        print(f"Adv Top1 Conf:      {adv_conf:.4f}")
        print(f"Clean Top1 Margin:  {clean_margin:.4f}")
        print(f"Adv Top1 Margin:    {adv_margin:.4f}")

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

    attack_cfg = SquareAttackConfig(
        eps=8 / 255,
        n_iters=200,
        eot_M=1,      # deterministic defense
        min_square=1,
        max_square=64,
        seed=0,
    )

    batch_size = 16
    num_workers = 4
    subset_size = 1000
    subset_seed = 0

    for name, ds in datasets.items():
        print(f"\nPreparing: {name}")
        class_names = get_class_list(name, ds)
        text_features = build_text_features(class_names, clip_model, device, dataset_name=name)

        eval_all_defenses(
            name=name,
            ds=ds,
            clip_model=clip_model,
            device=device,
            text_features=text_features,
            attack_cfg=attack_cfg,
            batch_size=batch_size,
            num_workers=num_workers,
            subset_size=subset_size,
            subset_seed=subset_seed,
        )


if __name__ == "__main__":
    main()