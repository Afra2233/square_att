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
            "a photo of {}, a type of food",
            "a dish of {}",
            "a photo of a plate of {}",
            "a close-up photo of {}",
            "a photo of cooked {}",
        ]
    elif dname == "stl10":
        templates = [
            "a photo of a {}",
            "a good photo of a {}",
            "a close-up photo of a {}",
            "a blurry photo of a {}",
        ]
    else:
        templates = [
            "a photo of a {}",
            "a blurry photo of a {}",
            "a close-up photo of a {}",
            "a photo of the {}",
            "a good photo of a {}",
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
# Views
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


def get_view_list(view_mode: ViewMode, strong: bool = True) -> List[ViewType]:
    if view_mode == "deterministic":
        if strong:
            return [
                "identity",
                "horizontal_flip",
                "resize_pad_96",
                "center_crop_96",
                "rotation_fixed_plus8",
                "blur_fixed_sigma_0p6",
            ]
        return [
            "identity",
            "horizontal_flip",
            "resize_pad_96",
            "center_crop_96",
        ]

    elif view_mode == "stochastic":
        if strong:
            return [
                "identity",
                "horizontal_flip",
                "resize_pad_96",
                "center_crop_96",
                "rotation_rand_8",
                "blur_rand_light",
            ]
        return [
            "identity",
            "horizontal_flip",
            "resize_pad_96",
            "center_crop_96",
        ]

    else:
        raise ValueError(f"Unknown view_mode: {view_mode}")


def is_deterministic_view_list(view_list: List[ViewType]) -> bool:
    stochastic_views = {"rotation_rand_8", "blur_rand_light"}
    return all(v not in stochastic_views for v in view_list)


# =========================================================
# Aggregation
# =========================================================
AggType = Literal["single", "avg_logits", "avg_features"]


@torch.inference_mode()
def collect_view_logits_and_features(
    model: CLIPZeroShot,
    x: torch.Tensor,
    view_list: List[ViewType],
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = []
    logits = []
    for vt in view_list:
        xv = apply_view_batch(x, vt)
        f = model.encode_image_features(xv)
        z = model.logits_from_features(f)
        feats.append(f)
        logits.append(z)
    feats = torch.stack(feats, dim=1)    # (B,K,D)
    logits = torch.stack(logits, dim=1)  # (B,K,C)
    return feats, logits


@torch.inference_mode()
def aggregate_from_view_cache(
    model: CLIPZeroShot,
    feats_bkd: torch.Tensor,
    logits_bkc: torch.Tensor,
    agg_type: AggType,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if agg_type == "avg_logits":
        logits = logits_bkc.mean(dim=1)
        cross_view_std = logits_bkc.std(dim=1).mean(dim=1)
        return logits, cross_view_std

    elif agg_type == "avg_features":
        f = feats_bkd.mean(dim=1)
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        logits = model.logits_from_features(f)
        cross_view_std = logits_bkc.std(dim=1).mean(dim=1)
        return logits, cross_view_std

    else:
        raise ValueError(f"Unsupported agg_type for view cache: {agg_type}")


# =========================================================
# AAA-inspired random shaping family
# =========================================================
ShapingType = Literal[
    "none",
    "linear",
    "sine",
    "competitor_drop",

]


@torch.inference_mode()
def margin_to_target_linear(margin: torch.Tensor, tau: float, alpha: float) -> torch.Tensor:
    attractor = (torch.floor(margin / tau) + 0.5) * tau
    return attractor - alpha * (margin - attractor)


@torch.inference_mode()
def margin_to_target_sine(margin: torch.Tensor, tau: float, alpha: float) -> torch.Tensor:
    attractor = (torch.floor(margin / tau) + 0.5) * tau
    return margin - alpha * tau * torch.sin(np.pi * (1.0 - 2.0 * (margin - attractor) / tau))


@torch.inference_mode()
def random_shape_logits(
    logits: torch.Tensor,
    shaping: ShapingType,
    tau: float = 6.0,
    alpha: float = 0.7,
    competitor_scale: float = 0.15,
) -> torch.Tensor:
    if shaping == "none":
        return logits

    if shaping not in {"linear", "sine", "competitor_drop"}:
        raise ValueError(f"Unsupported non-semantic shaping: {shaping}")

    B, C = logits.shape
    z = logits.clone()

    top2v, top2i = logits.topk(k=2, dim=1)
    top1_idx = top2i[:, 0]
    top2_idx = top2i[:, 1]
    top1_val = top2v[:, 0]
    top2_val = top2v[:, 1]

    # top2v: 每张图最大的两个 logit 值
    # top2i: 这两个值对应的类别下标
    # top1_idx: 当前预测类别
    # top2_idx: 当前第二名类别
    # top1_val: 第一名 logit
    # top2_val: 第二名 logit

    margin = top1_val - top2_val

    if shaping == "linear":
        target_margin = margin_to_target_linear(margin, tau=tau, alpha=alpha)
        delta = (target_margin - margin).clamp(min=-0.8 * margin, max=0.8 * margin)
        z.scatter_(1, top2_idx.view(-1, 1), (top2_val - delta).view(-1, 1))
        return z

    elif shaping == "sine":
        target_margin = margin_to_target_sine(margin, tau=tau, alpha=alpha)
        delta = (target_margin - margin).clamp(min=-0.8 * margin, max=0.8 * margin)
        z.scatter_(1, top2_idx.view(-1, 1), (top2_val - delta).view(-1, 1))
        return z

    elif shaping == "competitor_drop":
        topkv, topki = logits.topk(k=min(5, C), dim=1)
        for b in range(B):
            cands = topki[b, 1:].tolist()
            for c in cands:
                drop = competitor_scale * abs(logits[b, c].item() - top1_val[b].item())
                z[b, c] = logits[b, c] - drop
        return z

    else:
        raise ValueError(f"Unknown shaping: {shaping}")


@torch.inference_mode()
def semantic_random_shape_logits(
    logits: torch.Tensor,
    text_features: torch.Tensor,
    shaping: ShapingType,
    tau: float = 6.0,
    alpha: float = 0.5,
    competitor_scale: float = 0.10,
    semantic_topk: int = 5,
    semantic_thresh: float = 0.30,
    semantic_weighted: bool = True,
) -> torch.Tensor:
    """
    CLIP-aware semantic AAA-style shaping.
    Only shape competitors that are:
      1) in current top-k competitors
      2) semantically close to current top1 in text embedding space

    logits: (B, C)
    text_features: (C, D), assumed normalized
    """
    if shaping == "none":
        return logits

    allowed = {"semantic_linear", "semantic_sine", "semantic_competitor_drop"}
    if shaping not in allowed:
        raise ValueError(f"Unsupported semantic shaping: {shaping}")

    B, C = logits.shape
    z = logits.clone()

    # text_features already normalized in build_text_features()
    text_sim = text_features @ text_features.T  # (C, C)

    topkv, topki = logits.topk(k=min(semantic_topk, C), dim=1)
    top1_idx = topki[:, 0]
    top1_val = topkv[:, 0]

    for b in range(B):
        c_star = top1_idx[b].item()
        candidates = topki[b, 1:].tolist()  # top-k competitors excluding top1

        selected = []
        selected_sims = []

        for c in candidates:
            sim = text_sim[c_star, c].item()
            if sim >= semantic_thresh:
                selected.append(c)
                selected_sims.append(sim)

        # fallback to current top2 if nothing passes semantic filter
        if len(selected) == 0 and len(candidates) > 0:
            fallback = candidates[0]
            selected = [fallback]
            selected_sims = [max(text_sim[c_star, fallback].item(), 0.0)]

        if len(selected) == 0:
            continue

        z1 = top1_val[b]

        for c, sim in zip(selected, selected_sims):
            zc = logits[b, c]
            margin = (z1 - zc).view(1)

            # stronger shaping for semantically closer competitors
            w = max(sim, 0.0) if semantic_weighted else 1.0

            if shaping == "semantic_linear":
                target_margin = margin_to_target_linear(margin, tau=tau, alpha=alpha)
                delta = (target_margin - margin).clamp(
                    min=-0.8 * margin,
                    max=0.8 * margin,
                )
                # correct sign:
                # z'_c = z_c - (target_margin - margin)
                z[b, c] = zc - w * delta.item()

            elif shaping == "semantic_sine":
                target_margin = margin_to_target_sine(margin, tau=tau, alpha=alpha)
                delta = (target_margin - margin).clamp(
                    min=-0.8 * margin,
                    max=0.8 * margin,
                )
                z[b, c] = zc - w * delta.item()

            elif shaping == "semantic_competitor_drop":
                drop = competitor_scale * w * abs(zc.item() - z1.item())
                z[b, c] = zc - drop

    return z


@torch.inference_mode()
def sample_random_shaping(
    enable_random_shaping: bool,
    family: Optional[List[ShapingType]] = None,
) -> ShapingType:
    if not enable_random_shaping:
        return "none"
    family = family or ["semantic_linear", "semantic_sine", "semantic_competitor_drop"]
    return random.choice(family)


# =========================================================
# Unified defended prediction
# =========================================================
@torch.inference_mode()
def defended_predict(
    model: CLIPZeroShot,
    x: torch.Tensor,
    use_multiview: bool,
    agg_type: AggType,
    view_list: List[ViewType],
    use_random_shaping: bool,
    shaping_family: Optional[List[ShapingType]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    aux = {}

    if not use_multiview:
        feats = model.encode_image_features(x)
        logits = model.logits_from_features(feats)
        aux["cross_view_std"] = torch.zeros(x.size(0), device=x.device)

    else:
        feats_bkd, logits_bkc = collect_view_logits_and_features(model, x, view_list)
        logits, cross_view_std = aggregate_from_view_cache(model, feats_bkd, logits_bkc, agg_type)
        aux["cross_view_std"] = cross_view_std

    shaping = sample_random_shaping(use_random_shaping, shaping_family)
    logits_before_shape = logits.clone()

    if shaping.startswith("semantic_"):
        logits = semantic_random_shape_logits(
            logits=logits,
            text_features=model.text_features,
            shaping=shaping,
            tau=6.0,
            alpha=0.5,
            competitor_scale=0.10,
            semantic_topk=5,
            semantic_thresh=0.30,
            semantic_weighted=True,
        )
    else:
        logits = random_shape_logits(
            logits=logits,
            shaping=shaping,
        )

    shaping_id_map = {
        "none": 0,
        "linear": 1,
        "sine": 2,
        "competitor_drop": 3,
        "semantic_linear": 4,
        "semantic_sine": 5,
        "semantic_competitor_drop": 6,
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
# Fast attacker-side forward with EOT
# =========================================================
@torch.no_grad()
def defended_forward_for_attacker(
    model: CLIPZeroShot,
    x: torch.Tensor,
    use_multiview: bool,
    agg_type: AggType,
    view_list: List[ViewType],
    use_random_shaping: bool,
    shaping_family: Optional[List[ShapingType]],
    eot_M: int = 1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    M = max(1, eot_M)
    deterministic_views = is_deterministic_view_list(view_list)

    logits_sum = None
    score_shift_sum = None
    cross_view_std_sum = None

    if not use_multiview:
        base_logits = model(x) #(B,C)
        base_cross_view_std = torch.zeros(x.size(0), device=x.device)#这里是在构造一个全 0 的张量，长度等于 batch size
        #形状为（B,）,因为当前分支是 不使用 multiview
        for _ in range(M):#EOT循环
            #每次随机选的 shaping family 成员可能不同，所以攻击者不能只看一次输出，而要看：多次随机防御下输出的平均值
            shaping = sample_random_shaping(use_random_shaping, shaping_family)
            # 如果 use_random_shaping=False
            # 返回 "none"
            # 如果 use_random_shaping=True
            # 从 shaping_family 里随机选一个

            if shaping.startswith("semantic_"):
                shaped_logits = semantic_random_shape_logits(
                    logits=base_logits,
                    # 每一轮 shaping 都是基于 同一个 base_logits 做的，不是上一轮的结果继续 shaping。
                    text_features=model.text_features,
                    shaping=shaping,
                    tau=6.0,
                    alpha=0.5,
                    competitor_scale=0.10,
                    semantic_topk=5,
                    semantic_thresh=0.30,
                    semantic_weighted=True,
                )
            else:
                shaped_logits = random_shape_logits(base_logits, shaping=shaping)

            probs_before = base_logits.softmax(dim=-1) #把原始 logits 转成概率分布。
            probs_after = shaped_logits.softmax(dim=-1)# 同理，把 shaping 后的 logits 转成概率分布。
            score_shift = (probs_after - probs_before).abs().sum(dim=1) #score_shift.shape == (B,)
            #得到每个类别概率的变化量,取绝对值，不管增大还是减小都算变化,对类别维求和，得到每张图总共改动了多少。
            # 如果某张图：
            # shaping 前后概率变化很小
            # score_shift 小
            # shaping 前后概率变化很大
            # score_shift 大

            logits_sum = shaped_logits if logits_sum is None else (logits_sum + shaped_logits)#每轮的 shaped_logits 累加起来
            score_shift_sum = score_shift if score_shift_sum is None else (score_shift_sum + score_shift)#每张图在 M 次 shaping 下的 score_shift 总和
            cross_view_std_sum = base_cross_view_std if cross_view_std_sum is None else (cross_view_std_sum + base_cross_view_std)
            # 由于当前分支没有 multiview，所以 base_cross_view_std 永远是 0

        out_aux = {
            "score_shift_l1": score_shift_sum / float(M),
            "cross_view_std": cross_view_std_sum / float(M),
        }
            #         这里构造一个辅助字典。
            # "score_shift_l1"
            # 表示：
            # M 次随机 shaping 后，平均每张图的概率分布变化量
            # 形状是 (B,)
        return logits_sum / float(M), out_aux
        # 返回 M 次随机 shaping 后的平均 logits,同时记录 shaping 引起的概率变化强度。

    if deterministic_views:
        feats_bkd, logits_bkc = collect_view_logits_and_features(model, x, view_list)
        # 对 x 的每个 view 都跑一遍模型，会生成对应随机变换各生成一个view，然后算对所有类别的 logits z
        #feats_bkd =(B, K, D)    logits_bkc =(B, K, C)
        base_logits, base_cross_view_std = aggregate_from_view_cache(model, feats_bkd, logits_bkc, agg_type)
        #这里是把刚才 K 个视角的结果聚合起来，得到一个最终的多视角输出,得到的mean logits按 agg——type来。
        # base_cross_view_std （B,）每张图在不同 view 下，logits 波动有多大(这个值越大,这张图在不同 view 下预测更不稳定)
        for _ in range(M):
            shaping = sample_random_shaping(use_random_shaping, shaping_family)

            if shaping.startswith("semantic_"):
                shaped_logits = semantic_random_shape_logits(
                    logits=base_logits,
                    text_features=model.text_features,
                    shaping=shaping,
                    tau=6.0,
                    alpha=0.5,
                    competitor_scale=0.10,
                    semantic_topk=5,
                    semantic_thresh=0.30,
                    semantic_weighted=True,
                )
            else:
                shaped_logits = random_shape_logits(base_logits, shaping=shaping)

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

    for _ in range(M):
        _, logits, aux = defended_predict(
            model=model,
            x=x,
            use_multiview=use_multiview,
            agg_type=agg_type,
            view_list=view_list,
            use_random_shaping=use_random_shaping,
            shaping_family=shaping_family,
        )
        logits_sum = logits if logits_sum is None else (logits_sum + logits)
        score_shift_sum = aux["score_shift_l1"] if score_shift_sum is None else (score_shift_sum + aux["score_shift_l1"])
        cross_view_std_sum = aux["cross_view_std"] if cross_view_std_sum is None else (cross_view_std_sum + aux["cross_view_std"])

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
    n_iters: int = 200 #对每张图都进行200次patch生成，每次iter里都包含m个eot，最后200次后输出结果为那张图片的最终x_adv
    eot_M: int = 8
    min_square: int = 1
    max_square: int = 64
    seed: int = 0


@dataclass
class DefenseConfig:
    name: str
    use_multiview: bool
    agg_type: AggType
    use_random_shaping: bool
    shaping_family: Tuple[ShapingType, ...]


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
    view_list: List[ViewType],
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
        use_multiview=defense_cfg.use_multiview,
        agg_type=defense_cfg.agg_type,
        view_list=view_list,
        use_random_shaping=defense_cfg.use_random_shaping,
        shaping_family=list(defense_cfg.shaping_family),
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
            use_multiview=defense_cfg.use_multiview,
            agg_type=defense_cfg.agg_type,
            view_list=view_list,
            use_random_shaping=defense_cfg.use_random_shaping,
            shaping_family=list(defense_cfg.shaping_family),
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
    view_mode: ViewMode,
    strong_views: bool = True,
    batch_size: int = 8,
    num_workers: int = 4,
    subset_size: int = 1000,
    subset_seed: int = 0,
):
    view_list = get_view_list(view_mode=view_mode, strong=strong_views)

    print(f"\n===== {name} | subset={subset_size} | attack=adaptive square | view_mode={view_mode} =====")
    print(f"Views: {view_list}")
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

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"{name}-{view_mode}", ncols=120)):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        total += labels.numel()

        for dcfg in defense_list:
            clean_logits_eval, _ = defended_forward_for_attacker(
                model=model,
                x=images,
                use_multiview=dcfg.use_multiview,
                agg_type=dcfg.agg_type,
                view_list=view_list,
                use_random_shaping=dcfg.use_random_shaping,
                shaping_family=list(dcfg.shaping_family),
                eot_M=attack_cfg.eot_M,
            )
            # 你前面刚随机初始化了一个 x_adv，现在要先看看：
            # 这个初始扰动下模型输出是什么
            # 当前 loss（margin）是多少
            # 要想知道后面新提议的 x_new 有没有更好，必须先有一个“当前最好”的基准

            pred_clean = clean_logits_eval.argmax(dim=1)
            clean_ok = (pred_clean == labels)
            # 这里是在把预测标签和真实标签逐个比较。
            # 比较后得到一个布尔张量 （B,）
            stats[dcfg.name]["clean_correct"] += clean_ok.sum().item()
            # 这里是在累计“干净样本上预测正确的总数”。
            stats[dcfg.name]["clean_ok_count"] += clean_ok.sum().item()
            # clean_correct 和clean_ok_count是在算同样的东西，clean_correct 用来算 clean accuracy，clean_ok_count 是拿来算 ASR（attack success rate） 的分母。

            cfg_attack = replace(attack_cfg, seed=attack_cfg.seed + batch_idx)

            # 用 square attack 不断修改图片，最后产出一个“尽量能骗过模型”的 x_adv，
            # 它内部需要 defended_forward_for_attacker，来知道知道“新的扰动是不是更好”
            # 它会返回最终找到的对抗样本 x_adv，一些攻击过程统计信息 attack_log
            # 没有返回最终 x_adv 上的 logits / prediction
            x_adv, attack_log = square_attack_with_logging(
                model=model,
                x=images,
                y=labels,
                attack_cfg=cfg_attack,
                defense_cfg=dcfg,
                view_list=view_list,
            )
            # 好了，攻击已经结束了。现在拿最终得到的 x_adv 再认真测一次，看看模型到底有没有被攻破。
            adv_logits_eval, _ = defended_forward_for_attacker(
                model=model,
                x=x_adv,
                use_multiview=dcfg.use_multiview,
                agg_type=dcfg.agg_type,
                view_list=view_list,
                use_random_shaping=dcfg.use_random_shaping,
                shaping_family=list(dcfg.shaping_family),
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

    print(f"\nRESULT: {name} | view_mode={view_mode}")
    print(f"Samples: {total}")

    for dcfg in defense_list:
        val = stats[dcfg.name]
        clean_acc = val["clean_correct"] / max(total, 1)
        robust_acc = val["robust_correct"] / max(total, 1)
        asr = val["asr_num"] / (val["clean_ok_count"] + 1e-12)
        nb = max(val["num_batches"], 1.0)

        print(f"\n--- {dcfg.name} ---")
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
        # "cifar10": CIFAR10(
        #     root=f"{DATA_ROOT}/cifar10",
        #     train=False,
        #     download=True,
        #     transform=transform,
        # ),
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
            use_multiview=False,
            agg_type="single",
            use_random_shaping=False,
            shaping_family=("none",),
        ),
        DefenseConfig(
            name="multiview_only",
            use_multiview=True,
            agg_type="avg_logits",
            use_random_shaping=False,
            shaping_family=("none",),
        ),
        DefenseConfig(
            name="random_shaping_only",
            use_multiview=False,
            agg_type="single",
            use_random_shaping=True,
            # shaping_family=("linear", "sine", "competitor_drop"),
            shaping_family=("linear", "sine"),
        ),
        DefenseConfig(
            name="multiview_plus_random_shaping",
            use_multiview=True,
            agg_type="avg_logits",
            use_random_shaping=True,
            # shaping_family=("linear", "sine", "competitor_drop"),
            shaping_family=("linear", "sine"),
        ),
        # DefenseConfig(
        #     name="semantic_random_shaping_only",
        #     use_multiview=False,
        #     agg_type="single",
        #     use_random_shaping=True,
        #     shaping_family=("semantic_linear", "semantic_sine", "semantic_competitor_drop"),
        # ),
        # DefenseConfig(
        #     name="multiview_plus_semantic_random_shaping",
        #     use_multiview=True,
        #     agg_type="avg_logits",
        #     use_random_shaping=True,
        #     shaping_family=("semantic_linear", "semantic_sine", "semantic_competitor_drop"),
        # ),
    ]

    batch_size = 64
    num_workers = 4
    subset_size = 1000
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
            view_mode="deterministic",
            strong_views=True,
            batch_size=batch_size,
            num_workers=num_workers,
            subset_size=subset_size,
            subset_seed=subset_seed,
        )

        eval_defense_family(
            name=name,
            ds=ds,
            clip_model=clip_model,
            device=device,
            text_features=text_features,
            defense_list=defense_list,
            attack_cfg=attack_cfg,
            view_mode="stochastic",
            strong_views=True,
            batch_size=batch_size,
            num_workers=num_workers,
            subset_size=subset_size,
            subset_seed=subset_seed,
        )


if __name__ == "__main__":
    main()


# accepted_step_ratio

# 被接受的更新比例
# 越高说明攻击更容易找到有效方向

# mean_accepted_improvement

# 每次成功更新平均带来多大 loss 改善

# margin_trend_std

# margin 变化的波动程度

# margin_sign_flip_ratio

# 相邻 step 的改善方向是否频繁翻转
# 越高可能说明防御导致攻击优化更不稳定

# avg_score_shift_l1

# 防御 shaping 前后输出分布变化量

# avg_cross_view_std

# 不同视图预测差异

# final_margin_mean

# 攻击结束时的平均 margin


# import os
# import random
# import numpy as np
# from dataclasses import dataclass, replace
# from typing import Literal, Tuple, Dict, List, Optional

# from tqdm import tqdm

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, SubsetRandomSampler

# from torchvision import transforms
# from torchvision.transforms import InterpolationMode
# from torchvision.transforms import functional as TF
# from torchvision.datasets import CIFAR10, CIFAR100, Food101, STL10

# import clip


# # =========================================================
# # Utils
# # =========================================================
# def set_seed(seed: int = 0):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def get_class_list(name, ds):
#     if hasattr(ds, "classes") and ds.classes is not None:
#         return ds.classes

#     name = name.lower()
#     if name == "stl10":
#         return [
#             "airplane", "bird", "car", "cat", "deer",
#             "dog", "horse", "monkey", "ship", "truck"
#         ]

#     raise RuntimeError(f"[FATAL] Dataset {name} has no usable class names.")


# def make_subset_loader(
#     ds,
#     batch_size: int,
#     num_workers: int,
#     subset_size: int,
#     seed: int,
# ) -> DataLoader:
#     n = len(ds)
#     g = torch.Generator()
#     g.manual_seed(seed)
#     perm = torch.randperm(n, generator=g).tolist()
#     idx = perm[: min(subset_size, n)]
#     sampler = SubsetRandomSampler(idx)

#     loader = DataLoader(
#         ds,
#         batch_size=batch_size,
#         sampler=sampler,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=False,
#     )
#     return loader


# # =========================================================
# # Build CLIP Text Features
# # =========================================================
# @torch.inference_mode()
# def build_text_features(class_names, clip_model, device, dataset_name: str):
#     dname = dataset_name.lower()

#     if dname == "food101":
#         templates = [
#             "a photo of {}, a type of food",
#             "a dish of {}",
#             "a photo of a plate of {}",
#             "a close-up photo of {}",
#             "a photo of cooked {}",
#         ]
#     elif dname == "stl10":
#         templates = [
#             "a photo of a {}",
#             "a good photo of a {}",
#             "a close-up photo of a {}",
#             "a blurry photo of a {}",
#         ]
#     else:
#         templates = [
#             "a photo of a {}",
#             "a blurry photo of a {}",
#             "a close-up photo of a {}",
#             "a photo of the {}",
#             "a good photo of a {}",
#         ]

#     all_text_features = []
#     for c in class_names:
#         name = c.replace("_", " ")
#         prompts = [t.format(name) for t in templates]
#         tokens = clip.tokenize(prompts).to(device)
#         text_feats = clip_model.encode_text(tokens)
#         text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
#         text_feat = text_feats.mean(dim=0)
#         text_feat = text_feat / text_feat.norm()
#         all_text_features.append(text_feat)

#     text_features = torch.stack(all_text_features, dim=0)
#     return text_features


# # =========================================================
# # CLIP Zero-shot Classifier
# # =========================================================
# class CLIPZeroShot(nn.Module):
#     def __init__(self, clip_model, text_features, device):
#         super().__init__()
#         self.clip_model = clip_model
#         self.text_features = text_features.to(device)

#         mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
#         std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)
#         self.register_buffer("mean", mean.to(device))
#         self.register_buffer("std", std.to(device))

#     def preprocess(self, x: torch.Tensor) -> torch.Tensor:
#         return (x - self.mean) / self.std

#     def encode_image_features(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.preprocess(x)
#         f = self.clip_model.encode_image(x)
#         f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
#         return f

#     def logits_from_features(self, f: torch.Tensor) -> torch.Tensor:
#         return 100.0 * (f @ self.text_features.T)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         f = self.encode_image_features(x)
#         return self.logits_from_features(f)


# # =========================================================
# # Views
# # =========================================================
# ViewType = Literal[
#     "identity",
#     "horizontal_flip",
#     "resize_pad_96",
#     "center_crop_96",
#     "rotation_fixed_plus8",
#     "blur_fixed_sigma_0p6",
#     "rotation_rand_8",
#     "blur_rand_light",
# ]

# ViewMode = Literal["deterministic", "stochastic"]


# @torch.inference_mode()
# def apply_view_batch(x: torch.Tensor, view_t: ViewType) -> torch.Tensor:
#     if view_t == "identity":
#         return x

#     B, C, H, W = x.shape
#     out = []

#     for i in range(B):
#         xi = x[i]

#         if view_t == "horizontal_flip":
#             xo = TF.hflip(xi)

#         elif view_t == "resize_pad_96":
#             new_h = max(1, int(round(H * 0.96)))
#             new_w = max(1, int(round(W * 0.96)))
#             resized = TF.resize(
#                 xi,
#                 size=[new_h, new_w],
#                 interpolation=InterpolationMode.BILINEAR,
#                 antialias=True,
#             )
#             pad_top = (H - new_h) // 2
#             pad_bottom = H - new_h - pad_top
#             pad_left = (W - new_w) // 2
#             pad_right = W - new_w - pad_left
#             xo = TF.pad(
#                 resized,
#                 [pad_left, pad_top, pad_right, pad_bottom],
#                 fill=float(xi.mean()),
#             )

#         elif view_t == "center_crop_96":
#             crop_h = max(1, int(round(H * 0.96)))
#             crop_w = max(1, int(round(W * 0.96)))
#             top = (H - crop_h) // 2
#             left = (W - crop_w) // 2
#             cropped = xi[:, top:top + crop_h, left:left + crop_w]
#             xo = TF.resize(
#                 cropped,
#                 size=[H, W],
#                 interpolation=InterpolationMode.BILINEAR,
#                 antialias=True,
#             )

#         elif view_t == "rotation_fixed_plus8":
#             xo = TF.rotate(
#                 xi,
#                 angle=8.0,
#                 interpolation=InterpolationMode.BILINEAR,
#                 expand=False,
#                 fill=float(xi.mean()),
#             )

#         elif view_t == "blur_fixed_sigma_0p6":
#             kernel_size = 3 if min(H, W) < 128 else 5
#             xo = TF.gaussian_blur(xi, kernel_size=[kernel_size, kernel_size], sigma=0.6)

#         elif view_t == "rotation_rand_8":
#             angle = random.uniform(-8.0, 8.0)
#             xo = TF.rotate(
#                 xi,
#                 angle=angle,
#                 interpolation=InterpolationMode.BILINEAR,
#                 expand=False,
#                 fill=float(xi.mean()),
#             )

#         elif view_t == "blur_rand_light":
#             kernel_size = 3 if min(H, W) < 128 else 5
#             sigma = random.uniform(0.1, 1.0)
#             xo = TF.gaussian_blur(xi, kernel_size=[kernel_size, kernel_size], sigma=sigma)

#         else:
#             raise ValueError(f"Unknown view type: {view_t}")

#         out.append(xo.clamp(0.0, 1.0))

#     return torch.stack(out, dim=0)


# def get_view_list(view_mode: ViewMode, strong: bool = True) -> List[ViewType]:
#     if view_mode == "deterministic":
#         if strong:
#             return [
#                 "identity",
#                 "horizontal_flip",
#                 "resize_pad_96",
#                 "center_crop_96",
#                 "rotation_fixed_plus8",
#                 "blur_fixed_sigma_0p6",
#             ]
#         return [
#             "identity",
#             "horizontal_flip",
#             "resize_pad_96",
#             "center_crop_96",
#         ]

#     elif view_mode == "stochastic":
#         if strong:
#             return [
#                 "identity",
#                 "horizontal_flip",
#                 "resize_pad_96",
#                 "center_crop_96",
#                 "rotation_rand_8",
#                 "blur_rand_light",
#             ]
#         return [
#             "identity",
#             "horizontal_flip",
#             "resize_pad_96",
#             "center_crop_96",
#         ]

#     else:
#         raise ValueError(f"Unknown view_mode: {view_mode}")


# def is_deterministic_view_list(view_list: List[ViewType]) -> bool:
#     stochastic_views = {"rotation_rand_8", "blur_rand_light"}
#     return all(v not in stochastic_views for v in view_list)


# # =========================================================
# # Aggregation
# # =========================================================
# AggType = Literal["single", "avg_logits", "avg_features"]


# @torch.inference_mode()
# def collect_view_logits_and_features(
#     model: CLIPZeroShot,
#     x: torch.Tensor,
#     view_list: List[ViewType],
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     feats = []
#     logits = []
#     for vt in view_list:
#         xv = apply_view_batch(x, vt)
#         f = model.encode_image_features(xv)
#         z = model.logits_from_features(f)
#         feats.append(f)
#         logits.append(z)
#     feats = torch.stack(feats, dim=1)    # (B,K,D)
#     logits = torch.stack(logits, dim=1)  # (B,K,C)
#     return feats, logits


# @torch.inference_mode()
# def aggregate_from_view_cache(
#     model: CLIPZeroShot,
#     feats_bkd: torch.Tensor,
#     logits_bkc: torch.Tensor,
#     agg_type: AggType,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     if agg_type == "avg_logits":
#         logits = logits_bkc.mean(dim=1)
#         cross_view_std = logits_bkc.std(dim=1).mean(dim=1)
#         return logits, cross_view_std

#     elif agg_type == "avg_features":
#         f = feats_bkd.mean(dim=1)
#         f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
#         logits = model.logits_from_features(f)
#         cross_view_std = logits_bkc.std(dim=1).mean(dim=1)
#         return logits, cross_view_std

#     else:
#         raise ValueError(f"Unsupported agg_type for view cache: {agg_type}")


# # =========================================================
# # AAA-inspired random shaping family
# # heuristic, not paper-exact AAA
# # =========================================================
# ShapingType = Literal["none", "linear", "sine", "competitor_drop"]


# @torch.inference_mode()
# def margin_to_target_linear(margin: torch.Tensor, tau: float, alpha: float) -> torch.Tensor:
#     attractor = (torch.floor(margin / tau) + 0.5) * tau
#     return attractor - alpha * (margin - attractor)


# @torch.inference_mode()
# def margin_to_target_sine(margin: torch.Tensor, tau: float, alpha: float) -> torch.Tensor:
#     attractor = (torch.floor(margin / tau) + 0.5) * tau
#     return margin - alpha * tau * torch.sin(np.pi * (1.0 - 2.0 * (margin - attractor) / tau))


# @torch.inference_mode()
# def random_shape_logits(
#     logits: torch.Tensor,
#     shaping: ShapingType,
#     tau: float = 6.0,
#     alpha: float = 0.7,
#     competitor_scale: float = 0.15,
# ) -> torch.Tensor:
#     if shaping == "none":
#         return logits

#     B, C = logits.shape
#     z = logits.clone()

#     top2v, top2i = logits.topk(k=2, dim=1)
#     top1_idx = top2i[:, 0]
#     top2_idx = top2i[:, 1]
#     top1_val = top2v[:, 0]
#     top2_val = top2v[:, 1]

#     margin = top1_val - top2_val

#     if shaping == "linear":
#         target_margin = margin_to_target_linear(margin, tau=tau, alpha=alpha)
#         delta = (target_margin - margin).clamp(min=-0.8 * margin, max=0.8 * margin)
#         z.scatter_(1, top2_idx.view(-1, 1), (top2_val - delta).view(-1, 1))
#         return z

#     elif shaping == "sine":
#         target_margin = margin_to_target_sine(margin, tau=tau, alpha=alpha)
#         delta = (target_margin - margin).clamp(min=-0.8 * margin, max=0.8 * margin)
#         z.scatter_(1, top2_idx.view(-1, 1), (top2_val - delta).view(-1, 1))
#         return z

#     elif shaping == "competitor_drop":
#         topkv, topki = logits.topk(k=min(5, C), dim=1)
#         for b in range(B):
#             cands = topki[b, 1:].tolist()
#             for c in cands:
#                 drop = competitor_scale * abs(logits[b, c].item() - top1_val[b].item())
#                 z[b, c] = logits[b, c] - drop
#         return z

#     else:
#         raise ValueError(f"Unknown shaping: {shaping}")


# @torch.inference_mode()
# def sample_random_shaping(
#     enable_random_shaping: bool,
#     family: Optional[List[ShapingType]] = None,
# ) -> ShapingType:
#     if not enable_random_shaping:
#         return "none"
#     family = family or ["linear", "sine", "competitor_drop"]
#     return random.choice(family)


# # =========================================================
# # Unified defended prediction
# # =========================================================
# @torch.inference_mode()
# def defended_predict(
#     model: CLIPZeroShot,
#     x: torch.Tensor,
#     use_multiview: bool,
#     agg_type: AggType,
#     view_list: List[ViewType],
#     use_random_shaping: bool,
#     shaping_family: Optional[List[ShapingType]] = None,
# ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
#     aux = {}

#     if not use_multiview:
#         feats = model.encode_image_features(x)
#         logits = model.logits_from_features(feats)
#         aux["cross_view_std"] = torch.zeros(x.size(0), device=x.device)

#     else:
#         feats_bkd, logits_bkc = collect_view_logits_and_features(model, x, view_list)
#         logits, cross_view_std = aggregate_from_view_cache(model, feats_bkd, logits_bkc, agg_type)
#         aux["cross_view_std"] = cross_view_std

#     shaping = sample_random_shaping(use_random_shaping, shaping_family)
#     logits_before_shape = logits.clone()
#     logits = random_shape_logits(logits, shaping=shaping)

#     shaping_id_map = {"none": 0, "linear": 1, "sine": 2, "competitor_drop": 3}
#     aux["shaping_id"] = torch.full(
#         (x.size(0),), shaping_id_map[shaping], device=x.device, dtype=torch.long
#     )

#     probs_before = logits_before_shape.softmax(dim=-1)
#     probs_after = logits.softmax(dim=-1)
#     aux["score_shift_l1"] = (probs_after - probs_before).abs().sum(dim=1)

#     pred = logits.argmax(dim=1)
#     return pred, logits, aux


# # =========================================================
# # Fast attacker-side forward with EOT
# # - deterministic views: cache views once, EOT only repeats shaping
# # - stochastic views: rerun full pipeline each EOT sample
# # EOT includes random shaping
# # =========================================================
# @torch.no_grad()
# def defended_forward_for_attacker(
#     model: CLIPZeroShot,
#     x: torch.Tensor,
#     use_multiview: bool,
#     agg_type: AggType,
#     view_list: List[ViewType],
#     use_random_shaping: bool,
#     shaping_family: Optional[List[ShapingType]],
#     eot_M: int = 1,
# ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#     M = max(1, eot_M)
#     deterministic_views = is_deterministic_view_list(view_list)

#     logits_sum = None
#     score_shift_sum = None
#     cross_view_std_sum = None

#     if not use_multiview:
#         base_logits = model(x)
#         base_cross_view_std = torch.zeros(x.size(0), device=x.device)

#         for _ in range(M):
#             shaping = sample_random_shaping(use_random_shaping, shaping_family)
#             shaped_logits = random_shape_logits(base_logits, shaping=shaping)

#             probs_before = base_logits.softmax(dim=-1)
#             probs_after = shaped_logits.softmax(dim=-1)
#             score_shift = (probs_after - probs_before).abs().sum(dim=1)

#             logits_sum = shaped_logits if logits_sum is None else (logits_sum + shaped_logits)
#             score_shift_sum = score_shift if score_shift_sum is None else (score_shift_sum + score_shift)
#             cross_view_std_sum = base_cross_view_std if cross_view_std_sum is None else (cross_view_std_sum + base_cross_view_std)

#         out_aux = {
#             "score_shift_l1": score_shift_sum / float(M),
#             "cross_view_std": cross_view_std_sum / float(M),
#         }
#         return logits_sum / float(M), out_aux

#     if deterministic_views:
#         feats_bkd, logits_bkc = collect_view_logits_and_features(model, x, view_list)
#         base_logits, base_cross_view_std = aggregate_from_view_cache(model, feats_bkd, logits_bkc, agg_type)

#         for _ in range(M):
#             shaping = sample_random_shaping(use_random_shaping, shaping_family)
#             shaped_logits = random_shape_logits(base_logits, shaping=shaping)

#             probs_before = base_logits.softmax(dim=-1)
#             probs_after = shaped_logits.softmax(dim=-1)
#             score_shift = (probs_after - probs_before).abs().sum(dim=1)

#             logits_sum = shaped_logits if logits_sum is None else (logits_sum + shaped_logits)
#             score_shift_sum = score_shift if score_shift_sum is None else (score_shift_sum + score_shift)
#             cross_view_std_sum = base_cross_view_std if cross_view_std_sum is None else (cross_view_std_sum + base_cross_view_std)

#         out_aux = {
#             "score_shift_l1": score_shift_sum / float(M),
#             "cross_view_std": cross_view_std_sum / float(M),
#         }
#         return logits_sum / float(M), out_aux

#     for _ in range(M):
#         _, logits, aux = defended_predict(
#             model=model,
#             x=x,
#             use_multiview=use_multiview,
#             agg_type=agg_type,
#             view_list=view_list,
#             use_random_shaping=use_random_shaping,
#             shaping_family=shaping_family,
#         )
#         logits_sum = logits if logits_sum is None else (logits_sum + logits)
#         score_shift_sum = aux["score_shift_l1"] if score_shift_sum is None else (score_shift_sum + aux["score_shift_l1"])
#         cross_view_std_sum = aux["cross_view_std"] if cross_view_std_sum is None else (cross_view_std_sum + aux["cross_view_std"])

#     out_aux = {
#         "score_shift_l1": score_shift_sum / float(M),
#         "cross_view_std": cross_view_std_sum / float(M),
#     }
#     return logits_sum / float(M), out_aux


# # =========================================================
# # Loss
# # =========================================================
# def margin_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     true = logits.gather(1, y.view(-1, 1)).squeeze(1)
#     tmp = logits.clone()
#     tmp.scatter_(1, y.view(-1, 1), -1e9)
#     other = tmp.max(dim=1).values
#     return true - other


# # =========================================================
# # Square Attack with trajectory logging
# # =========================================================
# @dataclass
# class SquareAttackConfig:
#     eps: float = 8 / 255
#     n_iters: int = 200
#     eot_M: int = 8
#     min_square: int = 1
#     max_square: int = 64
#     seed: int = 0


# @dataclass
# class DefenseConfig:
#     name: str
#     use_multiview: bool
#     agg_type: AggType
#     use_random_shaping: bool
#     shaping_family: Tuple[ShapingType, ...]


# def square_size_schedule(i: int, n_iters: int, H: int, W: int, min_s: int, max_s: int) -> int:
#     frac = 1.0 - (i / max(n_iters - 1, 1))
#     s = int(round(min_s + (max_s - min_s) * (frac ** 2)))
#     s = max(min_s, min(s, min(H, W)))
#     return s


# @torch.no_grad()
# def square_attack_with_logging(
#     model: CLIPZeroShot,
#     x: torch.Tensor,
#     y: torch.Tensor,
#     attack_cfg: SquareAttackConfig,
#     defense_cfg: DefenseConfig,
#     view_list: List[ViewType],
# ) -> Tuple[torch.Tensor, Dict[str, float]]:
#     set_seed(attack_cfg.seed)

#     B, C, H, W = x.shape
#     max_s = min(attack_cfg.max_square, H, W)

#     x_adv = x + attack_cfg.eps * torch.sign(torch.randn_like(x))
#     x_adv = torch.max(torch.min(x_adv, x + attack_cfg.eps), x - attack_cfg.eps)
#     x_adv = x_adv.clamp(0.0, 1.0)

#     logits0, aux0 = defended_forward_for_attacker(
#         model=model,
#         x=x_adv,
#         use_multiview=defense_cfg.use_multiview,
#         agg_type=defense_cfg.agg_type,
#         view_list=view_list,
#         use_random_shaping=defense_cfg.use_random_shaping,
#         shaping_family=list(defense_cfg.shaping_family),
#         eot_M=attack_cfg.eot_M,
#     )
#     best = margin_loss(logits0, y)

#     accepted_steps = 0
#     total_steps = attack_cfg.n_iters * B
#     improvement_sum = 0.0
#     score_shift_sum = aux0["score_shift_l1"].sum().item()
#     cross_view_std_sum = aux0["cross_view_std"].sum().item()

#     margin_history = [best.detach().clone()]
#     delta_sign_flips = 0
#     prev_delta = None

#     for i in range(attack_cfg.n_iters):
#         s = square_size_schedule(i, attack_cfg.n_iters, H, W, attack_cfg.min_square, max_s)
#         x_new = x_adv.clone()

#         for b in range(B):
#             top = random.randint(0, H - s) if H > s else 0
#             left = random.randint(0, W - s) if W > s else 0
#             patch_sign = 1.0 if random.random() < 0.5 else -1.0
#             patch = (x[b, :, top:top + s, left:left + s] + patch_sign * attack_cfg.eps).clamp(0.0, 1.0)
#             x_new[b, :, top:top + s, left:left + s] = patch

#         x_new = torch.max(torch.min(x_new, x + attack_cfg.eps), x - attack_cfg.eps)
#         x_new = x_new.clamp(0.0, 1.0)

#         logits_new, aux_new = defended_forward_for_attacker(
#             model=model,
#             x=x_new,
#             use_multiview=defense_cfg.use_multiview,
#             agg_type=defense_cfg.agg_type,
#             view_list=view_list,
#             use_random_shaping=defense_cfg.use_random_shaping,
#             shaping_family=list(defense_cfg.shaping_family),
#             eot_M=attack_cfg.eot_M,
#         )
#         loss_new = margin_loss(logits_new, y)

#         delta = loss_new - best
#         margin_history.append(loss_new.detach().clone())

#         if prev_delta is not None:
#             sign_flip = ((delta * prev_delta) < 0).float().sum().item()
#             delta_sign_flips += sign_flip
#         prev_delta = delta.detach().clone()

#         improved = loss_new < best
#         if improved.any():
#             accepted_steps += improved.sum().item()
#             improvement_sum += (best[improved] - loss_new[improved]).sum().item()
#             x_adv[improved] = x_new[improved]
#             best[improved] = loss_new[improved]

#         score_shift_sum += aux_new["score_shift_l1"].sum().item()
#         cross_view_std_sum += aux_new["cross_view_std"].sum().item()

#     margin_hist = torch.stack(margin_history, dim=0)
#     margin_deltas = margin_hist[1:] - margin_hist[:-1]

#     log = {
#         "accepted_step_ratio": accepted_steps / max(total_steps, 1),
#         "mean_accepted_improvement": improvement_sum / max(accepted_steps, 1),
#         "margin_trend_std": margin_deltas.std().item(),
#         "margin_sign_flip_ratio": delta_sign_flips / max((attack_cfg.n_iters - 1) * B, 1),
#         "avg_score_shift_l1": score_shift_sum / max((attack_cfg.n_iters + 1) * B, 1),
#         "avg_cross_view_std": cross_view_std_sum / max((attack_cfg.n_iters + 1) * B, 1),
#         "final_margin_mean": best.mean().item(),
#     }

#     return x_adv, log


# # =========================================================
# # Eval
# # B: final clean/robust accuracy also use EOT-averaged defended accuracy
# # =========================================================
# @torch.inference_mode()
# def eval_defense_family(
#     name: str,
#     ds,
#     clip_model,
#     device: str,
#     text_features: torch.Tensor,
#     defense_list: List[DefenseConfig],
#     attack_cfg: SquareAttackConfig,
#     view_mode: ViewMode,
#     strong_views: bool = True,
#     batch_size: int = 8,
#     num_workers: int = 4,
#     subset_size: int = 100,
#     subset_seed: int = 0,
# ):
#     view_list = get_view_list(view_mode=view_mode, strong=strong_views)

#     print(f"\n===== {name} | subset={subset_size} | attack=adaptive square | view_mode={view_mode} =====")
#     print(f"Views: {view_list}")
#     print(f"Final clean/robust accuracy: EOT-averaged defended accuracy")
#     print(f"EOT includes random shaping: YES")

#     loader = make_subset_loader(
#         ds=ds,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         subset_size=subset_size,
#         seed=subset_seed,
#     )

#     model = CLIPZeroShot(clip_model, text_features, device).to(device).eval().float()

#     total = 0
#     stats: Dict[str, Dict[str, float]] = {}

#     for dcfg in defense_list:
#         stats[dcfg.name] = {
#             "clean_correct": 0.0,
#             "robust_correct": 0.0,
#             "clean_ok_count": 0.0,
#             "asr_num": 0.0,
#             "accepted_step_ratio_sum": 0.0,
#             "mean_accepted_improvement_sum": 0.0,
#             "margin_trend_std_sum": 0.0,
#             "margin_sign_flip_ratio_sum": 0.0,
#             "avg_score_shift_l1_sum": 0.0,
#             "avg_cross_view_std_sum": 0.0,
#             "final_margin_mean_sum": 0.0,
#             "num_batches": 0.0,
#         }

#     for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"{name}-{view_mode}", ncols=120)):
#         images = images.to(device, non_blocking=True).float()
#         labels = labels.to(device, non_blocking=True).long()
#         total += labels.numel()

#         for dcfg in defense_list:
#             clean_logits_eval, _ = defended_forward_for_attacker(
#                 model=model,
#                 x=images,
#                 use_multiview=dcfg.use_multiview,
#                 agg_type=dcfg.agg_type,
#                 view_list=view_list,
#                 use_random_shaping=dcfg.use_random_shaping,
#                 shaping_family=list(dcfg.shaping_family),
#                 eot_M=attack_cfg.eot_M,
#             )
#             pred_clean = clean_logits_eval.argmax(dim=1)
#             clean_ok = (pred_clean == labels)
#             stats[dcfg.name]["clean_correct"] += clean_ok.sum().item()
#             stats[dcfg.name]["clean_ok_count"] += clean_ok.sum().item()

#             cfg_attack = replace(attack_cfg, seed=attack_cfg.seed + batch_idx)

#             x_adv, attack_log = square_attack_with_logging(
#                 model=model,
#                 x=images,
#                 y=labels,
#                 attack_cfg=cfg_attack,
#                 defense_cfg=dcfg,
#                 view_list=view_list,
#             )

#             adv_logits_eval, _ = defended_forward_for_attacker(
#                 model=model,
#                 x=x_adv,
#                 use_multiview=dcfg.use_multiview,
#                 agg_type=dcfg.agg_type,
#                 view_list=view_list,
#                 use_random_shaping=dcfg.use_random_shaping,
#                 shaping_family=list(dcfg.shaping_family),
#                 eot_M=attack_cfg.eot_M,
#             )
#             pred_adv = adv_logits_eval.argmax(dim=1)
#             adv_ok = (pred_adv == labels)
#             stats[dcfg.name]["robust_correct"] += adv_ok.sum().item()
#             stats[dcfg.name]["asr_num"] += ((~adv_ok) & clean_ok).sum().item()

#             for k in [
#                 "accepted_step_ratio",
#                 "mean_accepted_improvement",
#                 "margin_trend_std",
#                 "margin_sign_flip_ratio",
#                 "avg_score_shift_l1",
#                 "avg_cross_view_std",
#                 "final_margin_mean",
#             ]:
#                 stats[dcfg.name][f"{k}_sum"] += attack_log[k]
#             stats[dcfg.name]["num_batches"] += 1.0

#         if device == "cuda":
#             torch.cuda.empty_cache()

#     print(f"\nRESULT: {name} | view_mode={view_mode}")
#     print(f"Samples: {total}")

#     for dcfg in defense_list:
#         val = stats[dcfg.name]
#         clean_acc = val["clean_correct"] / max(total, 1)
#         robust_acc = val["robust_correct"] / max(total, 1)
#         asr = val["asr_num"] / (val["clean_ok_count"] + 1e-12)
#         nb = max(val["num_batches"], 1.0)

#         print(f"\n--- {dcfg.name} ---")
#         print(f"Clean Acc:                 {clean_acc:.4f}")
#         print(f"Robust Acc:                {robust_acc:.4f}")
#         print(f"ASR(clean-correct):        {asr:.4f}")
#         print(f"Accepted step ratio:       {val['accepted_step_ratio_sum'] / nb:.4f}")
#         print(f"Mean accepted improvement: {val['mean_accepted_improvement_sum'] / nb:.6f}")
#         print(f"Margin trend std:          {val['margin_trend_std_sum'] / nb:.6f}")
#         print(f"Margin sign-flip ratio:    {val['margin_sign_flip_ratio_sum'] / nb:.4f}")
#         print(f"Avg score shift L1:        {val['avg_score_shift_l1_sum'] / nb:.6f}")
#         print(f"Avg cross-view std:        {val['avg_cross_view_std_sum'] / nb:.6f}")
#         print(f"Final margin mean:         {val['final_margin_mean_sum'] / nb:.6f}")

#     print("=" * 110)


# # =========================================================
# # Main
# # =========================================================
# def main():
#     set_seed(0)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"[DEBUG] Device: {device}")

#     clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
#     clip_model = clip_model.eval().float()

#     transform = transforms.Compose([
#         transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
#         transforms.ToTensor(),
#     ])

#     DATA_ROOT = "data"
#     os.makedirs(DATA_ROOT, exist_ok=True)

#     datasets = {
#         "cifar10": CIFAR10(
#             root=f"{DATA_ROOT}/cifar10",
#             train=False,
#             download=True,
#             transform=transform,
#         ),
#         "cifar100": CIFAR100(
#             root=f"{DATA_ROOT}/cifar100",
#             train=False,
#             download=True,
#             transform=transform,
#         ),
#         "food101": Food101(
#             root=f"{DATA_ROOT}/food101",
#             split="test",
#             download=True,
#             transform=transform,
#         ),
#         "stl10": STL10(
#             root=f"{DATA_ROOT}/stl10",
#             split="test",
#             download=True,
#             transform=transform,
#         ),
#     }

#     attack_cfg = SquareAttackConfig(
#         eps=8 / 255,
#         n_iters=50,
#         eot_M=8,
#         min_square=1,
#         max_square=64,
#         seed=0,
#     )

#     defense_list = [
#         DefenseConfig(
#             name="single",
#             use_multiview=False,
#             agg_type="single",
#             use_random_shaping=False,
#             shaping_family=("none",),
#         ),
#         DefenseConfig(
#             name="multiview_only",
#             use_multiview=True,
#             agg_type="avg_logits",
#             use_random_shaping=False,
#             shaping_family=("none",),
#         ),
#         DefenseConfig(
#             name="random_shaping_only",
#             use_multiview=False,
#             agg_type="single",
#             use_random_shaping=True,
#             shaping_family=("linear", "sine", "competitor_drop"),
#         ),
#         DefenseConfig(
#             name="multiview_plus_random_shaping",
#             use_multiview=True,
#             agg_type="avg_logits",
#             use_random_shaping=True,
#             shaping_family=("linear", "sine", "competitor_drop"),
#         ),
#     ]

#     batch_size = 8
#     num_workers = 4
#     subset_size = 500
#     subset_seed = 0

#     for name, ds in datasets.items():
#         print(f"\nPreparing: {name}")
#         class_names = get_class_list(name, ds)
#         text_features = build_text_features(class_names, clip_model, device, dataset_name=name)

#         eval_defense_family(
#             name=name,
#             ds=ds,
#             clip_model=clip_model,
#             device=device,
#             text_features=text_features,
#             defense_list=defense_list,
#             attack_cfg=attack_cfg,
#             view_mode="deterministic",
#             strong_views=True,
#             batch_size=batch_size,
#             num_workers=num_workers,
#             subset_size=subset_size,
#             subset_seed=subset_seed,
#         )

#         eval_defense_family(
#             name=name,
#             ds=ds,
#             clip_model=clip_model,
#             device=device,
#             text_features=text_features,
#             defense_list=defense_list,
#             attack_cfg=attack_cfg,
#             view_mode="stochastic",
#             strong_views=True,
#             batch_size=batch_size,
#             num_workers=num_workers,
#             subset_size=subset_size,
#             subset_seed=subset_seed,
#         )


# if __name__ == "__main__":
#     main()