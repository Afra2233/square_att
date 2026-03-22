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
# Randomized Transform Defense (inference-time)
# =========================================================
#TransformType = Literal["identity", "rotation_10", "crop_resize_80"] #Literal[]变量只能取固定几个value，这里指只能取"identity", "rotation_10", "crop_resize_80"
TransformType = Literal[
    "identity",
    "horizontal_flip",
    "rotation_10",
    "crop_resize_80",
    "color_jitter_light",
    "gaussian_blur_light",
    "gaussian_noise_light",
    "mixed_strong",
]

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

        if t == "horizontal_flip":
            if random.random() < 0.5:
                xo = TF.hflip(xi)
            else:
                xo = xi
        elif t == "rotation_10":
            angle = random.uniform(-10.0, 10.0)
            fill = float(xi.mean().item()) #因为图片旋转后，角落可能会空出来。这里不是填黑色 0，也不是填白色 1，而是：用这张图片的平均像素值来填充,这样通常会比纯黑边自然一些。
            xo = TF.rotate(
                xi,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR, #双线性插值，旋转更平滑
                expand=False,
                fill=fill,
            )

        elif t == "crop_resize_80":
            scale = random.uniform(0.8, 1.0)
            ch = max(1, int(round(H * scale)))
            cw = max(1, int(round(W * scale)))
            top = random.randint(0, H - ch) if H > ch else 0 #随机选择裁剪位置
            left = random.randint(0, W - cw) if W > cw else 0
            cropped = xi[:, top:top + ch, left:left + cw]
            xo = TF.resize(
                cropped,
                size=[H, W],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        elif t == "color_jitter_light":
            # 轻微亮度/对比度/饱和度变化，避免语义破坏过强
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            saturation = random.uniform(0.9, 1.1)
            hue = random.uniform(-0.03, 0.03)

            xo = TF.adjust_brightness(xi, brightness)
            xo = TF.adjust_contrast(xo, contrast)
            xo = TF.adjust_saturation(xo, saturation)
            xo = TF.adjust_hue(xo, hue)

        elif t == "gaussian_blur_light":
            # 轻模糊，破坏高频对抗扰动但不过度损伤语义
            kernel_size = 3 if min(H, W) < 128 else 5
            sigma = random.uniform(0.1, 1.0)
            xo = TF.gaussian_blur(xi, kernel_size=[kernel_size, kernel_size], sigma=sigma)
        elif t == "gaussian_noise_light":
            # 小噪声，建议控制在很轻的范围
            sigma = random.uniform(2.0 / 255.0, 6.0 / 255.0)
            noise = torch.randn_like(xi) * sigma
            xo = xi + noise

        elif t == "mixed_strong":
            # 从一组温和增强里随机采样 1~2 个串联
            xo = xi
            ops = [
                "horizontal_flip",
                "rotation_8",
                "crop_resize_85",
                "color_jitter_light",
                "gaussian_blur_light",
                "gaussian_noise_light",
            ]
            num_ops = random.choice([1, 2])
            chosen = random.sample(ops, k=num_ops)

            for op in chosen:
                if op == "horizontal_flip":
                    if random.random() < 0.5:
                        xo = TF.hflip(xo)

                elif op == "rotation_8":
                    angle = random.uniform(-8.0, 8.0)
                    fill = float(xo.mean().item())
                    xo = TF.rotate(
                        xo,
                        angle=angle,
                        interpolation=InterpolationMode.BILINEAR,
                        expand=False,
                        fill=fill,
                    )

                elif op == "crop_resize_85":
                    scale = random.uniform(0.85, 1.0)
                    ch = max(1, int(round(H * scale)))
                    cw = max(1, int(round(W * scale)))
                    top = random.randint(0, H - ch) if H > ch else 0
                    left = random.randint(0, W - cw) if W > cw else 0
                    cropped = xo[:, top:top + ch, left:left + cw]
                    xo = TF.resize(
                        cropped,
                        size=[H, W],
                        interpolation=InterpolationMode.BILINEAR,
                        antialias=True,
                    )

                elif op == "color_jitter_light":
                    brightness = random.uniform(0.9, 1.1)
                    contrast = random.uniform(0.9, 1.1)
                    saturation = random.uniform(0.9, 1.1)
                    hue = random.uniform(-0.03, 0.03)
                    xo = TF.adjust_brightness(xo, brightness)
                    xo = TF.adjust_contrast(xo, contrast)
                    xo = TF.adjust_saturation(xo, saturation)
                    xo = TF.adjust_hue(xo, hue)

                elif op == "gaussian_blur_light":
                    kernel_size = 3 if min(H, W) < 128 else 5
                    sigma = random.uniform(0.1, 1.0)
                    xo = TF.gaussian_blur(xo, kernel_size=[kernel_size, kernel_size], sigma=sigma)

                elif op == "gaussian_noise_light":
                    sigma = random.uniform(2.0 / 255.0, 6.0 / 255.0)
                    noise = torch.randn_like(xo) * sigma
                    xo = xo + noise

        else:
            raise ValueError(f"Unknown transform: {t}")

        out.append(xo.clamp(0.0, 1.0))

    return torch.stack(out, dim=0) #(B, 3, H, W)


# =========================================================
# Aggregation
# =========================================================
AggregationType = Literal[
    "single",
    "vote",
    "avg_logits",
    "avg_features",
    "semantic_weighted_logits",
    "semantic_topk_logits",
]


@torch.no_grad()
def collect_view_logits_and_features(
    model: CLIPZeroShot,
    x: torch.Tensor,
    defense_t: TransformType,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        feats:  (B,K,D), normalized
        logits: (B,K,C)
    """
    feats = []
    logits = []
    for _ in range(K):
        xt = apply_random_transform_batch(x, defense_t)
        f = model.encode_image_features(xt)
        z = model.logits_from_features(f)
        feats.append(f)
        logits.append(z)

    feats = torch.stack(feats, dim=1)    # (B,K,D)
    logits = torch.stack(logits, dim=1)  # (B,K,C)
    return feats, logits


@torch.no_grad()
def compute_semantic_logits_scores(
    logits_bkc: torch.Tensor,
    beta_entropy: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    logits_bkc: (B,K,C)

    Returns:
        scores_bk: (B,K)
        yhat:      (B,)
        conf0:     (B,)
    """
    B, K, C = logits_bkc.shape

    # first-pass pseudo-label from average logits
    logits0 = logits_bkc.mean(dim=1)            # (B,C)，对 K 个 view 的 logits 取平均
    probs0 = logits0.softmax(dim=-1)
    yhat = logits0.argmax(dim=1)                # (B,)得到一个“初步预测类别”
    conf0 = probs0.max(dim=1).values            # (B,)得到这个预测的置信度，整体来看，这张图我有多自信”

    # margin wrt pseudo-label
    yhat_idx = yhat.view(B, 1, 1).expand(-1, K, 1)  
    true_logit = logits_bkc.gather(dim=2, index=yhat_idx).squeeze(-1)  # (B,K) # 已经知道yhat是多少了，然后true_logits从logits_bkc中拿出所有那个类的logits，比如true_logit = [5, 4, 6]
    #gather 就是按照 yhat_idx 指定的类别编号，从每个 view 里取对应那一列
    tmp = logits_bkc.clone() 
    tmp.scatter_(2, yhat_idx, float("-inf")) #把 ychat第一强那列全部改成 -∞
    other_logit = tmp.max(dim=2).values3#现在没有最大的ychat了，最大的变成哥哥第二大的
    margin = true_logit - other_logit           # (B,K)

    # entropy (lower is better)
    probs_bkc = logits_bkc.softmax(dim=-1)
    entropy = -(probs_bkc * probs_bkc.clamp_min(1e-12).log()).sum(dim=-1)  # (B,K)

    # semantic-aware score based only on logits
    scores_bk = margin - beta_entropy * entropy
    return scores_bk, yhat, conf0


@torch.no_grad()
def predict_with_aggregation(
    model: CLIPZeroShot,
    x: torch.Tensor,
    defense_t: TransformType,
    aggregation: AggregationType,
    K: int = 10,
    tau: float = 0.3,
    beta_entropy: float = 0.5,
    topk_ratio: float = 0.5,
    conf_gate: float = 0.40,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        pred_final:   (B,)
        logits_final: (B,C)
    """
    if aggregation == "single" or (defense_t == "identity" and K <= 1):
        logits = model(x)
        pred = logits.argmax(dim=1)
        return pred, logits

    B = x.size(0)

    if aggregation == "vote": #vote对异常值不敏感，logits会被异常值影响，先vote，如果vote一样在看logits
        num_classes = model(x[:1]).size(1)
        votes = torch.zeros((B, num_classes), device=x.device, dtype=torch.int32) #(B, num_classes)
        logits_sum = torch.zeros((B, num_classes), device=x.device, dtype=torch.float32) #(B, num_classes)

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

    elif aggregation == "avg_features": #先把 K 次的“图像特征”平均 → 再做分类
        feat_sum = None
        for _ in range(K):
            xt = apply_random_transform_batch(x, defense_t)
            f = model.encode_image_features(xt)
            feat_sum = f if feat_sum is None else (feat_sum + f)
        f_bar = feat_sum / float(K)
        f_bar = f_bar / f_bar.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        logits_final = model.logits_from_features(f_bar)
        pred_final = logits_final.argmax(dim=1)
        return pred_final, logits_final

    elif aggregation == "semantic_weighted_logits":#所有 view 都用，但加权
        _, logits_bkc = collect_view_logits_and_features(model, x, defense_t, K)
        scores_bk, _, conf0 = compute_semantic_logits_scores(
            logits_bkc=logits_bkc,
            beta_entropy=beta_entropy,
        )

        weights_bk = torch.softmax(scores_bk / tau, dim=1)  # (B,K)
        logits_weighted = (logits_bkc * weights_bk.unsqueeze(-1)).sum(dim=1)  # (B,C)

        # fallback to avg_logits if first-pass confidence too low
        logits_avg = logits_bkc.mean(dim=1)
        use_weighted = (conf0 >= conf_gate).float().view(-1, 1) #use_weighted =0 或 1
        logits_final = use_weighted * logits_weighted + (1.0 - use_weighted) * logits_avg

        pred_final = logits_final.argmax(dim=1)
        return pred_final, logits_final

    elif aggregation == "semantic_topk_logits": #只用最好的 K 个 view，其它直接丢掉
        _, logits_bkc = collect_view_logits_and_features(model, x, defense_t, K)
        scores_bk, _, conf0 = compute_semantic_logits_scores(
            logits_bkc=logits_bkc,
            beta_entropy=beta_entropy,
        )

        k_keep = max(1, int(round(K * topk_ratio)))
        topk_idx = scores_bk.topk(k_keep, dim=1).indices  # (B,k_keep)

        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, logits_bkc.size(-1)) #把topk个 view 的 logits 取出来
        logits_topk = logits_bkc.gather(dim=1, index=gather_idx)  # (B,k_keep,C)

        logits_topk_mean = logits_topk.mean(dim=1)  # (B,C)
        logits_avg = logits_bkc.mean(dim=1)

        # fallback to avg_logits if first-pass confidence too low
        use_topk = (conf0 >= conf_gate).float().view(-1, 1) 
        logits_final = use_topk * logits_topk_mean + (1.0 - use_topk) * logits_avg

        pred_final = logits_final.argmax(dim=1)
        return pred_final, logits_final

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


# =========================================================
# Mechanism metrics: view consistency & image-text alignment stability
# =========================================================
@torch.no_grad()
def collect_view_features(
    model: CLIPZeroShot,
    x: torch.Tensor,
    defense_t: TransformType,
    K: int,
) -> torch.Tensor:
    """
    Returns:
        feats: (B, K, D), each feature L2-normalized
    """
    feats = []
    for _ in range(K):
        xt = apply_random_transform_batch(x, defense_t)
        f = model.encode_image_features(xt)
        feats.append(f)
    return torch.stack(feats, dim=1)


@torch.no_grad()
def compute_view_consistency(feats_bkd: torch.Tensor) -> torch.Tensor:
    """
    feats_bkd: (B, K, D), normalized
    Returns:
        consistency_per_sample: (B,)
        mean pairwise cosine similarity across views
    """
    B, K, D = feats_bkd.shape
    if K == 1:
        return torch.ones(B, device=feats_bkd.device)

    sim = feats_bkd @ feats_bkd.transpose(1, 2)
    eye = torch.eye(K, device=feats_bkd.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim.masked_select(~eye).view(B, K * K - K)
    consistency = off_diag.mean(dim=1)
    return consistency


@torch.no_grad()
def compute_alignment_stats(
    feats_bkd: torch.Tensor,
    labels: torch.Tensor,
    text_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    feats_bkd: (B, K, D), normalized
    labels:    (B,)
    text_features: (C,D), normalized

    Returns:
        align_mean_per_sample: (B,)
        align_var_per_sample:  (B,)
    """
    B, K, D = feats_bkd.shape
    correct_text = text_features[labels]
    sims = (feats_bkd * correct_text.unsqueeze(1)).sum(dim=-1)
    return sims.mean(dim=1), sims.var(dim=1, unbiased=False)


@torch.no_grad()
def compute_mechanism_metrics(
    model: CLIPZeroShot,
    x: torch.Tensor,
    labels: torch.Tensor,
    defense_t: TransformType,
    K: int,
) -> Dict[str, torch.Tensor]:
    """
    Returns per-sample tensors:
        view_consistency
        align_mean
        align_var
    """
    feats = collect_view_features(model, x, defense_t, K)
    view_consistency = compute_view_consistency(feats)
    align_mean, align_var = compute_alignment_stats(feats, labels, model.text_features)
    return {
        "view_consistency": view_consistency,
        "align_mean": align_mean,
        "align_var": align_var,
    }


# =========================================================
# Loss for attack
# =========================================================
def margin_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    tmp = logits.clone()
    tmp.scatter_(1, y.view(-1, 1), -1e9)
    other = tmp.max(dim=1).values
    return true - other


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
    """
    if aggregation == "single":
        return model(x)

    if t == "identity":
        return model(x)

    if aggregation in ("avg_logits", "vote"):
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
        f_bar = f_bar / f_bar.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return model.logits_from_features(f_bar)

    elif aggregation == "semantic_weighted_logits":
        _, logits = predict_with_aggregation(
            model=model,
            x=x,
            defense_t=t,
            aggregation="semantic_weighted_logits",
            K=max(1, eot_M),
            tau=0.3,
            beta_entropy=0.5,
            conf_gate=0.40,
        )
        return logits

    elif aggregation == "semantic_topk_logits":
        _, logits = predict_with_aggregation(
            model=model,
            x=x,
            defense_t=t,
            aggregation="semantic_topk_logits",
            K=max(1, eot_M),
            tau=0.3,
            beta_entropy=0.5,
            topk_ratio=0.5,
            conf_gate=0.40,
        )
        return logits

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


# =========================================================
# Confident Square Attack
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
    set_seed(cfg.seed)

    B, C, H, W = x.shape
    max_s = min(cfg.max_square, H, W)

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
# =========================================================
@torch.no_grad()
def eval_defenses_fair_with_mechanism(
    name: str,
    ds,
    clip_model,
    device: str,
    text_features: torch.Tensor,
    defenses: Tuple[TransformType, ...],
    aggregations: Tuple[AggregationType, ...],
    attack_cfg_base: SquareAttackConfig,
    batch_size: int = 16,
    num_workers: int = 0,
    subset_size: int = 200,
    subset_seed: int = 0,
    K_clean: int = 8,
    K_adv: int = 8,
    K_mech: int = 8,
):
    print(
        f"\n===== {name} | Defenses={defenses} | Aggregations={aggregations} "
        f"| K_clean={K_clean} K_adv={K_adv} K_mech={K_mech} | "
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

    mech_stats = {
        "clean_identity_view_consistency_sum": 0.0,
        "clean_identity_align_mean_sum": 0.0,
        "clean_identity_align_var_sum": 0.0,
        "adv_identity_view_consistency_sum": 0.0,
        "adv_identity_align_mean_sum": 0.0,
        "adv_identity_align_var_sum": 0.0,
    }

    stats: Dict[str, Dict[str, float]] = {}
    for d in defenses:
        for a in aggregations:
            key = f"{d}|{a}"
            stats[key] = {
                "def_clean_correct": 0.0,
                "def_robust_correct": 0.0,
                "asr_def_num": 0.0,
                "def_clean_ok": 0.0,
                "clean_view_consistency_sum": 0.0,
                "clean_align_mean_sum": 0.0,
                "clean_align_var_sum": 0.0,
                "adv_view_consistency_sum": 0.0,
                "adv_align_mean_sum": 0.0,
                "adv_align_var_sum": 0.0,
            }

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"{name}-eval", ncols=120)):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        n = labels.numel()
        total += n

        # ---------- Undefended clean ----------
        pred_uc, _ = predict_with_aggregation(
            model=model,
            x=images,
            defense_t="identity",
            aggregation="single",
            K=1,
        )
        uc = (pred_uc == labels)
        undef_clean_correct += uc.sum().item()

        mech_clean_id = compute_mechanism_metrics(
            model=model,
            x=images,
            labels=labels,
            defense_t="identity",
            K=1,
        )
        mech_stats["clean_identity_view_consistency_sum"] += mech_clean_id["view_consistency"].sum().item()
        mech_stats["clean_identity_align_mean_sum"] += mech_clean_id["align_mean"].sum().item()
        mech_stats["clean_identity_align_var_sum"] += mech_clean_id["align_var"].sum().item()

        # ---------- Undefended adaptive attack ----------
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

        mech_adv_id = compute_mechanism_metrics(
            model=model,
            x=x_adv_undef,
            labels=labels,
            defense_t="identity",
            K=1,
        )
        mech_stats["adv_identity_view_consistency_sum"] += mech_adv_id["view_consistency"].sum().item()
        mech_stats["adv_identity_align_mean_sum"] += mech_adv_id["align_mean"].sum().item()
        mech_stats["adv_identity_align_var_sum"] += mech_adv_id["align_var"].sum().item()

        # ---------- Defenses x Aggregations ----------
        for d in defenses:
            mech_clean = compute_mechanism_metrics(
                model=model,
                x=images,
                labels=labels,
                defense_t=d,
                K=K_mech,
            )

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

                stats[key]["clean_view_consistency_sum"] += mech_clean["view_consistency"].sum().item()
                stats[key]["clean_align_mean_sum"] += mech_clean["align_mean"].sum().item()
                stats[key]["clean_align_var_sum"] += mech_clean["align_var"].sum().item()

                cfg_d = replace(
                    attack_cfg_base,
                    seed=int(attack_cfg_base.seed) + int(batch_idx),
                    defense_transform_for_attacker=d,
                    aggregation_for_attacker=a,
                    eot_M=int(attack_cfg_base.eot_M),
                )
                x_adv_d = confident_square_attack_eot(model, images, labels, cfg_d)

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

                mech_adv = compute_mechanism_metrics(
                    model=model,
                    x=x_adv_d,
                    labels=labels,
                    defense_t=d,
                    K=K_mech,
                )
                stats[key]["adv_view_consistency_sum"] += mech_adv["view_consistency"].sum().item()
                stats[key]["adv_align_mean_sum"] += mech_adv["align_mean"].sum().item()
                stats[key]["adv_align_var_sum"] += mech_adv["align_var"].sum().item()

        if device == "cuda":
            torch.cuda.empty_cache()

    # ---------- print ----------
    undef_clean_acc = undef_clean_correct / max(total, 1)
    undef_robust_acc_ad = undef_robust_correct_adaptive / max(total, 1)

    print(f"\nRESULT: {name}")
    print(f"Samples (subset):                      {total}")
    print(f"Undefended Clean Accuracy:             {undef_clean_acc:.4f}")
    print(f"Undefended Robust Accuracy (adaptive): {undef_robust_acc_ad:.4f}")
    print(f"Undefended Clean View Consistency:     {mech_stats['clean_identity_view_consistency_sum'] / max(total, 1):.4f}")
    print(f"Undefended Clean Align Mean:           {mech_stats['clean_identity_align_mean_sum'] / max(total, 1):.4f}")
    print(f"Undefended Clean Align Variance:       {mech_stats['clean_identity_align_var_sum'] / max(total, 1):.6f}")
    print(f"Undefended Adv View Consistency:       {mech_stats['adv_identity_view_consistency_sum'] / max(total, 1):.4f}")
    print(f"Undefended Adv Align Mean:             {mech_stats['adv_identity_align_mean_sum'] / max(total, 1):.4f}")
    print(f"Undefended Adv Align Variance:         {mech_stats['adv_identity_align_var_sum'] / max(total, 1):.6f}")

    for d in defenses:
        for a in aggregations:
            key = f"{d}|{a}"
            def_clean_acc = stats[key]["def_clean_correct"] / max(total, 1)
            def_robust_acc = stats[key]["def_robust_correct"] / max(total, 1)
            asr_def = stats[key]["asr_def_num"] / (stats[key]["def_clean_ok"] + 1e-12)

            clean_view_consistency = stats[key]["clean_view_consistency_sum"] / max(total, 1)
            clean_align_mean = stats[key]["clean_align_mean_sum"] / max(total, 1)
            clean_align_var = stats[key]["clean_align_var_sum"] / max(total, 1)

            adv_view_consistency = stats[key]["adv_view_consistency_sum"] / max(total, 1)
            adv_align_mean = stats[key]["adv_align_mean_sum"] / max(total, 1)
            adv_align_var = stats[key]["adv_align_var_sum"] / max(total, 1)

            print(f"\n--- Defense: {d} | Aggregation: {a} ---")
            print(f"Defended Clean Accuracy:              {def_clean_acc:.4f}")
            print(f"Defended Robust Accuracy (adaptive):  {def_robust_acc:.4f}")
            print(f"ASR_def (on def-clean-ok):            {asr_def:.4f}")
            print(f"Clean View Consistency:               {clean_view_consistency:.4f}")
            print(f"Clean Align Mean:                     {clean_align_mean:.4f}")
            print(f"Clean Align Variance:                 {clean_align_var:.6f}")
            print(f"Adv View Consistency:                 {adv_view_consistency:.4f}")
            print(f"Adv Align Mean:                       {adv_align_mean:.4f}")
            print(f"Adv Align Variance:                   {adv_align_var:.6f}")

    print("=" * 110 + "\n")


# =========================================================
# Main
# =========================================================
def main():
    set_seed(0) #随机数固定
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
        # "cifar100": CIFAR100(f"{DATA_ROOT}/cifar100", train=False, download=True, transform=transform),
        # "food101": Food101(f"{DATA_ROOT}/food101", split="test", download=True, transform=transform),
        # "pets": OxfordIIITPet(f"{DATA_ROOT}/pets", split="test", download=True, transform=transform),
        # "fgvc_aircraft": FGVCAircraft(f"{DATA_ROOT}/fgvc_aircraft", split="test", download=True, transform=transform),
        # "stl10": STL10(f"{DATA_ROOT}/stl10", split="test", download=True, transform=transform),
    }

    defenses: Tuple[TransformType, ...] = (
        "horizontal_flip",
        "crop_resize_80",
        "rotation_10",
        "color_jitter_light",
        "gaussian_blur_light",
        "gaussian_noise_light",
        "mixed_strong",
    )

    aggregations: Tuple[AggregationType, ...] = (
        "vote",
        "avg_logits",
        "avg_features",
        "semantic_weighted_logits",
        "semantic_topk_logits",
    )

    attack_cfg = SquareAttackConfig(
        eps=8 / 255, #每个像素最多只能改8/255
        n_iters=200, #会迭代200次，即尝试 200 次修改，每次改一个方块，看有没有更好
        eot_M=8, #eot做8次，每次forward做8次随机变换再平均
        defense_transform_for_attacker="identity",
        aggregation_for_attacker="single",
        min_square=1,
        max_square=64,
        seed=0,
    )

    batch_size = 16
    num_workers = 0
    subset_size = 300
    subset_seed = 0

    K_clean = 8 #Clean inference      → 用 K_clean 个 view
    K_adv = 8 #Adversarial inference → 用 K_adv 个 view
    K_mech = 8 #Mechanism分析         → 用 K_mech 个 view
    

    for name, ds in datasets.items():
        print(f"\nPreparing: {name}")
        class_names = get_class_list(name, ds)
        text_features = build_text_features(class_names, clip_model, device)

        eval_defenses_fair_with_mechanism(
            name=name,
            ds=ds,
            clip_model=clip_model,
            device=device,
            text_features=text_features,
            defenses=defenses,
            aggregations=aggregations,
            attack_cfg_base=attack_cfg,
            batch_size=batch_size,
            num_workers=num_workers,
            subset_size=subset_size,
            subset_seed=subset_seed,
            K_clean=K_clean,
            K_adv=K_adv,
            K_mech=K_mech,
        )


if __name__ == "__main__":
    main()