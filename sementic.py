#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
semantic_score_shaping_square_eval.py

End-to-end runnable evaluation script for:
1) CLIP zero-shot clean accuracy
2) Undefended robust accuracy under Square Attack (score-based black-box)
3) Defended robust accuracy with semantic-constrained external score shaping
4) Calibration metrics (ECE)

Dependencies
------------
pip install torch torchvision open_clip_torch

Example
-------
python semantic_score_shaping_square_eval.py \
  --dataset cifar10 \
  --data-root ./data \
  --model ViT-B-32 \
  --pretrained openai \
  --batch-size 64 \
  --num-workers 4 \
  --epsilon 8/255 \
  --square-steps 200 \
  --samples-per-class 100 \
  --top-k 8 \
  --amplitude 0.35 \
  --frequency 1.2

Optional adaptive attack on defended model:
python semantic_score_shaping_square_eval.py \
  --dataset cifar10 \
  --data-root ./data \
  --adaptive-defense-attack

Supported datasets
------------------
- cifar10
- cifar100
- imagenetfolder   (expects ImageFolder layout under --data-root/val or --data-root)

Notes
-----
- This implementation is an untargeted Linf Square Attack using score queries.
- The loss used is the margin loss:
      loss = z_y - max_{c != y} z_c
  and the attack tries to MINIMIZE this loss.
- Query budget is approximated by number of square-attack update steps.
- If --dataset cifar10 and --samples-per-class > 0, the script samples that many
  images per class from the CIFAR-10 test set.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

import open_clip


# ----------------------------
# Utility
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_fraction_or_float(s: str) -> float:
    s = str(s).strip()
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def infer_image_size(model) -> int:
    image_size = getattr(getattr(model, "visual", None), "image_size", None)
    if isinstance(image_size, tuple):
        return int(image_size[-1])
    if image_size is None:
        return 224
    return int(image_size)


def get_clip_mean_std(model_name: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    return mean, std


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


# ----------------------------
# Calibration
# ----------------------------

class ExpectedCalibrationError:
    def __init__(self, n_bins: int = 15):
        self.n_bins = int(n_bins)
        self.reset()

    def reset(self):
        self.bin_total = torch.zeros(self.n_bins, dtype=torch.float64)
        self.bin_correct = torch.zeros(self.n_bins, dtype=torch.float64)
        self.bin_conf = torch.zeros(self.n_bins, dtype=torch.float64)

    @torch.no_grad()
    def update(self, probs: torch.Tensor, labels: torch.Tensor):
        conf, pred = probs.max(dim=-1)
        correct = pred.eq(labels)

        conf = conf.detach().cpu().to(torch.float64)
        correct = correct.detach().cpu().to(torch.float64)

        idx = torch.clamp((conf * self.n_bins).long(), max=self.n_bins - 1)

        for b in range(self.n_bins):
            mask = idx == b
            if mask.any():
                self.bin_total[b] += mask.sum().item()
                self.bin_correct[b] += correct[mask].sum().item()
                self.bin_conf[b] += conf[mask].sum().item()

    def compute(self) -> float:
        total = self.bin_total.sum().item()
        if total == 0:
            return 0.0
        ece = 0.0
        for b in range(self.n_bins):
            if self.bin_total[b] > 0:
                acc_b = self.bin_correct[b] / self.bin_total[b]
                conf_b = self.bin_conf[b] / self.bin_total[b]
                ece += (self.bin_total[b] / total) * abs(acc_b - conf_b)
        return float(ece)


# ----------------------------
# Semantic-constrained score shaper
# ----------------------------

@dataclass
class SemanticShapingConfig:
    top_k: int = 8
    similarity_temperature: float = 12.0
    weight_mode: str = "softmax"
    similarity_threshold: float = 0.5

    amplitude: float = 0.35
    frequency: float = 1.25
    phase: float = 0.0
    safe_margin: float = 1e-4
    budget_ratio: float = 0.45
    clamp_delta: bool = True

    enable_refinement: bool = True
    refinement_steps: int = 40
    refinement_lr: float = 0.05

    lambda_trend: float = 1.0
    lambda_sem: float = 0.25
    lambda_cal: float = 0.15
    calibration_temperature: float = 1.0

    eps: float = 1e-8


class SemanticConstrainedScoreShaper(nn.Module):
    def __init__(self, config: SemanticShapingConfig):
        super().__init__()
        self.cfg = config

    @staticmethod
    def _normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
        return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)

    def _compute_semantic_weights(self, top1_text: torch.Tensor, cand_text: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        top1_text = self._normalize(top1_text, dim=-1, eps=cfg.eps)
        cand_text = self._normalize(cand_text, dim=-1, eps=cfg.eps)
        sim = F.cosine_similarity(top1_text.unsqueeze(1), cand_text, dim=-1)

        if cfg.weight_mode == "softmax":
            w = F.softmax(cfg.similarity_temperature * sim, dim=-1)
        elif cfg.weight_mode == "relu":
            w = F.relu(sim)
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)
        elif cfg.weight_mode == "threshold":
            w = (sim >= cfg.similarity_threshold).float()
            zero_mask = (w.sum(dim=-1, keepdim=True) == 0)
            soft_w = F.softmax(cfg.similarity_temperature * sim, dim=-1)
            w = torch.where(zero_mask, soft_w, w / w.sum(dim=-1, keepdim=True).clamp_min(cfg.eps))
        else:
            raise ValueError(f"Unknown weight_mode: {cfg.weight_mode}")

        return w

    def _compute_safe_amplitude(self, margins: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        available = (margins - cfg.safe_margin).clamp_min(0.0)
        gamma_safe = cfg.budget_ratio * available / weights.clamp_min(cfg.eps)
        gamma_c = torch.minimum(torch.full_like(gamma_safe, cfg.amplitude), gamma_safe)
        return gamma_c

    def _closed_form_shape(self, top1_logits: torch.Tensor, cand_logits: torch.Tensor, weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        margins = top1_logits.unsqueeze(-1) - cand_logits
        gamma_c = self._compute_safe_amplitude(margins, weights)
        delta_cf = gamma_c * weights * torch.sin(cfg.frequency * margins + cfg.phase)

        if cfg.clamp_delta:
            max_allowed = (margins - cfg.safe_margin).clamp_min(0.0) * cfg.budget_ratio
            delta_cf = torch.sign(delta_cf) * torch.minimum(delta_cf.abs(), max_allowed)

        cand_logits_cf = cand_logits + delta_cf
        target_margins = top1_logits.unsqueeze(-1) - cand_logits_cf

        return {
            "cand_logits_cf": cand_logits_cf,
            "target_margins": target_margins,
        }

    def _refine_local_logits(
        self,
        local_logits_orig: torch.Tensor,
        local_logits_init: torch.Tensor,
        target_margins: torch.Tensor,
        local_text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        B, K = local_logits_orig.shape
        z = nn.Parameter(local_logits_init.clone())

        text_norm = self._normalize(local_text_embeds, dim=-1, eps=cfg.eps)
        A = torch.matmul(text_norm, text_norm.transpose(-1, -2)).clamp(min=0.0)
        eye = torch.eye(K, device=A.device, dtype=A.dtype).unsqueeze(0)
        A = A * (1.0 - eye)

        with torch.no_grad():
            orig_probs = F.softmax(local_logits_orig / cfg.calibration_temperature, dim=-1)

        optimizer = torch.optim.Adam([z], lr=cfg.refinement_lr)

        for _ in range(cfg.refinement_steps):
            optimizer.zero_grad()

            cur_margins = z[:, 0:1] - z[:, 1:]
            L_trend = F.mse_loss(cur_margins, target_margins)

            diff_orig = local_logits_orig.unsqueeze(-1) - local_logits_orig.unsqueeze(-2)
            diff_new = z.unsqueeze(-1) - z.unsqueeze(-2)
            sem_den = A.sum(dim=(-1, -2)).clamp_min(cfg.eps)
            L_sem = ((A * (diff_new - diff_orig).pow(2)).sum(dim=(-1, -2)) / sem_den).mean()

            new_probs = F.softmax(z / cfg.calibration_temperature, dim=-1)
            L_cal = F.kl_div(new_probs.log().clamp_min(math.log(cfg.eps)), orig_probs, reduction="batchmean")

            violation = F.relu(z[:, 1:] - z[:, 0:1] + cfg.safe_margin)
            L_top1 = violation.pow(2).mean()

            loss = cfg.lambda_trend * L_trend + cfg.lambda_sem * L_sem + cfg.lambda_cal * L_cal + 10.0 * L_top1
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            max_comp = z[:, 1:].max(dim=-1, keepdim=True).values
            need_fix = z[:, 0:1] < max_comp + cfg.safe_margin
            z[:, 0:1] = torch.where(need_fix, max_comp + cfg.safe_margin, z[:, 0:1])

        return z.detach()

    def forward(self, logits: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, C = logits.shape
        K = min(cfg.top_k, C)
        if K < 2:
            return logits

        topk_vals, topk_idx = torch.topk(logits, k=K, dim=-1, largest=True, sorted=True)
        top1_idx = topk_idx[:, 0]

        top1_logits = topk_vals[:, 0]
        cand_logits = topk_vals[:, 1:]

        top1_text = text_embeds[top1_idx]
        cand_text = text_embeds[topk_idx[:, 1:]]
        local_text = text_embeds[topk_idx]

        weights = self._compute_semantic_weights(top1_text, cand_text)
        cf = self._closed_form_shape(top1_logits, cand_logits, weights)

        local_logits_init = topk_vals.clone()
        local_logits_init[:, 1:] = cf["cand_logits_cf"]

        if cfg.enable_refinement and K > 2:
            local_logits_final = self._refine_local_logits(
                local_logits_orig=topk_vals,
                local_logits_init=local_logits_init,
                target_margins=cf["target_margins"],
                local_text_embeds=local_text,
            )
        else:
            local_logits_final = local_logits_init

        shaped_logits = logits.clone()
        shaped_logits.scatter_(1, topk_idx, local_logits_final)

        orig_pred = logits.argmax(dim=-1)
        new_pred = shaped_logits.argmax(dim=-1)
        flipped = orig_pred != new_pred
        if flipped.any():
            batch_idx = torch.nonzero(flipped, as_tuple=False).squeeze(-1)
            for b in batch_idx.tolist():
                y = orig_pred[b].item()
                max_other = torch.cat([shaped_logits[b, :y], shaped_logits[b, y + 1:]]).max()
                shaped_logits[b, y] = max_other + cfg.safe_margin

        return shaped_logits


# ----------------------------
# Dataset / prompts
# ----------------------------

def build_raw_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])


def build_dataset(name: str, data_root: str, image_size: int):
    raw_transform = build_raw_transform(image_size)
    name = name.lower()

    if name == "cifar10":
        ds = datasets.CIFAR10(root=data_root, train=False, transform=raw_transform, download=True)
        return ds, ds.classes

    if name == "cifar100":
        ds = datasets.CIFAR100(root=data_root, train=False, transform=raw_transform, download=True)
        return ds, ds.classes

    if name == "imagenetfolder":
        root = Path(data_root)
        candidate = root / "val"
        if candidate.exists():
            root = candidate
        ds = datasets.ImageFolder(root=str(root), transform=raw_transform)
        classnames = [c.replace("_", " ") for c in ds.classes]
        return ds, classnames

    raise ValueError(f"Unsupported dataset: {name}")


def build_class_balanced_subset(dataset, num_per_class: int, num_classes: int, seed: int = 0):
    rng = random.Random(seed)
    indices_by_class = [[] for _ in range(num_classes)]

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        indices_by_class[label].append(idx)

    selected_indices = []
    for c in range(num_classes):
        cls_indices = indices_by_class[c]
        if len(cls_indices) < num_per_class:
            raise ValueError(
                f"Class {c} only has {len(cls_indices)} samples, fewer than requested {num_per_class}."
            )
        rng.shuffle(cls_indices)
        selected_indices.extend(cls_indices[:num_per_class])

    rng.shuffle(selected_indices)
    return Subset(dataset, selected_indices)


def default_templates(dataset_name: str) -> List[str]:
    dataset_name = dataset_name.lower()
    if dataset_name in {"cifar10", "cifar100"}:
        return [
            "a photo of a {}.",
            "a blurry photo of a {}.",
            "a close-up photo of a {}.",
            "a bright photo of a {}.",
            "a clean photo of a {}.",
        ]
    return [
        "a photo of a {}.",
        "a photo of the {}.",
        "a blurry photo of a {}.",
        "a bright photo of a {}.",
        "a close-up photo of a {}.",
    ]


@torch.no_grad()
def build_zero_shot_classifier(model, tokenizer, classnames, templates, device):
    zeroshot_weights = []
    for classname in classnames:
        texts = [template.format(classname.replace("_", " ")) for template in templates]
        tokens = tokenizer(texts).to(device)
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        class_feature = text_features.mean(dim=0)
        class_feature = F.normalize(class_feature, dim=-1)
        zeroshot_weights.append(class_feature)
    return torch.stack(zeroshot_weights, dim=0)


# ----------------------------
# CLIP inference
# ----------------------------

class CLIPZeroShotWrapper(nn.Module):
    def __init__(self, model, text_features: torch.Tensor, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("text_features", text_features)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def preprocess(self, images_01: torch.Tensor) -> torch.Tensor:
        return (images_01 - self.mean) / self.std

    def logits(self, images_01: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(images_01)
        image_features = self.model.encode_image(x)
        image_features = F.normalize(image_features, dim=-1)
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        return self.logits(images_01)


class DefendedCLIPWrapper(nn.Module):
    def __init__(self, base_wrapper: CLIPZeroShotWrapper, shaper: SemanticConstrainedScoreShaper):
        super().__init__()
        self.base = base_wrapper
        self.shaper = shaper

    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        logits = self.base.logits(images_01)
        shaped_logits = self.shaper(logits, self.base.text_features)
        return shaped_logits


# ----------------------------
# Square attack
# ----------------------------

@torch.no_grad()
def margin_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Untargeted margin loss per sample:
        z_y - max_{c != y} z_c
    Attack minimizes this.
    """
    zy = logits.gather(1, labels[:, None]).squeeze(1)
    tmp = logits.clone()
    tmp[torch.arange(logits.size(0), device=logits.device), labels] = -1e9
    z_other = tmp.max(dim=1).values
    return zy - z_other


def p_selection(step: int, n_steps: int, p_init: float = 0.8) -> float:
    """
    Piecewise schedule inspired by Square Attack.
    Returns fraction of pixels per side length selection.
    """
    it = step / max(n_steps, 1)
    if it <= 0.1:
        p = p_init
    elif it <= 0.3:
        p = p_init / 2
    elif it <= 0.5:
        p = p_init / 4
    elif it <= 0.7:
        p = p_init / 8
    elif it <= 0.9:
        p = p_init / 16
    else:
        p = p_init / 32
    return max(p, 1.0 / (224 * 224))


def random_linf_square_delta_like(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Random ±eps init for Linf square attack.
    """
    return eps * torch.sign(torch.rand_like(x) * 2 - 1)


@torch.no_grad()
def square_attack_linf(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    n_queries: int = 200,
    p_init: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Untargeted Linf Square Attack (score-based black-box).

    Returns:
        adv_images: [B,C,H,W]
        query_counts: [B]
    """
    device = images.device
    B, C, H, W = images.shape

    x = images.detach()
    x_best = torch.clamp(x + random_linf_square_delta_like(x, epsilon), 0.0, 1.0)
    logits = model_fn(x_best)
    loss_best = margin_loss(logits, labels)
    preds = logits.argmax(dim=1)
    success = preds.ne(labels)

    query_counts = torch.ones(B, device=device, dtype=torch.long)

    for i in range(n_queries - 1):
        idx_to_fool = (~success).nonzero(as_tuple=False).squeeze(-1)
        if idx_to_fool.numel() == 0:
            break

        x_curr = x[idx_to_fool]
        x_best_curr = x_best[idx_to_fool]
        y_curr = labels[idx_to_fool]
        loss_best_curr = loss_best[idx_to_fool]

        p = p_selection(i, n_queries, p_init=p_init)
        s = int(round(math.sqrt(p * H * W / C)))
        s = max(min(s, H), 1)

        h_starts = torch.randint(0, H - s + 1, size=(idx_to_fool.numel(),), device=device)
        w_starts = torch.randint(0, W - s + 1, size=(idx_to_fool.numel(),), device=device)

        x_new = x_best_curr.clone()

        for j in range(idx_to_fool.numel()):
            hs = h_starts[j].item()
            ws = w_starts[j].item()

            patch = x_new[j, :, hs:hs+s, ws:ws+s]
            patch_sign = torch.sign(torch.rand_like(patch) * 2 - 1)
            proposal_patch = patch + 2.0 * epsilon * patch_sign

            x_window_orig = x_curr[j, :, hs:hs+s, ws:ws+s]
            proposal_patch = torch.max(torch.min(proposal_patch, x_window_orig + epsilon), x_window_orig - epsilon)
            proposal_patch = proposal_patch.clamp(0.0, 1.0)

            x_new[j, :, hs:hs+s, ws:ws+s] = proposal_patch

        logits_new = model_fn(x_new)
        loss_new = margin_loss(logits_new, y_curr)
        preds_new = logits_new.argmax(dim=1)

        improved = loss_new < loss_best_curr
        global_idx = idx_to_fool[improved]

        if improved.any():
            x_best[global_idx] = x_new[improved]
            loss_best[global_idx] = loss_new[improved]

        newly_successful = preds_new.ne(y_curr)
        success[idx_to_fool[newly_successful]] = True

        query_counts[idx_to_fool] += 1

    return x_best.detach(), query_counts.detach()


# ----------------------------
# Evaluation
# ----------------------------

@torch.no_grad()
def update_metrics(logits: torch.Tensor, labels: torch.Tensor, acc_meter: AverageMeter, ece: ExpectedCalibrationError):
    pred = logits.argmax(dim=-1)
    correct = pred.eq(labels).float().mean().item()
    acc_meter.update(correct, n=labels.size(0))
    probs = F.softmax(logits, dim=-1)
    ece.update(probs, labels)


def evaluate(
    clean_model: CLIPZeroShotWrapper,
    defended_model: DefendedCLIPWrapper,
    loader: DataLoader,
    epsilon: float,
    square_steps: int,
    adaptive_defense_attack: bool = False,
) -> Dict[str, float]:
    clean_acc = AverageMeter()
    clean_ece = ExpectedCalibrationError(n_bins=15)

    defend_clean_acc = AverageMeter()
    defend_clean_ece = ExpectedCalibrationError(n_bins=15)

    undef_rob_acc = AverageMeter()
    undef_rob_ece = ExpectedCalibrationError(n_bins=15)

    defend_rob_acc = AverageMeter()
    defend_rob_ece = ExpectedCalibrationError(n_bins=15)

    defend_adapt_rob_acc = AverageMeter()
    defend_adapt_rob_ece = ExpectedCalibrationError(n_bins=15)

    avg_queries_undef = AverageMeter()
    avg_queries_adapt = AverageMeter()

    clean_model.eval()
    defended_model.eval()

    for images, labels in loader:
        images = images.cuda(non_blocking=True) if torch.cuda.is_available() else images
        labels = labels.cuda(non_blocking=True) if torch.cuda.is_available() else labels

        with torch.no_grad():
            logits_clean = clean_model(images)
            logits_def_clean = defended_model(images)
            update_metrics(logits_clean, labels, clean_acc, clean_ece)
            update_metrics(logits_def_clean, labels, defend_clean_acc, defend_clean_ece)

        adv_undef, q_undef = square_attack_linf(
            model_fn=clean_model,
            images=images,
            labels=labels,
            epsilon=epsilon,
            n_queries=square_steps,
        )
        avg_queries_undef.update(q_undef.float().mean().item(), n=labels.size(0))

        with torch.no_grad():
            logits_undef_adv = clean_model(adv_undef)
            logits_def_adv = defended_model(adv_undef)
            update_metrics(logits_undef_adv, labels, undef_rob_acc, undef_rob_ece)
            update_metrics(logits_def_adv, labels, defend_rob_acc, defend_rob_ece)

        if adaptive_defense_attack:
            adv_def, q_adapt = square_attack_linf(
                model_fn=defended_model,
                images=images,
                labels=labels,
                epsilon=epsilon,
                n_queries=square_steps,
            )
            avg_queries_adapt.update(q_adapt.float().mean().item(), n=labels.size(0))

            with torch.no_grad():
                logits_def_adapt = defended_model(adv_def)
                update_metrics(logits_def_adapt, labels, defend_adapt_rob_acc, defend_adapt_rob_ece)

    results = {
        "clean_accuracy_undefended": clean_acc.avg,
        "clean_ece_undefended": clean_ece.compute(),
        "clean_accuracy_defended": defend_clean_acc.avg,
        "clean_ece_defended": defend_clean_ece.compute(),
        "robust_accuracy_undefended": undef_rob_acc.avg,
        "robust_ece_undefended": undef_rob_ece.compute(),
        "robust_accuracy_defended_nonadaptive": defend_rob_acc.avg,
        "robust_ece_defended_nonadaptive": defend_rob_ece.compute(),
        "avg_queries_nonadaptive": avg_queries_undef.avg,
    }

    if adaptive_defense_attack:
        results["robust_accuracy_defended_adaptive"] = defend_adapt_rob_acc.avg
        results["robust_ece_defended_adaptive"] = defend_adapt_rob_ece.compute()
        results["avg_queries_adaptive"] = avg_queries_adapt.avg

    return results


# ----------------------------
# Main
# ----------------------------

def build_argparser():
    parser = argparse.ArgumentParser(description="CLIP zero-shot evaluation with semantic score shaping defense + Square Attack")

    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenetfolder"])
    parser.add_argument("--data-root", type=str, required=True)

    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epsilon", type=str, default="8/255")
    parser.add_argument("--square-steps", type=int, default=200)
    parser.add_argument("--adaptive-defense-attack", action="store_true")

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=0,
        help="If > 0 and dataset is cifar10, sample this many test images per class.",
    )

    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--similarity-temperature", type=float, default=12.0)
    parser.add_argument("--weight-mode", type=str, default="softmax", choices=["softmax", "relu", "threshold"])
    parser.add_argument("--similarity-threshold", type=float, default=0.5)

    parser.add_argument("--amplitude", type=float, default=0.35)
    parser.add_argument("--frequency", type=float, default=1.25)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--safe-margin", type=float, default=1e-4)
    parser.add_argument("--budget-ratio", type=float, default=0.45)
    parser.add_argument("--no-clamp-delta", action="store_true")

    parser.add_argument("--disable-refinement", action="store_true")
    parser.add_argument("--refinement-steps", type=int, default=40)
    parser.add_argument("--refinement-lr", type=float, default=0.05)

    parser.add_argument("--lambda-trend", type=float, default=1.0)
    parser.add_argument("--lambda-sem", type=float, default=0.25)
    parser.add_argument("--lambda-cal", type=float, default=0.15)
    parser.add_argument("--calibration-temperature", type=float, default=1.0)

    parser.add_argument("--output-json", type=str, default="")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epsilon = parse_fraction_or_float(args.epsilon)

    print("=" * 80)
    print("Loading CLIP model...")
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(args.model)
    image_size = infer_image_size(model)
    mean, std = get_clip_mean_std(args.model)

    print(f"Model         : {args.model}")
    print(f"Pretrained    : {args.pretrained}")
    print(f"Image size    : {image_size}")
    print(f"Device        : {device}")

    print("=" * 80)
    print("Building dataset...")
    dataset, classnames = build_dataset(args.dataset, args.data_root, image_size)
    original_len = len(dataset)

    if args.dataset.lower() == "cifar10" and args.samples_per_class > 0:
        dataset = build_class_balanced_subset(
            dataset=dataset,
            num_per_class=args.samples_per_class,
            num_classes=10,
            seed=args.seed,
        )
        print(f"Using CIFAR10 balanced test subset: {args.samples_per_class} images/class")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Dataset       : {args.dataset}")
    print(f"Orig samples  : {original_len}")
    print(f"Eval samples  : {len(dataset)}")
    print(f"Num classes   : {len(classnames)}")

    print("=" * 80)
    print("Building zero-shot classifier...")
    templates = default_templates(args.dataset)
    with torch.no_grad():
        text_features = build_zero_shot_classifier(model, tokenizer, classnames, templates, device)

    clean_model = CLIPZeroShotWrapper(model, text_features, mean, std).to(device)

    defense_cfg = SemanticShapingConfig(
        top_k=args.top_k,
        similarity_temperature=args.similarity_temperature,
        weight_mode=args.weight_mode,
        similarity_threshold=args.similarity_threshold,
        amplitude=args.amplitude,
        frequency=args.frequency,
        phase=args.phase,
        safe_margin=args.safe_margin,
        budget_ratio=args.budget_ratio,
        clamp_delta=not args.no_clamp_delta,
        enable_refinement=not args.disable_refinement,
        refinement_steps=args.refinement_steps,
        refinement_lr=args.refinement_lr,
        lambda_trend=args.lambda_trend,
        lambda_sem=args.lambda_sem,
        lambda_cal=args.lambda_cal,
        calibration_temperature=args.calibration_temperature,
    )
    shaper = SemanticConstrainedScoreShaper(defense_cfg).to(device)
    defended_model = DefendedCLIPWrapper(clean_model, shaper).to(device)

    print("=" * 80)
    print("Running evaluation with Square Attack...")
    print(f"Epsilon       : {epsilon:.8f}")
    print(f"Square steps  : {args.square_steps}")
    print(f"Adaptive attk : {args.adaptive_defense_attack}")
    print("Defense cfg   :")
    print(json.dumps(asdict(defense_cfg), indent=2))

    results = evaluate(
        clean_model=clean_model,
        defended_model=defended_model,
        loader=loader,
        epsilon=epsilon,
        square_steps=args.square_steps,
        adaptive_defense_attack=args.adaptive_defense_attack,
    )

    print("=" * 80)
    print("RESULTS")
    for k, v in results.items():
        print(f"{k:40s}: {v:.6f}")

    if args.output_json:
        out = {
            "args": vars(args),
            "defense_config": asdict(defense_cfg),
            "results": results,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()