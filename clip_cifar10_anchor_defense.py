#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clip_cifar10_anchor_defense_stratified.py

功能:
1. 用 OpenCLIP 在 CIFAR-10 上做 zero-shot 分类
2. 从 CIFAR-10 test set 中按类别均匀抽样: 每类 samples_per_class 张
3. 用 undefended model 做第一次成功的 Square Attack
4. 取该次攻击后 adv top1 对应的 text embedding 作为 anchor
5. 后续 defend 前向时:
   - 把当前 top-k 图像特征轻微拉向 anchor
   - 对当前 top2-5 logits 加微小随机扰动
6. 评估:
   - clean accuracy
   - undefended robust accuracy
   - defended robust accuracy
   - ASR
   - top1-top2 margin

运行示例:
python clip_cifar10_anchor_defense_stratified.py \
    --data_root ./data \
    --batch_size 64 \
    --samples_per_class 100 \
    --eps 8 \
    --n_queries 1000 \
    --alpha 0.03 \
    --noise_std 0.01 \
    --topk_pull 1
"""

import time
import random
import argparse
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import open_clip
from autoattack.square import SquareAttack


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class AnchorState:
    vec: torch.Tensor = None
    cls_idx: int = None
    cls_name: str = None
    found: bool = False


class CLIPZeroShotDefense(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        text_features: torch.Tensor,
        alpha: float = 0.03,
        noise_std: float = 0.01,
        topk_pull: int = 1,
        logit_scale: float = 100.0,
    ):
        super().__init__()
        self.model = clip_model
        self.register_buffer("text_features", text_features)  # [C, D]
        self.alpha = alpha
        self.noise_std = noise_std
        self.topk_pull = topk_pull
        self.logit_scale = logit_scale
        self.anchor = AnchorState()

    @torch.no_grad()
    def encode_image_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model.encode_image(x)
        feat = F.normalize(feat, dim=-1)
        return feat

    @torch.no_grad()
    def clean_logits(self, x: torch.Tensor) -> torch.Tensor:
        img_feat = self.encode_image_features(x)
        logits = self.logit_scale * img_feat @ self.text_features.t()
        return logits

    @torch.no_grad()
    def defended_logits(self, x: torch.Tensor) -> torch.Tensor:
        img_feat = self.encode_image_features(x)
        logits = self.logit_scale * img_feat @ self.text_features.t()

        if not self.anchor.found or self.anchor.vec is None:
            return logits

        # 当前 top-k 预测
        k = min(self.topk_pull, logits.size(1))
        _ = logits.topk(k=k, dim=1).indices  # 保留接口含义

        # 轻微拉向 anchor
        anchor_vec = self.anchor.vec.unsqueeze(0)  # [1, D]
        pulled_feat = F.normalize(
            (1.0 - self.alpha) * img_feat + self.alpha * anchor_vec,
            dim=-1
        )

        defended_logits = self.logit_scale * pulled_feat @ self.text_features.t()

        # 对 top2-5 logits 加微小随机扰动
        if defended_logits.size(1) >= 5 and self.noise_std > 0:
            top5_idx = defended_logits.topk(k=5, dim=1).indices  # [B,5]
            idx_2to5 = top5_idx[:, 1:5]  # [B,4]

            noise = torch.randn(
                idx_2to5.shape,
                device=defended_logits.device,
                dtype=defended_logits.dtype
            ) * self.noise_std

            # 零均值，避免整体平移过大
            noise = noise - noise.mean(dim=1, keepdim=True)
            defended_logits.scatter_add_(1, idx_2to5, noise)

        return defended_logits

    def forward(self, x: torch.Tensor, defend: bool = False) -> torch.Tensor:
        if defend:
            return self.defended_logits(x)
        return self.clean_logits(x)


class SquareAttackWrapper(nn.Module):
    def __init__(self, defense_model: CLIPZeroShotDefense, defend: bool = False):
        super().__init__()
        self.defense_model = defense_model
        self.defend = defend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.defense_model(x, defend=self.defend)


def build_dataloader(
    data_root: str,
    batch_size: int,
    num_workers: int,
    samples_per_class: int = 0,
    seed: int = 0,
):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    testset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    if samples_per_class > 0:
        rng = np.random.RandomState(seed)
        class_to_indices = defaultdict(list)

        for idx, label in enumerate(testset.targets):
            class_to_indices[label].append(idx)

        selected_indices = []
        for cls in range(10):
            cls_indices = class_to_indices[cls]
            if samples_per_class > len(cls_indices):
                raise ValueError(
                    f"class {cls} only has {len(cls_indices)} samples, "
                    f"but samples_per_class={samples_per_class}"
                )
            picked = rng.choice(cls_indices, size=samples_per_class, replace=False)
            selected_indices.extend(picked.tolist())

        rng.shuffle(selected_indices)
        testset = Subset(testset, selected_indices)

    loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def build_model(device: str):
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(device)

    prompts = [f"a photo of a {c}" for c in CIFAR10_CLASSES]
    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)

    return model, text_features


@torch.no_grad()
def compute_top1_top2_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = logits.topk(k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


@torch.no_grad()
def evaluate_clean(defense_model, loader, device):
    total = 0
    correct = 0
    margin_sum = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = defense_model(x, defend=False)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)
        margin_sum += compute_top1_top2_margin(logits).sum().item()

    return {
        "clean_acc": correct / total,
        "clean_margin": margin_sum / total,
        "num_samples": total,
    }


def build_square_attack(model_for_attack, eps, n_queries, seed, device):
    attacker = SquareAttack(
        predict=model_for_attack,
        p_init=0.8,
        n_queries=n_queries,
        eps=eps,
        norm="Linf",
        n_restarts=1,
        seed=seed,
        verbose=False,
        device=device,
        resc_schedule=False,
    )
    return attacker


def find_first_successful_anchor(defense_model, loader, device, eps, n_queries, seed):
    attack_model = SquareAttackWrapper(defense_model, defend=False).to(device)
    attacker = build_square_attack(
        model_for_attack=attack_model,
        eps=eps,
        n_queries=n_queries,
        seed=seed,
        device=device,
    )

    seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            clean_logits = defense_model(x, defend=False)
            clean_pred = clean_logits.argmax(dim=1)

        x_adv = attacker.perturb(x, y)

        with torch.no_grad():
            adv_logits = defense_model(x_adv, defend=False)
            adv_pred = adv_logits.argmax(dim=1)

        success = (clean_pred == y) & (adv_pred != y)
        if success.any():
            idx = success.nonzero(as_tuple=False)[0].item()
            anchor_cls_idx = adv_pred[idx].item()

            defense_model.anchor.vec = defense_model.text_features[anchor_cls_idx].detach().clone()
            defense_model.anchor.cls_idx = anchor_cls_idx
            defense_model.anchor.cls_name = CIFAR10_CLASSES[anchor_cls_idx]
            defense_model.anchor.found = True

            return {
                "anchor_found": True,
                "anchor_cls_idx": anchor_cls_idx,
                "anchor_cls_name": CIFAR10_CLASSES[anchor_cls_idx],
                "global_sample_idx": seen + idx,
            }

        seen += y.size(0)

    return {
        "anchor_found": False,
        "anchor_cls_idx": None,
        "anchor_cls_name": None,
        "global_sample_idx": None,
    }


def evaluate_under_attack(defense_model, loader, device, eps, n_queries, seed, defend=False):
    attack_model = SquareAttackWrapper(defense_model, defend=defend).to(device)
    attacker = build_square_attack(
        model_for_attack=attack_model,
        eps=eps,
        n_queries=n_queries,
        seed=seed,
        device=device,
    )

    total = 0
    clean_correct = 0
    adv_correct = 0
    margin_sum = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            clean_logits = defense_model(x, defend=defend)
            clean_pred = clean_logits.argmax(dim=1)
            clean_correct += (clean_pred == y).sum().item()

        x_adv = attacker.perturb(x, y)

        with torch.no_grad():
            adv_logits = defense_model(x_adv, defend=defend)
            adv_pred = adv_logits.argmax(dim=1)
            adv_correct += (adv_pred == y).sum().item()
            margin_sum += compute_top1_top2_margin(adv_logits).sum().item()

        total += y.size(0)

    return {
        "acc_before_attack_in_this_mode": clean_correct / total,
        "robust_acc": adv_correct / total,
        "attack_success_rate": 1.0 - adv_correct / total,
        "adv_margin": margin_sum / total,
        "num_samples": total,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=0,
        help="0 means full CIFAR-10 test set; otherwise sample the same number from each class"
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=8.0,
        help="Linf epsilon in pixel scale, e.g. 8 means 8/255"
    )
    parser.add_argument("--n_queries", type=int, default=1000)

    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--topk_pull", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eps = args.eps / 255.0

    print("=" * 60)
    print("[Config]")
    print(f"device             : {device}")
    print(f"data_root          : {args.data_root}")
    print(f"batch_size         : {args.batch_size}")
    print(f"samples_per_class  : {args.samples_per_class}")
    print(f"eps                : {args.eps}/255 = {eps:.6f}")
    print(f"n_queries          : {args.n_queries}")
    print(f"alpha              : {args.alpha}")
    print(f"noise_std          : {args.noise_std}")
    print(f"topk_pull          : {args.topk_pull}")
    print(f"seed               : {args.seed}")
    print("=" * 60)

    loader = build_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    clip_model, text_features = build_model(device=device)

    defense_model = CLIPZeroShotDefense(
        clip_model=clip_model,
        text_features=text_features,
        alpha=args.alpha,
        noise_std=args.noise_std,
        topk_pull=args.topk_pull,
        logit_scale=100.0,
    ).to(device)

    t0 = time.time()

    # 1) clean
    clean_stats = evaluate_clean(defense_model, loader, device)
    print("\n===== Clean evaluation =====")
    print(f"clean_acc          : {clean_stats['clean_acc']:.4f}")
    print(f"clean_margin       : {clean_stats['clean_margin']:.4f}")
    print(f"num_samples        : {clean_stats['num_samples']}")

    # 2) 找 anchor
    anchor_stats = find_first_successful_anchor(
        defense_model=defense_model,
        loader=loader,
        device=device,
        eps=eps,
        n_queries=args.n_queries,
        seed=args.seed,
    )

    print("\n===== Anchor selection =====")
    print(f"anchor_found       : {anchor_stats['anchor_found']}")
    if not anchor_stats["anchor_found"]:
        print("No successful attacked sample found. Abort.")
        return

    print(f"anchor_class       : {anchor_stats['anchor_cls_name']} ({anchor_stats['anchor_cls_idx']})")
    print(f"sample_index       : {anchor_stats['global_sample_idx']}")

    # 3) undefended
    undef_stats = evaluate_under_attack(
        defense_model=defense_model,
        loader=loader,
        device=device,
        eps=eps,
        n_queries=args.n_queries,
        seed=args.seed,
        defend=False,
    )

    print("\n===== Under attack: undefended =====")
    print(f"mode_clean_acc     : {undef_stats['acc_before_attack_in_this_mode']:.4f}")
    print(f"robust_acc         : {undef_stats['robust_acc']:.4f}")
    print(f"ASR                : {undef_stats['attack_success_rate']:.4f}")
    print(f"adv_margin         : {undef_stats['adv_margin']:.4f}")
    print(f"num_samples        : {undef_stats['num_samples']}")

    # 4) defended
    defend_stats = evaluate_under_attack(
        defense_model=defense_model,
        loader=loader,
        device=device,
        eps=eps,
        n_queries=args.n_queries,
        seed=args.seed,
        defend=True,
    )

    print("\n===== Under attack: defended =====")
    print(f"mode_clean_acc     : {defend_stats['acc_before_attack_in_this_mode']:.4f}")
    print(f"robust_acc         : {defend_stats['robust_acc']:.4f}")
    print(f"ASR                : {defend_stats['attack_success_rate']:.4f}")
    print(f"adv_margin         : {defend_stats['adv_margin']:.4f}")
    print(f"num_samples        : {defend_stats['num_samples']}")

    print("\n===== Summary =====")
    print(f"clean accuracy     = {clean_stats['clean_acc']:.4f}")
    print(f"undef robust acc   = {undef_stats['robust_acc']:.4f}")
    print(f"defend robust acc  = {defend_stats['robust_acc']:.4f}")
    print(f"delta robust acc   = {defend_stats['robust_acc'] - undef_stats['robust_acc']:.4f}")
    print(f"total time (sec)   = {time.time() - t0:.2f}")


if __name__ == "__main__":
    main()