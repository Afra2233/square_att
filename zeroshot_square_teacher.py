
import os
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# pip install git+https://github.com/openai/CLIP.git
import clip

# pip install git+https://github.com/fra31/auto-attack
from autoattack import AutoAttack


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_image_transform(train: bool = False):
    # Keep it simple and deterministic for zero-shot CLIP
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def normalize_for_clip(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def square_local_perturb(
    x: torch.Tensor,
    eps: float,
    num_squares: int = 1,
    min_size: int = 16,
    max_size: int = 64,
) -> torch.Tensor:
    """
    Fast training-time local square perturbation.
    This is only for training regularization, not the test-time attack.
    x is in [0, 1].
    """
    x_adv = x.clone()
    b, c, h, w = x.shape
    for i in range(b):
        for _ in range(num_squares):
            s = random.randint(min_size, min(max_size, h, w))
            top = random.randint(0, h - s)
            left = random.randint(0, w - s)
            noise = torch.empty((c, s, s), device=x.device).uniform_(-eps, eps)
            x_adv[i, :, top:top + s, left:left + s] = torch.clamp(
                x_adv[i, :, top:top + s, left:left + s] + noise,
                0.0,
                1.0,
            )
    return x_adv


def prompt_ensemble(classname: str) -> List[str]:
    name = classname.replace("_", " ").replace("-", " ").lower()
    return [
        f"a photo of a {name}.",
        f"a blurry photo of a {name}.",
        f"a close-up photo of a {name}.",
        f"a bright photo of a {name}.",
    ]


def encode_text_features(model, device, classnames: List[str], batch_size: int = 64) -> torch.Tensor:
    prompts = []
    class_slices = []
    cursor = 0
    for c in classnames:
        local_prompts = prompt_ensemble(c)
        prompts.extend(local_prompts)
        class_slices.append((cursor, cursor + len(local_prompts)))
        cursor += len(local_prompts)

    with torch.no_grad():
        text_tokens = clip.tokenize(prompts, truncate=True).to(device)
        all_text_features = []
        for i in range(0, len(prompts), batch_size):
            feats = model.encode_text(text_tokens[i:i + batch_size])
            feats = F.normalize(feats, dim=-1)
            all_text_features.append(feats)
        all_text_features = torch.cat(all_text_features, dim=0)

        class_features = []
        for s, e in class_slices:
            feat = all_text_features[s:e].mean(dim=0, keepdim=True)
            feat = F.normalize(feat, dim=-1)
            class_features.append(feat)
        class_features = torch.cat(class_features, dim=0)
    return class_features


class ClipZeroShotModel(nn.Module):
    def __init__(self, device: str, model_name: str = "ViT-B/32", train_last_blocks: int = 1):
        super().__init__()
        self.device_name = device

        self.model, _ = clip.load(model_name, device=device, jit=False)
        self.model.float()

        self.original_model, _ = clip.load(model_name, device=device, jit=False)
        self.original_model.float()
        self.original_model.eval()
        for p in self.original_model.parameters():
            p.requires_grad = False

        # Freeze all params first
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze only a small tail of the visual encoder for speed
        visual = self.model.visual
        if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
            blocks = visual.transformer.resblocks
            train_last_blocks = max(0, min(train_last_blocks, len(blocks)))
            for blk in list(blocks)[-train_last_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True

        # Unfreeze final norm / projection / embeddings if present
        for name in ["ln_post", "proj"]:
            if hasattr(visual, name):
                obj = getattr(visual, name)
                if isinstance(obj, torch.Tensor):
                    obj.requires_grad = True
                elif obj is not None:
                    for p in obj.parameters():
                        p.requires_grad = True

        if hasattr(visual, "class_embedding") and isinstance(visual.class_embedding, torch.Tensor):
            visual.class_embedding.requires_grad = True
        if hasattr(visual, "positional_embedding") and isinstance(visual.positional_embedding, torch.Tensor):
            visual.positional_embedding.requires_grad = True

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        x = normalize_for_clip(x)
        feat = self.model.encode_image(x)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def encode_image_original(self, x: torch.Tensor) -> torch.Tensor:
        x = normalize_for_clip(x)
        feat = self.original_model.encode_image(x)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def build_text_features(self, classnames: List[str]) -> torch.Tensor:
        return encode_text_features(self.model, self.device_name, classnames)

    def logits_from_features(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        scale = self.model.logit_scale.exp()
        return scale * image_features @ text_features.t()

    def forward_with_text_features(self, x: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        image_features = self.encode_image(x)
        return self.logits_from_features(image_features, text_features)


def margin_stability_loss(logits_clean: torch.Tensor, logits_pert: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pos_clean = logits_clean.gather(1, labels[:, None]).squeeze(1)
    pos_pert = logits_pert.gather(1, labels[:, None]).squeeze(1)

    mask = F.one_hot(labels, num_classes=logits_clean.size(1)).bool()
    neg_clean = logits_clean.masked_fill(mask, float("-inf")).max(dim=1).values
    neg_pert = logits_pert.masked_fill(mask, float("-inf")).max(dim=1).values

    margin_clean = pos_clean - neg_clean
    margin_pert = pos_pert - neg_pert
    return torch.mean(torch.abs(margin_clean - margin_pert))


def rank_consistency_loss(logits_clean: torch.Tensor, logits_pert: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    p_clean = F.softmax(logits_clean / tau, dim=1).detach()
    log_p_pert = F.log_softmax(logits_pert / tau, dim=1)
    return F.kl_div(log_p_pert, p_clean, reduction="batchmean")


def anchor_loss(clean_feat: torch.Tensor, clean_feat_orig: torch.Tensor) -> torch.Tensor:
    return (1.0 - (clean_feat * clean_feat_orig.detach()).sum(dim=-1)).mean()


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 128
    lr: float = 2e-5
    weight_decay: float = 1e-4
    train_eps: float = 4 / 255
    train_num_squares: int = 1
    lambda_margin: float = 0.5
    lambda_rank: float = 1.0
    lambda_anchor: float = 0.5
    train_subset: int = 0
    num_workers: int = 4


def get_cifar10_train_loader(root: str, batch_size: int, num_workers: int, subset: int = 0):
    ds = datasets.CIFAR10(root=root, train=True, download=True, transform=build_image_transform(train=True))
    if subset and subset > 0:
        subset = min(subset, len(ds))
        ds = Subset(ds, list(range(subset)))
        classnames = ds.dataset.classes
    else:
        classnames = ds.classes
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return loader, classnames


def get_eval_dataset(root: str, name: str):
    tfm = build_image_transform(train=False)

    if name == "cifar100":
        ds = datasets.CIFAR100(root=root, train=False, download=True, transform=tfm)
        classnames = ds.classes
    elif name == "food101":
        ds = datasets.Food101(root=root, split="test", download=True, transform=tfm)
        classnames = ds.classes
    elif name == "stl10":
        ds = datasets.STL10(root=root, split="test", download=True, transform=tfm)
        classnames = ds.classes
    elif name == "oxfordpet":
        ds = datasets.OxfordIIITPet(root=root, split="test", download=True, transform=tfm)
        classnames = [c.lower() for c in ds.classes]
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return ds, classnames


def make_loader(ds, batch_size: int, num_workers: int):
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def evaluate_clean(model: ClipZeroShotModel, loader: DataLoader, text_features: torch.Tensor, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model.forward_with_text_features(images, text_features)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)


def evaluate_square_attack_autoattack(
    model: ClipZeroShotModel,
    loader: DataLoader,
    text_features: torch.Tensor,
    device: str,
    eps: float = 8 / 255,
    attack_queries: int = 1000,
) -> float:
    """
    Evaluate robust accuracy under AutoAttack Square Attack only.
    """
    model.eval()

    def forward_pass(x: torch.Tensor) -> torch.Tensor:
        return model.forward_with_text_features(x, text_features)

    adversary = AutoAttack(
        forward_pass,
        norm='Linf',
        eps=eps,
        version='custom',
        device=device,
        verbose=False
    )
    adversary.attacks_to_run = ['square']
    adversary.square.n_queries = attack_queries

    robust_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        x_adv = adversary.run_standard_evaluation(images, labels, bs=images.size(0))

        with torch.no_grad():
            logits_adv = forward_pass(x_adv)
            preds_adv = logits_adv.argmax(dim=1)

        robust_correct += (preds_adv == labels).sum().item()
        total += labels.numel()

    return 100.0 * robust_correct / max(total, 1)


def train_defense(
    model: ClipZeroShotModel,
    loader: DataLoader,
    classnames: List[str],
    device: str,
    cfg: TrainConfig
):
    model.train()
    text_features = model.build_text_features(classnames).detach()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        t0 = time.time()

        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                perturbed = square_local_perturb(
                    images,
                    eps=cfg.train_eps,
                    num_squares=cfg.train_num_squares,
                    min_size=16,
                    max_size=64,
                )

            clean_feat = model.encode_image(images)
            pert_feat = model.encode_image(perturbed)

            with torch.no_grad():
                orig_feat = model.encode_image_original(images)

            logits_clean = model.logits_from_features(clean_feat, text_features)
            logits_pert = model.logits_from_features(pert_feat, text_features)

            loss_match = F.cross_entropy(logits_clean, labels) + F.cross_entropy(logits_pert, labels)
            loss_margin = margin_stability_loss(logits_clean, logits_pert, labels)
            loss_rank = rank_consistency_loss(logits_clean, logits_pert)
            loss_anchor = anchor_loss(clean_feat, orig_feat)

            loss = (
                loss_match
                + cfg.lambda_margin * loss_margin
                + cfg.lambda_rank * loss_rank
                + cfg.lambda_anchor * loss_anchor
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_batches += 1

        print(
            f"[Train] epoch={epoch + 1}/{cfg.epochs} "
            f"loss={epoch_loss / max(epoch_batches, 1):.4f} "
            f"time={time.time() - t0:.1f}s"
        )


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_eps=args.train_eps,
        train_num_squares=args.train_num_squares,
        lambda_margin=args.lambda_margin,
        lambda_rank=args.lambda_rank,
        lambda_anchor=args.lambda_anchor,
        train_subset=args.train_subset,
        num_workers=args.num_workers,
    )

    train_loader, train_classnames = get_cifar10_train_loader(
        args.data_root,
        cfg.batch_size,
        cfg.num_workers,
        subset=cfg.train_subset,
    )

    print("Loading models...")
    base_model = ClipZeroShotModel(
        device=device,
        model_name=args.clip_model,
        train_last_blocks=args.train_last_blocks,
    ).to(device)

    defended_model = ClipZeroShotModel(
        device=device,
        model_name=args.clip_model,
        train_last_blocks=args.train_last_blocks,
    ).to(device)

    print("Fine-tuning defended model on CIFAR-10...")
    train_defense(defended_model, train_loader, train_classnames, device, cfg)

    eval_names = ["cifar100", "food101", "stl10", "oxfordpet"]
    results = {}

    for name in eval_names:
        print(f"\n=== Evaluating on {name} ===")
        ds, classnames = get_eval_dataset(args.data_root, name)
        if args.eval_subset > 0:
            ds = Subset(ds, list(range(min(args.eval_subset, len(ds)))))
        loader = make_loader(ds, args.eval_batch_size, args.num_workers)

        with torch.no_grad():
            base_text = base_model.build_text_features(classnames).detach()
            defended_text = defended_model.build_text_features(classnames).detach()

        clean_accuracy = evaluate_clean(defended_model, loader, defended_text, device)
        clean_base_accuracy = evaluate_clean(base_model, loader, base_text, device)

        print("  Running AutoAttack Square on undefended model...")
        undefended_robust_accuracy = evaluate_square_attack_autoattack(
            base_model,
            loader,
            base_text,
            device,
            eps=args.attack_eps,
            attack_queries=args.attack_queries,
        )

        print("  Running AutoAttack Square on defended model...")
        defend_accuracy = evaluate_square_attack_autoattack(
            defended_model,
            loader,
            defended_text,
            device,
            eps=args.attack_eps,
            attack_queries=args.attack_queries,
        )

        results[name] = {
            "clean_accuracy": clean_accuracy,
            "undefended_robust_accuracy": undefended_robust_accuracy,
            "defend_accuracy": defend_accuracy,
            "clean_base_accuracy": clean_base_accuracy,
        }

        print(f"{name}:")
        print(f"  clean accuracy             : {clean_accuracy:.2f}")
        print(f"  undefended robust accuracy : {undefended_robust_accuracy:.2f}")
        print(f"  defend accuracy            : {defend_accuracy:.2f}")
        print(f"  clean base accuracy        : {clean_base_accuracy:.2f}")

    print("\n=== Final Summary ===")
    for name, r in results.items():
        print(
            f"{name:10s} | clean={r['clean_accuracy']:.2f} | "
            f"undefended_robust={r['undefended_robust_accuracy']:.2f} | "
            f"defended={r['defend_accuracy']:.2f}"
        )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--clip-model", type=str, default="ViT-B/32")
    p.add_argument("--seed", type=int, default=42)

    # Training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--train-eps", type=float, default=4 / 255)
    p.add_argument("--train-num-squares", type=int, default=1)
    p.add_argument("--lambda-margin", type=float, default=0.5)
    p.add_argument("--lambda-rank", type=float, default=1.0)
    p.add_argument("--lambda-anchor", type=float, default=0.5)
    p.add_argument("--train-last-blocks", type=int, default=1)
    p.add_argument("--train-subset", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)

    # Eval / attack
    p.add_argument("--attack-eps", type=float, default=8 / 255)
    p.add_argument("--attack-queries", type=int, default=1000)
    p.add_argument("--eval-subset", type=int, default=0, help="0 means full test set")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
