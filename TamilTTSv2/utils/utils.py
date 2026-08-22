"""
Training Utilities for TamilTTSv2
=================================
- EMA: exponential moving average of model weights (eval/export quality boost).
- save_checkpoint / load_checkpoint: module-aware checkpoints with G + D optimizers and EMA.
- count_parameters, get_lr_scheduler: v1-compatible helpers.
"""
import os
import math
import torch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class EMA:
    """
    Exponential Moving Average over trainable parameters.
    shadow = decay * shadow + (1 - decay) * param
    """

    def __init__(self, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self, model):
        net = unwrap_model(model)
        self.shadow = {
            name: p.detach().clone()
            for name, p in net.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        net = unwrap_model(model)
        for name, p in net.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {name: v.detach().cpu().clone() for name, v in self.shadow.items()}

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        for name, value in state_dict.items():
            if name in self.shadow:
                self.shadow[name].copy_(value.to(self.shadow[name].device))
            else:
                self.shadow[name] = value.clone()

    @torch.no_grad()
    def copy_to(self, model):
        net = unwrap_model(model)
        for name, p in net.named_parameters():
            if name in self.shadow:
                p.copy_(self.shadow[name].to(p.device))

    @torch.no_grad()
    def store_backup(self, model):
        net = unwrap_model(model)
        self.backup = {
            name: p.detach().clone()
            for name, p in net.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def restore_backup(self, model):
        net = unwrap_model(model)
        for name, p in net.named_parameters():
            if name in self.backup:
                p.copy_(self.backup[name].to(p.device))
        self.backup = {}


def save_checkpoint(path, model, optimizer, scheduler=None, optimizer_disc=None, step=0, extra=None):
    """Save a module-aware checkpoint including both optimizers, scheduler, step and extras."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "optimizer_disc_state_dict": optimizer_disc.state_dict() if optimizer_disc is not None else None,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"  💾 Checkpoint saved: {path} (Step {step})")


def load_checkpoint(path, model, optimizer=None, scheduler=None, optimizer_disc=None):
    """Load a checkpoint with strict=False; returns the stored global step."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and ckpt.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            print("  ⚠️ Could not load generator optimizer state (param groups changed). Using fresh optimizer.")
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception:
            print("  ⚠️ Could not load scheduler state. Using fresh scheduler.")
    if optimizer_disc is not None and ckpt.get("optimizer_disc_state_dict"):
        try:
            optimizer_disc.load_state_dict(ckpt["optimizer_disc_state_dict"])
        except Exception:
            print("  ⚠️ Could not load discriminator optimizer state. Using fresh optimizer.")
    print(f"  ✅ Resumed from {path} at step {ckpt.get('step', 0)}")
    return int(ckpt.get("step", 0))


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Warmup + Cosine Annealing LR scheduler with a 0.05 floor (Kokoro / FastPitch standard)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
