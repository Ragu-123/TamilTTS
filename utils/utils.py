import os
import math
import torch


def save_checkpoint(path, model, optimizer, scheduler, step, loss):
    """Save model checkpoint to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "step": step, "loss": loss,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }, path)
    print(f"  💾 Checkpoint saved: {path} (Step {step}, Loss {loss:.4f})")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load model checkpoint from disk."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(model, "module"):
        model.module.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            print("  ⚠️ Could not load optimizer state (param groups changed). Using fresh optimizer.")
    if scheduler and ckpt.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception:
            print("  ⚠️ Could not load scheduler state. Using fresh scheduler.")
    print(f"  Resumed from step {ckpt['step']}, loss {ckpt['loss']:.4f}")
    return ckpt["step"]


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Warmup + Cosine Annealing LR scheduler (Kokoro / FastPitch standard)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
