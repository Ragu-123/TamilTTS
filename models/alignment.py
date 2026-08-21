"""
Monotonic Alignment Search & CTC Alignment Module
=================================================
Provides supervised alignment between Tamil text characters and 22.05kHz mel frames
using a Convolutional Alignment Head supervised via PyTorch CTC Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AlignmentModule(nn.Module):
    """
    Acoustic-to-Text Alignment Network (FastPitch / RAD-TTS standard).
    Projects 80-channel mel frames to character vocabulary distribution.
    Supervised via CTC Loss to guarantee grounded phoneme-to-frame alignment.
    """
    def __init__(self, mel_channels=80, hidden_dim=256, vocab_size=256):
        super().__init__()
        self.conv1 = nn.Conv1d(mel_channels, hidden_dim, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, hidden_dim)
        self.proj = nn.Conv1d(hidden_dim, vocab_size, 1)

    def forward(self, mel):
        """
        mel: [B, 80, T_mel]
        Returns:
            logits: [B, vocab_size, T_mel]
            log_probs: [T_mel, B, vocab_size] for torch.nn.functional.ctc_loss
        """
        h = F.leaky_relu(self.norm1(self.conv1(mel)), 0.1)
        h = F.leaky_relu(self.norm2(self.conv2(h)), 0.1)
        logits = self.proj(h)  # [B, vocab_size, T_mel]
        log_probs = F.log_softmax(logits.permute(2, 0, 1), dim=-1)  # [T_mel, B, vocab_size]
        return logits, log_probs


def maximum_path_numpy(value, mask, max_neg_val=-1e9):
    """
    Fast Dynamic Programming for Monotonic Alignment Search.
    value: [B, T_text, T_mel] — log-likelihood matrix
    mask:  [B, T_text, T_mel] — boolean mask
    """
    B, T_text, T_mel = value.shape
    path = np.zeros((B, T_text, T_mel), dtype=np.float32)

    for b in range(B):
        val = value[b].copy()
        m = mask[b]
        val[~m] = max_neg_val

        t_text_len = int(m.sum(axis=1).astype(bool).sum())
        t_mel_len = int(m.sum(axis=0).astype(bool).sum())

        if t_text_len == 0 or t_mel_len == 0:
            continue

        Q = np.full((t_text_len, t_mel_len), max_neg_val, dtype=np.float32)
        Q[0, 0] = val[0, 0]

        for j in range(1, t_mel_len):
            for i in range(min(j + 1, t_text_len)):
                if i == 0:
                    Q[0, j] = Q[0, j - 1] + val[0, j]
                else:
                    Q[i, j] = max(Q[i, j - 1], Q[i - 1, j - 1]) + val[i, j]

        curr_i = t_text_len - 1
        for j in range(t_mel_len - 1, -1, -1):
            path[b, curr_i, j] = 1.0
            if curr_i > 0:
                if j == 0 or Q[curr_i - 1, j - 1] >= Q[curr_i, j - 1]:
                    curr_i -= 1

    return path


def extract_alignment_durations(text_tokens, ctc_logits, text_mask=None):
    """
    Extracts true frame counts per character from CTC alignment posterior.
    
    text_tokens: [B, T_text]
    ctc_logits:  [B, vocab_size, T_mel]
    text_mask:   [B, T_text] (True for PAD)
    
    Returns:
        durations: [B, T_text] — number of mel frames per character
    """
    device = text_tokens.device
    B, T_text = text_tokens.shape
    T_mel = ctc_logits.shape[2]

    # Gather log-probabilities for the actual text tokens across all mel frames
    log_probs = F.log_softmax(ctc_logits, dim=1)  # [B, vocab_size, T_mel]
    
    # Expand text tokens to gather from log_probs: [B, T_text, T_mel]
    tokens_expanded = text_tokens.unsqueeze(-1).expand(-1, -1, T_mel)  # [B, T_text, T_mel]
    
    # Gather emission log-likelihood: [B, T_text, T_mel]
    # For each batch item, gather probabilities of each character across time
    sim = torch.gather(log_probs, 1, tokens_expanded)  # [B, T_text, T_mel]

    # Create joint mask
    if text_mask is not None:
        t_mask = (~text_mask).unsqueeze(2).expand(-1, -1, T_mel)  # [B, T_text, T_mel]
    else:
        t_mask = torch.ones(B, T_text, T_mel, dtype=torch.bool, device=device)

    sim_np = sim.detach().cpu().numpy()
    mask_np = t_mask.cpu().numpy()

    # Dynamic Programming MAS
    path_np = maximum_path_numpy(sim_np, mask_np)
    path = torch.from_numpy(path_np).to(device)

    # Sum along mel frames to get exact duration per character
    durations = path.sum(dim=-1)  # [B, T_text]
    return durations, path
