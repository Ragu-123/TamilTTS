"""
RAD-TTS / FastPitch Exact Monotonic Alignment Network
======================================================
Based on 'One TTS Alignment To Rule Them All' (Badlani et al., NVIDIA / NeMo FastPitch).

1. Computes soft pairwise attention matrix log_A between text representations and mel frames.
2. Computes exact Forward-Sum Loss (differentiable forward DP over monotonic paths).
3. Extracts exact Viterbi MAS hard alignment path and duration counts on the SAME log_A matrix.
4. Computes true RAD-TTS Binarization Loss (cross-entropy between soft attention and hard path).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def viterbi_mas(log_attn_np, mask_np):
    """
    Viterbi Dynamic Programming for Monotonic Alignment Search (MAS).
    Traverses the EXACT SAME log-attention matrix as Forward-Sum.

    log_attn_np: [B, T_text, T_mel]
    mask_np:     [B, T_text, T_mel] (boolean joint mask)

    Returns:
    hard_path:   [B, T_text, T_mel] (binary matrix with 1.0 along optimal path)
    """
    B, T_text, T_mel = log_attn_np.shape
    hard_path = np.zeros((B, T_text, T_mel), dtype=np.float32)

    for b in range(B):
        t_len = int(mask_np[b].sum(axis=1).astype(bool).sum())
        m_len = int(mask_np[b].sum(axis=0).astype(bool).sum())
        if t_len == 0 or m_len == 0:
            continue

        log_A = log_attn_np[b, :t_len, :m_len]
        Q = np.full((t_len, m_len), -1e9, dtype=np.float32)
        Q[0, 0] = log_A[0, 0]

        for t in range(1, m_len):
            for s in range(min(t + 1, t_len)):
                if s == 0:
                    Q[0, t] = Q[0, t - 1] + log_A[0, t]
                else:
                    Q[s, t] = max(Q[s, t - 1], Q[s - 1, t - 1]) + log_A[s, t]

        curr_s = t_len - 1
        for t in range(m_len - 1, -1, -1):
            hard_path[b, curr_s, t] = 1.0
            if curr_s > 0:
                if t == 0 or Q[curr_s - 1, t - 1] >= Q[curr_s, t - 1]:
                    curr_s -= 1

    return hard_path


def forward_sum_loss_exact(log_attn, text_lens, mel_lens):
    """
    Exact Differentiable Forward-Sum Loss (RAD-TTS / FastPitch Standard).
    Computes -log sum_{pi} P(pi) over all valid monotonic alignments.
    """
    B, T_text, T_mel = log_attn.shape
    device = log_attn.device
    loss_list = []

    for b in range(B):
        t_len = text_lens[b].item()
        m_len = mel_lens[b].item()
        log_A = log_attn[b, :t_len, :m_len]  # [t_len, m_len]

        # Forward DP recursion
        alpha = torch.full((t_len,), -1e9, device=device)
        alpha[0] = log_A[0, 0]

        for t in range(1, m_len):
            prev_diag = F.pad(alpha[:-1], (1, 0), value=-1e9)
            stacked = torch.stack([alpha, prev_diag], dim=0)
            alpha = log_A[:, t] + torch.logsumexp(stacked, dim=0)

        total_log_prob = alpha[t_len - 1]
        loss_list.append(-total_log_prob / m_len)

    return torch.stack(loss_list).mean()


def binarization_loss_exact(log_attn, hard_path, text_lens, mel_lens):
    """
    Exact RAD-TTS Binarization Loss.
    Cross-entropy between the soft attention log_A and the hard Viterbi alignment path.
    Forces soft attention to strictly follow the hard path.
    """
    B = log_attn.shape[0]
    loss_list = []
    for b in range(B):
        t_len = text_lens[b].item()
        m_len = mel_lens[b].item()
        h_path = hard_path[b, :t_len, :m_len]
        l_attn = log_attn[b, :t_len, :m_len]
        loss_b = -(h_path * l_attn).sum() / m_len
        loss_list.append(loss_b)
    return torch.stack(loss_list).mean()


class AlignmentNetwork(nn.Module):
    """
    Unified RAD-TTS / FastPitch Alignment Network.
    Uses the EXACT SAME attention matrix for Forward-Sum, Viterbi extraction, and Binarization.
    """
    def __init__(self, text_dim=512, mel_dim=80, attn_dim=128):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, attn_dim)
        self.mel_conv = nn.Sequential(
            nn.Conv1d(mel_dim, attn_dim, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(attn_dim, attn_dim, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(attn_dim, attn_dim, 1),
        )
        self.scale = attn_dim ** 0.5

    def forward(self, text_emb, mel, text_lens, mel_lens):
        """
        text_emb:  [B, T_text, text_dim]
        mel:       [B, mel_dim, T_mel]
        text_lens: [B] (actual character lengths per sample)
        mel_lens:  [B] (actual mel frame lengths per sample)

        Returns:
            durations:        [B, T_text] (exact alignment-derived frame counts per token)
            hard_path:        [B, T_text, T_mel] (binary alignment path)
            forward_sum_loss: scalar tensor (Exact Forward-Sum Loss)
            bin_loss:         scalar tensor (Exact RAD-TTS Binarization Loss)
        """
        B, T_text, _ = text_emb.shape
        T_mel = mel.shape[2]
        device = text_emb.device

        # 1. Project representations to shared alignment space
        t_proj = self.text_proj(text_emb)               # [B, T_text, attn_dim]
        m_proj = self.mel_conv(mel).transpose(1, 2)     # [B, T_mel, attn_dim]

        # 2. Pairwise energy matrix: [B, T_text, T_mel]
        energy = torch.bmm(t_proj, m_proj.transpose(1, 2)) / self.scale

        # 3. Dynamic sequence masking
        t_mask = torch.arange(T_text, device=device).unsqueeze(0) < text_lens.unsqueeze(1)
        m_mask = torch.arange(T_mel, device=device).unsqueeze(0) < mel_lens.unsqueeze(1)
        joint_mask = t_mask.unsqueeze(2) & m_mask.unsqueeze(1)

        energy_masked = energy.masked_fill(~joint_mask, -1e9)

        # 4. Soft attention distribution log_A (Softmax over text tokens)
        log_attn = F.log_softmax(energy_masked, dim=1)

        # 5. Exact Forward-Sum Loss on log_A
        fwd_loss = forward_sum_loss_exact(log_attn, text_lens, mel_lens)

        # 6. Extract Viterbi hard path from the EXACT SAME log_A matrix
        log_attn_np = log_attn.detach().cpu().numpy()
        mask_np = joint_mask.cpu().numpy()
        hard_path_np = viterbi_mas(log_attn_np, mask_np)
        hard_path = torch.from_numpy(hard_path_np).to(device)

        # Alignment-derived durations per character: [B, T_text]
        durations = hard_path.sum(dim=-1)

        # 7. Exact RAD-TTS Binarization Loss (cross entropy between log_A and hard_path)
        bin_loss = binarization_loss_exact(log_attn, hard_path, text_lens, mel_lens)

        return durations, hard_path, fwd_loss, bin_loss
