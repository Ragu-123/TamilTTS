"""
Monotonic Alignment Search (MAS)
================================
Dynamic Programming algorithm for finding the optimal monotonic path
between text tokens and mel-spectrogram frames.

Standard implementation used in VITS, StyleTTS 2, and AI4Bharat FastPitch.
"""
import torch
import numpy as np


def maximum_path_numpy(value, mask, max_neg_val=-1e9):
    """
    Vectorized/Fast Dynamic Programming for Monotonic Alignment Search.
    value: [B, T_text, T_mel] — log-likelihood / similarity matrix
    mask:  [B, T_text, T_mel] — boolean mask
    Returns:
    path:  [B, T_text, T_mel] — binary matrix with 1.0 along the optimal monotonic path
    """
    B, T_text, T_mel = value.shape
    path = np.zeros((B, T_text, T_mel), dtype=np.float32)

    for b in range(B):
        val = value[b].copy()
        m = mask[b]
        val[~m] = max_neg_val

        # Find actual lengths for batch item
        t_text_len = int(m.sum(axis=1).astype(bool).sum())
        t_mel_len = int(m.sum(axis=0).astype(bool).sum())

        if t_text_len == 0 or t_mel_len == 0:
            continue

        # DP Table Initialization
        Q = np.full((t_text_len, t_mel_len), max_neg_val, dtype=np.float32)
        Q[0, 0] = val[0, 0]

        # Fill DP table
        for j in range(1, t_mel_len):
            for i in range(min(j + 1, t_text_len)):
                if i == 0:
                    Q[0, j] = Q[0, j - 1] + val[0, j]
                else:
                    Q[i, j] = max(Q[i, j - 1], Q[i - 1, j - 1]) + val[i, j]

        # Backtrack optimal monotonic path
        curr_i = t_text_len - 1
        for j in range(t_mel_len - 1, -1, -1):
            path[b, curr_i, j] = 1.0
            if curr_i > 0:
                if j == 0 or Q[curr_i - 1, j - 1] >= Q[curr_i, j - 1]:
                    curr_i -= 1

    return path


def monotonic_alignment_search(text_emb, mel_emb, text_mask=None, mel_mask=None):
    """
    Computes Monotonic Alignment Search between text representations and mel representations.

    text_emb:  [B, T_text, H]
    mel_emb:   [B, T_mel, H]
    text_mask: [B, T_text] (True for PAD)
    mel_mask:  [B, T_mel] (True for PAD)

    Returns:
    durations: [B, T_text] — True duration (in mel frames) for each text token!
    alignment: [B, T_text, T_mel] — Monotonic alignment map
    """
    device = text_emb.device
    B, T_text, H = text_emb.shape
    T_mel = mel_emb.shape[1]

    # Normalize embeddings for cosine similarity
    text_norm = torch.nn.functional.normalize(text_emb, dim=-1)
    mel_norm = torch.nn.functional.normalize(mel_emb, dim=-1)

    # Compute similarity matrix: [B, T_text, T_mel]
    sim = torch.bmm(text_norm, mel_norm.transpose(1, 2))

    # Construct joint mask: [B, T_text, T_mel]
    if text_mask is not None:
        t_mask = (~text_mask).unsqueeze(2)  # [B, T_text, 1]
    else:
        t_mask = torch.ones(B, T_text, 1, dtype=torch.bool, device=device)

    if mel_mask is not None:
        m_mask = (~mel_mask).unsqueeze(1)  # [B, 1, T_mel]
    else:
        m_mask = torch.ones(B, 1, T_mel, dtype=torch.bool, device=device)

    joint_mask = (t_mask & m_mask).cpu().numpy()
    sim_np = sim.detach().cpu().numpy()

    # Run fast Dynamic Programming
    path_np = maximum_path_numpy(sim_np, joint_mask)
    path = torch.from_numpy(path_np).to(device)  # [B, T_text, T_mel]

    # Sum along mel axis to get true frame count per character
    durations = path.sum(dim=-1)  # [B, T_text]

    return durations, path
