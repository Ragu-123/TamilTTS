"""
RAD-TTS / FastPitch Alignment Network
====================================
Based on 'One TTS Alignment To Rule Them All' (Badlani et al., NVIDIA / NeMo FastPitch).
Computes a soft attention alignment matrix between text representations and mel frames,
trained using Forward-Sum Loss and Binarization Loss.
Extracts mathematically grounded, alignment-derived durations per character via Viterbi / MAS.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def maximum_path_numpy(value, mask, max_neg_val=-1e9):
    """
    Fast Dynamic Programming for Monotonic Alignment Search.
    value: [B, T_text, T_mel] — log-likelihood / energy matrix
    mask:  [B, T_text, T_mel] — boolean mask (True for valid region)
    Returns:
    path:  [B, T_text, T_mel] — binary matrix with 1.0 along the optimal monotonic path
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


class AlignmentNetwork(nn.Module):
    """
    Learned Text-to-Mel Alignment Network (RAD-TTS / FastPitch Standard).
    
    1. Projects text embeddings and mel features to a shared attention space.
    2. Computes pairwise energy: E = (T_proj * M_proj^T) / sqrt(D).
    3. Calculates Forward-Sum Loss (all valid monotonic alignments log-probability sum).
    4. Calculates Binarization Loss to enforce sharp, unambiguous character boundaries.
    5. Extracts exact alignment-derived durations per character via Viterbi MAS.
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
            forward_sum_loss: scalar tensor (Forward-Sum Loss)
            bin_loss:         scalar tensor (Binarization Loss)
        """
        B, T_text, _ = text_emb.shape
        T_mel = mel.shape[2]
        device = text_emb.device

        # 1. Project representations to alignment attention space
        t_proj = self.text_proj(text_emb)               # [B, T_text, attn_dim]
        m_proj = self.mel_conv(mel).transpose(1, 2)     # [B, T_mel, attn_dim]

        # 2. Pairwise energy matrix: [B, T_text, T_mel]
        energy = torch.bmm(t_proj, m_proj.transpose(1, 2)) / self.scale

        # 3. Dynamic sequence masking
        t_mask = torch.arange(T_text, device=device).unsqueeze(0) < text_lens.unsqueeze(1)  # [B, T_text]
        m_mask = torch.arange(T_mel, device=device).unsqueeze(0) < mel_lens.unsqueeze(1)    # [B, T_mel]
        joint_mask = t_mask.unsqueeze(2) & m_mask.unsqueeze(1)                             # [B, T_text, T_mel]

        energy_masked = energy.masked_fill(~joint_mask, -1e9)

        # 4. Soft attention distribution
        attn_soft = F.softmax(energy_masked, dim=1)  # Softmax over text tokens

        # 5. Binarization Loss: pushes soft attention weights towards 0 or 1
        bin_loss = -torch.mean(torch.abs(attn_soft - 0.5))

        # 6. Extract Viterbi hard monotonic path
        attn_log = F.log_softmax(energy_masked, dim=1)
        sim_np = attn_log.detach().cpu().numpy()
        mask_np = joint_mask.cpu().numpy()
        hard_path_np = maximum_path_numpy(sim_np, mask_np)
        hard_path = torch.from_numpy(hard_path_np).to(device)  # [B, T_text, T_mel]

        # Alignment-derived durations per character: [B, T_text]
        durations = hard_path.sum(dim=-1)

        # 7. Forward-Sum Loss (CTC formulation on monotonic state lattice)
        targets = torch.stack([
            torch.cat([
                torch.arange(1, text_lens[b] + 1, device=device),
                torch.zeros(T_text - text_lens[b], dtype=torch.long, device=device)
            ])
            for b in range(B)
        ])

        blank_energy = torch.zeros(B, 1, T_mel, device=device)
        full_energy = torch.cat([blank_energy, energy_masked], dim=1)         # [B, T_text+1, T_mel]
        log_probs = F.log_softmax(full_energy.permute(2, 0, 1), dim=-1)       # [T_mel, B, T_text+1]

        forward_sum_loss = F.ctc_loss(log_probs, targets, mel_lens, text_lens, blank=0, zero_infinity=True)

        return durations, hard_path, forward_sum_loss, bin_loss
