from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_collapse_score(h: torch.Tensor, n_intents: int = 4, eps: float = 1e-8) -> tuple[float, float]:
    """
    Compute R(l) = Tr(Sigma_W) / Tr(Sigma_B) from a hidden-state matrix.

    Uses a fast approximate partitioning via sign of PCA directions rather
    than k-means, so that it runs efficiently inside the training loop.

    Args:
        h: shape [B, D] - batch of hidden representations at one layer.
        n_intents: number of latent intent clusters.
    Returns:
        (R_score, evidence_proxy) both as Python floats.
    """
    h_t = h.detach().float()
    B, D = h_t.shape
    if B < n_intents:
        return 0.0, 0.0

    # Approximate partitioning: split on sign of top-2 PCA projections.
    centered = h_t - h_t.mean(dim=0, keepdim=True)
    try:
        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        proj1 = (centered @ Vt[0]) > 0
        proj2 = (centered @ Vt[1]) > 0
        # 4 quadrant clusters from cross-product of two binary splits.
        labels = proj1.long() * 2 + proj2.long()
    except Exception:
        return 0.0, 0.0

    global_mean = h_t.mean(dim=0)
    within_var, between_var = 0.0, 0.0
    cluster_means = []
    for k in range(n_intents):
        mask = labels == k
        if mask.sum() < 2:
            cluster_means.append(global_mean)
            continue
        cluster = h_t[mask]
        mu_k = cluster.mean(dim=0)
        cluster_means.append(mu_k)
        within_var += (cluster - mu_k).pow(2).sum().item()
        between_var += float(mask.sum()) * (mu_k - global_mean).pow(2).sum().item()

    within_tr = within_var / max(B, 1)
    between_tr = between_var / max(B, 1)
    r = within_tr / (between_tr + eps)

    means_t = torch.stack(cluster_means, dim=0)
    evidence_proxy = float(means_t.var(dim=0).mean().item())
    return float(r), evidence_proxy


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with causal self-attention and 4× FFN expansion."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# Backward-compatible alias.
MinimalTransformerBlock = TransformerBlock


class CARRBackbone(nn.Module):
    """
    Production sequential recommender backbone for CARR-v2 experiments.

    Architecture: 12-layer pre-norm transformer, d_model=256, 8 attention heads,
    4× FFN expansion, causal masking, embedding dropout, and layer-norm output
    projection. Exposes per-layer last-token hidden states for collapse diagnostics.
    """

    def __init__(
        self,
        num_items: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 12,
        max_seq: int = 200,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.n_layers = n_layers
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_items)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask.bool(), float("-inf"))

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            seq: [B, T] integer item IDs (0 = padding)
        Returns:
            logits: [B, num_items]
            hiddens: list of L tensors each [B, D] — last-token hidden per layer
        """
        B, T = seq.shape
        pos = torch.arange(T, device=seq.device).unsqueeze(0)
        x = self.emb_dropout(self.item_emb(seq) + self.pos_emb(pos))
        causal = self._causal_mask(T, seq.device)
        hiddens: list[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x, causal_mask=causal)
            hiddens.append(x[:, -1, :])
        logits = self.head(self.final_norm(x[:, -1, :]))
        return logits, hiddens


# Backward-compatible alias so existing imports of MiniBackbone continue to work.
MiniBackbone = CARRBackbone
