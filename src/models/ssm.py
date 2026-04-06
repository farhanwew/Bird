"""
ssm.py — ProtoSSM v5 architecture for BirdCLEF 2026.

Models:
  - SelectiveSSM       : Mamba-style input-dependent SSM
  - TemporalCrossAttention : Multi-head self-attention across windows
  - ProtoSSMv5         : Bidirectional SSM + cross-attn + prototypical classification
  - ResidualSSM        : Lightweight second-pass error correction model

All models process file-level sequences: (batch, n_windows=12, features).
This is the key difference from per-window EfficientNet approach.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SelectiveSSM — Mamba-style input-dependent state space model
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """
    Simplified Mamba-style selective state space model.
    Input-dependent discretization of continuous-time SSMs.
    Efficient for short sequences (T=12 windows per file).
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, d_conv,
            padding=d_conv - 1, groups=d_model
        )
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        # SSM parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            y: (B, T, d_model)
        """
        B, T, _ = x.shape

        xz = self.in_proj(x)                                  # (B, T, 2*d_model)
        x_ssm, z = xz.chunk(2, dim=-1)                        # each (B, T, d_model)

        # Depthwise conv along time
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)  # (B, T, d_model)

        # Input-dependent SSM parameters
        dt = F.softplus(self.dt_proj(x_conv))                 # (B, T, d_model)
        B_mat = self.B_proj(x_conv)                           # (B, T, d_state)
        C_mat = self.C_proj(x_conv)                           # (B, T, d_state)

        # State recurrence
        A = -torch.exp(self.A_log.float())                    # (d_model, d_state)
        h = torch.zeros(B, self.d_model, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            dt_t = dt[:, t, :].unsqueeze(-1)                  # (B, d_model, 1)
            dA = torch.exp(A.unsqueeze(0) * dt_t)             # (B, d_model, d_state)
            dB = dt_t * B_mat[:, t, :].unsqueeze(1)           # (B, d_model, d_state)
            x_t = x_conv[:, t, :]                             # (B, d_model)

            h = h * dA + x_t.unsqueeze(-1) * dB
            y_t = (h * C_mat[:, t, :].unsqueeze(1)).sum(-1)  # (B, d_model)
            y_t = y_t + x_t * self.D
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)                       # (B, T, d_model)
        y = y * F.silu(z)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# TemporalCrossAttention — non-local patterns across windows
# ---------------------------------------------------------------------------

class TemporalCrossAttention(nn.Module):
    """
    Multi-head self-attention across the 12 temporal windows.
    Captures long-range dependencies that sequential SSMs might miss.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ---------------------------------------------------------------------------
# ProtoSSMv5 — Full temporal model
# ---------------------------------------------------------------------------

class ProtoSSMv5(nn.Module):
    """
    Prototypical State Space Model v5.

    Pipeline:
      input_proj(d_input → d_model) + positional_enc + metadata_emb
      → N × BiSSM layers (forward + backward SSM, merged)
      → temporal cross-attention (optional)
      → prototype similarity + gated fusion with Perch logits
      → species_logits (B, T, n_classes)

    Key difference from per-window CNN: processes the entire 60s file
    (12 × 5s windows) as a sequence, capturing temporal context.
    """
    def __init__(
        self,
        d_input: int = 1536,
        d_model: int = 320,
        d_state: int = 32,
        n_ssm_layers: int = 4,
        n_classes: int = 234,
        n_windows: int = 12,
        dropout: float = 0.12,
        n_sites: int = 20,
        meta_dim: int = 24,
        use_cross_attn: bool = True,
        cross_attn_heads: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.use_cross_attn = use_cross_attn

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learned positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        # Metadata embeddings (site, hour)
        self.site_emb = nn.Embedding(n_sites + 1, meta_dim)   # +1 for unknown site
        self.hour_emb = nn.Embedding(25, meta_dim)             # 0-23 + unknown
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        # Bidirectional SSM stack
        self.ssm_fwd = nn.ModuleList()
        self.ssm_bwd = nn.ModuleList()
        self.ssm_merge = nn.ModuleList()
        self.ssm_norm = nn.ModuleList()
        self.ssm_drop = nn.ModuleList()
        for _ in range(n_ssm_layers):
            self.ssm_fwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_bwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_merge.append(nn.Linear(2 * d_model, d_model))
            self.ssm_norm.append(nn.LayerNorm(d_model))
            self.ssm_drop.append(nn.Dropout(dropout))

        # Temporal cross-attention
        if use_cross_attn:
            self.cross_attn = TemporalCrossAttention(d_model, n_heads=cross_attn_heads, dropout=dropout)

        # Prototypical classification head
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temperature = nn.Parameter(torch.ones(n_classes))

        # Gated fusion: alpha * proto_logits + (1 - alpha) * perch_logits
        self.fusion_alpha = nn.Parameter(torch.full((n_classes,), 0.5))

        # Family auxiliary head (initialized lazily)
        self.family_head = None

    def init_prototypes_from_data(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Data-driven prototype initialization.
        Args:
            embeddings: (N, d_model) — hidden states
            labels: (N, n_classes) — binary labels
        """
        with torch.no_grad():
            emb_norm = F.normalize(embeddings, dim=-1)
            for c in range(self.n_classes):
                mask = labels[:, c] > 0.5
                if mask.sum() > 0:
                    proto = emb_norm[mask].mean(0)
                    self.prototypes.data[c] = F.normalize(proto, dim=0)

    def init_family_head(self, n_families: int):
        """Add auxiliary taxonomic family prediction head."""
        self.family_head = nn.Linear(self.d_model, n_families)

    def forward(
        self,
        emb: torch.Tensor,
        perch_logits: torch.Tensor = None,
        site_ids: torch.Tensor = None,
        hours: torch.Tensor = None,
    ):
        """
        Args:
            emb:          (B, T, 1536) Perch embeddings
            perch_logits: (B, T, n_classes) mapped Perch logits (optional)
            site_ids:     (B,) site index (optional)
            hours:        (B,) hour of day 0-23 (optional)
        Returns:
            species_logits: (B, T, n_classes)
            family_logits:  (B, n_families) or None
            h:              (B, T, d_model) hidden states
        """
        B, T, _ = emb.shape

        # Project input + positional encoding
        h = self.input_proj(emb) + self.pos_enc[:, :T, :]     # (B, T, d_model)

        # Metadata conditioning
        if site_ids is not None or hours is not None:
            s = site_ids if site_ids is not None else torch.zeros(B, dtype=torch.long, device=emb.device)
            hr = hours if hours is not None else torch.zeros(B, dtype=torch.long, device=emb.device)
            meta = torch.cat([self.site_emb(s), self.hour_emb(hr)], dim=-1)  # (B, 2*meta_dim)
            h = h + self.meta_proj(meta).unsqueeze(1)          # broadcast over T

        # Bidirectional SSM layers
        for i in range(len(self.ssm_fwd)):
            h_fwd = self.ssm_fwd[i](h)
            h_bwd = self.ssm_bwd[i](h.flip(1)).flip(1)
            h_merged = self.ssm_merge[i](torch.cat([h_fwd, h_bwd], dim=-1))
            h = self.ssm_norm[i](h + self.ssm_drop[i](h_merged))

        # Cross-attention
        if self.use_cross_attn and self.cross_attn is not None:
            h = self.cross_attn(h)

        # Prototypical classification
        h_norm = F.normalize(h, dim=-1)                         # (B, T, d_model)
        proto_norm = F.normalize(self.prototypes, dim=-1)       # (n_classes, d_model)
        sim = torch.einsum('btd,cd->btc', h_norm, proto_norm)  # (B, T, n_classes)
        temp = self.proto_temperature.clamp(0.1, 10.0)
        proto_logits = sim * temp.unsqueeze(0).unsqueeze(0)

        # Gated fusion with Perch base logits
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)
            species_logits = alpha * proto_logits + (1 - alpha) * perch_logits
        else:
            species_logits = proto_logits

        # Auxiliary family head (mean over time)
        family_logits = None
        if self.family_head is not None:
            family_logits = self.family_head(h.mean(dim=1))    # (B, n_families)

        return species_logits, family_logits, h


# ---------------------------------------------------------------------------
# ResidualSSM — lightweight second-pass error correction
# ---------------------------------------------------------------------------

class ResidualSSM(nn.Module):
    """
    Lightweight SSM trained on prediction residuals of the first-pass ensemble.
    Learns to correct systematic biases (e.g., under-prediction at specific sites/times).

    Input: concat(Perch embeddings, first-pass logits) → BiSSM → correction
    Loss: MSE(correction, ground_truth - sigmoid(first_pass))
    """
    def __init__(
        self,
        d_input: int = 1536,
        d_scores: int = 234,
        d_model: int = 128,
        d_state: int = 16,
        n_classes: int = 234,
        n_windows: int = 12,
        dropout: float = 0.1,
        n_sites: int = 20,
        meta_dim: int = 8,
    ):
        super().__init__()
        self.d_model = d_model

        # Project concat(emb, scores) to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Metadata conditioning
        self.site_emb = nn.Embedding(n_sites + 1, meta_dim)
        self.hour_emb = nn.Embedding(25, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        # Single BiSSM layer
        self.ssm_fwd = SelectiveSSM(d_model, d_state)
        self.ssm_bwd = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # Correction head — initialized near zero for stable training start
        self.output_head = nn.Linear(d_model, n_classes)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        emb: torch.Tensor,
        first_pass_scores: torch.Tensor,
        site_ids: torch.Tensor = None,
        hours: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            emb:               (B, T, 1536)
            first_pass_scores: (B, T, n_classes) logits from ensemble
            site_ids:          (B,) optional
            hours:             (B,) optional
        Returns:
            correction: (B, T, n_classes) — add to first_pass_scores
        """
        B, T, _ = emb.shape

        x = torch.cat([emb, first_pass_scores], dim=-1)       # (B, T, d_input+d_scores)
        h = self.input_proj(x) + self.pos_enc[:, :T, :]

        if site_ids is not None or hours is not None:
            s = site_ids if site_ids is not None else torch.zeros(B, dtype=torch.long, device=emb.device)
            hr = hours if hours is not None else torch.zeros(B, dtype=torch.long, device=emb.device)
            meta = torch.cat([self.site_emb(s), self.hour_emb(hr)], dim=-1)
            h = h + self.meta_proj(meta).unsqueeze(1)

        h_fwd = self.ssm_fwd(h)
        h_bwd = self.ssm_bwd(h.flip(1)).flip(1)
        h_merged = self.ssm_merge(torch.cat([h_fwd, h_bwd], dim=-1))
        h = self.ssm_norm(h + self.drop(h_merged))

        return self.output_head(h)                             # (B, T, n_classes)
