"""
train_ssm.py — Training pipeline for the Perch + ProtoSSM v5 pipeline.

Prerequisites:
  1. Run train.py first to generate output/label_list.json
  2. Run extract_perch.py to cache Perch embeddings:
       python extract_perch.py --source train_soundscapes
       python extract_perch.py --source train_audio

Pipeline:
  - Load cached Perch embeddings (no audio reloading)
  - Train ProtoSSMv5 with focal BCE, distillation, SWA
  - Train per-class sklearn MLP probes on Perch embeddings
  - First-pass ensemble: 0.5 × ProtoSSM + 0.5 × MLP
  - Train ResidualSSM on prediction residuals
  - Save all artifacts to OUTPUT_DIR/ssm/

Usage:
    python train_ssm.py
    python train_ssm.py --config config.yaml --epochs 20
"""

import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.models.ssm import ProtoSSMv5, ResidualSSM
from src.taxonomy import TaxonomyManager
from src.prior import PriorAndProbeManager
from src.training_utils import (
    focal_bce_with_logits, build_pos_weights, mixup_files
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cache(cache_dir: str):
    """Load .npz arrays and parquet metadata from a perch_cache directory."""
    arrays = np.load(os.path.join(cache_dir, 'perch_arrays.npz'))
    meta_df = pd.read_parquet(os.path.join(cache_dir, 'perch_meta.parquet'))
    return arrays['emb_full'], arrays['scores_full_raw'], meta_df


def reshape_to_files(flat: np.ndarray, meta_df: pd.DataFrame, n_windows: int = 12):
    """
    Reshape flat (N_windows, feat) → (N_files, n_windows, feat).
    Also returns per-file metadata arrays.
    """
    filenames = meta_df['filename'].tolist()
    groups = []         # file index per window (for GroupKFold)
    file_starts = []    # start row in flat for each file
    unique_files = []

    i = 0
    file_idx = 0
    while i < len(filenames):
        fname = filenames[i]
        start = i
        while i < len(filenames) and filenames[i] == fname:
            i += 1
        file_starts.append(start)
        unique_files.append(fname)
        for _ in range(start, i):
            groups.append(file_idx)
        file_idx += 1

    n_files = len(unique_files)
    feat = flat.shape[1]

    out = np.zeros((n_files, n_windows, feat), dtype=flat.dtype)
    for fi, start in enumerate(file_starts):
        end = start + n_windows
        actual = flat[start:end]
        T = min(actual.shape[0], n_windows)
        out[fi, :T] = actual[:T]

    return out, np.array(groups, dtype=np.int64), unique_files, file_starts


# ---------------------------------------------------------------------------
# ProtoSSM training loop
# ---------------------------------------------------------------------------

def train_proto_ssm(
    model: ProtoSSMv5,
    emb: np.ndarray,                    # (N_files, 12, 1536)
    scores: np.ndarray,                 # (N_files, 12, n_classes) Perch logits
    labels: np.ndarray,                 # (N_files, 12, n_classes)
    site_ids: np.ndarray,               # (N_files,)
    hours: np.ndarray,                  # (N_files,)
    config: dict,
    device: torch.device,
    val_mask: np.ndarray = None,
    class_to_family: np.ndarray = None, # (n_classes,) species → family index
) -> ProtoSSMv5:
    """Train ProtoSSMv5 with focal BCE + distillation + family aux + SWA."""
    ssm_cfg = config.get('SSM', {})
    n_epochs = ssm_cfg.get('N_EPOCHS', 80)
    lr = ssm_cfg.get('LR', 8e-4)
    wd = ssm_cfg.get('WEIGHT_DECAY', 1e-3)
    patience = ssm_cfg.get('PATIENCE', 20)
    distill_w = ssm_cfg.get('DISTILL_WEIGHT', 0.15)
    family_w = 0.15
    label_smooth = ssm_cfg.get('LABEL_SMOOTHING', 0.03)
    focal_gamma = ssm_cfg.get('FOCAL_GAMMA', 2.5)
    mixup_alpha = ssm_cfg.get('MIXUP_ALPHA', 0.4)
    pos_weight_cap = ssm_cfg.get('POS_WEIGHT_CAP', 25.0)
    swa_start_frac = ssm_cfg.get('SWA_START_FRAC', 0.65)
    swa_lr = ssm_cfg.get('SWA_LR', 4e-4)
    warmup_epochs = 5  # skip mixup for first N epochs

    n_families = (int(class_to_family.max()) + 1) if class_to_family is not None else 0
    ctf_t = torch.from_numpy(class_to_family).long().to(device) if class_to_family is not None else None

    # Split train/val
    if val_mask is None:
        val_mask = np.zeros(len(emb), dtype=bool)
        val_mask[::10] = True  # simple 10% val fallback

    train_mask = ~val_mask
    y_flat = labels[train_mask].reshape(-1, labels.shape[-1])
    pos_weight = build_pos_weights(y_flat, cap=pos_weight_cap).to(device)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=n_epochs,
        steps_per_epoch=1, pct_start=0.1, anneal_strategy='cos'
    )

    # SWA
    swa_start = int(n_epochs * swa_start_frac)
    swa_model = optim.swa_utils.AveragedModel(model)
    swa_scheduler = optim.swa_utils.SWALR(optimizer, swa_lr=swa_lr)

    # Convert to tensors
    emb_t = torch.from_numpy(emb).to(device)
    scores_t = torch.from_numpy(scores).to(device)
    labels_t = torch.from_numpy(labels).to(device)
    site_t = torch.from_numpy(site_ids).long().to(device)
    hour_t = torch.from_numpy(hours).long().to(device)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    best_val_loss = float('inf')
    patience_count = 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        perm = np.random.permutation(train_idx)

        e_tr = emb_t[perm]
        s_tr = scores_t[perm]
        l_tr = labels_t[perm]
        si_tr = site_t[perm]
        hr_tr = hour_t[perm]

        # Apply mixup after warmup
        if epoch >= warmup_epochs and mixup_alpha > 0:
            e_tr, s_tr, l_tr, si_tr, hr_tr = mixup_files(
                e_tr, s_tr, l_tr, si_tr, hr_tr, alpha=mixup_alpha
            )

        # Apply label smoothing
        l_smooth = l_tr * (1 - label_smooth) + 0.5 * label_smooth

        optimizer.zero_grad()
        species_logits, family_logits, _ = model(e_tr, s_tr, si_tr, hr_tr)

        # Focal BCE loss
        main_loss = focal_bce_with_logits(
            species_logits, l_smooth, gamma=focal_gamma
        )

        # Distillation: match Perch logits
        distill_loss = focal_bce_with_logits(
            species_logits, torch.sigmoid(s_tr), gamma=focal_gamma
        ) if distill_w > 0 else 0.0

        # Family auxiliary loss — max over windows per file, then per-family presence
        family_loss = torch.tensor(0.0, device=device)
        if family_logits is not None and ctf_t is not None and n_families > 0:
            B_sz = l_tr.shape[0]
            # Any species active in any window → family present
            l_any = (l_tr > 0.5).float().max(dim=1).values  # (B, n_classes)
            fam_labels = torch.zeros(B_sz, n_families, device=device)
            for c_idx in range(l_any.shape[1]):
                f_idx = ctf_t[c_idx].item()
                fam_labels[:, f_idx] = torch.max(fam_labels[:, f_idx], l_any[:, c_idx])
            family_loss = F.binary_cross_entropy_with_logits(family_logits, fam_labels)

        loss = main_loss + distill_w * distill_loss + family_w * family_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # SWA or regular scheduler step
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits, _, _ = model(emb_t[val_idx], scores_t[val_idx],
                                   site_t[val_idx], hour_t[val_idx])
            val_loss = focal_bce_with_logits(
                v_logits, labels_t[val_idx], gamma=focal_gamma
            ).item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | train_loss={loss.item():.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Use SWA model if trained long enough
    if epoch >= swa_start:
        optim.swa_utils.update_bn(
            [(emb_t[train_idx], scores_t[train_idx], site_t[train_idx], hour_t[train_idx])],
            swa_model
        )
        return swa_model.module
    else:
        model.load_state_dict(best_state)
        return model


# ---------------------------------------------------------------------------
# MLP probe training (per-class sklearn)
# ---------------------------------------------------------------------------

def train_mlp_probes(
    emb: np.ndarray,       # (N_files, 12, 1536)
    labels: np.ndarray,    # (N_files, 12, n_classes)
    filenames: list,
    n_pca: int = 128,
):
    """
    Train per-class sklearn MLPClassifier on PCA-compressed Perch embeddings.
    Uses GroupKFold to avoid data leakage.

    Returns:
        probes: list of fitted MLPClassifier (one per class)
        pca:    fitted PCA transformer
        scaler: fitted StandardScaler
        oof_probs: (N_files, 12, n_classes) out-of-fold probabilities
    """
    N, T, D = emb.shape
    n_classes = labels.shape[-1]

    emb_flat = emb.reshape(-1, D)          # (N*T, 1536)
    labels_flat = labels.reshape(-1, n_classes)  # (N*T, n_classes)
    groups_flat = np.repeat(np.arange(N), T)     # (N*T,) file index

    # PCA + scale
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb_flat)
    pca = PCA(n_components=n_pca, random_state=42)
    emb_pca = pca.fit_transform(emb_scaled)    # (N*T, n_pca)

    oof_probs = np.zeros((N * T, n_classes), dtype=np.float32)
    probes = []

    print(f"  Training {n_classes} MLP probes (PCA-{n_pca} features)...")
    for c in tqdm(range(n_classes), desc="MLP probes"):
        y_c = labels_flat[:, c]
        n_pos = int(y_c.sum())

        if n_pos < 5:
            probes.append(None)
            continue

        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=500,
            learning_rate_init=5e-4,
            alpha=0.005,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            verbose=False,
        )

        gkf = GroupKFold(n_splits=5)
        for tr_idx, val_idx in gkf.split(emb_pca, y_c, groups=groups_flat):
            clf_fold = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=300,
                learning_rate_init=5e-4,
                alpha=0.005,
                random_state=42,
                verbose=False,
            )
            clf_fold.fit(emb_pca[tr_idx], y_c[tr_idx])
            prob = clf_fold.predict_proba(emb_pca[val_idx])
            if prob.shape[1] == 2:
                oof_probs[val_idx, c] = prob[:, 1]

        # Final probe trained on all data
        clf.fit(emb_pca, y_c)
        probes.append(clf)

    oof_probs_files = oof_probs.reshape(N, T, n_classes)
    return probes, pca, scaler, oof_probs_files


# ---------------------------------------------------------------------------
# ResidualSSM training
# ---------------------------------------------------------------------------

def train_residual_ssm(
    residual_model: ResidualSSM,
    emb: np.ndarray,           # (N, 12, 1536)
    first_pass: np.ndarray,    # (N, 12, n_classes) logits
    labels: np.ndarray,        # (N, 12, n_classes)
    site_ids: np.ndarray,
    hours: np.ndarray,
    config: dict,
    device: torch.device,
) -> ResidualSSM:
    """Train ResidualSSM on GT - sigmoid(first_pass) residuals via MSE."""
    ssm_cfg = config.get('SSM', {})
    n_epochs = min(ssm_cfg.get('N_EPOCHS', 80), 40)  # fewer epochs for residual
    lr = ssm_cfg.get('SWA_LR', 4e-4)
    wd = ssm_cfg.get('WEIGHT_DECAY', 1e-3)
    patience = min(ssm_cfg.get('PATIENCE', 20), 12)

    residual_model = residual_model.to(device)
    optimizer = optim.AdamW(residual_model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=n_epochs, steps_per_epoch=1
    )

    # Compute residuals
    residuals = labels - 1.0 / (1.0 + np.exp(-first_pass))  # GT - sigmoid(logits)
    residuals = residuals.astype(np.float32)

    # Simple 85/15 split
    n = len(emb)
    n_val = max(1, int(n * 0.15))
    val_idx = np.arange(n - n_val, n)
    train_idx = np.arange(n - n_val)

    emb_t = torch.from_numpy(emb).to(device)
    fp_t = torch.from_numpy(first_pass).to(device)
    res_t = torch.from_numpy(residuals).to(device)
    site_t = torch.from_numpy(site_ids).long().to(device)
    hour_t = torch.from_numpy(hours).long().to(device)

    best_val = float('inf')
    patience_count = 0
    best_state = None
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        residual_model.train()
        perm = np.random.permutation(train_idx)

        optimizer.zero_grad()
        correction = residual_model(emb_t[perm], fp_t[perm], site_t[perm], hour_t[perm])
        loss = criterion(correction, res_t[perm])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(residual_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        residual_model.eval()
        with torch.no_grad():
            corr_val = residual_model(emb_t[val_idx], fp_t[val_idx],
                                      site_t[val_idx], hour_t[val_idx])
            val_loss = criterion(corr_val, res_t[val_idx]).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in residual_model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    residual_model.load_state_dict(best_state)
    return residual_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--epochs', type=int, default=None, help="Override SSM.N_EPOCHS")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config.setdefault('SSM', {})['N_EPOCHS'] = args.epochs

    perch_cfg = config.get('PERCH', {})
    base_cache = perch_cfg.get('CACHE_DIR', 'output/perch_cache')
    output_dir = os.path.join(config.get('OUTPUT_DIR', 'output'), 'ssm')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load label list
    label_json = os.path.join(config.get('OUTPUT_DIR', 'output'), 'label_list.json')
    if not os.path.exists(label_json):
        raise FileNotFoundError(f"{label_json} not found. Run train.py first to generate it.")
    with open(label_json) as f:
        label_list = json.load(f)
    n_classes = len(label_list)
    print(f"Classes: {n_classes}")

    # Initialize taxonomy manager
    tax = TaxonomyManager(
        base_dir=config.get('OUTPUT_DIR', 'output'),
        taxonomy_csv=config.get('TAXONOMY_CSV', ''),
        label_list=label_list,
    )

    # Load cached embeddings — try soundscapes first, then audio
    sources = []
    for src in ('train_soundscapes', 'train_audio'):
        src_cache = os.path.join(base_cache, src)
        if os.path.exists(os.path.join(src_cache, 'perch_arrays.npz')):
            sources.append(src_cache)

    if not sources:
        raise FileNotFoundError(
            f"No Perch cache found in {base_cache}. "
            "Run: python extract_perch.py --source train_soundscapes"
        )

    print(f"Loading Perch cache from: {sources}")
    emb_list, scores_list, meta_list = [], [], []
    for src in sources:
        emb, scores, meta = load_cache(src)
        emb_list.append(emb)
        scores_list.append(scores)
        meta_list.append(meta)

    emb_flat = np.vstack(emb_list).astype(np.float32)
    scores_flat = np.vstack(scores_list).astype(np.float32)
    meta_df = pd.concat(meta_list, ignore_index=True)
    print(f"Total windows: {len(emb_flat)}")

    # Build site/hour metadata
    site_ids_flat, hours_flat = tax.build_site_index(meta_df)

    # Build label matrix
    labels_flat = np.zeros((len(emb_flat), n_classes), dtype=np.float32)
    soundscape_labels_csv = config.get('TRAIN_SOUNDSCAPES_LABELS', '')
    if soundscape_labels_csv and os.path.exists(soundscape_labels_csv):
        print("Building label matrix from soundscape labels...")
        labels_flat = tax.build_label_matrix(soundscape_labels_csv, meta_df)
    else:
        print("Warning: TRAIN_SOUNDSCAPES_LABELS not found. Labels will be empty.")

    # Reshape flat windows → per-file sequences (N_files, 12, feat)
    n_windows = perch_cfg.get('N_WINDOWS', 12)
    emb_files, groups, filenames, file_starts = reshape_to_files(emb_flat, meta_df, n_windows)
    scores_files = reshape_to_files(scores_flat, meta_df, n_windows)[0]
    labels_files = reshape_to_files(labels_flat, meta_df, n_windows)[0]

    # Per-file metadata
    site_ids_files = np.array([site_ids_flat[s] for s in file_starts], dtype=np.int64)
    hours_files = np.array([hours_flat[s] for s in file_starts], dtype=np.int64)

    n_files = len(emb_files)
    print(f"Files: {n_files}, Shape: emb={emb_files.shape}, labels={labels_files.shape}")

    # Validation split — GroupKFold(5) fold 0 ≈ 20% val, prevents leakage
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    _, val_file_idx = next(iter(kf.split(np.arange(n_files))))
    val_mask = np.zeros(n_files, dtype=bool)
    val_mask[val_file_idx] = True
    print(f"Val split: {val_mask.sum()}/{n_files} files (KFold-5 fold 0)")

    # -----------------------------------------------------------------------
    # Step 1: Train ProtoSSMv5
    # -----------------------------------------------------------------------
    ssm_cfg = config.get('SSM', {})
    print("\n=== Training ProtoSSMv5 ===")
    proto_model = ProtoSSMv5(
        d_input=1536,
        d_model=ssm_cfg.get('D_MODEL', 320),
        d_state=ssm_cfg.get('D_STATE', 32),
        n_ssm_layers=ssm_cfg.get('N_SSM_LAYERS', 4),
        n_classes=n_classes,
        n_windows=n_windows,
        dropout=ssm_cfg.get('DROPOUT', 0.12),
        n_sites=tax.n_sites + 1,
        use_cross_attn=ssm_cfg.get('USE_CROSS_ATTN', True),
        cross_attn_heads=ssm_cfg.get('CROSS_ATTN_HEADS', 8),
    )

    # Taxonomy auxiliary head
    class_to_family, n_families = tax.build_taxonomy_groups()
    proto_model.init_family_head(n_families)
    print(f"ProtoSSM params: {sum(p.numel() for p in proto_model.parameters()):,}")

    # Prototype initialization from training data hidden states (Task 11)
    print("Initializing prototypes from training data...")
    proto_model = proto_model.to(device)
    proto_model.eval()
    train_mask = ~val_mask
    with torch.no_grad():
        init_emb_t = torch.from_numpy(emb_files[train_mask]).to(device)
        init_sc_t = torch.from_numpy(scores_files[train_mask]).to(device)
        init_site_t = torch.from_numpy(site_ids_files[train_mask]).long().to(device)
        init_hour_t = torch.from_numpy(hours_files[train_mask]).long().to(device)
        _, _, h_init = proto_model(init_emb_t, init_sc_t, init_site_t, init_hour_t)
        h_flat = h_init.reshape(-1, h_init.shape[-1])             # (N_train*T, d_model)
        lab_flat = torch.from_numpy(labels_files[train_mask]).to(device).reshape(-1, n_classes)
        proto_model.init_prototypes_from_data(h_flat, lab_flat)
        del init_emb_t, init_sc_t, init_site_t, init_hour_t, h_init, h_flat, lab_flat
    print("  Prototypes initialized.")

    proto_model = train_proto_ssm(
        proto_model, emb_files, scores_files, labels_files,
        site_ids_files, hours_files, config, device, val_mask,
        class_to_family=class_to_family,
    )

    # Save ProtoSSM
    proto_path = os.path.join(output_dir, 'proto_ssm.pt')
    torch.save(proto_model.state_dict(), proto_path)
    print(f"Saved ProtoSSM → {proto_path}")

    # -----------------------------------------------------------------------
    # Step 2: Train MLP probes
    # -----------------------------------------------------------------------
    print("\n=== Training MLP Probes ===")
    probes, pca, scaler, oof_mlp = train_mlp_probes(
        emb_files[~val_mask], labels_files[~val_mask],
        [filenames[i] for i in np.where(~val_mask)[0]],
        n_pca=128,
    )

    # Save probes
    probes_path = os.path.join(output_dir, 'mlp_probes.pkl')
    with open(probes_path, 'wb') as f:
        pickle.dump({'probes': probes, 'pca': pca, 'scaler': scaler}, f)
    print(f"Saved MLP probes → {probes_path}")

    # -----------------------------------------------------------------------
    # Step 3: Generate first-pass logits on training set
    # -----------------------------------------------------------------------
    print("\n=== Building first-pass ensemble ===")
    proto_model.eval()
    emb_t = torch.from_numpy(emb_files).to(device)
    sc_t = torch.from_numpy(scores_files).to(device)
    site_t = torch.from_numpy(site_ids_files).long().to(device)
    hour_t = torch.from_numpy(hours_files).long().to(device)

    with torch.no_grad():
        proto_logits, _, _ = proto_model(emb_t, sc_t, site_t, hour_t)
        proto_logits = proto_logits.cpu().numpy()  # (N, 12, n_classes)

    # MLP probe predictions on full training set
    N, T, D = emb_files.shape
    emb_flat2 = emb_files.reshape(-1, D)
    emb_scaled2 = scaler.transform(emb_flat2)
    emb_pca2 = pca.transform(emb_scaled2)

    mlp_probs = np.zeros((N * T, n_classes), dtype=np.float32)
    for c, clf in enumerate(probes):
        if clf is not None:
            prob = clf.predict_proba(emb_pca2)
            if prob.shape[1] == 2:
                mlp_probs[:, c] = prob[:, 1]

    mlp_logits = np.log(mlp_probs.clip(1e-7, 1 - 1e-7) / (1 - mlp_probs.clip(1e-7, 1 - 1e-7)))
    mlp_logits = mlp_logits.reshape(N, T, n_classes)

    # Ensemble: 0.5 * ProtoSSM + 0.5 * MLP
    first_pass = 0.5 * proto_logits + 0.5 * mlp_logits

    # -----------------------------------------------------------------------
    # Step 4: Train ResidualSSM
    # -----------------------------------------------------------------------
    print("\n=== Training ResidualSSM ===")
    residual_model = ResidualSSM(
        d_input=1536,
        d_scores=n_classes,
        d_model=ssm_cfg.get('RESIDUAL_D_MODEL', 128),
        d_state=16,
        n_classes=n_classes,
        n_windows=n_windows,
        n_sites=tax.n_sites + 1,
    )
    print(f"ResidualSSM params: {sum(p.numel() for p in residual_model.parameters()):,}")

    residual_model = train_residual_ssm(
        residual_model, emb_files, first_pass, labels_files,
        site_ids_files, hours_files, config, device
    )

    residual_path = os.path.join(output_dir, 'residual_ssm.pt')
    torch.save(residual_model.state_dict(), residual_path)
    print(f"Saved ResidualSSM → {residual_path}")

    # -----------------------------------------------------------------------
    # Save metadata needed for inference
    # -----------------------------------------------------------------------
    meta_info = {
        'n_classes': n_classes,
        'n_windows': n_windows,
        'n_sites': tax.n_sites + 1,
        'site_to_idx': tax.site_to_idx,
        'ssm_config': ssm_cfg,
    }
    with open(os.path.join(output_dir, 'ssm_meta.json'), 'w') as f:
        json.dump(meta_info, f, indent=2)

    # -----------------------------------------------------------------------
    # Save prior tables (site × hour priors — needed for inference)
    # -----------------------------------------------------------------------
    print("\n=== Fitting prior tables ===")
    prior_mgr = PriorAndProbeManager(label_list)
    prior_tables = prior_mgr.fit_prior_tables(meta_df, labels_flat)
    prior_path = os.path.join(output_dir, 'prior_tables.pkl')
    with open(prior_path, 'wb') as f:
        pickle.dump({'prior_tables': prior_tables, 'label_list': label_list}, f)
    print(f"Saved prior tables → {prior_path}")

    print(f"\nAll SSM artifacts saved to {output_dir}/")
    print("Files to upload as Kaggle dataset:")
    for fname in ['proto_ssm.pt', 'residual_ssm.pt', 'mlp_probes.pkl',
                  'prior_tables.pkl', 'ssm_meta.json']:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / 1e6
            print(f"  {fname}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
