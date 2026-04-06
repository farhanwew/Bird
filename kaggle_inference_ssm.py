# ── Cell 1: Install TF + dependencies ────────────────────────────────────────
# TF wheel must be pre-loaded in a Kaggle dataset (no internet allowed)
import subprocess, os, sys

# Try loading TF from Kaggle dataset (pre-downloaded wheel)
TF_WHEEL_DIR = '/kaggle/input/birdclef2026-perch-model/wheels'
if os.path.isdir(TF_WHEEL_DIR):
    for wheel in os.listdir(TF_WHEEL_DIR):
        if wheel.endswith('.whl'):
            subprocess.run(['pip', 'install', '--quiet', '--no-deps',
                            os.path.join(TF_WHEEL_DIR, wheel)])

subprocess.run(['pip', 'install', 'soundfile', '-q'], capture_output=True)

# ── Cell 2: Paths & Imports ───────────────────────────────────────────────────
import json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

# ── PATHS — adjust to your Kaggle dataset names ──────────────────────────────
COMP_DIR      = '/kaggle/input/birdclef-2026'
CODE_DIR      = '/kaggle/input/birdclef2026-ssm-code'     # src/ directory
CACHE_DIR     = '/kaggle/input/birdclef2026-perch-cache'  # perch_arrays.npz etc.
PERCH_DIR     = '/kaggle/input/birdclef2026-perch-model/perch_v2_cpu/1'
WORK_DIR      = '/kaggle/working'
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, CODE_DIR)
os.makedirs(WORK_DIR, exist_ok=True)

N_WINDOWS   = 12
WINDOW_SEC  = 5
SAMPLE_RATE = 32000
N_CLASSES   = 234
RESIDUAL_W  = 0.35   # ResidualSSM correction weight
MLP_W       = 0.50   # MLP probe ensemble weight
PROTO_W     = 0.50   # ProtoSSM ensemble weight


# ── Cell 3: Load label list ───────────────────────────────────────────────────
label_list_path = os.path.join(CACHE_DIR, 'label_list.json')
if os.path.exists(label_list_path):
    with open(label_list_path) as f:
        label_list = json.load(f)
else:
    sub_df = pd.read_csv(f'{COMP_DIR}/sample_submission.csv')
    label_list = [c for c in sub_df.columns if c != 'row_id']
n_classes = len(label_list)
print(f"Classes: {n_classes}")


# ── Cell 4: Load cached training embeddings ───────────────────────────────────
# These are pre-extracted by running extract_perch.py on training data
print("Loading training embeddings...")

# Merge soundscapes + audio caches if both available
emb_list, scores_list, meta_list = [], [], []
for src in ('train_soundscapes', 'train_audio'):
    src_dir = os.path.join(CACHE_DIR, src)
    arrays_path = os.path.join(src_dir, 'perch_arrays.npz')
    meta_path = os.path.join(src_dir, 'perch_meta.parquet')
    if os.path.exists(arrays_path) and os.path.exists(meta_path):
        arr = np.load(arrays_path)
        emb_list.append(arr['emb_full'])
        scores_list.append(arr['scores_full_raw'])
        meta_list.append(pd.read_parquet(meta_path))
        print(f"  Loaded {src}: {arr['emb_full'].shape[0]} windows")

if not emb_list:
    raise FileNotFoundError(f"No Perch cache found in {CACHE_DIR}.")

emb_train_flat   = np.vstack(emb_list).astype(np.float32)
scores_train_flat = np.vstack(scores_list).astype(np.float32)
meta_train       = pd.concat(meta_list, ignore_index=True)
print(f"Total training windows: {len(emb_train_flat)}")


# ── Cell 5: Build label matrix ────────────────────────────────────────────────
from src.taxonomy import TaxonomyManager

label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}
labels_train_flat = np.zeros((len(emb_train_flat), n_classes), dtype=np.float32)

soundscape_labels_csv = f'{COMP_DIR}/train_soundscapes_labels.csv'
if os.path.exists(soundscape_labels_csv):
    tax = TaxonomyManager(
        base_dir=WORK_DIR,
        taxonomy_csv=f'{COMP_DIR}/taxonomy.csv',
        label_list=label_list,
    )
    labels_train_flat = tax.build_label_matrix(soundscape_labels_csv, meta_train)
    site_ids_flat, hours_flat = tax.build_site_index(meta_train)
else:
    site_ids_flat = np.zeros(len(emb_train_flat), dtype=np.int64)
    hours_flat    = np.zeros(len(emb_train_flat), dtype=np.int64)
    tax = TaxonomyManager(base_dir=WORK_DIR, label_list=label_list)

print(f"Label matrix: {labels_train_flat.shape}, positive rate: {labels_train_flat.mean():.4f}")


# ── Cell 6: Fit prior tables ──────────────────────────────────────────────────
from src.prior import PriorAndProbeManager

prior_mgr = PriorAndProbeManager(label_list)
prior_tables = prior_mgr.fit_prior_tables(meta_train, labels_train_flat)
print(f"Prior tables fitted. Sites: {prior_tables['n_sites']}")


# ── Cell 7: Reshape flat → file-level sequences ───────────────────────────────
def reshape_to_files(flat, meta_df, n_windows=12):
    filenames = meta_df['filename'].tolist()
    file_info, groups = [], []
    i, fi = 0, 0
    while i < len(filenames):
        fname = filenames[i]
        start = i
        while i < len(filenames) and filenames[i] == fname:
            i += 1
        file_info.append({'filename': fname, 'start': start, 'end': i})
        for _ in range(start, i):
            groups.append(fi)
        fi += 1

    n_files = len(file_info)
    feat = flat.shape[1]
    out = np.zeros((n_files, n_windows, feat), dtype=flat.dtype)
    for k, info in enumerate(file_info):
        s, e = info['start'], info['end']
        T = min(e - s, n_windows)
        out[k, :T] = flat[s:e][:T]
    return out, np.array(groups), file_info

emb_files,    groups, file_info = reshape_to_files(emb_train_flat, meta_train, N_WINDOWS)
scores_files, _,      _         = reshape_to_files(scores_train_flat, meta_train, N_WINDOWS)
labels_files, _,      _         = reshape_to_files(labels_train_flat, meta_train, N_WINDOWS)
starts = [f['start'] for f in file_info]
site_ids_files = np.array([site_ids_flat[s] for s in starts], dtype=np.int64)
hours_files    = np.array([hours_flat[s]    for s in starts], dtype=np.int64)
n_files_train  = len(emb_files)
print(f"Training files: {n_files_train}, shape: {emb_files.shape}")


# ── Cell 8: Train MLP probes ─────────────────────────────────────────────────
print("Training MLP probes...")
N, T, D = emb_files.shape
emb_flat2 = emb_files.reshape(-1, D)
labels_flat2 = labels_files.reshape(-1, n_classes)
groups_flat = np.repeat(np.arange(N), T)

scaler = StandardScaler()
emb_scaled = scaler.fit_transform(emb_flat2)
pca = PCA(n_components=128, random_state=42)
emb_pca = pca.fit_transform(emb_scaled)

probes = []
for c in tqdm(range(n_classes), desc="MLP probes"):
    y_c = labels_flat2[:, c]
    if y_c.sum() < 5:
        probes.append(None)
        continue
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128), max_iter=300,
        learning_rate_init=5e-4, alpha=0.005,
        early_stopping=True, validation_fraction=0.15,
        random_state=42, verbose=False,
    )
    clf.fit(emb_pca, y_c)
    probes.append(clf)

print(f"Trained {sum(p is not None for p in probes)}/{n_classes} probes")


# ── Cell 9: Train ProtoSSMv5 ─────────────────────────────────────────────────
from src.models.ssm import ProtoSSMv5, ResidualSSM
from src.training_utils import focal_bce_with_logits, build_pos_weights, mixup_files as mixup_fn

device = torch.device('cpu')
n_sites = int(site_ids_files.max()) + 2

proto_model = ProtoSSMv5(
    d_input=1536, d_model=320, d_state=32, n_ssm_layers=4,
    n_classes=n_classes, n_windows=N_WINDOWS,
    dropout=0.12, n_sites=n_sites, use_cross_attn=True, cross_attn_heads=8,
)
print(f"ProtoSSM params: {sum(p.numel() for p in proto_model.parameters()):,}")

N_EPOCHS = 80
LR = 8e-4
FOCAL_GAMMA = 2.5
SWA_START = int(N_EPOCHS * 0.65)

y_flat_train = labels_files.reshape(-1, n_classes)
pos_weight = build_pos_weights(y_flat_train, cap=25.0).to(device)

optimizer = torch.optim.AdamW(proto_model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, epochs=N_EPOCHS, steps_per_epoch=1, pct_start=0.1
)
swa_model = torch.optim.swa_utils.AveragedModel(proto_model)
swa_sched  = torch.optim.swa_utils.SWALR(optimizer, swa_lr=4e-4)

emb_t    = torch.from_numpy(emb_files)
scores_t = torch.from_numpy(scores_files)
labels_t = torch.from_numpy(labels_files)
site_t   = torch.from_numpy(site_ids_files).long()
hour_t   = torch.from_numpy(hours_files).long()

from sklearn.model_selection import KFold as _KFold
_kf = _KFold(n_splits=5, shuffle=True, random_state=42)
_, _val_file_idx = next(iter(_kf.split(np.arange(n_files_train))))
val_mask = np.zeros(n_files_train, dtype=bool)
val_mask[_val_file_idx] = True
tr_idx = np.where(~val_mask)[0]
vl_idx = np.where(val_mask)[0]

# Initialize prototypes from training data hidden states
print("Initializing prototypes from training data...")
proto_model.eval()
with torch.no_grad():
    _, _, h_init = proto_model(emb_t[tr_idx], scores_t[tr_idx], site_t[tr_idx], hour_t[tr_idx])
    h_flat = h_init.reshape(-1, h_init.shape[-1])
    lab_flat = labels_t[tr_idx].reshape(-1, n_classes)
    proto_model.init_prototypes_from_data(h_flat, lab_flat)
    del h_init, h_flat, lab_flat
print("  Prototypes initialized.")

proto_model.train()
best_val, patience, best_state = float('inf'), 0, None

for epoch in range(N_EPOCHS):
    perm = np.random.permutation(tr_idx)
    e_tr, s_tr, l_tr = emb_t[perm], scores_t[perm], labels_t[perm]
    si_tr, hr_tr = site_t[perm], hour_t[perm]

    if epoch >= 5:
        e_tr, s_tr, l_tr, si_tr, hr_tr = mixup_fn(e_tr, s_tr, l_tr, si_tr, hr_tr, alpha=0.4)

    l_smooth = l_tr * 0.97 + 0.015

    optimizer.zero_grad()
    sp_logits, _, _ = proto_model(e_tr, s_tr, si_tr, hr_tr)
    loss = focal_bce_with_logits(sp_logits, l_smooth, gamma=FOCAL_GAMMA)
    loss += 0.15 * focal_bce_with_logits(sp_logits, torch.sigmoid(s_tr), gamma=FOCAL_GAMMA)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(proto_model.parameters(), 1.0)
    optimizer.step()

    proto_model.eval()
    with torch.no_grad():
        vl_logits, _, _ = proto_model(emb_t[vl_idx], scores_t[vl_idx], site_t[vl_idx], hour_t[vl_idx])
        val_loss = focal_bce_with_logits(vl_logits, labels_t[vl_idx], gamma=FOCAL_GAMMA).item()
    proto_model.train()

    if epoch >= SWA_START:
        swa_model.update_parameters(proto_model)
        swa_sched.step()
    else:
        scheduler.step()

    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in proto_model.state_dict().items()}
        patience = 0
    else:
        patience += 1
        if patience >= 20:
            print(f"Early stop @ epoch {epoch+1}")
            break

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{N_EPOCHS} | train={loss.item():.4f} | val={val_loss:.4f}")

proto_model.load_state_dict(best_state)
proto_model.eval()
print("ProtoSSM training done.")


# ── Cell 10: Build first-pass ensemble ────────────────────────────────────────
print("Building first-pass ensemble logits...")
with torch.no_grad():
    proto_logits_train, _, _ = proto_model(emb_t, scores_t, site_t, hour_t)
    proto_logits_train = proto_logits_train.numpy()

mlp_probs_train = np.zeros((N * T, n_classes), dtype=np.float32)
for c, clf in enumerate(probes):
    if clf is not None:
        prob = clf.predict_proba(emb_pca)
        if prob.shape[1] == 2:
            mlp_probs_train[:, c] = prob[:, 1]

mlp_logits_train = np.log(
    mlp_probs_train.clip(1e-7, 1-1e-7) / (1 - mlp_probs_train.clip(1e-7, 1-1e-7))
).reshape(N, T, n_classes)

first_pass_train = PROTO_W * proto_logits_train + MLP_W * mlp_logits_train


# ── Cell 11: Train ResidualSSM ────────────────────────────────────────────────
print("Training ResidualSSM...")
residual_model = ResidualSSM(
    d_input=1536, d_scores=n_classes, d_model=128, d_state=16,
    n_classes=n_classes, n_windows=N_WINDOWS, n_sites=n_sites,
)
residuals_train = labels_files - 1.0 / (1.0 + np.exp(-first_pass_train))

res_optimizer = torch.optim.AdamW(residual_model.parameters(), lr=4e-4, weight_decay=1e-3)
fp_t_all  = torch.from_numpy(first_pass_train.astype(np.float32))
res_t_all = torch.from_numpy(residuals_train.astype(np.float32))
mse = nn.MSELoss()

n_val_res = max(1, N // 7)
tr_res = np.arange(N - n_val_res)
vl_res = np.arange(N - n_val_res, N)
best_res_val, best_res_state = float('inf'), None

for epoch in range(40):
    perm = np.random.permutation(tr_res)
    residual_model.train()
    res_optimizer.zero_grad()
    corr = residual_model(emb_t[perm], fp_t_all[perm], site_t[perm], hour_t[perm])
    loss = mse(corr, res_t_all[perm])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(residual_model.parameters(), 1.0)
    res_optimizer.step()

    residual_model.eval()
    with torch.no_grad():
        corr_v = residual_model(emb_t[vl_res], fp_t_all[vl_res], site_t[vl_res], hour_t[vl_res])
        vl = mse(corr_v, res_t_all[vl_res]).item()
    if vl < best_res_val:
        best_res_val = vl
        best_res_state = {k: v.clone() for k, v in residual_model.state_dict().items()}

residual_model.load_state_dict(best_res_state)
residual_model.eval()
print("ResidualSSM done.")


# ── Cell 12: Extract Perch embeddings for test soundscapes ────────────────────
print("Extracting Perch embeddings for test soundscapes...")

try:
    import tensorflow as tf
    from extract_perch import PerchExtractor, parse_filename_metadata

    extractor = PerchExtractor(
        model_dir=PERCH_DIR,
        sample_rate=SAMPLE_RATE,
        window_sec=WINDOW_SEC,
        n_windows=N_WINDOWS,
    )

    test_paths = sorted(Path(f'{COMP_DIR}/test_soundscapes').glob('*.ogg'))
    print(f"Test files: {len(test_paths)}")

    emb_test_list, scores_test_list = [], []
    meta_test_rows = []

    for path in tqdm(test_paths, desc="Perch extract"):
        fname = path.name
        stem  = path.stem
        meta  = parse_filename_metadata(fname)
        try:
            emb, logits = extractor.extract_file(str(path))
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            emb    = np.zeros((N_WINDOWS, 1536), dtype=np.float32)
            logits = np.zeros((N_WINDOWS, n_classes), dtype=np.float32)

        emb_test_list.append(emb)
        scores_test_list.append(logits)
        for t in range(N_WINDOWS):
            end_sec = (t + 1) * WINDOW_SEC
            meta_test_rows.append({
                'row_id': f"{stem}_{end_sec}",
                'filename': fname,
                'site': meta['site'],
                'hour_utc': meta['hour_utc'],
            })

    emb_test_flat    = np.vstack(emb_test_list).astype(np.float32)
    scores_test_flat = np.vstack(scores_test_list).astype(np.float32)
    meta_test        = pd.DataFrame(meta_test_rows)

except ImportError:
    print("TF not available — loading test cache if pre-extracted...")
    test_cache = os.path.join(CACHE_DIR, 'test_soundscapes')
    arr = np.load(os.path.join(test_cache, 'perch_arrays.npz'))
    emb_test_flat    = arr['emb_full'].astype(np.float32)
    scores_test_flat = arr['scores_full_raw'].astype(np.float32)
    meta_test        = pd.read_parquet(os.path.join(test_cache, 'perch_meta.parquet'))

print(f"Test windows: {len(emb_test_flat)}")


# ── Cell 13: Reshape test to file sequences ───────────────────────────────────
emb_test_files, _, test_file_info = reshape_to_files(emb_test_flat, meta_test, N_WINDOWS)
scores_test_files, _, _ = reshape_to_files(scores_test_flat, meta_test, N_WINDOWS)
n_test_files = len(emb_test_files)

site_ids_test = np.array([
    tax.site_to_idx.get(test_file_info[i]['filename']
        .split('_')[3] if '_' in test_file_info[i]['filename'] else 'UNKNOWN', 0)
    for i in range(n_test_files)
], dtype=np.int64)

hours_test = np.zeros(n_test_files, dtype=np.int64)
for i, info in enumerate(test_file_info):
    row = meta_test[meta_test['filename'] == info['filename']].iloc[0]
    hours_test[i] = int(row.get('hour_utc', 0))


# ── Cell 14: ProtoSSM inference with TTA ──────────────────────────────────────
print("Running ProtoSSM inference + TTA...")
emb_test_t    = torch.from_numpy(emb_test_files)
scores_test_t = torch.from_numpy(scores_test_files)
site_test_t   = torch.from_numpy(site_ids_test).long()
hour_test_t   = torch.from_numpy(hours_test).long()

TTA_SHIFTS = [0, 1, -1, 2, -2]
tta_preds = []

with torch.no_grad():
    for shift in TTA_SHIFTS:
        e_shift = torch.roll(emb_test_t, shift, dims=1)
        s_shift = torch.roll(scores_test_t, shift, dims=1)
        logits_tta, _, _ = proto_model(e_shift, s_shift, site_test_t, hour_test_t)
        # Roll predictions back
        logits_tta = torch.roll(logits_tta, -shift, dims=1)
        tta_preds.append(logits_tta.numpy())

proto_logits_test = np.mean(tta_preds, axis=0)  # (n_test_files, 12, n_classes)


# ── Cell 15: MLP probe inference ─────────────────────────────────────────────
print("Running MLP probe inference...")
N_t, T_t, D_t = emb_test_files.shape
emb_test_flat2 = emb_test_files.reshape(-1, D_t)
emb_test_scaled = scaler.transform(emb_test_flat2)
emb_test_pca = pca.transform(emb_test_scaled)

mlp_probs_test = np.zeros((N_t * T_t, n_classes), dtype=np.float32)
for c, clf in enumerate(probes):
    if clf is not None:
        prob = clf.predict_proba(emb_test_pca)
        if prob.shape[1] == 2:
            mlp_probs_test[:, c] = prob[:, 1]

mlp_logits_test = np.log(
    mlp_probs_test.clip(1e-7, 1-1e-7) / (1 - mlp_probs_test.clip(1e-7, 1-1e-7))
).reshape(N_t, T_t, n_classes)


# ── Cell 16: Ensemble + ResidualSSM correction ────────────────────────────────
first_pass_test = PROTO_W * proto_logits_test + MLP_W * mlp_logits_test

fp_test_t = torch.from_numpy(first_pass_test.astype(np.float32))
with torch.no_grad():
    correction = residual_model(emb_test_t, fp_test_t, site_test_t, hour_test_t).numpy()

final_logits = first_pass_test + RESIDUAL_W * correction  # (N_test, 12, n_classes)


# ── Cell 17: Post-processing ──────────────────────────────────────────────────
from src.postprocessing import PostProcessor
from src.prior import PriorAndProbeManager as _PriorMgr

final_logits_flat = final_logits.reshape(-1, n_classes)  # (N_test*12, n_classes)

# Fuse prior log-odds into logits (site × hour context)
test_sites = meta_test['site'].tolist()
test_hours = meta_test['hour_utc'].tolist()
prior_logits_test = prior_mgr.prior_logits_from_tables(test_sites, test_hours, prior_tables)
fused_logits_flat = _PriorMgr.fuse_scores(final_logits_flat, prior_logits_test, weight=0.35)

# Per-class smoothing: Aves/Mammalia=0.15 (event), Amphibia/Insecta=0.35 (texture)
taxonomy_csv_path = f'{COMP_DIR}/taxonomy.csv'
pp = PostProcessor(n_windows=N_WINDOWS)
class_alphas = PostProcessor.build_class_alphas(
    taxonomy_csv_path, label_list, alpha_event=0.15, alpha_texture=0.35
)

probs_final = pp.process(
    fused_logits_flat,
    class_temperatures=None,
    do_confidence_scale=True,
    do_rank_scale=True,
    do_smooth=True,
    smooth_alpha=0.20,        # fallback if taxonomy.csv missing
    class_alphas=class_alphas,
    rank_power=0.4,
)


# ── Cell 18: Build & save submission.csv ─────────────────────────────────────
row_ids = meta_test['row_id'].tolist()

# Reorder to match meta_test row order (file-contiguous)
test_row_ids = []
for info in test_file_info:
    s, e = info['start'], info['end']
    test_row_ids.extend(meta_test.iloc[s:e]['row_id'].tolist())

result_df = pd.DataFrame(probs_final, columns=label_list)
result_df.insert(0, 'row_id', test_row_ids)

# Align to sample_submission format
sample_sub = pd.read_csv(f'{COMP_DIR}/sample_submission.csv')
sub_cols = [c for c in sample_sub.columns if c != 'row_id']
for col in sub_cols:
    if col not in result_df.columns:
        result_df[col] = 0.0
result_df = result_df[['row_id'] + sub_cols]

out_path = f'{WORK_DIR}/submission.csv'
result_df.to_csv(out_path, index=False)
print(f"Saved: {out_path}  shape: {result_df.shape}")
result_df.head(3)
