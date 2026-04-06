# ── Cell 1: Install dependencies ─────────────────────────────────────────────
# TF is needed only for test soundscape extraction.
# Wheel must be pre-loaded in a Kaggle dataset (no internet allowed).
import subprocess, os, sys

TF_WHEEL_DIR = '/kaggle/input/birdclef2026-perch-model/wheels'
if os.path.isdir(TF_WHEEL_DIR):
    for wheel in os.listdir(TF_WHEEL_DIR):
        if wheel.endswith('.whl'):
            subprocess.run(['pip', 'install', '--quiet', '--no-deps',
                            os.path.join(TF_WHEEL_DIR, wheel)])

subprocess.run(['pip', 'install', 'soundfile', '-q'], capture_output=True)

# ── Cell 2: Paths & imports ───────────────────────────────────────────────────
import json, pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

# ── PATHS — adjust to match your uploaded Kaggle dataset names ───────────────
COMP_DIR    = '/kaggle/input/birdclef-2026'
CODE_DIR    = '/kaggle/input/birdclef2026-ssm-code'      # contains src/
WEIGHTS_DIR = '/kaggle/input/birdclef2026-ssm-weights'   # proto_ssm.pt etc.
PERCH_DIR   = '/kaggle/input/birdclef2026-perch-model/perch_v2_cpu/1'
WORK_DIR    = '/kaggle/working'
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, CODE_DIR)
os.makedirs(WORK_DIR, exist_ok=True)

N_WINDOWS   = 12
WINDOW_SEC  = 5
SAMPLE_RATE = 32000
RESIDUAL_W  = 0.35
MLP_W       = 0.50
PROTO_W     = 0.50


# ── Cell 3: Load label list & model config ────────────────────────────────────
with open(os.path.join(WEIGHTS_DIR, 'ssm_meta.json')) as f:
    ssm_meta = json.load(f)

n_classes  = ssm_meta['n_classes']
n_windows  = ssm_meta.get('n_windows', N_WINDOWS)
n_sites    = ssm_meta['n_sites']
site_to_idx = ssm_meta['site_to_idx']

# label_list: prefer from WEIGHTS_DIR, fallback to sample_submission
label_list_path = os.path.join(WEIGHTS_DIR, 'label_list.json')
if os.path.exists(label_list_path):
    with open(label_list_path) as f:
        label_list = json.load(f)
else:
    sub_df = pd.read_csv(f'{COMP_DIR}/sample_submission.csv')
    label_list = [c for c in sub_df.columns if c != 'row_id']

print(f"Classes: {n_classes}, Sites: {n_sites}, Windows: {n_windows}")


# ── Cell 4: Load pre-trained model weights ────────────────────────────────────
from src.models.ssm import ProtoSSMv5, ResidualSSM

ssm_cfg = ssm_meta.get('ssm_config', {})

proto_model = ProtoSSMv5(
    d_input=1536,
    d_model=ssm_cfg.get('D_MODEL', 320),
    d_state=ssm_cfg.get('D_STATE', 32),
    n_ssm_layers=ssm_cfg.get('N_SSM_LAYERS', 4),
    n_classes=n_classes,
    n_windows=n_windows,
    dropout=0.0,
    n_sites=n_sites,
    use_cross_attn=ssm_cfg.get('USE_CROSS_ATTN', True),
    cross_attn_heads=ssm_cfg.get('CROSS_ATTN_HEADS', 8),
)
proto_model.load_state_dict(
    torch.load(os.path.join(WEIGHTS_DIR, 'proto_ssm.pt'), map_location='cpu')
)
proto_model.eval()
print(f"ProtoSSM loaded  ({sum(p.numel() for p in proto_model.parameters()):,} params)")

residual_model = ResidualSSM(
    d_input=1536,
    d_scores=n_classes,
    d_model=ssm_cfg.get('RESIDUAL_D_MODEL', 128),
    d_state=16,
    n_classes=n_classes,
    n_windows=n_windows,
    n_sites=n_sites,
)
residual_model.load_state_dict(
    torch.load(os.path.join(WEIGHTS_DIR, 'residual_ssm.pt'), map_location='cpu')
)
residual_model.eval()
print(f"ResidualSSM loaded  ({sum(p.numel() for p in residual_model.parameters()):,} params)")

with open(os.path.join(WEIGHTS_DIR, 'mlp_probes.pkl'), 'rb') as f:
    probe_bundle = pickle.load(f)
probes  = probe_bundle['probes']
pca     = probe_bundle['pca']
scaler  = probe_bundle['scaler']
print(f"MLP probes loaded  ({sum(p is not None for p in probes)}/{n_classes} active)")


# ── Cell 5: Load prior tables ─────────────────────────────────────────────────
from src.prior import PriorAndProbeManager

with open(os.path.join(WEIGHTS_DIR, 'prior_tables.pkl'), 'rb') as f:
    prior_bundle = pickle.load(f)

prior_tables = prior_bundle['prior_tables']
prior_mgr    = PriorAndProbeManager(label_list)
print(f"Prior tables loaded. Sites: {prior_tables.get('n_sites', '?')}")


# ── Cell 6: Extract test soundscapes with Perch ───────────────────────────────
# Test data is only available at submission time — must run Perch here.
print("Extracting test soundscapes with Perch...")

from extract_perch import PerchExtractor, parse_filename_metadata

extractor = PerchExtractor(
    model_dir=PERCH_DIR,
    sample_rate=SAMPLE_RATE,
    window_sec=WINDOW_SEC,
    n_windows=n_windows,
)

test_paths = sorted(Path(f'{COMP_DIR}/test_soundscapes').glob('*.ogg'))
print(f"Test files: {len(test_paths)}")

emb_test_list, scores_test_list, meta_test_rows = [], [], []

for path in tqdm(test_paths, desc="Perch extract", unit="file"):
    fname = path.name
    stem  = path.stem
    meta  = parse_filename_metadata(fname)
    try:
        emb, logits = extractor.extract_file(str(path))
    except Exception as e:
        tqdm.write(f"  SKIP {fname}: {e}")
        emb    = np.zeros((n_windows, 1536), dtype=np.float32)
        logits = np.zeros((n_windows, n_classes), dtype=np.float32)

    emb_test_list.append(emb)
    scores_test_list.append(logits)
    for t in range(n_windows):
        meta_test_rows.append({
            'row_id':   f"{stem}_{(t + 1) * WINDOW_SEC}",
            'filename': fname,
            'site':     meta['site'],
            'hour_utc': meta['hour_utc'],
        })

emb_test_flat    = np.vstack(emb_test_list).astype(np.float32)
scores_test_flat = np.vstack(scores_test_list).astype(np.float32)
meta_test        = pd.DataFrame(meta_test_rows)
print(f"Test windows extracted: {len(emb_test_flat)}")


# ── Cell 7: Reshape test → file sequences ────────────────────────────────────
def reshape_to_files(flat, meta_df, n_win=12):
    filenames = meta_df['filename'].tolist()
    file_info, i, fi = [], 0, 0
    while i < len(filenames):
        fname = filenames[i]; start = i
        while i < len(filenames) and filenames[i] == fname:
            i += 1
        file_info.append({'filename': fname, 'start': start, 'end': i})
        fi += 1
    n_files = len(file_info)
    out = np.zeros((n_files, n_win, flat.shape[1]), dtype=flat.dtype)
    for k, info in enumerate(file_info):
        s, e = info['start'], info['end']
        T = min(e - s, n_win)
        out[k, :T] = flat[s:e][:T]
    return out, file_info

emb_test_files,    test_file_info = reshape_to_files(emb_test_flat, meta_test, n_windows)
scores_test_files, _              = reshape_to_files(scores_test_flat, meta_test, n_windows)
n_test_files = len(emb_test_files)

# Build site/hour index per file
site_ids_test = np.array([
    site_to_idx.get(
        meta_test[meta_test['filename'] == info['filename']].iloc[0]['site'],
        0
    )
    for info in test_file_info
], dtype=np.int64)

hours_test = np.array([
    int(meta_test[meta_test['filename'] == info['filename']].iloc[0]['hour_utc'])
    for info in test_file_info
], dtype=np.int64)

print(f"Test files: {n_test_files}, shape: {emb_test_files.shape}")


# ── Cell 8: ProtoSSM inference with TTA ──────────────────────────────────────
print("ProtoSSM inference (TTA)...")
emb_t    = torch.from_numpy(emb_test_files)
scores_t = torch.from_numpy(scores_test_files)
site_t   = torch.from_numpy(site_ids_test).long()
hour_t   = torch.from_numpy(hours_test).long()

TTA_SHIFTS = [0, 1, -1, 2, -2]
tta_preds  = []

with torch.no_grad():
    for shift in TTA_SHIFTS:
        e_s = torch.roll(emb_t, shift, dims=1)
        s_s = torch.roll(scores_t, shift, dims=1)
        logits_tta, _, _ = proto_model(e_s, s_s, site_t, hour_t)
        tta_preds.append(torch.roll(logits_tta, -shift, dims=1).numpy())

proto_logits_test = np.mean(tta_preds, axis=0)  # (N_test, 12, n_classes)
print("  done.")


# ── Cell 9: MLP probe inference ───────────────────────────────────────────────
print("MLP probe inference...")
N_t, T_t, D_t = emb_test_files.shape
emb_flat   = emb_test_files.reshape(-1, D_t)
emb_scaled = scaler.transform(emb_flat)
emb_pca    = pca.transform(emb_scaled)

mlp_probs = np.zeros((N_t * T_t, n_classes), dtype=np.float32)
for c, clf in enumerate(probes):
    if clf is not None:
        prob = clf.predict_proba(emb_pca)
        if prob.shape[1] == 2:
            mlp_probs[:, c] = prob[:, 1]

mlp_logits = np.log(
    mlp_probs.clip(1e-7, 1 - 1e-7) / (1 - mlp_probs.clip(1e-7, 1 - 1e-7))
).reshape(N_t, T_t, n_classes)
print("  done.")


# ── Cell 10: Ensemble + ResidualSSM correction ────────────────────────────────
first_pass = PROTO_W * proto_logits_test + MLP_W * mlp_logits

fp_t = torch.from_numpy(first_pass.astype(np.float32))
with torch.no_grad():
    correction = residual_model(emb_t, fp_t, site_t, hour_t).numpy()

final_logits = first_pass + RESIDUAL_W * correction  # (N_test, 12, n_classes)


# ── Cell 11: Post-processing ──────────────────────────────────────────────────
from src.postprocessing import PostProcessor

final_logits_flat = final_logits.reshape(-1, n_classes)

# Fuse site × hour priors
prior_logits = prior_mgr.prior_logits_from_tables(
    meta_test['site'].tolist(),
    meta_test['hour_utc'].tolist(),
    prior_tables,
)
fused_logits = PriorAndProbeManager.fuse_scores(final_logits_flat, prior_logits, weight=0.35)

# Per-class temporal smoothing
pp = PostProcessor(n_windows=n_windows)
class_alphas = PostProcessor.build_class_alphas(
    f'{COMP_DIR}/taxonomy.csv', label_list, alpha_event=0.15, alpha_texture=0.35
)
probs_final = pp.process(
    fused_logits,
    class_temperatures=None,
    do_confidence_scale=True,
    do_rank_scale=True,
    do_smooth=True,
    smooth_alpha=0.20,
    class_alphas=class_alphas,
    rank_power=0.4,
)


# ── Cell 12: Build & save submission.csv ──────────────────────────────────────
# Reconstruct row_ids in file-contiguous order (matches probs_final row order)
test_row_ids = []
for info in test_file_info:
    s, e = info['start'], info['end']
    test_row_ids.extend(meta_test.iloc[s:e]['row_id'].tolist())

result_df = pd.DataFrame(probs_final, columns=label_list)
result_df.insert(0, 'row_id', test_row_ids)

# Align columns to sample_submission format
sample_sub = pd.read_csv(f'{COMP_DIR}/sample_submission.csv')
sub_cols   = [c for c in sample_sub.columns if c != 'row_id']
for col in sub_cols:
    if col not in result_df.columns:
        result_df[col] = 0.0
result_df = result_df[['row_id'] + sub_cols]

out_path = f'{WORK_DIR}/submission.csv'
result_df.to_csv(out_path, index=False)
print(f"Saved: {out_path}  shape: {result_df.shape}")
result_df.head(3)
