# visualize_embedding_scatter.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="outputs/predictions_test.csv",
                    help="Path ke CSV prediksi (test atau all).")
parser.add_argument("--outdir", type=str, default="outputs",
                    help="Folder output PNG.")
parser.add_argument("--method", type=str, default="umap", choices=["umap", "pca"],
                    help="Metode embedding 2D.")
parser.add_argument("--n_neighbors", type=int, default=20,
                    help="(UMAP) jumlah tetangga untuk struktur lokal.")
parser.add_argument("--min_dist", type=float, default=0.1,
                    help="(UMAP) kedekatan minima antar titik (semakin kecil → cluster lebih rapat).")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ---------- Load ----------
df = pd.read_csv(args.csv)

# Tentukan label/pred untuk pewarnaan biner
label_col = None
for c in ["attack_detected", "label", "class"]:
    if c in df.columns:
        label_col = c
        break

if label_col is not None:
    y = df[label_col]
    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower().replace({
            "attack":1,"anomaly":1,"malicious":1,"yes":1,"true":1,
            "normal":0,"benign":0,"no":0,"false":0
        })
        y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    else:
        y = y.replace(-1,1).astype(int)
else:
    # fallback ke prediksi model
    if "anomaly_pred_tuned" in df.columns:
        y = df["anomaly_pred_tuned"].astype(int)
    elif "anomaly_pred" in df.columns:
        y = df["anomaly_pred"].astype(int)
    else:
        raise ValueError("Tidak ada kolom label atau prediksi untuk pewarnaan biner.")

# ---------- Siapkan fitur numerik ----------
drop_cols = {
    "anomaly_score", "anomaly_pred", "anomaly_pred_tuned",
    "attack_detected", "label", "class"
}
feature_cols = [c for c in df.columns if c not in drop_cols]
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

if len(feature_cols) < 2:
    raise ValueError("Butuh >=2 fitur numerik untuk embedding (cek CSV).")

X = df[feature_cols].copy()
# bersihkan & isi NA
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

# buang kolom konstan
const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
if const_cols:
    X = X.drop(columns=const_cols)

# ---------- Standardize ----------
scaler = StandardScaler()
Xp = scaler.fit_transform(X.values)

# ---------- Embedding 2D ----------
if args.method == "umap":
    try:
        import umap
    except ImportError:
        raise SystemExit("UMAP belum terinstal. Jalankan: pip install umap-learn")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
        metric="euclidean",
    )
    Z = reducer.fit_transform(Xp)
else:
    pca = PCA(n_components=2, random_state=args.seed)
    Z = pca.fit_transform(Xp)

# ---------- Ambil skor (opsional, untuk gradasi warna) ----------
score = df["anomaly_score"].values if "anomaly_score" in df.columns else None

# ---------- Plot 1: Biner Normal vs Outlier ----------
plt.figure(figsize=(12, 8))
mask_out = (y == 1)   # 1 = outlier/attack
mask_norm = (y == 0)

plt.scatter(Z[mask_norm, 0], Z[mask_norm, 1], s=14, alpha=0.65, c="#2f7bff", label="Normal")
plt.scatter(Z[mask_out, 0],  Z[mask_out, 1],  s=14, alpha=0.85, c="#ff493e", label="Outlier/Attack")

plt.title("Figure 2: Normal vs. Outliers", fontsize=22, color="red", weight="bold")
plt.xlabel("principal_component_1")
plt.ylabel("principal_component_2")
plt.legend(frameon=False, loc="upper right")
plt.tight_layout()
out1 = os.path.join(args.outdir, f"viz_{args.method}_scatter_binary.png")
plt.savefig(out1, dpi=150)
plt.close()
print("Saved:", out1)

# ---------- Plot 2: Gradasi skor anomali ----------
if score is not None:
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(Z[:, 0], Z[:, 1], s=14, c=score, cmap="coolwarm", alpha=0.85)
    plt.title("Anomaly Score Map (higher = more anomalous)", fontsize=18, weight="bold")
    plt.xlabel("principal_component_1")
    plt.ylabel("principal_component_2")
    cbar = plt.colorbar(sc)
    cbar.set_label("anomaly_score", rotation=90)
    plt.tight_layout()
    out2 = os.path.join(args.outdir, f"viz_{args.method}_scoremap.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print("Saved:", out2)
else:
    print("Kolom 'anomaly_score' tidak ditemukan—melewati score map.")
