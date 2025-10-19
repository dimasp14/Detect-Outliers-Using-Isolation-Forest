# analyze_umap_clusters.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.stats import ks_2samp

# ----------------- CLI -----------------
ap = argparse.ArgumentParser()
ap.add_argument("--csv", type=str, default="outputs/predictions_test.csv",
                help="Path ke CSV prediksi (test/all).")
ap.add_argument("--outdir", type=str, default="outputs",
                help="Direktori penyimpanan gambar & tabel.")
ap.add_argument("--clusters", type=int, default=3,
                help="Jumlah cluster KMeans di ruang UMAP.")
ap.add_argument("--method", type=str, default="umap", choices=["umap", "pca"],
                help="Metode embedding 2D.")
ap.add_argument("--n_neighbors", type=int, default=20,
                help="(UMAP) banyak tetangga.")
ap.add_argument("--min_dist", type=float, default=0.1,
                help="(UMAP) kerapatan antar titik.")
ap.add_argument("--seed", type=int, default=42, help="Random seed.")
ap.add_argument("--topk", type=int, default=8, help="Top-k fitur per cluster untuk heatmap.")
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ----------------- Load data -----------------
df = pd.read_csv(args.csv)

# pilih label untuk analisis anomali (urutan prioritas)
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
    # fallback ke prediksi
    if "anomaly_pred_tuned" in df.columns:
        y = df["anomaly_pred_tuned"].astype(int)
    elif "anomaly_pred" in df.columns:
        y = df["anomaly_pred"].astype(int)
    else:
        raise ValueError("Tidak ada kolom label/prediksi untuk analisis anomali.")

anomaly_score = df["anomaly_score"].values if "anomaly_score" in df.columns else None

# pilih fitur numerik saja (drop kolom non-fitur)
drop_cols = {"anomaly_score","anomaly_pred","anomaly_pred_tuned","attack_detected","label","class"}
feature_cols = [c for c in df.columns if c not in drop_cols]
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

if len(feature_cols) < 2:
    raise ValueError("Butuh >=2 fitur numerik untuk embedding & analisis cluster.")

X = df[feature_cols].copy()
X = X.replace([np.inf,-np.inf], np.nan).fillna(X.median(numeric_only=True))
const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
if const_cols:
    X = X.drop(columns=const_cols)
    feature_cols = [c for c in feature_cols if c not in const_cols]

# ----------------- Standardize & Embedding -----------------
scaler = StandardScaler()
Xp = scaler.fit_transform(X.values)

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
        metric="euclidean"
    )
    Z = reducer.fit_transform(Xp)
else:
    pca = PCA(n_components=2, random_state=args.seed)
    Z = pca.fit_transform(Xp)

# ----------------- Clustering di ruang embedding -----------------
km = KMeans(n_clusters=args.clusters, random_state=args.seed, n_init="auto")
cluster = km.fit_predict(Z)

df_out = df.copy()
df_out["embed_x"] = Z[:,0]
df_out["embed_y"] = Z[:,1]
df_out["umap_cluster"] = cluster
df_out.to_csv(os.path.join(args.outdir, "embedding_with_cluster.csv"), index=False)

# ----------------- Scatter per cluster -----------------
plt.figure(figsize=(14,9))
for k in range(args.clusters):
    mask = cluster==k
    # teks legend menampilkan proporsi anomali
    rate = y[mask].mean() if mask.sum()>0 else 0.0
    plt.scatter(Z[mask,0], Z[mask,1], s=12, alpha=0.7, label=f"Cluster {k} (n={mask.sum()}, anom={rate:.2f})")
plt.legend(frameon=False)
plt.title("Embedding clusters (UMAP/PCA)", fontsize=18)
plt.xlabel("principal_component_1")
plt.ylabel("principal_component_2")
plt.tight_layout()
out_sc = os.path.join(args.outdir, f"viz_{args.method}_scatter_cluster.png")
plt.savefig(out_sc, dpi=160); plt.close()
print("Saved:", out_sc)

# ----------------- Score map -----------------
if anomaly_score is not None:
    plt.figure(figsize=(14,9))
    sc = plt.scatter(Z[:,0], Z[:,1], c=anomaly_score, s=12, alpha=0.85, cmap="coolwarm")
    plt.colorbar(sc, label="anomaly_score")
    plt.title("Anomaly Score Map", fontsize=16)
    plt.xlabel("principal_component_1"); plt.ylabel("principal_component_2")
    plt.tight_layout()
    out_sm = os.path.join(args.outdir, f"viz_{args.method}_scoremap.png")
    plt.savefig(out_sm, dpi=160); plt.close()
    print("Saved:", out_sm)

# ----------------- Ringkasan per cluster -----------------
summary_rows = []
for k in range(args.clusters):
    mk = cluster==k
    row = {"cluster": k, "size": int(mk.sum()), "anomaly_rate": float(y[mk].mean())}
    # rata-rata beberapa fitur penting
    for col in feature_cols:
        row[f"mean_{col}"] = float(X.loc[mk, col].mean())
        row[f"median_{col}"] = float(X.loc[mk, col].median())
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(args.outdir, "cluster_summary.csv"), index=False)
print("Saved: cluster_summary.csv")

# ----------------- Ranking fitur per cluster -----------------
# skor: Cohen's d (effect size) & KS-distance antara distribusi cluster k vs sisanya
def cohens_d(a, b):
    a = np.asarray(a); b = np.asarray(b)
    mu1, mu2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    # pooled sd:
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / (n1+n2-2 + 1e-8))
    if sp == 0:
        return 0.0
    return float((mu1 - mu2) / sp)

rank_tables = []
for k in range(args.clusters):
    mk = cluster==k
    others = ~mk
    rows = []
    for col in feature_cols:
        a = X.loc[mk, col].values
        b = X.loc[others, col].values
        d = cohens_d(a, b)
        ks = ks_2samp(a, b).statistic
        rows.append({"feature": col, "cohens_d": d, "ks_distance": float(ks),
                     "mean_in": float(a.mean()), "mean_out": float(b.mean())})
    rank = pd.DataFrame(rows).sort_values(["ks_distance","cohens_d"], ascending=[False, False])
    rank["cluster"] = k
    rank_tables.append(rank)
    rank.to_csv(os.path.join(args.outdir, f"cluster_top_features_{k}.csv"), index=False)

# ----------------- Heatmap top-k fitur gabungan -----------------
# pilih top-k per cluster, gabungkan unik lalu plot mean-normalized heatmap
topk = args.topk
selected = []
for rank in rank_tables:
    selected += list(rank.head(topk)["feature"].values)
selected = list(dict.fromkeys(selected))  # unique & keep order

# z-score tiap fitur agar komparatif
Xz = pd.DataFrame(X, columns=feature_cols)
Xz = (Xz - Xz.mean())/(Xz.std(ddof=0) + 1e-8)

hm = []
index_names = []
for k in range(args.clusters):
    mk = cluster==k
    index_names.append(f"cluster_{k}")
    hm.append(Xz.loc[mk, selected].mean(axis=0).values)  # mean z-score di cluster
heat = np.vstack(hm)  # shape: (clusters, features_selected)

plt.figure(figsize=(max(10, len(selected)*0.6), 1.2*args.clusters + 3))
plt.imshow(heat, aspect="auto")
plt.colorbar(label="mean z-score (cluster)")
plt.yticks(range(args.clusters), index_names)
plt.xticks(range(len(selected)), selected, rotation=60, ha="right")
plt.title("Cluster Feature Profile (top-k per cluster)", fontsize=14)
plt.tight_layout()
out_hm = os.path.join(args.outdir, "viz_heatmap_top_features.png")
plt.savefig(out_hm, dpi=180); plt.close()
print("Saved:", out_hm)

print("\nDone. Files saved in:", os.path.abspath(args.outdir))
