# visualize_results.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

PRED_PATH = r"outputs/predictions_test.csv"   # ubah kalau perlu
OUT_DIR   = r"outputs"

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(PRED_PATH)

# pastikan kolom-kolom penting ada
assert "anomaly_score" in df.columns, "Kolom 'anomaly_score' tidak ditemukan."
assert "anomaly_pred" in df.columns, "Kolom 'anomaly_pred' tidak ditemukan."

# --- 1) Histogram skor anomali (seluruh data)
plt.figure(figsize=(8,5))
df["anomaly_score"].plot(kind="hist", bins=40, density=True)
plt.xlabel("anomaly_score (lebih tinggi = lebih anomali)")
plt.ylabel("density")
plt.title("Distribusi anomaly_score (test set)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "viz_hist_anomaly_score_all.png"))
plt.close()

# --- 2) Histogram skor anomali dipisah menurut prediksi model
plt.figure(figsize=(8,5))
df[df["anomaly_pred"]==0]["anomaly_score"].plot(kind="hist", bins=40, density=True, alpha=0.6, label="pred: normal")
df[df["anomaly_pred"]==1]["anomaly_score"].plot(kind="hist", bins=40, density=True, alpha=0.6, label="pred: anomaly")
plt.xlabel("anomaly_score")
plt.ylabel("density")
plt.title("Distribusi anomaly_score per prediksi model")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "viz_hist_anomaly_score_by_pred.png"))
plt.close()

# --- 3) Boxplot skor anomali per prediksi (normal vs anomaly)
plt.figure(figsize=(7,5))
data_box = [df[df["anomaly_pred"]==0]["anomaly_score"], df[df["anomaly_pred"]==1]["anomaly_score"]]
plt.boxplot(data_box, labels=["pred: normal", "pred: anomaly"], showfliers=False)
plt.ylabel("anomaly_score (lebih tinggi = lebih anomali)")
plt.title("Boxplot anomaly_score per prediksi model")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "viz_box_anomaly_score_by_pred.png"))
plt.close()

# --- Jika ada label ground-truth, buat ROC & PR curve
label_col = None
for c in ["attack_detected", "label", "class"]:
    if c in df.columns:
        label_col = c
        break

if label_col is not None:
    # normalisasi label ke {0,1} (1 = attack)
    y = df[label_col]
    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower().replace({
            "attack":1, "anomaly":1, "malicious":1, "yes":1, "true":1,
            "normal":0, "benign":0, "no":0, "false":0
        })
        y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    else:
        y = y.replace(-1, 1).astype(int)

    # decision_function semakin besar = normal, jadi kita balik untuk skor anomali
    scores = df["anomaly_score"].values

    # 4) ROC curve
    fpr, tpr, _ = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve (attack vs anomaly_score)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "viz_roc_curve.png"))
    plt.close()

    # 5) Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    plt.figure(figsize=(6,6))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (attack vs anomaly_score)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "viz_pr_curve.png"))
    plt.close()

    # 6) Distribusi skor anomali per ground-truth (normal vs attack)
    plt.figure(figsize=(8,5))
    df[y==0]["anomaly_score"].plot(kind="hist", bins=40, density=True, alpha=0.6, label="GT: normal")
    df[y==1]["anomaly_score"].plot(kind="hist", bins=40, density=True, alpha=0.6, label="GT: attack")
    plt.xlabel("anomaly_score")
    plt.ylabel("density")
    plt.title("Distribusi anomaly_score per ground-truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "viz_hist_anomaly_score_by_gt.png"))
    plt.close()

print("Selesai. PNG disimpan di folder 'outputs/'.")
