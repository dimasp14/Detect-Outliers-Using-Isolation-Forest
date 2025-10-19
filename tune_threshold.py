# tune_threshold.py
import numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

PATH = r"outputs/predictions_test.csv" 
LABEL = "attack_detected"

df = pd.read_csv(PATH)
y = df[LABEL]
# normalisasi label -> {0,1}
if y.dtype == object:
    y = y.astype(str).str.strip().str.lower().replace({
        "attack":1,"anomaly":1,"malicious":1,"yes":1,"true":1,
        "normal":0,"benign":0,"no":0,"false":0
    })
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
else:
    y = y.replace(-1,1).astype(int)

scores = df["anomaly_score"].values  # lebih tinggi = lebih anomali

# sweep threshold pada berbagai persentil
percentiles = np.linspace(50, 99.5, 100)  # dari median sampai 99.5th
best = None
for p in percentiles:
    thr = np.percentile(scores, p)
    pred = (scores >= thr).astype(int)
    # contoh: optimasi F1 untuk kelas attack (pos=1)
    # (bisa ganti kriteria: recall, precision, atau cost-based)
    tp = ((pred==1)&(y==1)).sum()
    fp = ((pred==1)&(y==0)).sum()
    fn = ((pred==0)&(y==1)).sum()
    precision = tp / (tp+fp) if (tp+fp)>0 else 0.0
    recall = tp / (tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    cand = {"p":p, "thr":float(thr), "precision":precision, "recall":recall, "f1":f1, "tp":int(tp), "fp":int(fp), "fn":int(fn)}
    if best is None or cand["f1"] > best["f1"]:
        best = cand

print("Best threshold by F1(attack):", best)

# Laporan detail di threshold terbaik
pred_best = (scores >= best["thr"]).astype(int)
print("\nConfusion Matrix at best threshold:")
print(confusion_matrix(y, pred_best))
print("\nClassification Report:")
print(classification_report(y, pred_best, digits=4))
print("\nROC-AUC (pakai skor, ambang bebas):", roc_auc_score(y, scores))
