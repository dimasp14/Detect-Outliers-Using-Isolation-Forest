#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deteksi Aktivitas Tidak Normal pada Sistem Keamanan Siber
menggunakan Unsupervised Learning Isolation Forest + Threshold Tuning

Mode threshold:
- default     : pakai predict() bawaan IsolationForest (ambang berdasar contamination)
- percentile  : pakai persentil skor anomali (lebih tinggi = lebih anomali)
- f1          : sweep ambang dan pilih yang memaksimalkan F1(attack) pada test set (butuh label)

Contoh pakai:
python implement.py ^
  --csv_path "dataset/cybersecurity_intrusion_data.csv" ^
  --output_dir "outputs" ^
  --include_features "network_packet_size,login_attempts,session_duration,ip_reputation_score,failed_logins,unusual_time_access" ^
  --drop_id_like ^
  --n_estimators 700 ^
  --max_samples 0.9 ^
  --contamination 0.08 ^
  --threshold_mode f1
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------
# Konfigurasi & util
# ------------------------------------------------------------

LABEL_CANDIDATES = ["attack_detected", "label", "class"]


@dataclass
class RunConfig:
    csv_path: str
    output_dir: str = "outputs"
    contamination: float = 0.05
    random_state: int = 42
    test_size: float = 0.25
    n_estimators: int = 300
    max_samples: str | int | float = "auto"

    # opsi subset fitur
    include_features: str = ""  # comma-separated
    exclude_features: str = ""  # comma-separated
    drop_id_like: bool = False  # drop kolom berbau id

    # tuning threshold
    threshold_mode: str = "default"  # default | percentile | f1
    threshold_value: float = 95.0    # jika percentile, gunakan persentil ini


def make_ohe() -> OneHotEncoder:
    """Kompatibel lintas versi sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def coerce_max_samples(v):
    """Ubah argumen max_samples menjadi tipe yang valid ('auto' | int>=1 | 0< float <=1)."""
    if isinstance(v, (int, float)):
        return v
    s = str(v).strip().lower()
    if s == "auto":
        return "auto"
    try:
        if any(ch in s for ch in [".", "e"]):
            return float(s)
        return int(s)
    except Exception:
        raise ValueError("Invalid --max_samples. Gunakan 'auto', int >=1, atau float dalam (0,1].")


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def coerce_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Konversi string yes/no true/false -> 0/1 jika memungkinkan."""
    bool_like = {"yes": 1, "no": 0, "true": 1, "false": 0, "y": 1, "n": 0, "1": 1, "0": 0}
    new_df = df.copy()
    for col in new_df.columns:
        if new_df[col].dtype == object:
            sample = new_df[col].astype(str).str.strip().str.lower().replace(bool_like)
            uniques = set(pd.Series(sample).dropna().unique())
            if uniques.issubset({0, 1}):
                new_df[col] = pd.to_numeric(sample, errors="coerce")
    return new_df


def normalize_label_series(y: pd.Series) -> pd.Series:
    """Normalisasi label ke {0,1}; 1=attack/anomaly, 0=normal."""
    if y.dtype != object:
        vals = y.dropna().unique()
        if set(vals).issubset({0, 1, -1}):
            y = y.replace(-1, 1)
        return y.astype(int)
    mapping = {
        "attack": 1, "anomaly": 1, "malicious": 1, "yes": 1, "true": 1,
        "normal": 0, "benign": 0, "no": 0, "false": 0
    }
    y_norm = y.astype(str).str.strip().str.lower().replace(mapping)
    if not set(pd.Series(y_norm).dropna().unique()).issubset({0, 1}):
        # fallback: factorize -> majority class dianggap normal (0)
        codes, _ = pd.factorize(y_norm)
        counts = pd.Series(codes).value_counts()
        if len(counts) >= 2:
            majority_code = counts.idxmax()
            y_bin = (codes != majority_code).astype(int)
            return pd.Series(y_bin, index=y.index)
        return pd.Series(codes, index=y.index)
    return y_norm.astype(int)


def parse_csv_list(s: Optional[str]) -> Optional[list]:
    if s is None or str(s).strip() == "":
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


def choose_features(
    X: pd.DataFrame, include: Optional[list], exclude: Optional[list], drop_id_like: bool
) -> pd.DataFrame:
    cols = list(X.columns)
    if drop_id_like:
        id_like = [c for c in cols if any(k in c.lower() for k in ["id", "uuid", "session_id"])]
        cols = [c for c in cols if c not in id_like]
    if include is not None:
        missing = [c for c in include if c not in X.columns]
        if missing:
            print(f"[Warn] include_features tidak ditemukan di data: {missing}")
        cols = [c for c in include if c in X.columns]
    if exclude is not None:
        cols = [c for c in cols if c not in exclude]
    if not cols:
        raise ValueError("Daftar fitur kosong setelah include/exclude. Mohon cek argumen.")
    return X[cols]


# ------------------------------------------------------------
# Pipeline & evaluasi
# ------------------------------------------------------------

def build_pipeline(numeric_features: List[str], categorical_features: List[str], cfg: RunConfig) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())]
    )
    transformers = []
    if len(numeric_features) > 0:
        transformers.append(("num", numeric_transformer, numeric_features))
    if len(categorical_features) > 0:
        transformers.append(("cat", categorical_transformer, categorical_features))
    if not transformers:
        raise ValueError("Tidak ada fitur numerik maupun kategorikal yang bisa dipakai.")
    preprocessor = ColumnTransformer(transformers=transformers)

    iso_forest = IsolationForest(
        n_estimators=cfg.n_estimators,
        max_samples=cfg.max_samples,
        contamination=cfg.contamination,
        random_state=cfg.random_state,
        n_jobs=-1,
        bootstrap=False,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", iso_forest)])


def evaluate_scores(y_true: pd.Series, scores: np.ndarray) -> float:
    """ROC-AUC menggunakan skor anomali (semakin besar = makin anomali)."""
    try:
        y_bin = normalize_label_series(y_true)
        return float(roc_auc_score(y_bin, scores))
    except Exception:
        return float("nan")


def evaluate_with_preds(y_true: pd.Series, preds_anom: np.ndarray) -> Tuple[dict, np.ndarray, str]:
    """Confusion matrix & classification report, asumsi preds_anom (0/1; 1=anomali)."""
    y_bin = normalize_label_series(y_true)
    cm = confusion_matrix(y_bin, preds_anom)
    report = classification_report(y_bin, preds_anom, digits=4)
    stats = {"confusion_matrix": cm.tolist(), "classification_report": report}
    return stats, cm, report


def sweep_f1_threshold(y_true: pd.Series, scores: np.ndarray) -> dict:
    """Cari ambang (berbasis persentil) yang memaksimalkan F1(attack=1)."""
    y_bin = normalize_label_series(y_true)
    percentiles = np.linspace(50, 99.5, 100)
    best = None
    for p in percentiles:
        thr = np.percentile(scores, p)
        pred = (scores >= thr).astype(int)
        tp = int(((pred == 1) & (y_bin == 1)).sum())
        fp = int(((pred == 1) & (y_bin == 0)).sum())
        fn = int(((pred == 0) & (y_bin == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        cand = {"p": float(p), "thr": float(thr), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        if best is None or cand["f1"] > best["f1"]:
            best = cand
    return best


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(cfg: RunConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("\n[1/5] Loading data from:", cfg.csv_path)
    # Auto-detect delimiter
    df = pd.read_csv(cfg.csv_path, sep=None, engine="python")
    print("Shape:", df.shape)

    df = coerce_booleans(df)

    # Split features / label
    label_col = find_label_column(df)
    if label_col is not None:
        y = df[label_col]
        X = df.drop(columns=[label_col])
        print(f"Label column detected: {label_col}")
    else:
        y = None
        X = df
        print("No label column found. Proceeding fully unsupervised.")

    # Feature selection
    include = parse_csv_list(cfg.include_features)
    exclude = parse_csv_list(cfg.exclude_features)
    if (include and len(include) > 0) or (exclude and len(exclude) > 0) or cfg.drop_id_like:
        X = choose_features(X, include, exclude, cfg.drop_id_like)
        print(f"[Feature Selection] Using {len(X.columns)} features: {list(X.columns)}")

    # Feature types
    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    print(f"Numeric features: {len(numeric_features)} | Categorical features: {len(categorical_features)}")

    # Split
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
        )
    else:
        X_train, X_test = train_test_split(X, test_size=cfg.test_size, random_state=cfg.random_state)
        y_train = y_test = None

    # Fit
    print("\n[2/5] Building pipeline & training Isolation Forest...")
    pipe = build_pipeline(numeric_features, categorical_features, cfg)
    pipe.fit(X_train)
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    # Scores/preds on TEST
    print("\n[3/5] Generating anomaly scores & predictions...")
    Z_test = preprocess.transform(X_test)
    scores_test = -model.decision_function(Z_test)  # lebih besar = lebih anomali
    preds_test_default = (model.predict(Z_test) == -1).astype(int)

    out_test = X_test.copy()
    out_test["anomaly_score"] = scores_test
    out_test["anomaly_pred"] = preds_test_default  # default
    if y_test is not None:
        out_test[label_col] = y_test.values

    pred_path = os.path.join(cfg.output_dir, "predictions_test.csv")
    out_test.sort_values("anomaly_score", ascending=False).to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")

    # Scores/preds on ALL (ranking)
    Z_all = preprocess.transform(X)
    scores_all = -model.decision_function(Z_all)
    preds_all_default = (model.predict(Z_all) == -1).astype(int)

    all_df = X.copy()
    all_df["anomaly_score"] = scores_all
    all_df["anomaly_pred"] = preds_all_default
    if y is not None:
        all_df[label_col] = y.values

    all_path = os.path.join(cfg.output_dir, "predictions_all.csv")
    all_df.sort_values("anomaly_score", ascending=False).to_csv(all_path, index=False)
    print(f"Saved full predictions to {all_path}")

    # Save model
    model_path = os.path.join(cfg.output_dir, "isoforest_pipeline.joblib")
    joblib.dump(pipe, model_path)
    print(f"Saved model pipeline to {model_path}")

    # ---- Evaluation (default) ----
    if y_test is not None:
        print("\n[4/5] Evaluating (default IF threshold)...")
        auc = evaluate_scores(y_test, scores_test)
        stats_default, cm_default, rep_default = evaluate_with_preds(y_test, preds_test_default)
        eval_default = {"roc_auc": auc, **stats_default}
        with open(os.path.join(cfg.output_dir, "evaluation.json"), "w", encoding="utf-8") as f:
            json.dump(eval_default, f, indent=2)
        print(json.dumps({"roc_auc": auc}, indent=2))
        print("Confusion Matrix (default):")
        print(cm_default)
        print("\nClassification report (default):\n", rep_default)
    else:
        print("\n[4/5] No labels detected — skipped default evaluation.")

    # ---- Threshold tuning ----
    tuned_info = None
    preds_test_tuned = None
    if cfg.threshold_mode.lower() in ["percentile", "f1"]:
        print("\n[4.1/5] Threshold tuning:", cfg.threshold_mode)

        if cfg.threshold_mode.lower() == "percentile":
            p = float(cfg.threshold_value)
            thr = float(np.percentile(scores_test, p))
            preds_test_tuned = (scores_test >= thr).astype(int)
            tuned_info = {"mode": "percentile", "percentile": p, "threshold": thr}

        elif cfg.threshold_mode.lower() == "f1":
            if y_test is None:
                print("[Warn] Tidak ada label pada test set. 'f1' tuning dilewati.")
            else:
                best = sweep_f1_threshold(y_test, scores_test)
                preds_test_tuned = (scores_test >= best["thr"]).astype(int)
                tuned_info = {"mode": "f1", **best}

        # simpan eval tuned jika ada
        if (y_test is not None) and (preds_test_tuned is not None):
            stats_tuned, cm_tuned, rep_tuned = evaluate_with_preds(y_test, preds_test_tuned)
            eval_tuned = {"roc_auc": evaluate_scores(y_test, scores_test), **stats_tuned, "tuned_info": tuned_info}
            with open(os.path.join(cfg.output_dir, "evaluation_tuned.json"), "w", encoding="utf-8") as f:
                json.dump(eval_tuned, f, indent=2)
            print("\n[4.2/5] Evaluation (tuned):")
            print("Tuned info:", json.dumps(tuned_info, indent=2))
            print("Confusion Matrix (tuned):")
            print(cm_tuned)
            print("\nClassification report (tuned):\n", rep_tuned)

            # tulis kolom prediksi_tuned ke file predictions_test.csv
            out_test_tuned = out_test.copy()
            out_test_tuned["anomaly_pred_tuned"] = preds_test_tuned
            out_test_tuned.to_csv(pred_path, index=False)

    # Top-N anomaly
    print("\n[5/5] Exporting top-rows for quick manual inspection...")
    topn_path = os.path.join(cfg.output_dir, "top20_anomalies.csv")
    all_df.nlargest(20, "anomaly_score").to_csv(topn_path, index=False)
    print(f"Saved {topn_path}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="unsupervised_learning/dataset/cybersecurity_intrusion_data.csv", help="Path ke file CSV dataset")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Direktori untuk menyimpan artefak")
    parser.add_argument("--contamination", type=float, default=0.05, help="Perkiraan proporsi anomali di data")
    parser.add_argument("--random_state", type=int, default=42, help="Seed random")
    parser.add_argument("--test_size", type=float, default=0.25, help="Proporsi test split")
    parser.add_argument("--n_estimators", type=int, default=300, help="Jumlah pohon IsolationForest")
    parser.add_argument("--max_samples", type=str, default="auto", help="Jumlah sampel per pohon (int/float/auto)")
    parser.add_argument("--include_features", type=str, default="", help="Comma-separated fitur yang DIPAKAI saja (prioritas). Kosong = semua.")
    parser.add_argument("--exclude_features", type=str, default="", help="Comma-separated fitur yang DIABAIKAN.")
    parser.add_argument("--drop_id_like", action="store_true", help="Drop kolom berbau ID (id, uuid, session_id).")

    # tuning
    parser.add_argument("--threshold_mode", type=str, default="default", choices=["default", "percentile", "f1"], help="Mode threshold")
    parser.add_argument("--threshold_value", type=float, default=95.0, help="Persentil untuk mode percentile (mis. 95 berarti top 5% skor dianggap anomali)")

    args = parser.parse_args()

    cfg = RunConfig(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        contamination=float(args.contamination),
        random_state=int(args.random_state),
        test_size=float(args.test_size),
        n_estimators=int(args.n_estimators),
        max_samples=coerce_max_samples(args.max_samples),
        include_features=args.include_features,
        exclude_features=args.exclude_features,
        drop_id_like=bool(args.drop_id_like),
        threshold_mode=args.threshold_mode,
        threshold_value=float(args.threshold_value),
    )

    if not (0 < cfg.contamination < 0.5):
        print("[Warning] contamination sebaiknya dalam (0, 0.5). Mengatur ke 0.05.")
        cfg.contamination = 0.05

    main(cfg)
