#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deteksi Aktivitas Tidak Normal pada Sistem Keamanan Siber
menggunakan Unsupervised Learning Isolation Forest

Catatan:
- Model dilatih TANPA menggunakan label (unsupervised).
- Jika kolom 'attack_detected' tersedia, ia hanya dipakai untuk evaluasi.
- Script fleksibel untuk menangani kolom kategorikal & numerik.

Struktur umum:
1) Load & inspect data
2) Preprocessing (imputation, encoding, scaling)
3) Train IsolationForest pada X_train
4) Prediksi & evaluasi (opsional jika label tersedia)
5) Simpan artefak (model, prediksi)

Contoh pakai:
python implement.py ^
  --csv_path "unsupervised_learning/dataset/cybersecurity_intrusion_data.csv" ^
  --output_dir "outputs" ^
  --contamination 0.05 ^
  --test_size 0.25 ^
  --include_features "network_packet_size,login_attempts,session_duration,ip_reputation_score,failed_logins,protocol_type,encryption_used" ^
  --drop_id_like
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional

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

LABEL_CANDIDATES = [
    "attack_detected",
    "label",
    "class",
]


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


def make_ohe() -> OneHotEncoder:
    """
    Helper agar kompatibel dengan berbagai versi scikit-learn.
    - sklearn >= 1.2: OneHotEncoder(..., sparse_output=False)
    - sklearn <  1.2: OneHotEncoder(..., sparse=False)
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def coerce_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Konversi string yes/no true/false -> 0/1 jika memungkinkan."""
    bool_like = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "y": 1,
        "n": 0,
        "1": 1,
        "0": 0,
    }
    new_df = df.copy()
    for col in new_df.columns:
        if new_df[col].dtype == object:
            sample = (
                new_df[col].astype(str).str.strip().str.lower().replace(bool_like)
            )
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
        "attack": 1,
        "anomaly": 1,
        "malicious": 1,
        "yes": 1,
        "true": 1,
        "normal": 0,
        "benign": 0,
        "no": 0,
        "false": 0,
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
# Pipeline
# ------------------------------------------------------------

def build_pipeline(numeric_features: List[str], categorical_features: List[str], cfg: RunConfig) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe()),
        ]
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

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", iso_forest)])
    return pipe


def evaluate_unsupervised(y_true: pd.Series, scores: np.ndarray, preds: np.ndarray) -> dict:
    """
    Evaluasi: pakai skor decision_function untuk ROC-AUC (dibalik tandanya),
    dan preds untuk Confusion Matrix / classification report.
    IsolationForest: predict() => 1 (normal), -1 (anomali). Kita map ke {0,1} dengan 1=anomali.
    """
    y_bin = normalize_label_series(y_true)
    preds_anom = (preds == -1).astype(int)

    try:
        auc = roc_auc_score(y_bin, -scores)  # -scores: lebih besar => lebih anomali
    except Exception:
        auc = np.nan

    cm = confusion_matrix(y_bin, preds_anom)
    report = classification_report(y_bin, preds_anom, digits=4)
    return {
        "roc_auc": float(auc) if auc == auc else None,  # NaN safe
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(cfg: RunConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("\n[1/5] Loading data from:", cfg.csv_path)
    # Auto-detect delimiter: hindari kasus file terbaca 1 kolom
    df = pd.read_csv(cfg.csv_path, sep=None, engine="python")
    print("Shape:", df.shape)

    # Coerce boolean-like strings
    df = coerce_booleans(df)

    # Split features / label jika ada
    label_col = find_label_column(df)
    if label_col is not None:
        y = df[label_col]
        X = df.drop(columns=[label_col])
        print(f"Label column detected: {label_col}")
    else:
        y = None
        X = df
        print("No label column found. Proceeding fully unsupervised.")

    # Terapkan subset fitur (include/exclude/drop_id_like)
    include = parse_csv_list(cfg.include_features)
    exclude = parse_csv_list(cfg.exclude_features)
    if (include and len(include) > 0) or (exclude and len(exclude) > 0) or cfg.drop_id_like:
        X = choose_features(X, include, exclude, cfg.drop_id_like)
        print(f"[Feature Selection] Using {len(X.columns)} features: {list(X.columns)}")

    # Identify feature types
    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    print(f"Numeric features: {len(numeric_features)} | Categorical features: {len(categorical_features)}")

    # Train/test split (label hanya untuk evaluasi)
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=y if y is not None else None,
        )
    else:
        X_train, X_test = train_test_split(
            X, test_size=cfg.test_size, random_state=cfg.random_state
        )
        y_train = y_test = None

    # Build pipeline & fit
    print("\n[2/5] Building pipeline & training Isolation Forest...")
    pipe = build_pipeline(numeric_features, categorical_features, cfg)
    pipe.fit(X_train)

    # Inference
    print("\n[3/5] Generating anomaly scores & predictions...")
    # gunakan named_steps agar kompatibel lintas sklearn
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    Z_test = preprocess.transform(X_test)
    scores_test = model.decision_function(Z_test)
    preds_test = model.predict(Z_test)

    # Output dataframe test
    out_test = X_test.copy()
    out_test["anomaly_score"] = -scores_test  # makin tinggi = makin anomali
    out_test["anomaly_pred"] = (preds_test == -1).astype(int)
    if y_test is not None:
        out_test[label_col] = y_test.values

    pred_path = os.path.join(cfg.output_dir, "predictions_test.csv")
    out_test.sort_values("anomaly_score", ascending=False).to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")

    # Full dataset scoring (ranking seluruh data)
    Z_all = preprocess.transform(X)
    scores_all = model.decision_function(Z_all)
    preds_all = model.predict(Z_all)

    all_df = X.copy()
    all_df["anomaly_score"] = -scores_all
    all_df["anomaly_pred"] = (preds_all == -1).astype(int)
    if y is not None:
        all_df[label_col] = y.values

    all_path = os.path.join(cfg.output_dir, "predictions_all.csv")
    all_df.sort_values("anomaly_score", ascending=False).to_csv(all_path, index=False)
    print(f"Saved full predictions to {all_path}")

    # Save model pipeline
    model_path = os.path.join(cfg.output_dir, "isoforest_pipeline.joblib")
    joblib.dump(pipe, model_path)
    print(f"Saved model pipeline to {model_path}")

    # Evaluation (jika label tersedia)
    if y_test is not None:
        print("\n[4/5] Evaluating against ground truth label (not used in training)...")
        eval_stats = evaluate_unsupervised(y_test, scores_test, preds_test)
        print(json.dumps({"roc_auc": eval_stats["roc_auc"]}, indent=2))
        print("Confusion Matrix (y: 1=attack, 0=normal | pred: 1=anomaly, 0=normal):")
        print(np.array(eval_stats["confusion_matrix"]))
        print("\nClassification report:\n", eval_stats["classification_report"])
        with open(os.path.join(cfg.output_dir, "evaluation.json"), "w", encoding="utf-8") as f:
            json.dump(eval_stats, f, indent=2)
        print("Saved evaluation.json")
    else:
        print("\n[4/5] No labels detected — skipped quantitative evaluation.")

    # Top-N anomaly untuk inspeksi cepat
    print("\n[5/5] Exporting top-rows for quick manual inspection...")
    topn_path = os.path.join(cfg.output_dir, "top20_anomalies.csv")
    all_df.nlargest(20, "anomaly_score").to_csv(topn_path, index=False)
    print(f"Saved {topn_path}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default="unsupervised_learning/dataset/cybersecurity_intrusion_data.csv",
        help="Path ke file CSV dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Direktori untuk menyimpan artefak",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Perkiraan proporsi anomali di data",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed random",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.25,
        help="Proporsi test split",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=300,
        help="Jumlah pohon IsolationForest",
    )
    parser.add_argument(
        "--max_samples",
        type=str,
        default="auto",
        help="Jumlah sampel per pohon (int/float/auto)",
    )
    parser.add_argument(
        "--include_features",
        type=str,
        default="",
        help="Comma-separated fitur yang DIPAKAI saja (prioritas). Kosong = semua.",
    )
    parser.add_argument(
        "--exclude_features",
        type=str,
        default="",
        help="Comma-separated fitur yang DIABAIKAN.",
    )
    parser.add_argument(
        "--drop_id_like",
        action="store_true",
        help="Drop kolom berbau ID (id, uuid, session_id).",
    )

    args = parser.parse_args()

    cfg = RunConfig(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        contamination=float(args.contamination),
        random_state=int(args.random_state),
        test_size=float(args.test_size),
        n_estimators=int(args.n_estimators),
        max_samples=args.max_samples if args.max_samples != "auto" else "auto",
        include_features=args.include_features,
        exclude_features=args.exclude_features,
        drop_id_like=bool(args.drop_id_like),
    )

    # Validasi contamination
    if not (0 < cfg.contamination < 0.5):
        print("[Warning] contamination sebaiknya dalam (0, 0.5). Mengatur ke 0.05.")
        cfg.contamination = 0.05

    main(cfg)
