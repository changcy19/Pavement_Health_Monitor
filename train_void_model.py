from __future__ import annotations

import argparse
import json
from pathlib import Path


import joblib  # type: ignore
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent


POINTS = ["P001", "P002", "P003", "P004"]
POINT_FEATURES = [
    "n_samples",
    "time_start",
    "time_end",
    "accel_mean",
    "accel_std",
    "accel_rms",
    "accel_peak",
    "accel_kurtosis",
    "strain_mean",
    "strain_std",
    "strain_rms",
    "strain_peak",
    "strain_kurtosis",
    "spectral_centroid",
    "dominant_freq",
    "spec_energy_0_200",
    *[f"band_ratio_{i}" for i in range(1, 9)],
]

FEATURE_COLS = ["tg", "h_cm", "e_gpa", "v_kmh", "m_t"]
for point in POINTS:
    FEATURE_COLS.extend([f"{point}_{name}" for name in POINT_FEATURES])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a void-classification baseline model.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset_void.csv", help="Feature table CSV")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model_out", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df = df.dropna(subset=["void_code"]).copy()
    if df.empty:
        raise SystemExit("Dataset is empty after dropping invalid labels.")

    y = df["void_code"].astype(str)
    groups = df["source_stem"] if "source_stem" in df.columns else df["file"]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset missing required feature columns: {missing[:10]}")
    X = df[FEATURE_COLS].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    test_groups = groups.iloc[test_idx].reset_index(drop=True)

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), FEATURE_COLS)],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=args.seed,
        class_weight="balanced",
        min_samples_leaf=3,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=sorted(y.unique()))

    test_detail = pd.DataFrame({
        "source_stem": test_groups,
        "y_true": y_test.reset_index(drop=True),
        "y_pred": pd.Series(pred),
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_detail.to_csv(args.output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    by_case = (
        test_detail.groupby("source_stem")
        .agg(
            y_true=("y_true", "first"),
            y_pred_mode=("y_pred", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
            n_rows=("y_pred", "size"),
        )
        .reset_index()
    )
    by_case["correct"] = by_case["y_true"] == by_case["y_pred_mode"]
    by_case.to_csv(args.output_dir / "test_case_summary.csv", index=False, encoding="utf-8-sig")

    joblib.dump(pipe, args.output_dir / "void_classifier.joblib")

    summary = {
        "dataset": str(args.dataset),
        "rows": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_cases": int(groups.iloc[train_idx].nunique()),
        "test_cases": int(groups.iloc[test_idx].nunique()),
        "features": FEATURE_COLS,
        "accuracy": float(acc),
        "labels": sorted(y.unique().tolist()),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(cm, index=sorted(y.unique()), columns=sorted(y.unique())).to_csv(args.output_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    print(f"Accuracy: {acc:.4f}")
    print(report)
    print(f"Saved test details to {args.output_dir / 'test_predictions.csv'}")
    print(f"Saved case summary to {args.output_dir / 'test_case_summary.csv'}")
    print(f"Saved model to {args.output_dir / 'void_classifier.joblib'}")


if __name__ == "__main__":
    main()
