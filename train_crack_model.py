from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib  # type: ignore
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def mode_or_first(series: pd.Series):
    mode = series.mode()
    return mode.iat[0] if not mode.empty else series.iloc[0]


def build_xy(df: pd.DataFrame):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in dataset_crack.csv: {missing[:10]}")

    if "crack_depth_pct" not in df.columns:
        raise SystemExit("dataset_crack.csv must contain crack_depth_pct as target column")

    groups = df["source_stem"] if "source_stem" in df.columns else df["file"]
    X = df[FEATURE_COLS].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
    y = df["crack_depth_pct"].astype(float)
    return X, y, groups


def run_train(dataset_path: Path, save_dir: Path, test_size: float = 0.2, seed: int = 42):
    df = pd.read_csv(dataset_path)
    if df.empty:
        raise SystemExit("dataset_crack.csv is empty")

    X, y, groups = build_xy(df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    save_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(save_dir / "train_split.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(save_dir / "test_split.csv", index=False, encoding="utf-8-sig")

    preprocessor = ColumnTransformer([("num", StandardScaler(), FEATURE_COLS)], remainder="drop")
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=seed,
        min_samples_leaf=3,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    joblib.dump(pipe, save_dir / "crack_regressor.joblib")

    test_pred = test_df[["source_stem", "file", "crack_depth_pct"]].copy()
    test_pred["pred_crack_depth_pct"] = pred
    test_pred["abs_error"] = (test_pred["pred_crack_depth_pct"] - test_pred["crack_depth_pct"]).abs()
    test_pred.to_csv(save_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    case_summary = (
        test_pred.groupby("source_stem")
        .agg(
            true_crack_depth_pct=("crack_depth_pct", "first"),
            pred_crack_depth_pct=("pred_crack_depth_pct", mode_or_first),
            n_rows=("pred_crack_depth_pct", "size"),
            mae=("abs_error", "mean"),
        )
        .reset_index()
    )
    case_summary.to_csv(save_dir / "test_case_summary.csv", index=False, encoding="utf-8-sig")

    mse = mean_squared_error(y_test, pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    metrics = {
        "dataset": str(dataset_path),
        "rows": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_cases": int(groups.iloc[train_idx].nunique()),
        "test_cases": int(groups.iloc[test_idx].nunique()),
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "r2": float(r2),
        "features": FEATURE_COLS,
        "target": "crack_depth_pct",
    }
    (save_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")
    print(f"Saved model to {save_dir / 'crack_regressor.joblib'}")
    print(f"Saved train split to {save_dir / 'train_split.csv'}")
    print(f"Saved test split to {save_dir / 'test_split.csv'}")
    print(f"Saved test predictions to {save_dir / 'test_predictions.csv'}")
    print(f"Saved case summary to {save_dir / 'test_case_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset_crack.csv")
    parser.add_argument("--save", type=Path, default=ROOT / "model_out_crack")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_train(args.dataset, args.save, args.test_size, args.seed)


if __name__ == "__main__":
    main()
