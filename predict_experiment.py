from __future__ import annotations

import argparse
from pathlib import Path

import joblib  # type: ignore
import pandas as pd


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
    parser = argparse.ArgumentParser(description="Predict void width for experiment data.")
    parser.add_argument("--model", type=Path, default=ROOT / "model_out" / "void_regressor.joblib")
    parser.add_argument("--input", type=Path, required=True, help="Experiment feature CSV")
    parser.add_argument("--output", type=Path, default=ROOT / "experiment_predictions.csv")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.input)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Input file missing columns: {missing[:10]}")

    X = df[FEATURE_COLS].copy()
    pred = model.predict(X)

    out = df.copy()
    out["pred_void_width_cm"] = pred
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
