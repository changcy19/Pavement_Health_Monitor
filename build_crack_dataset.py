from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


CASE_PATTERN = re.compile(
    r"^TG(?P<tg_sign>N|-)?(?P<tg>\d+(?:\.\d+)?)_H(?P<h>\d+(?:\.\d+)?)_E(?P<e>\d+(?:\.\d+)?)_"
    r"(?P<crack>25|50|75|100)_V(?P<v>\d+(?:\.\d+)?)_M(?P<m>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)


@dataclass
class CaseLabel:
    file: str
    stem: str
    tg: float
    h_cm: float
    e_gpa: float
    crack_depth_pct: float
    v_kmh: float
    m_t: float


def parse_case_name(path: Path) -> CaseLabel:
    match = CASE_PATTERN.match(path.stem)
    if not match:
        raise ValueError(f"Unsupported ODB name: {path.name}")
    tg = float(match.group("tg"))
    if (match.group("tg_sign") or "").upper() == "N":
        tg = -tg
    return CaseLabel(
        file=path.name,
        stem=path.stem,
        tg=tg,
        h_cm=float(match.group("h")),
        e_gpa=float(match.group("e")),
        crack_depth_pct=float(match.group("crack")),
        v_kmh=float(match.group("v")),
        m_t=float(match.group("m")),
    )


def load_csv_signals(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in frame.columns}
    if "col_1" not in cols or "col_2" not in cols or "col_3" not in cols:
        raise ValueError(f"Unexpected columns in {csv_path.name}: {list(frame.columns)}")
    frame = frame.rename(columns={cols["col_1"]: "time", cols["col_2"]: "accel", cols["col_3"]: "strain"})
    if "col_4" in cols:
        frame = frame.rename(columns={cols["col_4"]: "stress"})
    return frame


def band_energy_ratio(freq: np.ndarray, amp: np.ndarray, low: float = 0.0, high: float = 200.0, bands: int = 8) -> np.ndarray:
    edges = np.linspace(low, high, bands + 1)
    energies = []
    for i in range(bands):
        l, r = float(edges[i]), float(edges[i + 1])
        mask = (freq >= l) & (freq <= r) if i == bands - 1 else (freq >= l) & (freq < r)
        energies.append(float(np.sum(np.square(amp[mask]))))
    energies = np.asarray(energies, dtype=float)
    total = float(energies.sum())
    if total <= 0:
        return np.zeros_like(energies)
    return energies / total


def extract_features(csv_path: Path, label: CaseLabel, point: str) -> dict:
    frame = load_csv_signals(csv_path)
    numeric = frame[["time", "accel", "strain"]].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(numeric["time"]) & np.isfinite(numeric["accel"]) & np.isfinite(numeric["strain"])
    if not np.any(mask):
        raise ValueError(f"No valid numeric rows in {csv_path.name}")

    time = numeric.loc[mask, "time"].to_numpy(dtype=float)
    accel = numeric.loc[mask, "accel"].to_numpy(dtype=float)
    strain = numeric.loc[mask, "strain"].to_numpy(dtype=float)

    dt = float(np.median(np.diff(time))) if len(time) > 1 else 1.0
    accel_c = accel - float(np.mean(accel)) if len(accel) else accel
    amp = np.abs(np.fft.rfft(accel_c)) * 2.0 / max(len(accel_c), 1)
    freq = np.fft.rfftfreq(len(accel_c), d=dt) if len(accel_c) else np.array([], dtype=float)
    band_ratio = band_energy_ratio(freq, amp)

    feature = {
        "file": label.file,
        "point": point,
        **{k: v for k, v in asdict(label).items() if k not in {"file", "stem"}},
        "n_samples": int(len(time)),
        "time_start": float(time[0]) if len(time) else np.nan,
        "time_end": float(time[-1]) if len(time) else np.nan,
        "accel_mean": float(np.mean(accel)) if len(accel) else np.nan,
        "accel_std": float(np.std(accel)) if len(accel) else np.nan,
        "accel_rms": float(np.sqrt(np.mean(np.square(accel)))) if len(accel) else np.nan,
        "accel_peak": float(np.max(np.abs(accel))) if len(accel) else np.nan,
        "accel_kurtosis": float(pd.Series(accel).kurt()) if len(accel) > 3 else np.nan,
        "strain_mean": float(np.mean(strain)) if len(strain) else np.nan,
        "strain_std": float(np.std(strain)) if len(strain) else np.nan,
        "strain_rms": float(np.sqrt(np.mean(np.square(strain)))) if len(strain) else np.nan,
        "strain_peak": float(np.max(np.abs(strain))) if len(strain) else np.nan,
        "strain_kurtosis": float(pd.Series(strain).kurt()) if len(strain) > 3 else np.nan,
        "spectral_centroid": float(np.sum(freq * np.square(amp)) / (np.sum(np.square(amp)) + 1e-12)) if len(freq) else np.nan,
        "dominant_freq": float(freq[int(np.argmax(amp))]) if len(freq) else np.nan,
        "spec_energy_0_200": float(np.sum(np.square(amp[(freq >= 0) & (freq <= 200)]))) if len(freq) else np.nan,
    }

    for i, value in enumerate(band_ratio, start=1):
        feature[f"band_ratio_{i}"] = float(value)

    return feature


def build_dataset(data_dir: Path, csv_dir: Path, points: list[str]) -> pd.DataFrame:
    rows = []
    missing = []
    for odb_path in sorted(data_dir.glob("*.odb")):
        label = parse_case_name(odb_path)
        row = {**{k: v for k, v in asdict(label).items() if k not in {"stem"}}}
        row["source_stem"] = label.stem
        ok = True
        for point in points:
            csv_path = csv_dir / f"{odb_path.stem}__{point}.csv"
            if not csv_path.exists():
                ok = False
                missing.append(f"{odb_path.name}:{point}")
                break
            point_feat = extract_features(csv_path, label, point)
            for k, v in point_feat.items():
                row[f"{point}_{k}"] = v
        if ok:
            rows.append(row)
    if not rows:
        raise SystemExit("No CSV feature rows were built. Check export and conversion outputs.")
    if missing:
        print(f"[WARN] Skipped {len(missing)} missing point files; first missing: {missing[:5]}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a crack feature table from exported CSV signals.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="Directory containing ODB files")
    parser.add_argument("--csv-dir", type=Path, default=ROOT / "csv_data", help="Directory containing converted CSV files")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset_crack.csv", help="Output dataset CSV")
    parser.add_argument("--meta", type=Path, default=ROOT / "dataset_crack_meta.json", help="Output metadata JSON")
    parser.add_argument("--points", nargs="*", default=["P001", "P002", "P003", "P004"], help="Point names")
    args = parser.parse_args()

    frame = build_dataset(args.data_dir, args.csv_dir, args.points)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")

    meta = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "points": args.points,
        "target": "crack_depth_pct",
    }
    args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}")
    print(f"Saved: {args.meta}")


if __name__ == "__main__":
    main()
