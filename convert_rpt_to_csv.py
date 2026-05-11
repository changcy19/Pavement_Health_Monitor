from __future__ import annotations

import argparse
from pathlib import Path


def convert_one(rpt_path: Path, csv_path: Path) -> bool:
    rows = []
    max_cols = 0
    for line in rpt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        trim = line.strip()
        if not trim:
            continue
        parts = trim.split()
        vals = []
        all_numeric = True
        for p in parts:
            try:
                vals.append(f"{float(p):.17g}")
            except Exception:
                all_numeric = False
                break
        if all_numeric and len(vals) >= 2:
            rows.append(vals)
            max_cols = max(max_cols, len(vals))

    if not rows:
        return False

    header = [f"col_{i}" for i in range(1, max_cols + 1)]
    out_lines = [",".join(header)]
    for row in rows:
        padded = row + [""] * (max_cols - len(row))
        out_lines.append(",".join(padded))
    csv_path.write_text("\n".join(out_lines), encoding="utf-8-sig")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Abaqus RPT files to CSV.")
    parser.add_argument("--input-dir", type=Path, default=Path("rpt_data"), help="RPT directory")
    parser.add_argument("--output-dir", type=Path, default=Path("csv_data"), help="CSV directory")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    for rpt_path in sorted(args.input_dir.rglob("*.rpt")):
        rel = rpt_path.relative_to(args.input_dir)
        csv_path = (args.output_dir / rel).with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_path.exists():
            skipped += 1
            continue
        if convert_one(rpt_path, csv_path):
            converted += 1
            print(f"[OK] {rpt_path.name} -> {csv_path.name}")
        else:
            print(f"[SKIP] No numeric data: {rpt_path}")

    print(f"[DONE] Converted={converted}, SkippedExisting={skipped}")


if __name__ == "__main__":
    main()
