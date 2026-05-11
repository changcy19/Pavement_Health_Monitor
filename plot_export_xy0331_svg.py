from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "XY_DATA5"
OUTPUT_DIR = ROOT / "plots5"
PATTERN = re.compile(
    r"^valid(?:(?P<prefix>[0-9A-Za-z]+)-)?(?P<c>[A-Z])(?:-(?P<t>\d+))?__P(?P<p>\d+)\.csv$",
    re.IGNORECASE,
)
# Read mode: "all" | "with-2" | "without-2"
FILE_VARIANT = "without-2"
TARGET_POINTS = {
    "001": "P001",
    "002": "P002",
    "003": "P003",
    "004": "P004",
    "005": "P005",
    "006": "P006",
    "007": "P007",
    "008": "P008",
}
CASE_ORDER = ["A", "B", "C", "D"]


@dataclass
class SeriesData:
    point_code: str
    point_name: str
    t_code: str
    t_label: str
    t_value: float
    c_code: str
    c_label: str
    c_value: float
    time: np.ndarray
    accel: np.ndarray
    strain: np.ndarray
    freq: np.ndarray
    amp: np.ndarray
    source: Path


def format_t_label(code: str) -> str:
    """T2 -> 0.002, T5 -> 0.005"""
    if code == '2':
        return "0.002"
    elif code == '5':
        return "0.005"
    return code


def format_c_label(code: str) -> str:
    """Void size labels."""
    mapping = {"O": "0 m", "A": "0.5 m", "B": "1.0 m", "C": "1.5 m", "D": "2.0 m"}
    return mapping.get(code.upper(), code)


def read_series(path: Path) -> SeriesData:
    match = PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Unexpected file name: {path.name}")

    t_code = match.group("t") or "2"
    c_code = match.group("c")
    point_code = match.group("p")
    point_name = TARGET_POINTS[point_code]

    frame = pd.read_csv(path)
    time = frame["col_1"].to_numpy(dtype=float)
    accel = frame["col_2"].to_numpy(dtype=float)
    strain = frame["col_3"].to_numpy(dtype=float)

    # Use the actual spacing from the exported time column to avoid label-only assumptions.
    dt = float(np.median(np.diff(time)))
    centered = accel - float(np.mean(accel))
    freq = np.fft.rfftfreq(centered.size, d=dt)
    amp = np.abs(np.fft.rfft(centered)) * 2.0 / centered.size
    if amp.size:
        amp[0] /= 2.0

    return SeriesData(
        point_code=point_code,
        point_name=point_name,
        t_code=t_code,
        t_label=format_t_label(t_code),
        t_value=float(format_t_label(t_code)),
        c_code=c_code,
        c_label=format_c_label(c_code),
        c_value={"O": 0.0, "A": 0.5, "B": 1.0, "C": 1.5, "D": 2.0}.get(c_code, 0.0),
        time=time,
        accel=accel,
        strain=strain,
        freq=freq,
        amp=amp,
        source=path,
    )


def file_variant_matches(path: Path, mode: str) -> bool:
    name = path.name.lower()
    has_dash_t = bool(re.search(r"-[0-9]+__p", name))
    if mode == "with-2":
        return has_dash_t
    if mode == "without-2":
        return not has_dash_t
    return True


def escape(text: str) -> str:
    return html.escape(text, quote=True)


def nice_limits(values: list[np.ndarray]) -> tuple[float, float]:
    vmin = min(float(np.min(v)) for v in values)
    vmax = max(float(np.max(v)) for v in values)
    if math.isclose(vmin, vmax):
        pad = 1.0 if math.isclose(vmin, 0.0) else abs(vmin) * 0.1
        return vmin - pad, vmax + pad
    pad = (vmax - vmin) * 0.06
    return vmin - pad, vmax + pad


def expand_limits(vmin: float, vmax: float, factor: float = 1.2) -> tuple[float, float]:
    center = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin) * factor
    if math.isclose(half, 0.0):
        half = 1.0
    return center - half, center + half


def ticks_for_limits(vmin: float, vmax: float, count: int = 5) -> list[float]:
    if count < 2:
        return [vmin, vmax]
    return [vmin + i * (vmax - vmin) / (count - 1) for i in range(count)]


def fmt_tick(value: float) -> str:
    av = abs(value)
    if av >= 1000 or (0 < av < 0.01):
        return f"{value:.2e}"
    if av >= 100:
        return f"{value:.1f}"
    if av >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def polyline_points(
    xs: np.ndarray,
    ys: np.ndarray,
    x0: float,
    y0: float,
    width: float,
    height: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> str:
    xmin, xmax = xlim
    ymin, ymax = ylim
    if math.isclose(xmax, xmin):
        xmax = xmin + 1.0
    if math.isclose(ymax, ymin):
        ymax = ymin + 1.0

    xnorm = (xs - xmin) / (xmax - xmin)
    ynorm = (ys - ymin) / (ymax - ymin)
    px = x0 + xnorm * width
    py = y0 + height - ynorm * height
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(px, py, strict=True))


def draw_subplot(
    pieces: list[str],
    series: SeriesData,
    x: float,
    y: float,
    width: float,
    height: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    x_label: str,
    y_label: str,
    line_color: str,
    domain: str,
) -> None:
    left_pad = 70
    right_pad = 18
    top_pad = 28
    bottom_pad = 42
    plot_x = x + left_pad
    plot_y = y + top_pad
    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad

    pieces.append(
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        'fill="#ffffff" stroke="#d9dde5" stroke-width="1"/>'
    )
    pieces.append(
        f'<rect x="{plot_x:.2f}" y="{plot_y:.2f}" width="{plot_w:.2f}" height="{plot_h:.2f}" '
        'fill="#fbfcfe" stroke="#bcc5d3" stroke-width="1"/>'
    )

    xticks = ticks_for_limits(xlim[0], xlim[1])
    yticks = ticks_for_limits(ylim[0], ylim[1])

    for xt in xticks:
        px = plot_x + (xt - xlim[0]) / (xlim[1] - xlim[0]) * plot_w
        pieces.append(
            f'<line x1="{px:.2f}" y1="{plot_y:.2f}" x2="{px:.2f}" y2="{plot_y + plot_h:.2f}" '
            'stroke="#eef1f6" stroke-width="1"/>'
        )
        pieces.append(
            f'<text x="{px:.2f}" y="{plot_y + plot_h + 18:.2f}" text-anchor="middle" '
            'font-size="11" fill="#4b5563">'
            f"{escape(fmt_tick(xt))}</text>"
        )

    for yt in yticks:
        py = plot_y + plot_h - (yt - ylim[0]) / (ylim[1] - ylim[0]) * plot_h
        pieces.append(
            f'<line x1="{plot_x:.2f}" y1="{py:.2f}" x2="{plot_x + plot_w:.2f}" y2="{py:.2f}" '
            'stroke="#eef1f6" stroke-width="1"/>'
        )
        pieces.append(
            f'<text x="{plot_x - 8:.2f}" y="{py + 4:.2f}" text-anchor="end" '
            'font-size="11" fill="#4b5563">'
            f"{escape(fmt_tick(yt))}</text>"
        )

    if domain == "time_accel":
        xs = series.time
        ys = series.accel
    elif domain == "time_strain":
        xs = series.time
        ys = series.strain
    else:
        xs = series.freq
        ys = series.amp

    # Enforce plotting window so displayed data always matches the axis range.
    xmask = (xs >= xlim[0]) & (xs <= xlim[1])
    if np.any(xmask):
        xs = xs[xmask]
        ys = ys[xmask]

    points = polyline_points(xs, ys, plot_x, plot_y, plot_w, plot_h, xlim, ylim)
    pieces.append(
        f'<polyline fill="none" stroke="{line_color}" stroke-width="1.6" '
        f'points="{points}"/>'
    )

    pieces.append(
        f'<text x="{x + width / 2:.2f}" y="{y + 18:.2f}" text-anchor="middle" '
        'font-size="13" font-weight="600" fill="#1f2937">'
        f"{escape(f'{series.point_name} | Void={series.c_code} ({series.c_label})')}"
        "</text>"
    )
    pieces.append(
        f'<text x="{x + width / 2:.2f}" y="{y + height - 10:.2f}" text-anchor="middle" '
        'font-size="12" fill="#4b5563">'
        f"{escape(x_label)}</text>"
    )
    pieces.append(
        f'<text x="{x + 18:.2f}" y="{y + height / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 {x + 18:.2f} {y + height / 2:.2f})" '
        'font-size="12" fill="#4b5563">'
        f"{escape(y_label)}</text>"
    )


def band_energy_ratio(series: SeriesData, band_edges: np.ndarray) -> np.ndarray:
    energies = []
    for i in range(len(band_edges) - 1):
        low = float(band_edges[i])
        high = float(band_edges[i + 1])
        if i == len(band_edges) - 2:
            mask = (series.freq >= low) & (series.freq <= high)
        else:
            mask = (series.freq >= low) & (series.freq < high)
        band_amp = series.amp[mask]
        energies.append(float(np.sum(np.square(band_amp))))

    energy_array = np.asarray(energies, dtype=float)
    total = float(np.sum(energy_array))
    if total <= 0.0:
        return np.zeros_like(energy_array)
    return energy_array / total


def band_energy_stats(series: SeriesData, low: float, high: float) -> dict[str, float]:
    low_mask = (series.freq >= 0.0) & (series.freq < low)
    high_mask = (series.freq >= low) & (series.freq <= high)

    low_energy = float(np.sum(np.square(series.amp[low_mask])))
    high_energy = float(np.sum(np.square(series.amp[high_mask])))
    total_energy = low_energy + high_energy
    high_low_ratio = float(high_energy / low_energy) if low_energy > 0.0 else float("inf")
    high_total_ratio = float(high_energy / total_energy) if total_energy > 0.0 else 0.0

    return {
        "energy_0_50": low_energy,
        "energy_50_200": high_energy,
        "energy_ratio_50_200_to_0_50": high_low_ratio,
        "energy_ratio_50_200_to_total": high_total_ratio,
    }


def export_band_energy_csv(output_path: Path, series_list: list[SeriesData]) -> None:
    rows = []
    for series in series_list:
        if series.t_code != "2":
            continue
        stats = band_energy_stats(series, 50.0, 200.0)
        rows.append(
            {
                "point": series.point_name,
                "void_code": series.c_code,
                "void_label": series.c_label,
                "signal_type": "acceleration",
                **stats,
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["point", "void_code"], kind="stable").reset_index(drop=True)
    output_path.write_text(frame.to_csv(index=False), encoding="utf-8-sig")


def draw_band_subplot(
    pieces: list[str],
    series: SeriesData,
    x: float,
    y: float,
    width: float,
    height: float,
    band_edges: np.ndarray,
    line_color: str,
) -> None:
    left_pad = 70
    right_pad = 18
    top_pad = 28
    bottom_pad = 42
    plot_x = x + left_pad
    plot_y = y + top_pad
    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad

    pieces.append(
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        'fill="#ffffff" stroke="#d9dde5" stroke-width="1"/>'
    )
    pieces.append(
        f'<rect x="{plot_x:.2f}" y="{plot_y:.2f}" width="{plot_w:.2f}" height="{plot_h:.2f}" '
        'fill="#fbfcfe" stroke="#bcc5d3" stroke-width="1"/>'
    )

    ratios = band_energy_ratio(series, band_edges)
    y_ticks = np.linspace(0.0, 1.0, 5)
    for yt in y_ticks:
        py = plot_y + plot_h - yt * plot_h
        pieces.append(
            f'<line x1="{plot_x:.2f}" y1="{py:.2f}" x2="{plot_x + plot_w:.2f}" y2="{py:.2f}" '
            'stroke="#eef1f6" stroke-width="1"/>'
        )
        pieces.append(
            f'<text x="{plot_x - 8:.2f}" y="{py + 4:.2f}" text-anchor="end" '
            'font-size="11" fill="#4b5563">'
            f"{escape(f'{yt:.2f}')}</text>"
        )

    band_count = len(ratios)
    bar_gap = 4.0
    bar_w = (plot_w - (band_count + 1) * bar_gap) / band_count
    for i, ratio in enumerate(ratios):
        bx = plot_x + bar_gap + i * (bar_w + bar_gap)
        bh = max(0.0, min(1.0, float(ratio))) * plot_h
        by = plot_y + plot_h - bh
        pieces.append(
            f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" '
            f'fill="{line_color}" fill-opacity="0.85"/>'
        )

        label = f"{int(band_edges[i])}-{int(band_edges[i + 1])}"
        pieces.append(
            f'<text x="{bx + bar_w / 2:.2f}" y="{plot_y + plot_h + 16:.2f}" text-anchor="middle" '
            'font-size="9" fill="#4b5563">'
            f"{escape(label)}</text>"
        )

    pieces.append(
        f'<text x="{x + width / 2:.2f}" y="{y + 18:.2f}" text-anchor="middle" '
        'font-size="13" font-weight="600" fill="#1f2937">'
        f"{escape(f'{series.point_name} | Void={series.c_code} ({series.c_label})')}</text>"
    )
    pieces.append(
        f'<text x="{x + width / 2:.2f}" y="{y + height - 10:.2f}" text-anchor="middle" '
        'font-size="12" fill="#4b5563">Band (Hz)</text>'
    )
    pieces.append(
        f'<text x="{x + 18:.2f}" y="{y + height / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 {x + 18:.2f} {y + height / 2:.2f})" '
        'font-size="12" fill="#4b5563">Energy Ratio</text>'
    )


def write_band_energy_svg(
    output_path: Path,
    title: str,
    subtitle: str,
    series_grid: list[list[SeriesData]],
    band_edges: np.ndarray,
) -> None:
    cols = len(series_grid[0])
    rows = len(series_grid)
    cell_w = 430
    cell_h = 250
    left_margin = 28
    top_margin = 88
    gap_x = 18
    gap_y = 18
    width = left_margin * 2 + cols * cell_w + (cols - 1) * gap_x
    height = top_margin + 34 + rows * cell_h + (rows - 1) * gap_y + 30

    colors = {"O": "#2563eb", "A": "#16a34a", "B": "#d97706", "C": "#dc2626"}

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f5f7fb"/>',
        f'<text x="{width / 2:.2f}" y="34" text-anchor="middle" font-size="28" '
        'font-weight="700" fill="#111827">'
        f"{escape(title)}</text>",
        f'<text x="{width / 2:.2f}" y="60" text-anchor="middle" font-size="14" '
        'fill="#4b5563">'
        f"{escape(subtitle)}</text>",
    ]

    for r, row in enumerate(series_grid):
        for c, series in enumerate(row):
            x = left_margin + c * (cell_w + gap_x)
            y = top_margin + 20 + r * (cell_h + gap_y)
            draw_band_subplot(
                pieces=pieces,
                series=series,
                x=x,
                y=y,
                width=cell_w,
                height=cell_h,
                band_edges=band_edges,
                line_color=colors.get(series.c_code, "#334155"),
            )

    pieces.append("</svg>")
    output_path.write_text("\n".join(pieces), encoding="utf-8")


def write_svg(
    output_path: Path,
    title: str,
    subtitle: str,
    series_grid: list[list[SeriesData]],
    domain: str,
    x_label: str,
    y_label: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    cols = len(series_grid[0])
    rows = len(series_grid)
    cell_w = 430
    cell_h = 250
    left_margin = 28
    top_margin = 88
    gap_x = 18
    gap_y = 18
    width = left_margin * 2 + cols * cell_w + (cols - 1) * gap_x
    height = top_margin + 34 + rows * cell_h + (rows - 1) * gap_y + 30

    colors = {
        "O": "#2563eb",  # no void
        "A": "#16a34a",  # 0.5m
        "B": "#d97706",  # 1.0m
        "C": "#dc2626",  # 1.5m
    }

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f5f7fb"/>',
        f'<text x="{width / 2:.2f}" y="34" text-anchor="middle" font-size="28" '
        'font-weight="700" fill="#111827">'
        f"{escape(title)}</text>",
        f'<text x="{width / 2:.2f}" y="60" text-anchor="middle" font-size="14" '
        'fill="#4b5563">'
        f"{escape(subtitle)}</text>",
    ]

    for r, row in enumerate(series_grid):
        for c, series in enumerate(row):
            x = left_margin + c * (cell_w + gap_x)
            y = top_margin + 20 + r * (cell_h + gap_y)
            draw_subplot(
                pieces=pieces,
                series=series,
                x=x,
                y=y,
                width=cell_w,
                height=cell_h,
                xlim=xlim,
                ylim=ylim,
                x_label=x_label,
                y_label=y_label,
                line_color=colors.get(series.c_code, "#334155"),
                domain=domain,
            )

    pieces.append("</svg>")
    output_path.write_text("\n".join(pieces), encoding="utf-8")


def build_overview(
    output_path: Path,
    chart_pairs: list[tuple[str, Path, Path, Path, Path]],
) -> None:
    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
    <title>路面板振动响应特征对比</title>
  <style>
    body {{
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
      margin: 24px;
      color: #1f2937;
      background: #f3f5f9;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #d8dee9;
      border-radius: 14px;
      padding: 18px 18px 8px;
      margin-bottom: 20px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      background: #fff;
    }}
    .meta {{
      margin: 8px 0 20px;
      color: #4b5563;
      line-height: 1.6;
    }}
  </style>
</head>
<body>
  <div class="card">
                <h1>路面板振动响应特征对比</h1>
    <div class="meta">
            valid 后字母表示脱空宽度: O=0m, A=0.5m, B=1.0m, C=1.5m<br>
            数据列: col_1 时间(1-3.5s, dt=0.002s), col_2 加速度(m/s²), col_3 应变, col_4 应力(Pa)<br>
            频段能量图基于加速度频谱，范围 0-200Hz，每 25Hz 一段并归一化
    </div>
  </div>
    {''.join(
            f'''<div class="card">
                <h2>{escape(point_label)} 加速度时域图 (1-3.5s)</h2>
                <img src="{escape(accel_time_path.name)}" alt="{escape(point_label)}加速度时域图">
        </div>
        <div class="card">
                <h2>{escape(point_label)} 应变时域图 (1-3.5s)</h2>
                <img src="{escape(strain_time_path.name)}" alt="{escape(point_label)}应变时域图">
    </div>
    <div class="card">
                <h2>{escape(point_label)} 加速度频域图 (0-200Hz)</h2>
                <img src="{escape(freq_path.name)}" alt="{escape(point_label)}频域图">
    </div>
    <div class="card">
                <h2>{escape(point_label)} 分段能量占比图 (0-200Hz, 每25Hz一段)</h2>
                <img src="{escape(band_path.name)}" alt="{escape(point_label)}分段能量占比图">
    </div>'''
                        for point_label, accel_time_path, strain_time_path, freq_path, band_path in chart_pairs
    )}
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if FILE_VARIANT not in {"all", "with-2", "without-2"}:
        raise SystemExit(f"Invalid FILE_VARIANT: {FILE_VARIANT}")

    series_list: list[SeriesData] = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        if not file_variant_matches(path, FILE_VARIANT):
            continue
        match = PATTERN.match(path.name)
        if not match:
            continue
        point_code = match.group("p")
        if point_code not in TARGET_POINTS:
            continue
        series_list.append(read_series(path))

    if not series_list:
        raise SystemExit("No target CSV files found for configured point pairs.")

    # --- 1. 时域图处理：统一 1s - 3.5s ---
    time_xlim = (1.0, 3.5)

    # 提取该时间窗内的加速度和应变值，用于计算统一 Y 轴刻度
    visible_accels = []
    visible_strains = []
    for s in series_list:
        mask = (s.time >= time_xlim[0]) & (s.time <= time_xlim[1])
        if np.any(mask):
            visible_accels.append(s.accel[mask])
            visible_strains.append(s.strain[mask])

    # 如果该窗口内没数据，回退到全局范围
    time_ylim = nice_limits(visible_accels if visible_accels else [s.accel for s in series_list])
    time_ylim = expand_limits(*time_ylim, factor=1.35)
    strain_ylim = nice_limits(visible_strains if visible_strains else [s.strain for s in series_list])
    strain_ylim = expand_limits(*strain_ylim, factor=1.35)

    by_key = {(s.point_code, s.c_code): s for s in series_list if s.t_code == "2"}
    available_cases = {s.c_code for s in series_list if s.t_code == "2"}
    preferred_cases = [c for c in CASE_ORDER if c in available_cases]
    if len(preferred_cases) < 4 and "O" in available_cases:
        preferred_cases.append("O")
    effective_case_order = preferred_cases[:4]
    if len(effective_case_order) < 4:
        raise SystemExit(f"Insufficient case files. Found cases: {', '.join(sorted(available_cases))}")

    # --- 2. 频域图处理：统一 0 - 200Hz ---
    freq_limit = 200.0

    # 对所有序列进行截断，仅保留 200Hz 以内的频率分量，以便在频域图中保持一致的比较范围
    freq_series_list: list[SeriesData] = []
    for series in series_list:
        mask = series.freq <= freq_limit + 1e-9
        freq_series_list.append(
            SeriesData(
                point_code=series.point_code,
                point_name=series.point_name,
                t_code=series.t_code,
                t_label=series.t_label,
                t_value=series.t_value,
                c_code=series.c_code,
                c_label=series.c_label,
                c_value=series.c_value,
                time=series.time,
                accel=series.accel,
                strain=series.strain,
                freq=series.freq[mask],
                amp=series.amp[mask],
                source=series.source,
            )
        )
    # 计算 200Hz 范围内所有振幅的最大值，统一频域 Y 轴
    freq_ylim = nice_limits([s.amp for s in freq_series_list])

    freq_by_key = {(s.point_code, s.c_code): s for s in freq_series_list if s.t_code == "2"}

    band_edges = np.linspace(0.0, 200.0, 9)

    chart_pairs: list[tuple[str, Path, Path, Path, Path]] = []
    point_codes = sorted(TARGET_POINTS.keys())
    for point_code in point_codes:
        point_label = TARGET_POINTS[point_code]

        missing_keys = [
            (point_code, c_code)
            for c_code in effective_case_order
            if (point_code, c_code) not in by_key
        ]
        if missing_keys:
            missing_desc = ", ".join(f"{TARGET_POINTS[p]}-{c}" for p, c in missing_keys)
            raise SystemExit(f"Missing required series for {point_label}: {missing_desc}")

        case_series = [by_key[(point_code, c_code)] for c_code in effective_case_order]
        case_freq_series = [freq_by_key[(point_code, c_code)] for c_code in effective_case_order]

        accel_time_grid = [case_series[:2], case_series[2:]]
        strain_time_grid = [case_series[:2], case_series[2:]]
        freq_grid = [case_freq_series[:2], case_freq_series[2:]]

        accel_time_svg = OUTPUT_DIR / f"acceleration_time_{point_code}.svg"
        strain_time_svg = OUTPUT_DIR / f"strain_time_{point_code}.svg"
        freq_svg = OUTPUT_DIR / f"acceleration_frequency_{point_code}.svg"
        band_svg = OUTPUT_DIR / f"acceleration_band_energy_ratio_{point_code}.svg"

        write_svg(
            output_path=accel_time_svg,
            title=f"{point_label} Acceleration Time Signal",
            subtitle="Void cases O/A/B/C in 2x2 layout",
            series_grid=accel_time_grid,
            domain="time_accel",
            x_label="Time (s)",
            y_label="Acceleration (m/s²)",
            xlim=time_xlim,
            ylim=time_ylim,
        )

        write_svg(
            output_path=strain_time_svg,
            title=f"{point_label} Strain Time Signal",
            subtitle="Void cases O/A/B/C in 2x2 layout",
            series_grid=strain_time_grid,
            domain="time_strain",
            x_label="Time (s)",
            y_label="Strain",
            xlim=time_xlim,
            ylim=strain_ylim,
        )

        write_svg(
            output_path=freq_svg,
            title=f"{point_label} Acceleration Spectrum",
            subtitle="Void cases O/A/B/C in 2x2 layout",
            series_grid=freq_grid,
            domain="freq",
            x_label="Frequency (Hz)",
            y_label="Amplitude",
            xlim=(0.0, freq_limit),
            ylim=freq_ylim,
        )

        write_band_energy_svg(
            output_path=band_svg,
            title=f"{point_label} Band Energy Ratio",
            subtitle="0-200Hz split into 8 bands (25Hz each)",
            series_grid=freq_grid,
            band_edges=band_edges,
        )

        chart_pairs.append((point_label, accel_time_svg, strain_time_svg, freq_svg, band_svg))

    build_overview(
        output_path=OUTPUT_DIR / "comparison_overview.html",
        chart_pairs=chart_pairs,
    )

    export_band_energy_csv(OUTPUT_DIR / "band_energy_ratio_50_200_vs_0_50.csv", freq_series_list)

    for point_label, accel_time_svg, strain_time_svg, freq_svg, band_svg in chart_pairs:
        print(f"Generated ({point_label}): {accel_time_svg}")
        print(f"Generated ({point_label}): {strain_time_svg}")
        print(f"Generated ({point_label}): {freq_svg}")
        print(f"Generated ({point_label}): {band_svg}")
    print(f"Generated: {OUTPUT_DIR / 'comparison_overview.html'}")
    print(f"Generated: {OUTPUT_DIR / 'band_energy_ratio_50_200_vs_0_50.csv'}")


if __name__ == "__main__":
    main()
