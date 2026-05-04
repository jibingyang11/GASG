"""Merge right-view generation JSON files into CSV/JSON/Markdown summaries."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List


ORDER = [
    "identity",
    "shift_const",
    "dav2_warp",
    "moge_warp",
    "zerostereo_official",
    "stereospace_hf",
    "oracle_warp",
    "gasg_untrained",
    "gasg_warp_only",
    "gasg_direct_only",
    "gasg_no_refine",
    "gasg_full",
]

COLS = [
    "method",
    "psnr",
    "ssim",
    "lpips",
    "inference_time_ms_mean",
    "inference_time_ms_median",
    "params_M",
    "n_eval",
]


def load_rows(results_dir: str) -> List[Dict]:
    rows = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path)
        if name.startswith("summary"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        if {"method", "psnr", "ssim", "lpips"}.issubset(row):
            rows[row["method"]] = row
    return [rows[m] for m in ORDER if m in rows]


def write_outputs(rows: List[Dict], results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in COLS})

    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    md = [
        "# Right-view generation results",
        "",
        "| Method | N | PSNR | SSIM | LPIPS | Time (ms) | Params (M) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['method']} | {int(row.get('n_eval', 0))} | "
            f"{float(row['psnr']):.2f} | {float(row['ssim']):.4f} | "
            f"{float(row['lpips']):.4f} | "
            f"{float(row.get('inference_time_ms_mean', 0.0)):.1f} | "
            f"{float(row.get('params_M', 0.0)):.2f} |"
        )
    with open(os.path.join(results_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/rightview_generation")
    args = parser.parse_args()
    rows = load_rows(args.results_dir)
    write_outputs(rows, args.results_dir)
    print(f"Merged {len(rows)} right-view result files in {args.results_dir}")


if __name__ == "__main__":
    main()
