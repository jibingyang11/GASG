"""Run the PersonDataset depth-model suite in separate Python processes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List


DEFAULT_METHODS = [
    "defom_identity",
    "defom_shift",
    "defom_dav2_warp",
    "defom_moge_warp",
    "defom_zerostereo",
    "gasg_defom",
    "defom_real",
]


def run(cmd: List[str], continue_on_error: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if not continue_on_error:
            raise
        print("WARNING: method failed; continuing.", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--person_root", default="data/PersonDataset")
    p.add_argument("--save_dir", default="results/pseudostereo_depth")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_visuals", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    args, extra = p.parse_known_args()

    py = sys.executable
    os.makedirs(args.save_dir, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    for method in methods:
        out_json = os.path.join(args.save_dir, f"{method}.json")
        if args.skip_existing and os.path.exists(out_json):
            print(f"Skip existing: {out_json}")
            continue
        cmd = [
            py, "scripts/eval_person_depth_model.py",
            "--method", method,
            "--person_root", args.person_root,
            "--save_dir", args.save_dir,
            "--height", str(args.height),
            "--width", str(args.width),
            "--max_samples", str(args.max_samples),
            "--device", args.device,
        ]
        if args.save_visuals:
            cmd.append("--save_visuals")
        cmd += extra
        run(cmd, args.continue_on_error)

    run([py, "scripts/merge_person_depth_results.py",
         "--results_dir", args.save_dir], False)


if __name__ == "__main__":
    main()
