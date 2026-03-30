"""
Plot exported TensorBoard scalar CSVs (one plot per file).

By default, uses step on x-axis and value on y-axis. You can switch x-axis to
other columns (for example time_spend) using --x-axis.

Run from project root:
  python simulations/rl_dbr/plot_tensorboard_logs.py 2603282120_training_rldbr
  python simulations/rl_dbr/plot_tensorboard_logs.py 2603282120_training_rldbr --x-axis time_spend
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_REPO_ROOT = _SCRIPT_DIR.parent.parent


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").is_dir():
            return p
    return _DEFAULT_REPO_ROOT.resolve()


def resolve_experiment_dir(repo_root: Path, experiment_name: str) -> Path:
    candidates = [
        repo_root / "data" / "experiments" / experiment_name,
        repo_root / "simulations" / "rl_dbr" / "data" / "experiments" / experiment_name,
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    raise FileNotFoundError(
        "Experiment directory not found. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def resolve_input_output_dirs(
    experiment_dir: Path, input_dir: Path | None, out_dir: Path | None
) -> tuple[Path, Path]:
    resolved_input = input_dir.resolve() if input_dir else experiment_dir / "tensorboard_logs"
    resolved_output = out_dir.resolve() if out_dir else resolved_input / "plots"
    return resolved_input, resolved_output


def plot_csv_file(csv_path: Path, out_dir: Path, x_axis: str) -> tuple[bool, str]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"{csv_path.name}: failed to read CSV ({e})"

    required = {x_axis, "value"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"{csv_path.name}: missing column(s): {', '.join(missing)}"

    plot_df = df[[x_axis, "value"]].copy()
    x_num = pd.to_numeric(plot_df[x_axis], errors="coerce")
    if x_num.notna().any():
        plot_df = plot_df.assign(_x_num=x_num).sort_values("_x_num").drop(columns="_x_num")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df[x_axis], plot_df["value"], linewidth=1.5)
    ax.set_xlabel(x_axis)
    ax.set_ylabel("value")
    ax.set_title(csv_path.stem)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{csv_path.stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True, f"{csv_path.name}: plotted"


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot tensorboard_logs CSV files")
    parser.add_argument(
        "experiment_name",
        help="Hydra output folder basename (e.g. 2603282120_training_rldbr)",
    )
    parser.add_argument(
        "--x-axis",
        default="step",
        help="Column to use as x-axis (default: step; e.g. time_spend)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: search upward for .git)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input CSV directory (default: {experiment}/tensorboard_logs)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output image directory (default: {input-dir}/plots)",
    )
    args = parser.parse_args()

    if not HAS_MPL:
        print("matplotlib not installed. Install with: pip install matplotlib", file=sys.stderr)
        return 1

    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path.cwd())
    try:
        experiment_dir = resolve_experiment_dir(repo_root, args.experiment_name)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    input_dir, out_dir = resolve_input_output_dirs(experiment_dir, args.input_dir, args.out_dir)
    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        return 1

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"Error: no CSV files found in {input_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    warnings: list[str] = []

    for csv_path in csv_files:
        ok, msg = plot_csv_file(csv_path, out_dir, args.x_axis)
        if ok:
            created += 1
        else:
            skipped += 1
            warnings.append(msg)

    print("TensorBoard CSV plotting")
    print(f"  Experiment: {experiment_dir}")
    print(f"  Input:      {input_dir}")
    print(f"  Output:     {out_dir}")
    print(f"  X-axis:     {args.x_axis}")
    print(f"  CSV files:  {len(csv_files)}")
    print(f"  Created:    {created}")
    print(f"  Skipped:    {skipped}")
    for w in warnings:
        print(f"  Warning:    {w}")

    return 0 if created > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
