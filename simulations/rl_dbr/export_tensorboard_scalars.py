"""
Export TensorBoard scalar runs for an RL-DBR Hydra experiment to CSV (one file per tag).

Resolves the experiment folder by name (searches data/** recursively, including
data/experiment and data/experiments), finds matching SB3 log dirs under the tensorboard root, and writes CSVs to
{experiment_dir}/tensorboard_logs/.
Each CSV includes wall_time plus timestamp (UTC ISO-8601) and time_spend
(minutes since the earliest scalar event across all exported tags).

Run from project root:
  python simulations/rl_dbr/export_tensorboard_scalars.py 2603282120_training_rldbr
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    STORE_EVERYTHING_SIZE_GUIDANCE,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_REPO_ROOT = _SCRIPT_DIR.parent.parent


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").is_dir():
            return p
    return cur


def resolve_experiment_dir(repo_root: Path, experiment_name: str) -> Path:
    """Resolve to …/<experiment_name> under data/** (and rl_dbr/data/**), any depth."""
    search_bases = [
        repo_root / "data",
        repo_root / "simulations" / "rl_dbr" / "data",
    ]
    matches: list[Path] = []
    for base in search_bases:
        if not base.is_dir():
            continue
        for p in base.rglob(experiment_name):
            if p.is_dir() and p.name == experiment_name:
                matches.append(p.resolve())

    seen: set[Path] = set()
    unique: list[Path] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise FileNotFoundError(
            f"Multiple experiment directories named {experiment_name!r} found:\n  "
            + "\n  ".join(str(p) for p in unique)
        )
    tried = "\n  ".join(str(b / experiment_name) for b in search_bases)
    raise FileNotFoundError(
        "Experiment directory not found (searched recursively under data/ and "
        "simulations/rl_dbr/data/). "
        f"Expected a folder named {experiment_name!r}. Example direct paths:\n  {tried}"
    )


def tensorboard_root_from_hydra(experiment_dir: Path, repo_root: Path) -> Path | None:
    cfg_path = experiment_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        return None
    cfg = OmegaConf.load(cfg_path)
    raw = OmegaConf.select(cfg, "training.tensorboard_log")
    if raw is None:
        return None
    p = Path(str(raw))
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def _dir_has_event_files(d: Path) -> bool:
    return d.is_dir() and any(d.glob("events.out.tfevents.*"))


def discover_tb_run_dirs(tb_root: Path, experiment_name: str) -> list[Path]:
    runs: list[Path] = []
    exact = tb_root / experiment_name
    if _dir_has_event_files(exact):
        runs.append(exact)
    for p in sorted(tb_root.glob(f"{experiment_name}_*")):
        if _dir_has_event_files(p):
            runs.append(p)
    # Stable order: exact first, then by name
    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in runs:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            ordered.append(rp)
    return ordered


def merge_scalar_events(run_dirs: list[Path]) -> dict[str, list]:
    by_tag: dict[str, list] = defaultdict(list)
    for d in run_dirs:
        ea = EventAccumulator(str(d), size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            by_tag[tag].extend(ea.Scalars(tag))
    for tag, events in by_tag.items():
        events.sort(key=lambda e: (e.step, e.wall_time))
    return dict(by_tag)


def min_wall_time_across_tags(by_tag: dict[str, list]) -> float:
    return min(e.wall_time for events in by_tag.values() for e in events)


def sanitize_tag_for_filename(tag: str) -> str:
    out = []
    for ch in tag:
        if ch in r'\/:*?"<>|':
            out.append("_")
        else:
            out.append(ch)
    s = "".join(out).strip("._")
    return s or "metric"


def export_scalars_to_csv(
    by_tag: dict[str, list], out_dir: Path, wall_time_start: float
) -> tuple[int, list[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for tag, events in sorted(by_tag.items()):
        name = sanitize_tag_for_filename(tag.replace("/", "__"))
        path = out_dir / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "step",
                    "wall_time",
                    "timestamp",
                    "time_spend",
                    "time_step",
                    "value",
                ]
            )
            prev_wall_time = None
            for e in events:
                ts = datetime.fromtimestamp(e.wall_time, tz=timezone.utc).isoformat()
                time_relative_min = (e.wall_time - wall_time_start) / 60.0
                time_step_min = (
                    0.0
                    if prev_wall_time is None
                    else (e.wall_time - prev_wall_time) / 60.0
                )
                w.writerow(
                    [e.step, e.wall_time, ts, time_relative_min, time_step_min, e.value]
                )
                prev_wall_time = e.wall_time
        written.append(path)
    return len(by_tag), written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export TensorBoard scalar logs for a Hydra experiment to CSV files."
    )
    parser.add_argument(
        "experiment_name",
        help="Hydra output folder basename (e.g. 2603282120_training_rldbr)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: search upward for .git, else rl_dbr/../..)",
    )
    parser.add_argument(
        "--tensorboard-root",
        type=Path,
        default=None,
        help="Override TensorBoard log root (default: training.tensorboard_log from .hydra/config.yaml)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root
    if repo_root is None:
        repo_root = find_repo_root(Path.cwd())
    else:
        repo_root = repo_root.resolve()

    try:
        experiment_dir = resolve_experiment_dir(repo_root, args.experiment_name)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.tensorboard_root is not None:
        tb_root = args.tensorboard_root.resolve()
    else:
        tb_root = tensorboard_root_from_hydra(experiment_dir, repo_root)
        if tb_root is None:
            tb_root = (repo_root / "data" / "tensorboard_log").resolve()

    if not tb_root.is_dir():
        print(f"Error: TensorBoard root is not a directory: {tb_root}", file=sys.stderr)
        return 1

    run_dirs = discover_tb_run_dirs(tb_root, args.experiment_name)
    if not run_dirs:
        print(
            f"Error: No TensorBoard run directories under {tb_root} "
            f"matching {args.experiment_name!r} or {args.experiment_name}_*",
            file=sys.stderr,
        )
        return 1

    by_tag = merge_scalar_events(run_dirs)
    if not by_tag:
        print(
            "Error: No scalar tags found in the selected TensorBoard event files.",
            file=sys.stderr,
        )
        return 1

    wall_time_start = min_wall_time_across_tags(by_tag)
    out_dir = experiment_dir / "tensorboard_logs"
    n_tags, paths = export_scalars_to_csv(by_tag, out_dir, wall_time_start)

    print("TensorBoard scalar export")
    print(f"  Experiment:  {experiment_dir}")
    print(f"  TB root:     {tb_root}")
    print(f"  TB run dirs: {len(run_dirs)}")
    for d in run_dirs:
        print(f"    - {d}")
    print(f"  Output:      {out_dir}")
    print(f"  Tags:        {n_tags} CSV file(s) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
