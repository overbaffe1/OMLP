"""Перебор порогов AF на реальных данных.

Ключи ``--grid имя=v1,v2`` перебираются, ``--fix имя=значение`` фиксируются.
Примеры:
    python tools/sweep_af.py --data-dir /tmp/firesdata/real/train \
        --fix delta_day=25 --fix candidate_t4=330 --grid delta_night=10,15,20 --grid k_sigma=2,3
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fires import af as afm  # noqa: E402
from fires.chips import Dataset  # noqa: E402


def f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 1.0


def parse_kv(items: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    valid = {f.name for f in fields(afm.AFParams)}
    for item in items:
        name, _, values = item.partition("=")
        if name not in valid:
            raise SystemExit(f"неизвестный параметр: {name}")
        out[name] = [float(v) for v in values.split(",")] if values else [1.0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--grid", action="append", default=[], help="имя=v1,v2[,v3]")
    ap.add_argument("--fix", action="append", default=[], help="имя=значение")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    grid = parse_kv(args.grid)
    fixed = {k: v[0] for k, v in parse_kv(args.fix).items()}
    if not grid:
        raise SystemExit("нужен хотя бы один --grid")

    ds = Dataset(args.data_dir)
    ids = ds.chip_ids("af")
    if args.limit:
        ids = ids[: args.limit]
    chips = []
    for cid in ids:
        chip = ds.load(cid)
        chips.append((chip, bool(float(np.nanmedian(chip.layers["sun_zenith"])) >= afm.AFParams().night_sun_zenith)))

    names = list(grid)
    print(" ".join(f"{n:>18s}" for n in names) + " |  day F1 (tp/fp/fn)          night F1 (tp/fp/fn)         all F1")
    rows = []
    for combo in itertools.product(*(grid[n] for n in names)):
        params = replace(afm.AFParams(), **fixed, **dict(zip(names, combo)))
        day = dict(tp=0, fp=0, fn=0)
        night = dict(tp=0, fp=0, fn=0)
        for chip, is_night in chips:
            pred = afm.detect_chip(chip, params).astype(bool)
            truth = np.asarray(chip.mask).astype(bool)
            b = night if is_night else day
            b["tp"] += int((pred & truth).sum())
            b["fp"] += int((pred & ~truth).sum())
            b["fn"] += int((truth & ~pred).sum())
        fd, fng = f1(**day), f1(**night)
        allc = {k: day[k] + night[k] for k in day}
        fa = f1(**allc)
        rows.append((fa, combo, dict(day), dict(night)))
        print(" ".join(f"{c:18.1f}" for c in combo)
              + f" | {fd:.4f} ({day['tp']}/{day['fp']}/{day['fn']})".ljust(30)
              + f" | {fng:.4f} ({night['tp']}/{night['fp']}/{night['fn']})".ljust(32) + f" | {fa:.4f}")
    rows.sort(key=lambda r: -r[0])
    best = rows[0]
    print("\nЛучшее:", dict(zip(names, best[1])), "| all F1", round(best[0], 5),
          "| day", best[2], "| night", best[3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
