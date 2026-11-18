"""Быстрая диагностика AF-ветки на реальных данных: F1 и разрез день/ночь.

Пример:
    python tools/diag_af.py --data-dir /tmp/firesdata/real/train [--limit 60]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fires import af as afm  # noqa: E402
from fires.chips import Dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default="")
    ap.add_argument("--blob-filter", default="")
    args = ap.parse_args()

    ds = Dataset(args.data_dir)
    params = afm.AFParams()
    blob_filter = None
    if args.blob_filter:
        from fires.blobfilter import BlobFilter

        blob_filter = BlobFilter.load(args.blob_filter)

    ids = ds.chip_ids("af")
    if args.limit:
        ids = ids[: args.limit]

    tp = fp = fn = 0
    day = dict(n=0, tp=0, fp=0, fn=0, chips=0, chips_true=0)
    night = dict(n=0, tp=0, fp=0, fn=0, chips=0, chips_true=0)
    per_chip = []
    for cid in ids:
        chip = ds.load(cid)
        pred = afm.detect_chip(chip, params, blob_filter=blob_filter).astype(bool)
        truth = np.asarray(chip.mask).astype(bool)
        t = int((pred & truth).sum())
        p = int((pred & ~truth).sum())
        f = int((truth & ~pred).sum())
        tp += t
        fp += p
        fn += f
        sun = float(np.nanmedian(chip.layers["sun_zenith"]))
        bucket = night if sun >= params.night_sun_zenith else day
        bucket["n"] += 1
        bucket["chips_true"] += int(truth.any())
        bucket["tp"] += t
        bucket["fp"] += p
        bucket["fn"] += f
        per_chip.append(dict(chip=cid, pred=int(pred.sum()), tp=t, fp=p, fn=f))

    def f1(stats: dict) -> float:
        denom = 2 * stats["tp"] + stats["fp"] + stats["fn"]
        return (2 * stats["tp"] / denom) if denom else 1.0

    total = dict(n=len(ids), tp=tp, fp=fp, fn=fn)
    out = {
        "n_chips": len(ids),
        "f1_af": round(f1(total), 5),
        "counts": total,
        "day": dict(day, f1=round(f1(day), 5)),
        "night": dict(night, f1=round(f1(night), 5)),
        "worst": sorted(per_chip, key=lambda r: -r["fp"])[:10],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps({**out, "per_chip": per_chip}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
