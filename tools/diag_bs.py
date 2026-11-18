"""Диагностика BS-ветки на реальных данных: распределения dNBR по классам и покрову.

Запуск: ``python tools/diag_bs.py --data-dir /tmp/firesdata/real/train --limit 60``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fires import bs as bsm  # noqa: E402
from fires.chips import Dataset  # noqa: E402

QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--stride", type=int, default=1, help="шаг по чипам (для скорости)")
    args = ap.parse_args()

    ds = Dataset(args.data_dir)
    ids = ds.chip_ids("bs")[:: max(1, args.stride)]
    if args.limit:
        ids = ids[: args.limit]

    codes = bsm.group_codes()
    buckets: dict[str, list[np.ndarray]] = {}
    conf = np.zeros((4, 4), dtype=np.int64)
    for cid in ids:
        chip = ds.load(cid)
        layers = chip.layers
        pre, post = chip.pre, chip.post
        nbr_pre = bsm.nbr(bsm.scale_reflectance(pre["B8A"]), bsm.scale_reflectance(pre["B12"]))
        nbr_post = bsm.nbr(bsm.scale_reflectance(post["B8A"]), bsm.scale_reflectance(post["B12"]))
        dnbr = nbr_pre - nbr_post
        scl = layers.get("SCL")
        ok = np.isfinite(dnbr)
        if scl is not None:
            ok &= ~np.isin(np.asarray(scl), bsm.SCL_BAD) & (np.asarray(scl) != bsm.SCL_WATER)
        ok &= (nbr_pre != 0.0) | (nbr_post != 0.0)
        truth = np.asarray(chip.mask_sev).astype(np.uint8)
        pred = bsm.classify_chip(chip, bsm.BSParams())
        for t in range(4):
            for p in range(4):
                conf[t, p] += int(((truth == t) & (pred == p)).sum())
        lc = np.asarray(layers["landcover"])
        for code, group in codes.items():
            sel = ok & (lc == code) & (truth > 0)
            if sel.any():
                buckets.setdefault(group, []).append(np.asarray(dnbr[sel], dtype=np.float32))
        for cls in (1, 2, 3):
            sel = ok & (truth == cls)
            if sel.any():
                buckets.setdefault(f"класс {cls}", []).append(np.asarray(dnbr[sel], dtype=np.float32))

    print("Строки — истина, столбцы — предсказание (0..3):")
    print(conf)
    print("IoU_burn:", round(conf[1:, 1:].sum() / max(1, conf[:, 1:].sum() + conf[1:, :].sum() - conf[1:, 1:].sum()), 4))
    print("\nРаспределение dNBR (квантили, %) по группам покрова и истинным классам:")
    header = "группа" + "".join(f"{q:>8d}" for q in QUANTILES) + f"{'n':>12}"
    print(header)
    for name in sorted(buckets):
        values = np.concatenate(buckets[name])
        qs = np.percentile(values, QUANTILES)
        print(f"{name:<14}" + "".join(f"{v:8.3f}" for v in qs) + f"{values.size:12d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
