#!/usr/env python3
"""Исследование BS-чипов ТЕСТА: запас под блокировкой SCL (как с перистой на трейне).

Эталона на тесте нет, поэтому показываем только то, что считается без него:
- покрытие классами-капельками (8/9/10 и прочими плохими) в pre/post;
- расхождение cloud_frac из meta с фактическим покрытием (индикатор глюка слоя);
- dNBR-квантили заблокированных пикселей по группам против открытого фона чипа;
- оценка резерва: сколько пикселей вошло бы, если размаскировать группу
  (dNBR > 0.15 — явная гарь, > 0.3 — ядро).

Запуск::

    py -3 tools\\probe_test_scl.py --data-dir <путь>\\test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fires import bs as bsm  # noqa: E402
from fires.chips import Dataset, find_data_dir  # noqa: E402


def scl_layers(chip):
    scl = chip.layers.get("SCL") if chip.layers else None
    scl_pre = chip.layers.get("SCL_pre") if chip.layers else None
    if scl is None and chip.post and "SCL" in chip.post:
        scl = chip.post["SCL"]
    if scl_pre is None and chip.pre and "SCL" in chip.pre:
        scl_pre = chip.pre["SCL"]
    return (np.asarray(scl) if scl is not None else None,
            np.asarray(scl_pre) if scl is not None else None)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def dnbr_of(chip):
    scale = bsm.scale_reflectance
    return (bsm.nbr(scale(chip.pre["B8A"]), scale(chip.pre["B12"]))
            - bsm.nbr(scale(chip.post["B8A"]), scale(chip.post["B12"])))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Запас под блокировкой SCL на тесте")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    dataset = Dataset(find_data_dir(args.data_dir, prefer="masks"))
    ids = dataset.chip_ids("bs")
    if args.limit:
        ids = ids[: args.limit]
    print(f"BS-чипов теста: {len(ids)}")

    stats = {g: {"px": 0, "hot": 0, "core": 0, "dnbr": []}
             for g in ("post 8/9", "post 10", "post other-bad", "pre-only")}
    bg_dnbr: list[np.ndarray] = []
    rows = []
    for cid in ids:
        try:
            chip = dataset.load(cid)
        except Exception as exc:
            print(f"    [warn] {cid}: {exc}", file=sys.stderr)
            continue
        shape = chip.post["B8A"].shape
        scl_post, scl_pre = scl_layers(chip)
        post_bad = np.isin(scl_post, bsm.SCL_BAD) | (scl_post == bsm.SCL_WATER)
        pre_bad = np.isin(scl_pre, bsm.SCL_BAD) | (scl_pre == bsm.SCL_WATER)
        masked = post_bad | pre_bad
        try:
            dnbr = np.asarray(dnbr_of(chip), dtype=np.float32)
            finite = np.isfinite(dnbr)
        except (KeyError, TypeError):
            continue
        bg = (~masked) & finite
        if bg.any():
            bg_dnbr.append(np.asarray(dnbr[bg], dtype=np.float32))
        grp = np.full(shape, "", dtype=object)
        grp[post_bad & np.isin(scl_post, (8, 9))] = "post 8/9"
        grp[post_bad & (scl_post == 10)] = "post 10"
        grp[post_bad & (grp == "")] = "post other-bad"
        grp[(grp == "") & masked] = "pre-only"
        for gname in stats:
            m = (grp == gname) & finite
            if not m.any():
                continue
            px = int(m.sum())
            hot = int((m & (dnbr > 0.15)).sum())
            core = int((m & (dnbr > 0.30)).sum())
            stats[gname]["px"] += px
            stats[gname]["hot"] += hot
            stats[gname]["core"] += core
            if hot:
                stats[gname]["dnbr"].append(np.asarray(dnbr[m & (dnbr > 0.15)], dtype=np.float32))
        frac = float(masked.mean())
        cloud = dataset.meta.get(cid, {}).get("cloud_frac")
        rows.append({"cid": cid, "masked": frac, "cloud": cloud,
                     "hist": " ".join(
                         f"{v}({int(c / shape[0] / shape[1] * 100)}%)" for v, c in
                         zip(*np.unique(scl_post[scl_post == scl_post], return_counts=True))
                     ) if scl_post is not None else "нет слоя"})

    total = sum(r["masked"] for r in rows) / max(1, len(rows))
    print(f"среднее покрытие плохими классами (pre+post, на чип): {total:.1%}")
    heavy = [r for r in rows if r["masked"] > 0.3]
    print(f"чипов с покрытием > 30%: {len(heavy)} из {len(rows)}")
    mismatch = [r for r in rows
                if _f(r["cloud"]) < 0.1 and r["masked"] > 0.3]
    print(f"индикатор глюка слоя (cloud_frac < 10%, покрытие > 30%): {len(mismatch)} чипов")

    allbg = np.concatenate(bg_dnbr) if bg_dnbr else np.array([])
    print("\nоткрытый фон (не заблокирован):",
          " / ".join(f"{v:.3f}" for v in np.percentile(allbg, (50, 75, 95, 99)))
          if allbg.size else "—", f"(n={allbg.size})")
    print(f"\n{'группа':<16}{'px':>10}{'dNBR>0.15':>11}{'dNBR>0.3':>10}  медиана горячего хвоста")
    for gname in stats:
        s = stats[gname]
        tail = np.concatenate(s["dnbr"]) if s["dnbr"] else np.array([])
        med = f"{np.median(tail):.3f}" if tail.size else "—"
        print(f"{gname:<16}{s['px']:10d}{s['hot']:11d}{s['core']:10d}  {med}")
    hot_total = sum(s["hot"] for s in stats.values())
    print(f"\nитог: горячий хвост (dNBR>0.15) под блокировкой: {hot_total} px — "
          f"это верхняя оценка резерва (там может быть и настоящая облачность)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
