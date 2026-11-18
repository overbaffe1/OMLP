#!/usr/bin/env python3
"""Зонд маски SCL: сколько истины гари запрещает предсказывать слой SCL.

Мотивация: чипы BS_tr_000111/000024/000069 дают IoU=0.000 при отличном сигнале
dNBR — 97–100% чипа помечены SCL_BAD, хотя cloud_frac в meta 0–6% (облаков нет).

Показывает по каждому BS-чипу с эталоном: гистограмму классов SCL (pre/post),
долю истины под маской, медианный dNBR в замаскированной гари (если сигнал хороший —
пиксели восстановимы ослаблением маски). Обучения и модели нет — только маски и dNBR.

Запуск::

    py -3 tools\\diag_scl.py --data-dir <путь>\\train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fires import bs as bsm  # noqa: E402
from fires.chips import Dataset, find_data_dir  # noqa: E402

SCL_NAMES = {0: "nodata", 1: "satur", 2: "dark", 3: "shadow", 4: "veg", 5: "bare",
             6: "water", 7: "unclass", 8: "cloud", 9: "cirrus+", 10: "cirrus", 11: "snow"}


def _scl_layers(chip):
    layers = chip.layers
    scl = layers.get("SCL")
    scl_pre = layers.get("SCL_pre")
    if scl is None and chip.post and "SCL" in chip.post:
        scl = chip.post["SCL"]
    if scl_pre is None and chip.pre and "SCL" in chip.pre:
        scl_pre = chip.pre["SCL"]
    return (np.asarray(scl) if scl is not None else None,
            np.asarray(scl_pre) if scl_pre is not None else None)


def _hist_top(scl: np.ndarray | None) -> str:
    if scl is None:
        return "нет слоя"
    vals, counts = np.unique(scl, return_counts=True)
    order = np.argsort(-counts)[:4]
    parts = []
    for i in order:
        v = int(vals[i])
        name = SCL_NAMES.get(v, "?")
        bad = "*" if (v in bsm.SCL_BAD or v == bsm.SCL_WATER) else " "
        parts.append(f"{v}({name}){bad}:{counts[i] / scl.size:.0%}")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Зонд маски SCL на BS-чипах")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    dataset = Dataset(find_data_dir(args.data_dir, prefer="masks"))
    ids = dataset.chip_ids("bs")
    if args.limit:
        ids = ids[: args.limit]

    rows = []
    masked_burn_px = 0
    total_burn_px = 0
    masked_dnbr: list[np.ndarray] = []
    # разделимость под маской по группам: гарь vs фон (dNBR) — что можно размаскировать
    sep: dict[str, dict[str, list[np.ndarray]]] = {
        g: {"burn": [], "bg": []} for g in ("post 8/9", "post 10", "post other-bad", "pre-only")}
    for cid in ids:
        try:
            chip = dataset.load(cid)
        except Exception as exc:
            print(f"    [warn] {cid}: {exc}", file=sys.stderr)
            continue
        if chip.mask_sev is None:
            continue
        truth = np.asarray(chip.mask_sev, dtype=np.uint8)
        burn = truth > 0
        n_burn = int(burn.sum())
        total_burn_px += n_burn
        scl_post, scl_pre = _scl_layers(chip)

        def bad_frac(scl):
            if scl is None or scl.shape != truth.shape:
                return float("nan")
            return float((np.isin(scl, bsm.SCL_BAD) | (scl == bsm.SCL_WATER)).mean())

        def burn_under(scl):
            if scl is None or scl.shape != truth.shape or n_burn == 0:
                return float("nan")
            bad = np.isin(scl, bsm.SCL_BAD) | (scl == bsm.SCL_WATER)
            return float((bad & burn).sum() / n_burn)

        bp, bpr = bad_frac(scl_post), bad_frac(scl_pre)
        masked = np.zeros(truth.shape, bool)
        for scl in (scl_post, scl_pre):
            if scl is not None and scl.shape == truth.shape:
                masked |= np.isin(scl, bsm.SCL_BAD) | (scl == bsm.SCL_WATER)
        under = float((masked & burn).sum() / max(1, n_burn))
        masked_burn_px += int((masked & burn).sum())
        med_burn = float("nan")
        try:
            scale = bsm.scale_reflectance
            dnbr = (bsm.nbr(scale(chip.pre["B8A"]), scale(chip.pre["B12"]))
                    - bsm.nbr(scale(chip.post["B8A"]), scale(chip.post["B12"])))
            finite = np.isfinite(dnbr)
            sel = masked & burn & finite
            if sel.any():
                masked_dnbr.append(np.asarray(dnbr[sel], dtype=np.float32))
                med_burn = float(np.median(dnbr[sel]))
            # группа пикселя — по причине маскировки (пост-дата первична для картирования)
            post_bad = (np.isin(scl_post, bsm.SCL_BAD) | (scl_post == bsm.SCL_WATER)
                        if scl_post is not None and scl_post.shape == truth.shape
                        else np.zeros(truth.shape, bool))
            grp = np.full(truth.shape, "", dtype=object)
            grp[post_bad & np.isin(scl_post, (8, 9))] = "post 8/9"
            grp[post_bad & (scl_post == 10)] = "post 10"
            grp[post_bad & (grp == "")] = "post other-bad"
            grp[(grp == "") & masked] = "pre-only"
            for gname in sep:
                m = (grp == gname) & finite
                if m.any():
                    sep[gname]["burn"].append(np.asarray(dnbr[m & burn], dtype=np.float32))
                    sep[gname]["bg"].append(np.asarray(dnbr[m & (truth == 0)], dtype=np.float32))
        except (KeyError, TypeError):
            pass
        rows.append({"chip_id": cid, "burn_px": n_burn,
                     "bad_post": bp, "bad_pre": bpr, "under": under,
                     "med_dnbr_burn": med_burn,
                     "hist_post": _hist_top(scl_post), "hist_pre": _hist_top(scl_pre),
                     "cloud": dataset.meta.get(cid, {}).get("cloud_frac")})

    print(f"чипов с эталоном: {len(rows)}")
    print(f"истины гари под маской SCL (непредсказуемо сейчас): "
          f"{masked_burn_px / max(1, total_burn_px):.1%} ({masked_burn_px} px)")
    if masked_dnbr:
        all_masked = np.concatenate(masked_dnbr)
        qs = np.percentile(all_masked, (5, 25, 50, 75, 95))
        print(f"dNBR в замаскированной гари (5/25/50/75/95): "
              f"{' / '.join(f'{v:.3f}' for v in qs)}  (n={all_masked.size})")
        print("    медиана > 0.1 = сигнал хороший, пиксели восстановимы ослаблением маски")
    print("\nчипы, где под маской больше половины истины (кандидаты на восстановление):")
    print(f"  {'chip':<16}{'burn_px':>9}{'%под маской':>11}{'med_dNBR':>10}{'cloud':>8}")
    bad_chips = [r for r in rows if np.isfinite(r["under"]) and r["under"] > 0.5]
    bad_chips.sort(key=lambda r: -r["under"])
    for r in bad_chips:
        print(f"  {r['chip_id']:<16}{r['burn_px']:9d}{r['under']:11.0%}"
              f"{r['med_dnbr_burn']:10.3f}{str(r['cloud']):>8}")
        print(f"    post: {r['hist_post']}")
        print(f"    pre:  {r['hist_pre']}")
    if not bad_chips:
        print("  (таких нет)")

    print("\nразделимость под маской: dNBR гари vs фона по причине маскировки")
    print("(медиана гари заметно выше фона = группу можно размаскировать)")
    print(f"  {'группа':<16}{'burn_px':>10}{'bg_px':>10}  "
          f"{'burn 5/25/50/75/95':<34}{'bg 5/25/50/75/95'}")
    for gname in ("post 8/9", "post 10", "post other-bad", "pre-only"):
        bb = np.concatenate(sep[gname]["burn"]) if sep[gname]["burn"] else np.array([])
        gg = np.concatenate(sep[gname]["bg"]) if sep[gname]["bg"] else np.array([])
        if bb.size == 0:
            print(f"  {gname:<16}{'—':>10}{'—':>10}  нет замаскированной гари")
            continue
        qb = " / ".join(f"{v:.3f}" for v in np.percentile(bb, (5, 25, 50, 75, 95)))
        qg = " / ".join(f"{v:.3f}" for v in np.percentile(gg, (5, 25, 50, 75, 95))) if gg.size else "—"
        print(f"  {gname:<16}{bb.size:10d}{gg.size:10d}  {qb:<34}{qg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
