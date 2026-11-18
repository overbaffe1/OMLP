"""Плотность добавок релаксаций AF-правил на трейне.

Измеряемые множества (каждое — «что добавится, если правило ослабить»):
  S1 marg>=thr    — пиксели пашни/голой почвы, отрезанные правкой marginal,
                    но с delta уже порога дня (близко к огню);
  S2 marg>=thr+1  — более узкое окно той же правки;
  S3 edge         — продление кромки: пустой пиксель рядом с огнём,
                    i4 >= candidate - 8 и delta >= threshold - 5;
  S4 edge-bright  — то же, но i4 >= absolute (совсем яркий).

Для каждого множества: добавится пикселей, из них истина (density).
Запуск::

    py -3 tools\\diag_af_rescue.py --data-dir <путь>\\train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fires import af as afm  # noqa: E402
from fires.chips import Dataset, find_data_dir  # noqa: E402
from fires.weights import load_weights  # noqa: E402


def morph_box(mask):
    m = mask.astype(np.float32)
    total = np.zeros_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            total += np.roll(np.roll(m, dy, 0), dx, 1)
    return total


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args(argv)

    dataset = Dataset(find_data_dir(args.data_dir, prefer="masks"))
    weights = load_weights()
    p = afm.to_params(weights.get("af", {}))

    def add_sets(chip):
        layers = chip.layers
        i4 = np.asarray(layers["I4"], dtype=np.float32)
        i5 = np.asarray(layers["I5"], dtype=np.float32)
        i1 = np.asarray(layers["I1"], dtype=np.float32)
        i3 = np.asarray(layers["I3"], dtype=np.float32)
        landcover = np.asarray(layers["landcover"])
        valid = np.asarray(layers.get("valid", np.ones(i4.shape, dtype=bool))).astype(bool)
        sun = layers.get("sun_zenith")

        night = sun is not None and float(np.nanmedian(np.asarray(sun, dtype=np.float32))) >= p.night_sun_zenith
        sat_thr = p.sat_t4_night if night else p.sat_t4
        abs_thr = p.absolute_t4_night if night else p.absolute_t4
        cand_thr = p.candidate_t4_night if night else p.candidate_t4
        delta_thr = p.delta_night if night else p.delta_day

        valid2 = valid & np.isfinite(i4) & np.isfinite(i5)
        screen = valid2 & ((landcover != afm.LC_WATER) if p.exclude_water else True)
        saturated = i4 >= sat_thr
        absolute = i4 >= abs_thr
        candidate = screen & ((i4 >= cand_thr) | absolute | saturated)
        mean_bg, std_bg, _counts = afm._background_stats(i4, candidate, screen, p.bg_window)
        delta = i4 - i5
        contextual = candidate & (i4 - mean_bg > p.k_sigma * std_bg.clip(0.5, None)) & (delta > delta_thr)

        fire_pre = ((absolute & screen) | (saturated & screen & p.sat_always_fire) | contextual) & screen

        # marginal-правка из detect(): отрезает thermal-пиксели пашни/голой почвы
        marginal = (landcover == afm.LC_CROPLAND) | (landcover == afm.LC_BARE)
        pruned = fire_pre & marginal & (delta < delta_thr + 2.0) & (i4 < abs_thr)
        fire_post = fire_pre & ~pruned

        near = (morph_box(fire_post) > 0) & ~fire_post & screen
        return fire_post, {
            "S1 marg>=thr": pruned & (delta >= delta_thr),
            "S2 marg>=thr+1": pruned & (delta >= delta_thr + 1.0),
            "S3 edge": near & (i4 >= cand_thr - 8.0) & (delta >= delta_thr - 5.0),
            "S4 edge-bright": near & (i4 >= abs_thr),
        }

    stats = {}
    base_n = base_t = 0
    for cid in dataset.chip_ids("af"):
        chip = dataset.load(cid)
        if chip.mask is None:
            continue
        mask = chip.mask.astype(bool)
        try:
            fire, adds = add_sets(chip)
        except (KeyError, ValueError):
            continue
        base_n += int(fire.sum())
        base_t += int((fire & mask).sum())
        for name, m in adds.items():
            s = stats.setdefault(name, {"n": 0, "t": 0})
            s["n"] += int(m.sum())
            s["t"] += int((m & mask).sum())

    print(f"база: пикселей {base_n}, истина {base_t}, density {base_t / max(1, base_n):.3f}")
    for name, s in sorted(stats.items()):
        d = s["t"] / max(1, s["n"])
        print(f"{name:<16} добавится {s['n']:6d}  истина {s['t']:6d}  density {d:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
