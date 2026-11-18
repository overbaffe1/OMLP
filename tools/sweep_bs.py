#!/usr/bin/env python3
"""Быстрый подбор порогов dNBR по типам покрова на реальных чипах.

    python tools/sweep_bs.py --data-dir /tmp/firesdata/real/train --max-chips 96

Пиксельные данные кэшируются один раз (dNBR, маска валидности, группа покрова, эталон), после
чего перебор порогов идёт мгновенно — это позволяет искать границы классов по всей сетке, а не
по трём заранее выбранным тройкам. Постобработка (заполнение дыр, чистка мелких контуров) в этом
инструменте не применяется: он ищет рабочую область порогов, окончательные значения проверяются
полным `train.py` и `evaluate.py`.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fires import bs as bs_module  # noqa: E402
from fires.chips import Dataset  # noqa: E402


def cache(dataset: Dataset, max_chips: int, use_sar: bool) -> list[dict]:
    """Считает dNBR и вспомогательные маски для выборки чипов."""

    ids = dataset.chip_ids("bs")[:max_chips]
    out: list[dict] = []
    for n, chip_id in enumerate(ids, 1):
        chip = dataset.load(chip_id)
        pre, post = chip.pre, chip.post
        if "B8A" not in pre or "B8A" not in post:
            continue
        nbr_pre = bs_module.nbr(bs_module.scale_reflectance(pre["B8A"]), bs_module.scale_reflectance(pre["B12"]))
        nbr_post = bs_module.nbr(bs_module.scale_reflectance(post["B8A"]), bs_module.scale_reflectance(post["B12"]))
        dnbr = (nbr_pre - nbr_post).astype(np.float32)

        ok = np.ones(dnbr.shape, dtype=bool)
        layers = chip.layers
        landcover = layers.get("landcover")
        if landcover is None:
            landcover = np.full(dnbr.shape, 30, dtype=np.uint8)
        landcover = np.asarray(landcover)
        ok &= landcover != bs_module.LC_WATER
        for key in ("SCL", "SCL_pre"):
            scl = layers.get(key)
            if scl is not None:
                ok &= ~np.isin(np.asarray(scl), bs_module.SCL_BAD)
                ok &= np.asarray(scl) != bs_module.SCL_WATER
        ok &= (nbr_pre != 0.0) | (nbr_post != 0.0)

        codes = bs_module.group_codes()
        group = np.zeros(dnbr.shape, dtype=np.int8)  # 0 — группа не определена
        names = sorted({v for v in codes.values()})
        for index, name in enumerate(names):
            group |= np.isin(landcover, [c for c, n in codes.items() if n == name]) * np.int8(index + 1)

        truth = np.asarray(chip.mask_sev if chip.mask_sev is not None else chip.mask, dtype=np.uint8)
        out.append({"dnbr": dnbr, "ok": ok, "group": group, "truth": truth, "names": names})
        if n % 24 == 0:
            print(f"  обработано {n} чипов")
    return out


def report_quantiles(cached: list[dict]) -> None:
    """Печатает квантили dNBR по (группа покрова × истинный класс) — обоснование порогов."""

    print("\n=== распределение dNBR: истинный класс × группа покрова (медиана [p10 p90], тыс. px) ===")
    for name in cached[0]["names"]:
        for cls in (1, 2, 3):
            values = []
            for item in cached:
                sel = (item["ok"] & (item["group"] == item["names"].index(name) + 1) & (item["truth"] == cls)
                       & np.isfinite(item["dnbr"]))
                if sel.any():
                    values.append(item["dnbr"][sel])
            if not values:
                continue
            allv = np.concatenate(values)
            print(
                f"  {name:9s} класс {cls}: медиана {np.median(allv):.3f} "
                f"[{np.percentile(allv, 10):.3f} {np.percentile(allv, 90):.3f}], {allv.size / 1000:.1f} тыс. px"
            )
    # Сколько ложных пикселей даёт каждая группа при текущих порогах
    print("\n=== ложные срабатывания по группам (dNBR выше порога класса 1, эталон = 0) ===")
    for name in cached[0]["names"]:
        index = cached[0]["names"].index(name) + 1
        for t1 in (0.10, 0.15, 0.20, 0.25):
            fp = sum(int(((item["group"] == index) & item["ok"] & (item["truth"] == 0) & (item["dnbr"] >= t1)).sum())
                     for item in cached)
            tp = sum(int(((item["group"] == index) & item["ok"] & (item["truth"] > 0) & (item["dnbr"] >= t1)).sum())
                     for item in cached)
            print(f"  {name:9s} t1={t1:.2f}: fp {fp / 1000:8.1f} тыс.  tp {tp / 1000:8.1f} тыс.")


def score_for(cached: list[dict], thresholds: dict[str, tuple[float, float, float]]) -> dict:
    """Считает IoU_burn и mIoU_sev по пулу пикселей для заданных порогов по группам."""

    tp_b = fp_b = fn_b = 0
    cls_tp = {1: 0, 2: 0, 3: 0}
    cls_fp = {1: 0, 2: 0, 3: 0}
    cls_fn = {1: 0, 2: 0, 3: 0}
    for item in cached:
        index_name = {i + 1: name for i, name in enumerate(item["names"])}
        pred = np.zeros(item["truth"].shape, dtype=np.uint8)
        for idx, name in index_name.items():
            if name == "other" and thresholds["other"][0] >= 1.0:
                continue
            t1, t2, t3 = thresholds[name]
            d = np.where((item["group"] == idx) & item["ok"], item["dnbr"], -np.inf)
            pred = np.where((d >= t1) & (d < t2), 1, pred).astype(np.uint8)
            pred = np.where((d >= t2) & (d < t3), 2, pred).astype(np.uint8)
            pred = np.where(d >= t3, 3, pred).astype(np.uint8)
        truth = item["truth"]
        tp_b += int(((pred > 0) & (truth > 0)).sum())
        fp_b += int(((pred > 0) & (truth == 0)).sum())
        fn_b += int(((pred == 0) & (truth > 0)).sum())
        for cls in (1, 2, 3):
            cls_tp[cls] += int(((pred == cls) & (truth == cls)).sum())
            cls_fp[cls] += int(((pred == cls) & (truth != cls)).sum())
            cls_fn[cls] += int(((pred != cls) & (truth == cls)).sum())

    def iou(tp: int, fp: int, fn: int) -> float:
        den = tp + fp + fn
        return 1.0 if den == 0 else tp / den

    iou_burn = iou(tp_b, fp_b, fn_b)
    per_class = {cls: iou(cls_tp[cls], cls_fp[cls], cls_fn[cls]) for cls in (1, 2, 3)}
    miou = float(np.mean(list(per_class.values())))
    return {
        "iou_burn": round(float(iou_burn), 5),
        "miou_sev": round(float(miou), 5),
        "objective": round(0.35 * iou_burn + 0.30 * miou, 5),
        "per_class": {k: round(v, 4) for k, v in per_class.items()},
        "counts": {"tp": tp_b, "fp": fp_b, "fn": fn_b},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Подбор порогов dNBR по типам покрова")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-chips", type=int, default=96)
    ap.add_argument("--search", action="store_true", help="перебрать сетку порогов (иначе только отчёт)")
    args = ap.parse_args(argv)

    dataset = Dataset(args.data_dir)
    print(f"чипов BS: {len(dataset.chip_ids('bs'))}, берём {args.max_chips}")
    cached = cache(dataset, args.max_chips, use_sar=True)
    print(f"закэшировано чипов: {len(cached)}")
    report_quantiles(cached)

    current = {
        "forest": bs_module.BSParams().forest,
        "semi": bs_module.BSParams().semi,
        "cropland": bs_module.BSParams().cropland,
        "other": bs_module.BSParams().other,
    }
    print("\n=== текущие пороги ===")
    print(" ", current, score_for(cached, current))

    if not args.search:
        return 0

    # Порог класса 1 ищем по всей сетке, границы классов 2/3 — вокруг измеренных медиан
    grid_t1 = [0.06, 0.08, 0.10, 0.12, 0.15, 0.18]
    grid_t2 = [0.15, 0.19, 0.22, 0.26, 0.30]
    grid_t3 = [0.32, 0.37, 0.42, 0.48, 0.55]
    results = []
    for t1, t2, t3 in itertools.product(grid_t1, grid_t2, grid_t3):
        if not (t1 < t2 < t3):
            continue
        triple = (t1, t2, t3)
        metrics = score_for(cached, {"forest": triple, "semi": triple, "cropland": triple, "other": (1.0, 1.0, 1.0)})
        results.append((metrics["objective"], triple, metrics))
    results.sort(key=lambda row: -row[0])
    print("\n=== единая тройка для всех групп (топ-10) ===")
    for objective, triple, metrics in results[:10]:
        print(f"  t1/t2/t3={triple} objective={objective:.4f} IoU={metrics['iou_burn']:.4f} "
              f"mIoU={metrics['miou_sev']:.4f} fp={metrics['counts']['fp'] / 1000:.1f} тыс.")

    # Дообучение по группам: берём лучшую единую тройку и сдвигаем границы по каждой группе
    best_triple = results[0][1]
    print(f"\n=== покоординатный подъём от {best_triple} ===")
    thresholds = {"forest": best_triple, "semi": best_triple, "cropland": best_triple, "other": (1.0, 1.0, 1.0)}
    best_score = score_for(cached, thresholds)["objective"]
    for _ in range(3):
        improved = False
        for group in ("forest", "semi", "cropland"):
            base = thresholds[group]
            for pos, values in enumerate((grid_t1, grid_t2, grid_t3)):
                for value in values:
                    candidate = list(base)
                    candidate[pos] = value
                    if not (candidate[0] < candidate[1] < candidate[2]):
                        continue
                    trial = dict(thresholds)
                    trial[group] = tuple(candidate)
                    metrics = score_for(cached, trial)
                    if metrics["objective"] > best_score + 1e-6:
                        best_score = metrics["objective"]
                        thresholds = trial
                        improved = True
                        print(f"  {group} -> {tuple(candidate)}: objective={best_score:.4f} "
                              f"IoU={metrics['iou_burn']:.4f} mIoU={metrics['miou_sev']:.4f}")
        if not improved:
            break
    print("\n=== итог поиска ===")
    print(" ", {k: tuple(v) for k, v in thresholds.items()})
    print(" ", score_for(cached, thresholds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
