#!/usr/bin/env python3
"""Диагностика ошибок BS без обучения: где решение теряет Score.

Отвечает на вопросы, которые нельзя решить перебором вслепую:

* какие переходы путаются (confusion 0↔1, 1↔2);
* где ложные срабатывания и пропуски по группам покрова;
* сколько эталонной гари каждого класса режет ROI-гейт модели (dNBR <= cut);
* как качество зависит от облачности чипа;
* какие чипы самые проблемные (кандидаты на ручной разбор).

Запуск (с текущей конфигурацией из весов)::

    py -3 tools\\diag_errors.py --data-dir <путь>\\train
    py -3 tools\\diag_errors.py --data-dir <путь>\\train --thresholds-only --limit 60

Предсказания считаются той же конфигурацией, что инференс (веса + модель степени,
если она загрузилась). Обучения нет — только проход по чипам с масками.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import load_severity_model  # noqa: E402
from fires import bs as bsm  # noqa: E402
from fires.chips import Dataset, find_data_dir  # noqa: E402
from fires.weights import load_weights  # noqa: E402

CLOUD_BUCKETS = ((0.0, 0.05, "<5%"), (0.05, 0.25, "5–25%"), (0.25, 2.0, ">25%"))
QUANTILES = (5, 25, 50, 75, 95)


def _cloud_frac(dataset: Dataset, chip_id: str) -> float:
    raw = dataset.meta.get(chip_id, {}).get("cloud_frac")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    return value


def _dnbr(chip) -> np.ndarray | None:
    try:
        pre, post = chip.pre, chip.post
        scale = bsm.scale_reflectance
        return (bsm.nbr(scale(pre["B8A"]), scale(pre["B12"]))
                - bsm.nbr(scale(post["B8A"]), scale(post["B12"]))).astype(np.float32)
    except (KeyError, TypeError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Диагностика ошибок BS без обучения")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--weights", default=None, help="путь к weights.json")
    ap.add_argument("--thresholds-only", action="store_true",
                    help="диагностировать пороговый путь без модели (быстро)")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    ap.add_argument("--limit", type=int, default=0, help="первые N чипов (0 — все)")
    ap.add_argument("--stride", type=int, default=1, help="шаг по чипам")
    ap.add_argument("--cuts", default="0.0,0.03,0.05,0.08,0.10,0.15",
                    help="пороги dNBR для таблицы потерь ROI-гейта")
    ap.add_argument("--json-out", default=None, help="дополнительно записать сводку в JSON")
    args = ap.parse_args(argv)

    cuts = sorted({float(v) for v in args.cuts.split(",") if v.strip() != ""})
    resolved = find_data_dir(args.data_dir, prefer="masks")
    dataset = Dataset(resolved)
    ids = dataset.chip_ids("bs")[:: max(1, args.stride)]
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        print(f"в каталоге {args.data_dir} нет BS-чипов")
        return 1

    weights = load_weights(args.weights)
    bs_params = bsm.to_params(weights.get("bs", {}))
    sev_model = None if args.thresholds_only else load_severity_model(weights, args.weights, args.device)
    mode = "пороги" if sev_model is None else (
        f"модель ({getattr(sev_model, 'backend', '?')}/{getattr(sev_model, 'device', '?')})")
    print(f"=== Диагностика BS: {mode}, чипов: {len(ids)} ===")

    codes = bsm.group_codes()
    conf = np.zeros((4, 4), dtype=np.int64)
    # по группам покрова: пиксели истины, TP/FP/FN гари, пропуски по классам
    groups: dict[str, dict[str, int]] = {}
    # потери ROI: для каждого среза — сколько пикселей истины каждого класса ниже среза
    roi_lost = {c: np.zeros(4, dtype=np.int64) for c in cuts}
    roi_total = np.zeros(4, dtype=np.int64)
    roi_inside = 0
    roi_all = 0
    # облачность: на чип — IoU гари и mIoU
    per_chip: list[dict] = []
    fp_dnbr: list[np.ndarray] = []
    tp1_dnbr: list[np.ndarray] = []
    t0 = time.perf_counter()
    n_pred = 0
    n_err = 0
    for i, cid in enumerate(ids, 1):
        if i % 20 == 0 or i == len(ids):
            print(f"    ... {i}/{len(ids)} ({time.perf_counter() - t0:.0f} с)", flush=True)
        try:
            chip = dataset.load(cid)
            if chip.mask_sev is None:
                continue
            truth = np.asarray(chip.mask_sev, dtype=np.uint8)
            pred = np.asarray(bsm.classify_chip(chip, bs_params, severity_model=sev_model),
                              dtype=np.uint8)
        except Exception as exc:  # один битый чип не должен ронять всю диагностику
            n_err += 1
            print(f"    [warn] чип {cid}: {exc}", file=sys.stderr)
            continue
        n_pred += 1
        for t in range(4):
            row = truth == t
            if row.any():
                for p in range(4):
                    conf[t, p] += int((row & (pred == p)).sum())
        lc = np.asarray(chip.layers.get("landcover")) if chip.layers.get("landcover") is not None else None
        if lc is not None and lc.shape == truth.shape:
            for gname in ("forest", "semi", "cropland", "other", "unknown"):
                groups.setdefault(gname, {"truth_burn": 0, "tp": 0, "fp": 0, "fn": 0,
                                          "t1": 0, "t1_missed": 0, "t2": 0, "t2_missed": 0})
            flat_lc = lc.reshape(-1)
            flat_t = truth.reshape(-1)
            flat_p = pred.reshape(-1)
            for code in np.unique(flat_lc):
                gname = codes.get(int(code))
                if gname is None:
                    gname = "unknown" if int(code) != bsm.LC_WATER else "other"
                sel = flat_lc == code
                g = groups[gname]
                tb = sel & (flat_t > 0)
                pb = sel & (flat_p > 0)
                g["truth_burn"] += int(tb.sum())
                g["tp"] += int((tb & pb).sum())
                g["fp"] += int(((flat_t == 0) & pb & sel).sum())
                g["fn"] += int((tb & (flat_p == 0)).sum())
                for cls, key in ((1, "t1"), (2, "t2")):
                    m = sel & (flat_t == cls)
                    g[key] += int(m.sum())
                    g[key + "_missed"] += int((m & (flat_p == 0)).sum())
        dnbr = _dnbr(chip)
        if dnbr is not None and dnbr.shape == truth.shape:
            finite = np.isfinite(dnbr)
            roi_all += int(finite.sum())
            roi_inside += int((finite & (dnbr > 0.05)).sum())
            for cls in range(4):
                sel = finite & (truth == cls)
                roi_total[cls] += int(sel.sum())
                for c in cuts:
                    roi_lost[c][cls] += int((sel & (dnbr <= c)).sum())
            fp = (truth == 0) & (pred > 0) & finite
            tp1 = (truth == 1) & (pred == 1) & finite
            if fp.any():
                fp_dnbr.append(np.asarray(dnbr[fp], dtype=np.float32))
            if tp1.any():
                tp1_dnbr.append(np.asarray(dnbr[tp1], dtype=np.float32))
        tp = int(((truth > 0) & (pred > 0)).sum())
        fp_n = int(((truth == 0) & (pred > 0)).sum())
        fn_n = int(((truth > 0) & (pred == 0)).sum())
        iou_burn = tp / max(1, tp + fp_n + fn_n)
        ious = []
        for cls in (1, 2, 3):
            tpx = int(((truth == cls) & (pred == cls)).sum())
            fpx = int(((truth != cls) & (pred == cls)).sum())
            fnx = int(((truth == cls) & (pred != cls)).sum())
            ious.append(tpx / max(1, tpx + fpx + fnx))
        per_chip.append({"chip_id": cid, "iou_burn": iou_burn, "miou": float(np.mean(ious)),
                         "cloud": _cloud_frac(dataset, cid),
                         "burn_px": int((truth > 0).sum())})

    print(f"\nпредсказано чипов: {n_pred}, пропущено с ошибкой: {n_err} "
          f"({time.perf_counter() - t0:.1f} с)")

    print("\n--- 1. Confusion (строки — истина, столбцы — предсказание) ---")
    print("        " + "".join(f"pred{c:>10d}" for c in range(4)) + "     recall")
    for t in range(4):
        row = conf[t]
        rec = row[t] / max(1, row.sum())
        print(f"truth{t}: " + "".join(f"{v:14d}" for v in row) + f"  {rec:6.3f}")
    ious = [conf[c, c] / max(1, conf[c, :].sum() + conf[:, c].sum() - conf[c, c]) for c in (1, 2, 3)]
    tp_all = int(conf[1:, 1:].sum())
    denom = int(conf[:, 1:].sum() + conf[1:, :].sum() - conf[1:, 1:].sum())
    print(f"IoU_burn={tp_all / max(1, denom):.4f}  "
          f"IoU(1/2/3)={'/'.join(f'{v:.4f}' for v in ious)}")

    print("\n--- 2. Ошибки гари по группам покрова ---")
    print(f"{'группа':<10}{'truth_px':>12}{'TP':>12}{'FP':>12}{'FN':>12}"
          f"{'IoU':>8}{'доля FP':>9}")
    fp_total = sum(g["fp"] for g in groups.values()) or 1
    for gname in ("forest", "semi", "cropland", "other", "unknown"):
        g = groups.get(gname)
        if g is None or (g["truth_burn"] + g["fp"] + g["fn"] == 0):
            continue
        iou = g["tp"] / max(1, g["tp"] + g["fp"] + g["fn"])
        print(f"{gname:<10}{g['truth_burn']:12d}{g['tp']:12d}{g['fp']:12d}{g['fn']:12d}"
              f"{iou:8.3f}{g['fp'] / fp_total:9.1%}")
    print("пропуски слабых классов в фон (truth=k, pred=0) по покровам:")
    for gname in ("forest", "semi", "cropland", "other", "unknown"):
        g = groups.get(gname)
        if g is None or (g["t1"] + g["t2"] == 0):
            continue
        r1 = g["t1_missed"] / max(1, g["t1"])
        r2 = g["t2_missed"] / max(1, g["t2"])
        print(f"    {gname:<10} класс 1: {g['t1_missed']:8d}/{g['t1']:<8d} ({r1:5.1%} в фоне)   "
              f"класс 2: {g['t2_missed']:8d}/{g['t2']:<8d} ({r2:5.1%} в фоне)")

    print("\n--- 3. Потери ROI-гейта: доля истины каждого класса с dNBR <= среза ---")
    print("(строка 0.05 — текущий гейт модели; что ниже — модель вообще не видит)")
    header = f"{'срез':>6}" + "".join(f"{f'класс {c}':>14}" for c in range(4))
    print(header)
    for c in cuts:
        cells = "".join(f"{roi_lost[c][k] / max(1, roi_total[k]):14.1%}" for k in range(4))
        print(f"{c:6.2f}{cells}")
    print(f"пикселей чипа внутри ROI(0.05): {roi_inside / max(1, roi_all):.1%} (цена ослабления гейта)")

    print("\n--- 4. Качество по облачности чипа (cloud_frac из meta) ---")
    print(f"{'облака':<8}{'чипов':>7}{'IoU_burn':>10}{'mIoU':>8}")
    for lo, hi, label in CLOUD_BUCKETS:
        bucket = [c for c in per_chip if np.isfinite(c["cloud"]) and lo <= c["cloud"] < hi]
        if not bucket:
            print(f"{label:<8}{0:7d}{'—':>10}{'—':>8}")
            continue
        ib = float(np.mean([c["iou_burn"] for c in bucket]))
        mi = float(np.mean([c["miou"] for c in bucket]))
        print(f"{label:<8}{len(bucket):7d}{ib:10.4f}{mi:8.4f}")
    unknown_cl = [c for c in per_chip if not np.isfinite(c["cloud"])]
    if unknown_cl:
        print(f"(без cloud_frac в meta: {len(unknown_cl)} чипов)")

    if fp_dnbr and tp1_dnbr:
        fp_all = np.concatenate(fp_dnbr)
        tp1_all = np.concatenate(tp1_dnbr)
        print("\n--- 5. Разделимость: dNBR ложных срабатываний vs верного класса 1 ---")
        print("        " + "".join(f"{q:>8d}" for q in QUANTILES) + f"{'n':>12}")
        for name, vals in (("FP", fp_all), ("TP-1", tp1_all)):
            qs = np.percentile(vals, QUANTILES)
            print(f"{name:<8}" + "".join(f"{v:8.3f}" for v in qs) + f"{vals.size:12d}")
        print("(если распределения сильно пересекаются — одним dNBR их не разделить, "
              "нужны контекст/NDVI/SAR)")

    per_chip.sort(key=lambda c: c["iou_burn"])
    print("\n--- 6. Худшие 10 чипов по IoU гари (кандидаты на ручной разбор) ---")
    for c in per_chip[:10]:
        cl = f"{c['cloud']:.1%}" if np.isfinite(c["cloud"]) else "—"
        print(f"    {c['chip_id']}: IoU_burn={c['iou_burn']:.3f} mIoU={c['miou']:.3f} "
              f"облака={cl} burn_px={c['burn_px']}")

    if args.json_out:
        summary = {
            "mode": mode, "n_chips": n_pred,
            "confusion": conf.tolist(),
            "groups": {k: {kk: int(vv) for kk, vv in g.items()} for k, g in groups.items()},
            "roi": {"cuts": cuts,
                    "lost": {str(c): roi_lost[c].tolist() for c in cuts},
                    "total": roi_total.tolist()},
            "worst_chips": per_chip[:10],
        }
        Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"\nсводка записана: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
