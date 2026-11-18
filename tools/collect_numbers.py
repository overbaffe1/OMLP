#!/usr/bin/env python3
"""Собирает все числа решения в один JSON: метрики, абляции, скорость, статистика набора.

    python tools/collect_numbers.py --data-dir /tmp/firesdata/real/train --out docs/numbers.json

Один прогон даёт и итоговую конфигурацию, и полный набор абляций (тем же кодом, что
``evaluate.py``), поэтому числа в отчёте и презентации не могут «разъехаться» с кодом.
Скорость инференса измеряется отдельно — вызовом ``inference.process`` на том же каталоге.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate as evaluate_module  # noqa: E402
from fires import af as af_module  # noqa: E402
from fires import bs as bs_module  # noqa: E402
from fires.chips import Dataset  # noqa: E402
from fires.severity import SeverityModel  # noqa: E402
from fires.weights import load_weights  # noqa: E402


def dataset_stats(dataset: Dataset) -> dict:
    """Статистика набора: сколько чипов, пикселей горения, площадь гари и доли классов."""

    af_ids, bs_ids = dataset.chip_ids("af"), dataset.chip_ids("bs")
    fire_px = 0
    chips_with_fire = 0
    for chip_id in af_ids:
        mask = dataset.load(chip_id).mask
        if mask is None:
            continue
        count = int(np.asarray(mask).sum())
        fire_px += count
        chips_with_fire += int(count > 0)

    sev = {1: 0, 2: 0, 3: 0}
    burn_ha = 0.0
    cloud = []
    for chip_id in bs_ids:
        chip = dataset.load(chip_id)
        if chip.mask_sev is not None:
            values = np.asarray(chip.mask_sev)
            for cls in (1, 2, 3):
                sev[cls] += int((values == cls).sum())
        row = dataset.meta.get(chip_id, {})
        burn_ha += float(row.get("burn_area_ha") or 0.0)
        value = row.get("cloud_frac")
        if value is not None:
            try:
                cloud.append(float(value))
            except (TypeError, ValueError):
                pass

    total_sev = sum(sev.values()) or 1
    return {
        "n_af": len(af_ids),
        "n_bs": len(bs_ids),
        "af_chips_with_fire": chips_with_fire,
        "af_fire_px": fire_px,
        "bs_burn_area_ha": round(burn_ha, 1),
        "bs_severity_px": sev,
        "bs_severity_share": {str(k): round(v / total_sev, 4) for k, v in sev.items()},
        "bs_cloud_frac_median": round(float(np.median(cloud)), 4) if cloud else None,
        "bs_cloud_frac_max": round(float(np.max(cloud)), 4) if cloud else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Сбор чисел решения в JSON для отчёта и презентации")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--weights", default="weights/weights.json")
    ap.add_argument("--out", default="docs/numbers.json")
    ap.add_argument("--no-ablation", action="store_true", help="только итоговые метрики (быстро)")
    ap.add_argument("--no-inference", action="store_true", help="не мерить скорость инференса")
    args = ap.parse_args(argv)

    dataset = Dataset(args.data_dir)
    weights = load_weights(args.weights)
    af_params = af_module.to_params(weights.get("af", {}))
    bs_params = bs_module.to_params(weights.get("bs", {}))

    severity_model = None
    if weights.get("bs_mode") == "model" and weights.get("severity_model"):
        model_path = Path(args.weights).parent / weights["severity_model"]
        if model_path.exists():
            severity_model = SeverityModel.load(model_path)
            print(f"модель степени поражения: {model_path}")
        else:
            print(f"[warn] не найдена {model_path} — считаю на порогах", file=sys.stderr)

    print("== итоговая конфигурация ==")
    final = evaluate_module.evaluate(dataset, af_params, bs_params, severity_model=severity_model)
    print(json.dumps(final, ensure_ascii=False))

    rows: list[dict] = []
    if not args.no_ablation:
        print("== абляции ==")
        variants = evaluate_module.ablation_variants(af_params, bs_params, None, severity_model=severity_model)
        for name, af_p, bs_p, blob, sev in variants:
            result = evaluate_module.evaluate(dataset, af_p, bs_p, blob_filter=blob, severity_model=sev)
            rows.append(
                {
                    "name": name,
                    "score": result["score"],
                    "f1_af": result["f1_af"],
                    "iou_burn": result["iou_burn"],
                    "miou_sev": result["miou_sev"],
                    "delta_score": round(result["score"] - final["score"], 5),
                }
            )
            print(f"  {name:48s} Score={result['score']:.4f}  Δ={rows[-1]['delta_score']:+.4f}")

    speed = None
    if not args.no_inference:
        print("== скорость инференса ==")
        from inference import process as inference_process

        t0 = time.perf_counter()
        report = inference_process(Path(args.data_dir), Path("/tmp/_numbers_submission.csv"), Path(args.weights), verbose=False)
        speed = {
            "chips": report["chips_processed"],
            "rows": report["rows_written"],
            "seconds": round(time.perf_counter() - t0, 2),
            "jobs": report["jobs"],
        }
        print(json.dumps(speed, ensure_ascii=False))

    numbers = {
        "data_dir": str(args.data_dir),
        "weights": str(args.weights),
        "weights_content": {
            "af_mode": weights.get("af_mode"),
            "bs_mode": weights.get("bs_mode"),
            "af": weights.get("af"),
            "bs": weights.get("bs"),
            "train_metrics": weights.get("train_metrics"),
        },
        "dataset": dataset_stats(dataset),
        "final": final,
        "ablations": rows,
        "inference": speed,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(numbers, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"числа сохранены: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
