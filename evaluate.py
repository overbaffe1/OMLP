#!/usr/bin/env python3
"""Локальная оценка решения на чипах с эталонными масками.

    python evaluate.py --data-dir /path/to/train [--weights weights/weights.json] [--ablation]

Печатает компоненты метрики соревнования (F1_af, IoU_burn, mIoU_sev и Score) и, при
``--ablation``, вклад отдельных механизмов: отсева техногенных источников, блика,
подтверждения по Sentinel-1, порогов по типу покрова. Абляции нужны для отчёта
(критерий «Эксперименты и сравнения»).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from fires import af as af_module
from fires import bs as bs_module
from fires.blobfilter import BlobFilter
from fires.chips import Dataset, find_data_dir
from fires.metrics import score_from_pairs
from fires.severity import SeverityModel
from fires.weights import load_weights


def collect(dataset: Dataset, af_params, bs_params, limit: int | None = None, blob_filter=None,
            severity_model=None):
    af_pairs, bs_pairs = {}, {}
    for chip_id in dataset.chip_ids():
        chip = dataset.load(chip_id)
        if chip.kind == "af" and chip.mask is not None:
            af_pairs[chip_id] = (chip.mask.astype(bool), af_module.detect_chip(chip, af_params, blob_filter=blob_filter))
        elif chip.kind == "bs" and chip.mask_sev is not None:
            bs_pairs[chip_id] = (
                chip.mask_sev.astype(np.uint8),
                bs_module.classify_chip(chip, bs_params, severity_model=severity_model),
            )
        if limit and len(af_pairs) + len(bs_pairs) >= limit:
            break
    return af_pairs, bs_pairs


def evaluate(dataset: Dataset, af_params, bs_params, blob_filter=None, severity_model=None) -> dict:
    af_pairs, bs_pairs = collect(dataset, af_params, bs_params, blob_filter=blob_filter, severity_model=severity_model)
    result = score_from_pairs(af_pairs, bs_pairs)
    result["n_af"] = len(af_pairs)
    result["n_bs"] = len(bs_pairs)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Оценка решения на чипах с масками")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--blob-filter", default=None, help="путь к weights/blob_filter.pkl")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None,
                    help="где считать модель степени поражения (по умолчанию — как обучена)")
    ap.add_argument("--strict", action="store_true",
                    help="падать, если настроенные обучаемые компоненты (модель/классификатор) не загрузились")
    args = ap.parse_args(argv)

    resolved = find_data_dir(args.data_dir, prefer="masks")
    if resolved != Path(args.data_dir):
        print(f"в {args.data_dir} чипов нет; использую найденный рядом каталог {resolved}")
    dataset = Dataset(resolved)
    if not dataset.chip_ids():
        print(f"в каталоге {args.data_dir} не нашлось ни одного чипа — "
              f"проверьте путь: py -3 -m fires.chips --data-dir \"{args.data_dir}\"")
        return 1
    weights = load_weights(args.weights)
    af_params = af_module.to_params(weights.get("af", {}))
    bs_params = bs_module.to_params(weights.get("bs", {}))

    try:
        if args.blob_filter:
            blob_filter = BlobFilter.load(args.blob_filter)
        else:
            blob_filter = load_blob_filter(weights, args.weights, strict=args.strict)
        severity_model = load_severity_model(weights, args.weights, args.device, strict=args.strict)
    except RuntimeError as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1

    print("=== Активная конфигурация ===")
    print("очаги: " + ("blobs (классификатор)" if blob_filter is not None else "правила"))
    if severity_model is not None:
        print("степень поражения: обученная модель")
    else:
        hint = " (модель степени поражения НЕ загрузилась)" if weights.get("bs_mode") == "model" else ""
        print(f"степень поражения: пороги{hint}")
    print(json.dumps(evaluate(dataset, af_params, bs_params, blob_filter=blob_filter,
                              severity_model=severity_model), ensure_ascii=False, indent=2))

    if not args.ablation:
        return 0

    print("\n=== Абляции ===")
    variants = ablation_variants(af_params, bs_params, blob_filter, severity_model=severity_model)
    for name, af_p, bs_p, bf, sm in variants:
        res = evaluate(dataset, af_p, bs_p, blob_filter=bf, severity_model=sm)
        print(f"{name:48s} Score={res['score']:.4f}  F1_af={res['f1_af']:.4f}  IoU_burn={res['iou_burn']:.4f}  mIoU_sev={res['miou_sev']:.4f}")
    return 0


def load_blob_filter(weights: dict, weights_path: str | Path | None, strict: bool = False) -> "BlobFilter | None":
    """Загружает обученный классификатор очагов, если решение выбрало режим ``blobs``."""

    if weights.get("af_mode") != "blobs":
        return None
    base = Path(weights_path).parent if weights_path else Path(__file__).resolve().parent / "weights"
    path = base / weights.get("blob_filter", "blob_filter.pkl")
    if not path.exists():
        message = f"[warn] классификатор очагов {path} не найден; работаю на правилах"
        print(message, file=sys.stderr)
        if strict:
            raise RuntimeError(message)
        return None
    try:
        return BlobFilter.load(path)
    except Exception as exc:
        message = f"[warn] классификатор очагов {path} не загрузился ({exc}); работаю на правилах"
        print(message, file=sys.stderr)
        if strict:
            raise RuntimeError(message) from exc
        return None


def load_severity_model(weights: dict, weights_path: str | Path | None, device: str | None = None,
                        strict: bool = False) -> "SeverityModel | None":
    """Загружает обученную модель степени поражения, если решение выбрало режим ``model``."""

    name = weights.get("severity_model")
    if not name or weights.get("bs_mode") != "model":
        return None
    base = Path(weights_path).parent if weights_path else Path(__file__).resolve().parent / "weights"
    path = base / name
    if not path.exists():
        message = f"[warn] модель степени поражения {path} не найдена; работаю на порогах"
        print(message, file=sys.stderr)
        if strict:
            raise RuntimeError(message)
        return None
    try:
        model = SeverityModel.load(path)
        if weights.get("severity_class_bias"):
            model.class_bias = tuple(float(x) for x in weights["severity_class_bias"])
        if device:
            from fires.severity import resolve_device

            model.device = resolve_device(device)
        return model
    except Exception as exc:  # модель собрана другой версией sklearn — считаем на порогах
        message = f"[warn] модель степени поражения {path} не загрузилась ({exc}); работаю на порогах"
        print(message, file=sys.stderr)
        if strict:
            raise RuntimeError(message) from exc
        return None


def ablation_variants(af_params, bs_params, blob_filter=None, severity_model=None):
    """Список абляций: каждая строка выключает ровно один механизм решения.

    Возвращает список ``(название, af_params, bs_params, blob_filter, severity_model)``;
    используется и в ``evaluate.py --ablation``, и в ``tools/collect_numbers.py`` — числа отчёта
    и презентации считаются одним и тем же набором вариантов.
    """

    import copy

    # В режиме классификатора очагов правила AF задают состав кандидатов, а решение
    # принимает модель, поэтому варианты отключают именно механизмы классификатора.
    def variant(**config_changes):
        if blob_filter is None:
            return None
        clone = copy.deepcopy(blob_filter)
        for key, value in config_changes.items():
            setattr(clone.config, key, value)
        return clone

    threshold_filter = blob_filter
    if blob_filter is not None:
        threshold_filter = copy.deepcopy(blob_filter)
        threshold_filter.threshold = 0.95

    day_only = replace(
        af_params,
        sat_t4_night=af_params.sat_t4,
        absolute_t4_night=af_params.absolute_t4,
        candidate_t4_night=af_params.candidate_t4,
        delta_night=af_params.delta_day,
    )
    tw = severity_model  # сокращение: модель присутствует во всех вариантах, кроме явной абляции
    # Абляция морфологии должна выключать активный механизм, а не повторять его: в режиме модели
    # дыры не заполняются (модель их различает), поэтому проверяется обратный вариант.
    fill_label = ("BS: с заполнением дыр (как в пороговом режиме)" if not bs_params.fill_holes_mask
                  else "BS: без заполнения дыр")
    patch_label = ("BS: без фильтрации мелких контуров" if bs_params.min_patch_px != 1
                   else "BS: с фильтрацией мелких контуров (6 px)")
    variants = [
        ("AF: без ночных порогов (единая дневная шкала)", day_only, bs_params, blob_filter, tw),
        ("AF: без режима блика (обычные пороги в зоне блика)", replace(af_params, glint_margin=0.0, glint_t4_min=0.0), bs_params, blob_filter, tw),
        ("AF: без отсева техногенных источников", replace(af_params, exclude_built=False), bs_params, blob_filter, tw),
        ("AF: без отсева одиночных пикселей", replace(af_params, isolated_guard=False), bs_params, blob_filter, tw),
        ("AF: только абсолютные пороги (без контекстного теста)", replace(af_params, k_sigma=1e6), bs_params, blob_filter, tw),
        ("AF: без маски воды", replace(af_params, exclude_water=False), bs_params, blob_filter, tw),
        ("BS: без Sentinel-1", af_params, replace(bs_params, use_sar=False), blob_filter, tw),
        ("BS: без маскировки облаков по SCL", af_params, replace(bs_params, mask_scl=False), blob_filter, tw),
        ("BS: SAR с восстановлением под облаком", af_params, replace(bs_params, sar_recovery=True), blob_filter, tw),
        ("BS: без порогов по покрову (только USGS)", af_params, replace(bs_params, semi=bs_params.forest, cropland=bs_params.forest), blob_filter, tw),
        (fill_label, af_params, replace(bs_params, fill_holes_mask=not bs_params.fill_holes_mask), blob_filter, tw),
        (patch_label, af_params, replace(bs_params, min_patch_px=1 if bs_params.min_patch_px != 1 else 6), blob_filter, tw),
        ("BS: без сглаживания dNBR", af_params, replace(bs_params, smooth_dnbr=0), blob_filter, tw),
    ]
    if severity_model is not None:
        # Ключевая развилка BS: если решение выбрало обучаемую модель, вклад порогов виден
        # только так — тем же набором параметров, но без модели.
        variants.insert(0, ("BS: пороги вместо обученной модели", af_params, bs_params, blob_filter, None))
    if blob_filter is not None:  # варианты классификатора очагов имеют смысл только в режиме blobs
        variants = [
            ("AF: без классификатора очагов (только правила)", af_params, bs_params, None, tw),
            ("AF: классификатор с порогом 0.95", af_params, bs_params, threshold_filter, tw),
            ("AF: кандидаты только по мягким порогам", af_params, bs_params, variant(use_soft_candidates=False), tw),
            ("AF: без пиксельного гейта в блобе", af_params, bs_params, variant(pixel_t4=-1000.0, pixel_delta=-1000.0), tw),
        ] + variants
    return variants


if __name__ == "__main__":
    raise SystemExit(main())
