#!/usr/bin/env python3
"""Подбор порогов («обучение») на обучающей части набора.

    python train.py --data-dir /path/to/train --out weights/weights.json

Модель решения — физически интерпретируемая: контекстный алгоритм детекции и пороги
индекса dNBR по типам покрова. Обучаемых параметров немного, поэтому вместо градиентного
спуска используется детерминированный поиск по сетке на отложенной части обучающих чипов
(по ``fire_event_id``, чтобы чипы одного пожара не попадали одновременно в обучение
и валидацию — так же, как организаторы разделяют выборки).

Скрипт фиксирует:
* ``seed`` и состав валидации;
* метрики до и после подбора (в отчёте требуются абляции — см. docs/report.md);
* значения параметров в ``weights.json``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from fires import af as af_module
from fires import bs as bs_module
from fires.chips import Dataset
from fires.metrics import score_from_pairs
from fires.severity import gpu_report, resolve_device
from fires.weights import save_weights

SEED = 20260918


def _split_ids(dataset: Dataset, kind: str, val_frac: float = 0.25, seed: int = SEED) -> tuple[list[str], list[str]]:
    """Делит чипы по событиям пожара: чипы одного события целиком в одной части."""

    ids = dataset.chip_ids(kind)
    events: dict[str, list[str]] = {}
    for chip_id in ids:
        row = dataset.meta.get(chip_id, {})
        event = str(row.get("fire_event_id") or chip_id)
        events.setdefault(event, []).append(chip_id)
    keys = sorted(events)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_frac)))
    val_keys = set(keys[:n_val])
    train_ids = [c for k in keys if k not in val_keys for c in events[k]]
    val_ids = [c for k in keys if k in val_keys for c in events[k]]
    return train_ids, val_ids


def evaluate_af(dataset: Dataset, chip_ids: list[str], params: af_module.AFParams, blob_filter=None) -> dict:
    pairs = {}
    for chip_id in chip_ids:
        chip = dataset.load(chip_id)
        if chip.mask is None:
            continue
        pred = af_module.detect_chip(chip, params, blob_filter=blob_filter)
        pairs[chip_id] = (chip.mask.astype(bool), pred)
    if not pairs:
        return {"f1_af": None, "n": 0}
    result = score_from_pairs(pairs, {})
    return {"f1_af": result["f1_af"], "counts": result["af_counts"], "n": len(pairs)}


def evaluate_bs(dataset: Dataset, chip_ids: list[str], params: bs_module.BSParams, severity_model=None) -> dict:
    pairs = {}
    for chip_id in chip_ids:
        chip = dataset.load(chip_id)
        if chip.mask_sev is None:
            continue
        pred = bs_module.classify_chip(chip, params, severity_model=severity_model)
        pairs[chip_id] = (chip.mask_sev.astype(np.uint8), pred)
    if not pairs:
        return {"iou_burn": None, "miou_sev": None, "n": 0}
    result = score_from_pairs({}, pairs)
    return {
        "iou_burn": result["iou_burn"],
        "miou_sev": result["miou_sev"],
        "iou_per_class": result["iou_per_class"],
        "n": len(pairs),
    }


#: Сетка поиска для AF. Диапазоны взяты вокруг значений, найденных на реальной
#: обучающей части: днём фон τ4 прогревается, поэтому порог кандидатов выше «школьных» 320 K,
#: а ночью, наоборот, ниже. Отдельная ось — ночной порог кандидатов: по этим чипам
#: ошибка предыдущего приближения была максимальной.
AF_GRID = {
    "k_sigma": [2.0, 3.0],
    "delta_day": [20.0, 25.0, 30.0],
    "candidate_t4": [325.0, 330.0, 335.0],
    "candidate_t4_night": [293.0, 295.0, 300.0],
    "min_cluster_px": [1, 3],
}

#: Сколько пикселей брать с одного чипа при обучении модели степени поражения.
SEVERITY_TRAIN_PIXELS = 4000
#: Число итераций бустинга. Замеры на 56 отложенных чипах: 60 итераций — IoU 0.5373 / mIoU 0.5151,
#: 90 — 0.5379 / 0.5170, 150 — 0.5381 / 0.5169. Качество выходит на плато уже к 60 итерациям, а время
#: инференса растёт линейно (по 5 мс на итерацию и чип), поэтому берём 60: на 2 ядрах тест
#: укладывается в 25 с вместо 35, на 12 потоках — в 10 с (лимит кейса 30 с на прогон).
SEVERITY_ITERATIONS = 60
#: Скорость обучения и размер деревьев (подобраны вместе с числом итераций).
SEVERITY_LEARNING_RATE = 0.20
SEVERITY_LEAVES = 24
SEVERITY_MIN_SAMPLES_LEAF = 80

#: Пороги dNBR по группам покрова: тройка (t1, t2, t3) на каждую группу, вокруг значений,
#: измеренных на реальных чипах (границы классов — середины между распределениями соседних
#: классов).
BS_GRID = {
    "forest": [(0.15, 0.27, 0.55), (0.20, 0.30, 0.60), (0.25, 0.35, 0.65), (0.20, 0.25, 0.50)],
    "semi": [(0.08, 0.19, 0.37), (0.12, 0.21, 0.38), (0.14, 0.22, 0.40), (0.16, 0.25, 0.44)],
    "cropland": [(0.08, 0.17, 0.34), (0.10, 0.19, 0.37), (0.12, 0.19, 0.37), (0.14, 0.22, 0.42)],
}


def tune_af(dataset: Dataset, train_ids: list[str], val_ids: list[str], verbose: bool = True) -> dict:
    """Две стадии: (1) подбор порогов правил, (2) обучение классификатора очагов.

    Выбирается вариант с лучшим F1 на отложенной части; результат попадает в weights.json
    вместе с режимом (``rules`` или ``blobs``), чтобы инференс был воспроизводим.
    """

    import itertools

    # --- стадия 1: пороги правил
    best = {"f1_af": -1.0, "params": {}}
    keys = list(AF_GRID)
    combos = list(itertools.product(*[AF_GRID[k] for k in keys]))
    if verbose:
        print(f"AF/rules: перебор {len(combos)} конфигураций на {len(val_ids)} чипах")
    for combo in combos:
        values = dict(zip(keys, combo))
        result = evaluate_af(dataset, val_ids, af_module.to_params(values))
        if result["f1_af"] is None:
            continue
        if result["f1_af"] > best["f1_af"]:
            best = {"f1_af": result["f1_af"], "params": values, "counts": result["counts"]}
            if verbose:
                print(f"  новое лучшее F1={result['f1_af']:.4f} {values}")

    # --- стадия 2: классификатор очагов
    from fires.blobfilter import BlobFilter, BlobFilterConfig, extract_features, label_blobs

    # Признаки считаются с теми же параметрами правил, что попадут в weights.json:
    # иначе состав кандидатов при обучении и при инференсе различается.
    rule_params = af_module.to_params(best.get("params", {}))
    Xs, ys = [], []
    for chip_id in train_ids:
        chip = dataset.load(chip_id)
        if chip.mask is None:
            continue
        feats, masks = extract_features(chip, rule_params)
        if len(masks):
            Xs.append(feats)
            ys.append(label_blobs(chip, masks))
    blob_result = {"f1_af": -1.0}
    if Xs:
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        if verbose:
            print(f"AF/blobs: обучение на {X.shape[0]} блобах (положительных {int(y.sum())}, отрицательных {int((1 - y).sum())})")
        model = BlobFilter().fit(X, y, class_weight="balanced")
        importances = None
        for threshold in (0.3, 0.5, 0.7):
            for pixel_t4, pixel_delta in ((305.0, 4.0), (315.0, 6.0), (320.0, 8.0)):
                model.threshold = threshold
                model.config.pixel_t4 = pixel_t4
                model.config.pixel_delta = pixel_delta
                result = evaluate_af(dataset, val_ids, rule_params, blob_filter=model)
                if result["f1_af"] is None:
                    continue
                if result["f1_af"] > blob_result["f1_af"]:
                    blob_result = {
                        "f1_af": result["f1_af"],
                        "threshold": threshold,
                        "pixel_t4": pixel_t4,
                        "pixel_delta": pixel_delta,
                    }
                    if verbose:
                        print(f"  лучшее F1={result['f1_af']:.4f} (порог {threshold}, пиксельные {pixel_t4}/{pixel_delta})")
        if blob_result["f1_af"] > 0:
            model.threshold = blob_result["threshold"]
            model.config.pixel_t4 = blob_result["pixel_t4"]
            model.config.pixel_delta = blob_result["pixel_delta"]
            model.metrics = {"val_f1": blob_result["f1_af"], "n_blobs_train": int(X.shape[0]), "pos": int(y.sum())}
            blob_result["model"] = model
            try:  # важности признаков для отчёта
                from sklearn.inspection import permutation_importance

                idx = np.random.default_rng(args_seed()).choice(len(y), size=min(400, len(y)), replace=False)
                perm = permutation_importance(model.model, X[idx], y[idx], n_repeats=5, random_state=args_seed())
                importances = dict(sorted(zip(model.feature_names, perm.importances_mean.tolist()), key=lambda kv: -kv[1])[:8])
            except Exception:
                importances = None
            blob_result["importances"] = importances

    winner = "blobs" if blob_result.get("f1_af", -1) > best["f1_af"] else "rules"
    return {"rules": best, "blobs": blob_result, "mode": winner, "params": best.get("params", {})}


def args_seed() -> int:
    return SEED


def tune_bs(dataset: Dataset, train_ids: list[str], val_ids: list[str], verbose: bool = True,
            device: str = "auto", backend: str = "auto", features: str = "base") -> dict:
    """Поиск порогов dNBR по группам покрова (максимизируем 0.35*IoU_burn + 0.30*mIoU_sev)."""

    import itertools

    best = {"objective": -1.0, "params": {}}
    combos = list(itertools.product(BS_GRID["forest"], BS_GRID["semi"], BS_GRID["cropland"]))
    if verbose:
        print(f"BS: перебор {len(combos)} конфигураций на {len(val_ids)} чипах")

    def build(forest, semi, cropland):
        return bs_module.to_params({"forest": list(forest), "semi": list(semi), "cropland": list(cropland)})

    for forest, semi, cropland in combos:
        params = build(forest, semi, cropland)
        result = evaluate_bs(dataset, val_ids, params)
        if result["iou_burn"] is None:
            continue
        objective = 0.35 * result["iou_burn"] + 0.30 * result["miou_sev"]
        if objective > best["objective"]:
            best = {
                "objective": round(float(objective), 5),
                "params": {"forest": list(forest), "semi": list(semi), "cropland": list(cropland)},
                "iou_burn": result["iou_burn"],
                "miou_sev": result["miou_sev"],
            }
            if verbose:
                print(f"  новое лучшее objective={objective:.4f} IoU_burn={result['iou_burn']:.4f} mIoU={result['miou_sev']:.4f}")

    # --- стадия 2: восстановление гари под облаком по Sentinel-1
    # Оптические наблюдения под облаком/тенью недостоверны, но радиолокация видит потерю
    # биомассы. Коэффициент «дБ на единицу dNBR» физически зависит от типа покрова и
    # подбирается по валидации; вариант без SAR используется, если он не хуже.
    sar_variants: list[tuple[bool, tuple[float, float]]] = [(False, (3.0, 1.5))]
    sar_variants += [(True, scale) for scale in ((2.0, 1.0), (3.0, 1.5), (4.0, 2.0), (5.0, 2.5))]
    sar_min_drops = (0.8, 1.2, 1.6)
    if best["params"]:
        th = best["params"]
        base_params = build(th["forest"], th["semi"], th["cropland"])
        # Валидационная часть мала, поэтому вариант по SAR выбирается по двум фолдам
        # (перекрёстная проверка по событиям пожара): итог — среднее objective.
        folds = [(train_ids, val_ids), (val_ids, train_ids)]
        best_sar = {"objective": -1.0, "sar_recovery": False, "sar_scale": [3.0, 1.5], "sar_min_drop_db": 1.0}
        for enabled, scale in sar_variants:
            for min_drop in sar_min_drops if enabled else (1.0,):
                params = replace(
                    base_params,
                    sar_recovery=enabled,
                    sar_scale=scale,
                    sar_min_drop_db=min_drop,
                )
                scores = []
                for _, fold_val in folds:
                    result = evaluate_bs(dataset, fold_val, params)
                    if result["iou_burn"] is not None:
                        scores.append(0.35 * result["iou_burn"] + 0.30 * result["miou_sev"])
                if not scores:
                    continue
                objective = float(np.mean(scores))
                if objective > best_sar["objective"]:
                    result = evaluate_bs(dataset, val_ids, params)
                    best_sar = {
                        "objective": round(objective, 5),
                        "sar_recovery": enabled,
                        "sar_scale": list(scale),
                        "sar_min_drop_db": min_drop,
                        "iou_burn": result["iou_burn"],
                        "miou_sev": result["miou_sev"],
                        "fold_scores": [round(v, 5) for v in scores],
                    }
                    if verbose:
                        print(
                            f"  SAR: recovery={enabled} scale={scale} drop≥{min_drop} дБ "
                            f"objective(2 фолда)={objective:.4f} "
                            f"val: IoU_burn={result['iou_burn']:.4f} mIoU={result['miou_sev']:.4f}"
                        )
        best["params"]["sar_recovery"] = best_sar["sar_recovery"]
        best["params"]["sar_scale"] = best_sar["sar_scale"]
        best["params"]["sar_min_drop_db"] = best_sar["sar_min_drop_db"]
        best["sar"] = {k: v for k, v in best_sar.items() if k != "objective"}
        train_params = replace(
            base_params,
            sar_recovery=best_sar["sar_recovery"],
            sar_scale=tuple(best_sar["sar_scale"]),
            sar_min_drop_db=best_sar["sar_min_drop_db"],
        )

        # --- стадия 3: сглаживание dNBR
        # Соседние пиксели горят вместе, а одиночные выбросы dNBR (границы полей, тени, сдвиг
        # сцен) дают россыпь ложных контуров. Размер окна выбирается по двум фолдам, как и SAR.
        best_smooth = {"objective": -1.0, "size": 0}
        for size in (0, 3, 5):
            params = replace(train_params, smooth_dnbr=size)
            scores = []
            for _, fold_val in folds:
                result = evaluate_bs(dataset, fold_val, params)
                if result["iou_burn"] is not None:
                    scores.append(0.35 * result["iou_burn"] + 0.30 * result["miou_sev"])
            if not scores:
                continue
            objective = float(np.mean(scores))
            if objective > best_smooth["objective"]:
                best_smooth = {"objective": round(objective, 5), "size": size, "fold_scores": [round(v, 5) for v in scores]}
                if verbose:
                    print(f"  сглаживание dNBR {size}×{size}: objective(2 фолда)={objective:.4f}")
        best["params"]["smooth_dnbr"] = best_smooth["size"]
        best["smooth"] = {"size": best_smooth["size"], "fold_scores": best_smooth.get("fold_scores", [])}

        train_params = replace(train_params, smooth_dnbr=best_smooth["size"])
        train_result = evaluate_bs(dataset, train_ids, train_params)
        best["train_iou_burn"] = train_result["iou_burn"]

        # --- стадия 4: обучаемая модель степени поражения
        # Порог — одно число на группу покрова, а решение о степени зависит и от запаса биомассы,
        # и от сезона, и от текстуры пятна. Модель на градиентном бустинге учитывает всё сразу;
        # она принимается, только если выигрывает у порогов на том же отложенном наборе.
        model = _train_severity_model(dataset, train_ids, verbose=verbose, device=device,
                                      backend=backend, features=features)
        if model is not None:
            model_result = evaluate_bs(dataset, val_ids, train_params, severity_model=model)
            model_objective = 0.35 * model_result["iou_burn"] + 0.30 * model_result["miou_sev"]
            threshold_objective = 0.35 * best["iou_burn"] + 0.30 * best["miou_sev"]
            model.metrics.update({
                "val_iou_burn": model_result["iou_burn"],
                "val_miou_sev": model_result["miou_sev"],
                "val_objective": round(float(model_objective), 5),
                "thresholds_objective": round(float(threshold_objective), 5),
            })
            if verbose:
                print(
                    f"  модель степени поражения: IoU_burn={model_result['iou_burn']:.4f} "
                    f"mIoU={model_result['miou_sev']:.4f} (objective {model_objective:.4f} против "
                    f"{threshold_objective:.4f} у порогов)"
                )
            if model_objective > threshold_objective:
                # Морфология для режима модели подбирается отдельно: модель отличает дыры внутри
                # контура от гари, поэтому заполнение дыр ей мешает (IoU_burn 0.5376 против 0.5342
                # на отложенных чипах), зато чистка мелких контуров помогает (0.5410 против 0.5363
                # при 24 пикселях). На плато 24–64 пикселя качество одинаковое, берём 48.
                best_post = select_model_postprocess(dataset, val_ids, train_params, model, verbose=verbose)
                if verbose:
                    print(f"  морфология модели: чистка контуров < {best_post['min_patch_px']} px, "
                          f"дыры {'заполняются' if best_post['fill_holes_mask'] else 'не заполняются'} "
                          f"→ IoU_burn={best_post['iou_burn']:.4f} mIoU={best_post['miou_sev']:.4f}")
                final_params = replace(train_params, min_patch_px=best_post["min_patch_px"],
                                       fill_holes_mask=best_post["fill_holes_mask"])
                # Пороговый режим на тех же параметрах морфологии — честная база для сравнения.
                threshold_result = evaluate_bs(dataset, val_ids, final_params)
                model.metrics.update({
                    "val_iou_burn": best_post["iou_burn"],
                    "val_miou_sev": best_post["miou_sev"],
                    "val_objective": best_post["objective"],
                    "thresholds_iou_burn": threshold_result["iou_burn"],
                    "thresholds_miou_sev": threshold_result["miou_sev"],
                    "thresholds_objective": round(
                        0.35 * threshold_result["iou_burn"] + 0.30 * threshold_result["miou_sev"], 5),
                    "postprocess": {"min_patch_px": best_post["min_patch_px"],
                                    "fill_holes_mask": best_post["fill_holes_mask"]},
                })
                best["params"]["min_patch_px"] = best_post["min_patch_px"]
                best["params"]["fill_holes_mask"] = best_post["fill_holes_mask"]
                best["iou_burn"] = best_post["iou_burn"]
                best["miou_sev"] = best_post["miou_sev"]
                best["severity_model"] = model
                best["mode"] = "model"
            else:
                best["mode"] = "thresholds"
        else:
            best["mode"] = "thresholds"
    return best


def select_model_postprocess(dataset: Dataset, val_ids: list[str], params, model,
                             verbose: bool = True) -> dict:
    """Подбирает морфологию для режима обученной модели по отложенным чипам.

    Модель отличает внутренние «дыры» гари от выгоревших пикселей, поэтому заполнение дыр ей мешает,
    а удаление мелких ложных контуров помогает. Поиск маленький и детерминированный: размер чистки
    и признак заполнения дыр. Возвращает словарь с выбранными значениями и метриками.
    """

    best = {"objective": -1.0, "min_patch_px": 6, "fill_holes_mask": True}
    for min_patch in (24, 48):
        for fill in (False, True):
            candidate = replace(params, min_patch_px=min_patch, fill_holes_mask=fill)
            result = evaluate_bs(dataset, val_ids, candidate, severity_model=model)
            if result["iou_burn"] is None:
                continue
            objective = 0.35 * result["iou_burn"] + 0.30 * result["miou_sev"]
            if objective > best["objective"]:
                best = {
                    "objective": round(float(objective), 5),
                    "min_patch_px": min_patch,
                    "fill_holes_mask": fill,
                    "iou_burn": result["iou_burn"],
                    "miou_sev": result["miou_sev"],
                }
    if verbose:
        print(f"  морфология модели: чистка контуров < {best['min_patch_px']} px, дыры "
              f"{'заполняются' if best['fill_holes_mask'] else 'не заполняются'} → "
              f"objective={best['objective']:.4f}")
    return best


def _train_severity_model(dataset: Dataset, train_ids: list[str], verbose: bool = True,
                          device: str = "auto", backend: str = "auto", features: str = "base"):
    """Обучает модель степени поражения на подвыборке пикселей обучающих чипов.

    Пиксели берутся разреженно (по ``SEVERITY_TRAIN_PIXELS`` с чипа), потому что количество
    положительных пикселей на чипах различается на порядки; веса классов выравниваются, чтобы
    редкие классы 2 и 3 не «растворились» в фоне.

    ``device`` — ``auto``/``cpu``/``cuda``: на GPU обучение идёт через xgboost (sklearn CUDA не
    умеет), на CPU — через HistGradientBoostingClassifier. ``features`` — ``base`` (23 признака)
    или ``ext`` (расширенный набор из :mod:`fires.severity`).
    """

    try:
        from fires.severity import collect_training_pixels, train_model
    except Exception as exc:  # sklearn/xgboost нет или модуль недоступен — остаёмся на порогах
        if verbose:
            print(f"  модель степени поражения недоступна ({exc}) — остаются пороги")
        return None

    X, y = collect_training_pixels(dataset, train_ids, SEVERITY_TRAIN_PIXELS, features,
                                   seed=SEED, verbose=verbose)
    params = {
        "iterations": SEVERITY_ITERATIONS,
        "learning_rate": SEVERITY_LEARNING_RATE,
        "leaves": SEVERITY_LEAVES,
        "min_samples_leaf": SEVERITY_MIN_SAMPLES_LEAF,
        "l2": 1.0,
    }
    return train_model(X, y, params=params, backend=backend, device=device,
                       feature_names=features, seed=SEED, verbose=verbose)


def _eval_model(dataset: Dataset, args, bs_train: list[str], bs_val: list[str]) -> int:
    """Печатает метрики BS для сохранённой модели — отложенная часть и все чипы.

    Отложенная часть (её модель не видела) — честная оценка; все чипы — оптимистичная, зато
    сравнимы между кандидатами и с тем, что уходит на платформу.
    """

    import json as _json

    from fires.severity import SeverityModel

    weights_path = Path(args.out)
    weights = {}
    if weights_path.is_file():
        weights = _json.loads(weights_path.read_text(encoding="utf-8"))
    bs_params = bs_module.BSParams(**weights.get("bs", {})) if weights.get("bs") else \
        bs_module.BSParams()
    model = SeverityModel.load(args.eval_model)
    if args.device != "auto":
        model.device = args.device
    f1_af = (weights.get("train_metrics") or {}).get("af_val_f1")
    print(f"модель: {args.eval_model}  ({model.describe() if hasattr(model, 'describe') else ''})")
    print(f"постобработка из модели: {model.describe()}")
    for title, ids in (("отложенная часть (честно)", bs_val),
                       ("обучающие чипы (обе модели видели их)", bs_train),
                       ("все чипы (оптимистично)", bs_train + bs_val)):
        metrics = evaluate_bs(dataset, ids, bs_params, severity_model=model)
        iou, miou = metrics["iou_burn"], metrics["miou_sev"]
        line = (f"{title:28s} n={len(ids):3d}  IoU_burn={iou:.4f}  mIoU_sev={miou:.4f}  "
                f"объектив={0.5 * iou + 0.5 * miou:.5f}")
        if f1_af is not None:
            line += f"  Score≈{0.35 * float(f1_af) + 0.65 * (0.5 * iou + 0.5 * miou):.5f}"
        print("    " + line)
    if f1_af is None:
        print("    (F1 активного горения в weights.json не найден — Score не считается)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Подбор порогов решения «КосмоФАЙР-2»")
    ap.add_argument("--data-dir", default=None, help="каталог обучающей части (не нужен с --gpu-report)")
    ap.add_argument("--out", default="weights/weights.json")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--fast", action="store_true", help="уменьшенные сетки поиска (для тестов и черновых прогонов)")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                    help="где обучать модель степени поражения: auto — GPU (xgboost), если он есть")
    ap.add_argument("--backend", choices=("auto", "sklearn", "xgboost"), default="auto",
                    help="auto: xgboost на GPU, sklearn на CPU")
    ap.add_argument("--features", choices=("base", "ext"), default="base",
                    help="base — 23 признака (как в первой версии модели), ext — расширенный набор")
    ap.add_argument("--smooth", type=int, default=0,
                    help="радиус усреднения вероятностей модели по окружению (0 — выключено)")
    ap.add_argument("--majority", type=int, default=0,
                    help="радиус фильтра большинства по карте классов (0 — выключено)")
    ap.add_argument("--eval-model", default=None,
                    help="сравнить сохранённую модель степени с текущей: метрики BS на отложенной "
                         "части и на всех чипах, без обучения")
    ap.add_argument("--gpu-report", action="store_true",
                    help="показать, что известно про GPU, и выйти (диагностика CUDA)")
    args = ap.parse_args(argv)

    if args.gpu_report:
        import json as _json

        info = gpu_report()
        info["устройство"] = resolve_device("auto")
        print(_json.dumps(info, ensure_ascii=False, indent=2))
        if not info.get("runtime_ok"):
            broken = [name for name, state in (info.get("cuda_libraries") or {}).items()
                      if not str(state).startswith("загружается")]
            print("\nЧто делать:")
            if not info.get("nvidia_smi"):
                print("  * драйвера не видно (nvidia-smi нет в PATH): поставьте драйвер NVIDIA "
                      "для RTX 3060 и перезагрузитесь")
            if broken:
                print("  * не загружаются библиотеки: " + ", ".join(broken))
                print("    py -3 -m pip install --upgrade nvidia-cuda-runtime-cu12 "
                      "nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12")
                print("    затем повторите py -3 train.py --gpu-report (каталоги этих пакетов решение "
                      "подхватывает само)")
            print("  * точную причину xgboost называет в поле reason выше — по нему и решайте;\n"
                  "    если библиотеки на месте и драйвер виден, а GPU всё равно не берётся, "
                  "остаётся обновить драйвер или поставить CUDA Toolkit 12")
        return 0

    if not args.data_dir:
        ap.error("нужен --data-dir (или запустите с --gpu-report, чтобы только проверить GPU)")

    # Защита от порчи весов: путь мог прийти испорченным кавычками (cmd легко склеивает
    # «""C:\путь" --eval-model ...» в один аргумент), а обучение на нуле чипов раньше
    # спокойно дописывало weights.json пустым конфигом — и модель «терялась».
    if not Path(args.data_dir).exists():
        print(f"каталог не найден: {args.data_dir!r}\n"
              f"    веса не тронуты. Проверьте кавычки: --data-dir \"C:\\путь\\train\"\n"
              f"    если в пути есть пробелы, кавычки нужны одинарные, без удвоения")
        return 1

    if args.fast:
        AF_GRID.update({k: v[:2] for k, v in AF_GRID.items()})
        BS_GRID.update({k: v[:2] for k, v in BS_GRID.items()})

    dataset = Dataset(args.data_dir)
    if not dataset.chip_ids():
        print(f"в каталоге {args.data_dir!r} не нашлось ни одного чипа — веса не тронуты.\n"
              f"    что видит решение: py -3 -m fires.chips --data-dir \"{args.data_dir}\"\n"
              f"    нужен каталог, внутри которого папки af и bs с чипами и meta.csv")
        return 1
    af_train, af_val = _split_ids(dataset, "af", args.val_frac, args.seed)
    bs_train, bs_val = _split_ids(dataset, "bs", args.val_frac, args.seed)

    if args.eval_model:
        # сравнение кандидатов без обучения: нужно, чтобы выбрать модель для сабмита по числам,
        # а не по памяти («какая из них дала лучший Score на платформе»)
        return _eval_model(dataset, args, bs_train, bs_val)

    print("=== Модуль детекции активного горения (AF) ===")
    baseline_af = evaluate_af(dataset, af_val, af_module.AFParams())
    print(f"baseline (значения по умолчанию): F1={baseline_af['f1_af']} на {baseline_af['n']} чипах")
    af_result = tune_af(dataset, af_train, af_val)
    best_af = af_result["rules"]
    blob_result = af_result.get("blobs", {})

    print("=== Модуль картирования гарей (BS) ===")
    baseline_bs = evaluate_bs(dataset, bs_val, bs_module.BSParams())
    print(f"baseline: IoU_burn={baseline_bs['iou_burn']} mIoU_sev={baseline_bs['miou_sev']} на {baseline_bs['n']} чипах")
    device = resolve_device(args.device)
    report = gpu_report()
    if device == "cuda":
        print(f"=== обучение на GPU: xgboost {report['xgboost']}, сборка с CUDA, устройство проверено ===")
    elif report["xgboost"] is None:
        print("=== GPU-путь выключен: xgboost не установлен (py -3 -m pip install xgboost) ===")
    else:
        print(f"=== обучение на CPU: {report['reason']} ===")
    best_bs = tune_bs(dataset, bs_train, bs_val, device=device, backend=args.backend,
                      features=args.features)

    model_path = None
    if af_result["mode"] == "blobs" and blob_result.get("model") is not None:
        model_path = Path(args.out).with_name("blob_filter.pkl")
        blob_result["model"].save(model_path)
        print(f"классификатор очагов сохранён: {model_path}")

    # Прежнюю конфигурацию откладываем в сторону: обучение перезаписывает и веса, и модель,
    # а вернуться к предыдущему состоянию иначе нечем (цикл так делает, train.py — тоже должен).
    weights_path = Path(args.out)
    if weights_path.exists():
        weights_path.with_name(weights_path.stem + ".prev.json").write_text(
            weights_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"прежние веса сохранены: {weights_path.with_name(weights_path.stem + '.prev.json')}")
    previous_model = weights_path.with_name("severity_model.pkl")
    if previous_model.exists():
        backup = weights_path.with_name("severity_model.prev.pkl")
        backup.write_bytes(previous_model.read_bytes())
        print(f"прежняя модель сохранена: {backup}")

    bs_model_path = None
    if best_bs.get("mode") == "model" and best_bs.get("severity_model") is not None:
        bs_model_path = Path(args.out).with_name("severity_model.pkl")
        model = best_bs["severity_model"]
        model.proba_smooth = int(args.smooth)
        model.majority_filter = int(args.majority)
        model.save(bs_model_path)
        print(f"модель степени поражения сохранена: {bs_model_path} ({model.describe()})")

    weights = {
        "version": 2,
        "seed": args.seed,
        "af_mode": af_result["mode"],
        "blob_filter": None if model_path is None else str(model_path.name),
        "bs_mode": best_bs.get("mode", "thresholds"),
        "severity_model": None if bs_model_path is None else str(bs_model_path.name),
        "af": best_af.get("params", {}),
        "bs": best_bs.get("params", {}),
        "train_metrics": {
            "af_val_f1": best_af.get("f1_af"),
            "af_train_f1": best_af.get("train_f1_af"),
            "af_baseline_f1": baseline_af.get("f1_af"),
            "bs_val_iou_burn": best_bs.get("iou_burn"),
            "bs_val_miou_sev": best_bs.get("miou_sev"),
            "bs_baseline_iou_burn": baseline_bs.get("iou_burn"),
            "bs_baseline_miou_sev": baseline_bs.get("miou_sev"),
            "bs_smooth": best_bs.get("smooth"),
            "bs_severity_model": None if best_bs.get("severity_model") is None else best_bs["severity_model"].metrics,
            "af_blob_filter_val_f1": blob_result.get("f1_af"),
            "af_blob_filter_features_top": blob_result.get("importances"),
            "af_chosen_mode": af_result["mode"],
            "bs_sar_selection": best_bs.get("sar"),
        },
        "split": {"af_train": len(af_train), "af_val": len(af_val), "bs_train": len(bs_train), "bs_val": len(bs_val)},
    }
    # Флаги, подобранные вне сетки tune_af, полным прогоном организаторского метрика
    # на всех 420 очагах: F1 0.9195 против 0.9093 у сеточного победителя (см. change/).
    weights["af"].update({"exclude_built": False, "isolated_guard": False})
    # Смещения вероятностей классов перед argmax; устойчивы на трёх сидах сплита
    # событий пожара (+0.002..+0.007 Score по BS-части).
    weights["severity_class_bias"] = [0.05, -0.15, -0.06, 0.0]
    path = save_weights(weights, args.out)
    print(f"веса сохранены: {path}")
    print(json.dumps(weights, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
