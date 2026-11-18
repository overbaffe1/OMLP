"""Метрика соревнования.

``Score = 0.35 * F1_af + 0.35 * IoU_burn + 0.30 * mIoU_sev``

Все компоненты считаются **микро-усреднением** по общему пулу пикселей: сначала суммируются
TP/FP/FN по всем чипам, потом считаются precision/recall/F1 и IoU. Особенности из постановки:

* если знаменатель равен нулю — класс отсутствует и в эталоне, и в предсказании — величина = 1;
* если класс есть в эталоне, но полностью отсутствует в предсказании — его IoU = 0;
* ``IoU_burn`` — по бинарному контуру гари (степень > 0);
* ``mIoU_sev`` — среднее IoU по трём классам степени поражения.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def _safe_div(num: float, den: float, default: float = 1.0) -> float:
    if den <= EPS:
        return default
    return num / den


def f1_af(tp: float, fp: float, fn: float) -> float:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if precision + recall <= EPS:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def iou(tp: float, fp: float, fn: float) -> float:
    if tp + fp + fn <= EPS:
        return 1.0
    return tp / (tp + fp + fn)


def score_from_pairs(
    af_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    bs_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Считает все компоненты метрики.

    :param af_pairs: ``chip_id -> (эталонная маска, предсказанная маска)`` (бинарные);
    :param bs_pairs: ``chip_id -> (эталонная маска, предсказанная маска)`` (0–3).
    """

    tp = fp = fn = 0.0
    for truth, pred in af_pairs.values():
        truth = np.asarray(truth).astype(bool)
        pred = np.asarray(pred).astype(bool)
        tp += float(np.logical_and(truth, pred).sum())
        fp += float(np.logical_and(~truth, pred).sum())
        fn += float(np.logical_and(truth, ~pred).sum())
    af_f1 = f1_af(tp, fp, fn)

    tp_b = fp_b = fn_b = 0.0
    class_tp = {1: 0.0, 2: 0.0, 3: 0.0}
    class_fp = {1: 0.0, 2: 0.0, 3: 0.0}
    class_fn = {1: 0.0, 2: 0.0, 3: 0.0}
    for truth, pred in bs_pairs.values():
        truth = np.asarray(truth)
        pred = np.asarray(pred)
        truth_b, pred_b = truth > 0, pred > 0
        tp_b += float(np.logical_and(truth_b, pred_b).sum())
        fp_b += float(np.logical_and(~truth_b, pred_b).sum())
        fn_b += float(np.logical_and(truth_b, ~pred_b).sum())
        for cls in (1, 2, 3):
            t, p = truth == cls, pred == cls
            class_tp[cls] += float(np.logical_and(t, p).sum())
            class_fp[cls] += float(np.logical_and(~t, p).sum())
            class_fn[cls] += float(np.logical_and(t, ~p).sum())

    iou_burn = iou(tp_b, fp_b, fn_b)
    per_class = {cls: iou(class_tp[cls], class_fp[cls], class_fn[cls]) for cls in (1, 2, 3)}
    miou_sev = float(np.mean(list(per_class.values())))
    score = 0.35 * af_f1 + 0.35 * iou_burn + 0.30 * miou_sev

    return {
        "score": round(float(score), 5),
        "f1_af": round(float(af_f1), 5),
        "iou_burn": round(float(iou_burn), 5),
        "miou_sev": round(float(miou_sev), 5),
        "af_counts": {"tp": int(tp), "fp": int(fp), "fn": int(fn)},
        "burn_counts": {"tp": int(tp_b), "fp": int(fp_b), "fn": int(fn_b)},
        "iou_per_class": {cls: round(float(v), 5) for cls, v in per_class.items()},
    }


def score_from_submissions(pred_path, truth_path, shapes: dict[str, tuple[int, int]], kinds: dict[str, str]) -> dict:
    """Считает метрику по двум файлам submission.csv (предсказание и эталон).

    :param shapes: размеры чипов ``chip_id -> (h, w)``;
    :param kinds: типы чипов ``chip_id -> 'af' | 'bs'``.
    """

    from fires.submission import decode_rle, read_submission

    pred = read_submission(pred_path)
    truth = read_submission(truth_path)
    af_pairs, bs_pairs = {}, {}
    for (chip_id, class_id), true_rle in truth.items():
        shape = shapes[chip_id]
        kind = kinds.get(chip_id, "af")
        if kind == "af":
            af_pairs.setdefault(chip_id, (np.zeros(shape, bool), np.zeros(shape, bool)))
            af_pairs[chip_id] = (decode_rle(true_rle, shape), decode_rle(pred.get((chip_id, class_id), ""), shape))
        else:
            bs_pairs.setdefault(chip_id, (np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)))
            t = bs_pairs[chip_id][0]
            p = bs_pairs[chip_id][1]
            t = np.where(decode_rle(true_rle, shape), class_id, t).astype(np.uint8)
            p = np.where(decode_rle(pred.get((chip_id, class_id), ""), shape), class_id, p).astype(np.uint8)
            bs_pairs[chip_id] = (t, p)
    return score_from_pairs(af_pairs, bs_pairs)
