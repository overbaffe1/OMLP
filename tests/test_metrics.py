"""Тесты метрики соревнования: F1 по AF, IoU гари и mIoU по степеням поражения."""

from __future__ import annotations

import numpy as np
import pytest

from fires.metrics import f1_af, iou, score_from_pairs, score_from_submissions
from fires.submission import write_submission


def test_f1_and_iou_basics() -> None:
    assert f1_af(10, 0, 0) == 1.0
    assert iou(0, 0, 0) == 1.0  # пустой класс считается идеальным по правилам кейса
    assert f1_af(1, 1, 3) == pytest.approx(2 / 6)
    assert iou(3, 1, 1) == pytest.approx(0.6)
    # деление на ноль не должно ронять расчёт
    assert f1_af(0, 0, 5) == 0.0


def test_score_weights() -> None:
    pred = np.zeros((2, 2), dtype=bool)
    pred[0, 0] = True
    truth = pred.copy()
    peers = {"c1": (truth, pred)}
    result = score_from_pairs(peers, {})
    assert result["f1_af"] == 1.0
    assert result["score"] == pytest.approx(0.35 * 1.0 + 0.35 * 1.0 + 0.30 * 1.0)


def test_score_partial_and_empty_bs() -> None:
    truth = np.zeros((2, 2), dtype=bool)
    truth[0, :2] = True
    pred = np.zeros((2, 2), dtype=bool)
    pred[0, 0] = True
    score = score_from_pairs({"c1": (truth, pred)}, {})
    assert score["af_counts"] == {"tp": 1, "fp": 0, "fn": 1}
    assert score["f1_af"] == pytest.approx(2 / 3, abs=1e-4)
    # BS-часть отсутствует → по правилам кейса обе компоненты равны 1
    assert score["iou_burn"] == 1.0
    assert score["miou_sev"] == 1.0


def test_score_bs_multiclass() -> None:
    truth = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    pred = np.array([[0, 1, 3, 3]], dtype=np.uint8)
    score = score_from_pairs({}, {"b1": (truth, pred)})
    # классы: 1 — точное совпадение, 2 — пропуск, 3 — один пиксель верный и один лишний
    assert score["iou_per_class"][1] == 1.0
    assert score["iou_per_class"][2] == 0.0
    assert score["iou_per_class"][3] == pytest.approx(0.5)
    assert score["miou_sev"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    # бинарная «гарь»: эталон 3 px, предсказано 3 px, совпало 3
    assert score["iou_burn"] == 1.0
    assert score["burn_counts"]["tp"] == 3


def test_score_from_submissions_matches_pairs(tmp_path) -> None:
    af_truth = np.zeros((3, 3), dtype=bool)
    af_truth[1, 1] = True
    bs_truth = np.array([[0, 1, 1], [2, 2, 0], [3, 0, 3]], dtype=np.uint8)
    bs_pred = np.array([[0, 1, 1], [2, 3, 0], [3, 0, 3]], dtype=np.uint8)

    truth_path = write_submission(
        [
            {"chip_id": "af1", "class_id": 1, "rle": "5 1"},
            {"chip_id": "bs1", "class_id": 1, "rle": "2 2"},
            {"chip_id": "bs1", "class_id": 2, "rle": "4 2"},
            {"chip_id": "bs1", "class_id": 3, "rle": "7 1 9 1"},
        ],
        tmp_path / "truth.csv",
    )
    pred_path = write_submission(
        [
            {"chip_id": "af1", "class_id": 1, "rle": "5 1"},
            {"chip_id": "bs1", "class_id": 1, "rle": "2 2"},
            {"chip_id": "bs1", "class_id": 2, "rle": "4 1"},
            {"chip_id": "bs1", "class_id": 3, "rle": "5 1 7 1 9 1"},
        ],
        tmp_path / "pred.csv",
    )
    shapes = {"af1": (3, 3), "bs1": (3, 3)}
    from_submissions = score_from_submissions(pred_path, truth_path, shapes, {"af1": "af", "bs1": "bs"})
    from_pairs = score_from_pairs({"af1": (af_truth, af_truth)}, {"bs1": (bs_truth, bs_pred)})
    assert from_submissions["score"] == pytest.approx(from_pairs["score"])
