"""Тесты кодека RLE и валидации файла решения."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from fires.submission import (
    decode_rle,
    encode_rle,
    read_submission,
    read_template,
    validate_submission,
    write_submission,
)


def test_rle_roundtrip_simple() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[1, 1:3] = True
    rle = encode_rle(mask)
    assert rle == "1 1 6 2"
    assert np.array_equal(decode_rle(rle, mask.shape), mask)


def test_rle_merges_series_across_rows() -> None:
    """Соприкасающиеся серии склеиваются: иначе нарушается требование «без соприкосновений»."""

    mask = np.zeros((2, 2), dtype=bool)
    mask[:] = True
    assert encode_rle(mask) == "1 4"
    assert np.array_equal(decode_rle("1 4", mask.shape), mask)
    assert np.array_equal(decode_rle("1 2 3 2", mask.shape), mask)  # семантика та же


def test_rle_empty_and_full_mask() -> None:
    shape = (3, 3)
    assert encode_rle(np.zeros(shape, dtype=bool)) == ""
    assert decode_rle("", shape).sum() == 0
    assert decode_rle(None, shape).sum() == 0
    full = encode_rle(np.ones(shape, dtype=bool))
    assert full == "1 9"
    assert decode_rle(full, shape).all()


def test_rle_invalid_input() -> None:
    with pytest.raises(ValueError):
        decode_rle("1", (2, 2))
    with pytest.raises(ValueError):
        decode_rle("0 1", (2, 2))
    with pytest.raises(ValueError):
        decode_rle("4 5", (2, 2))
    with pytest.raises(ValueError):
        decode_rle("1 3 2 1", (2, 2))  # серии пересекаются


def test_write_and_read_submission(tmp_path) -> None:
    rows = [
        {"chip_id": "af_1", "class_id": 1, "rle": "1 2"},
        {"chip_id": "bs_1", "class_id": 1, "rle": ""},
        {"chip_id": "bs_1", "class_id": 2, "rle": "5 2"},
        {"chip_id": "bs_1", "class_id": 3, "rle": ""},
    ]
    path = write_submission(rows, tmp_path / "submission.csv")
    parsed = read_submission(path)
    assert parsed[("af_1", 1)] == "1 2"
    assert parsed[("bs_1", 3)] == ""
    with path.open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == ["chip_id", "class_id", "rle"]


def test_empty_rle_is_quoted(tmp_path) -> None:
    """Постановка требует пустую строку в кавычках, а не пустую ячейку (иначе это «пропуск»)."""

    rows = [
        {"chip_id": "af_1", "class_id": 1, "rle": ""},
        {"chip_id": "bs_1", "class_id": 2, "rle": "1 4 10 2"},
    ]
    path = write_submission(rows, tmp_path / "submission.csv")
    text = path.read_text(encoding="utf-8")
    assert 'af_1,1,""' in text, "пустая маска должна быть пустой строкой в кавычках"
    assert 'bs_1,2,"1 4 10 2"' in text, "RLE всегда в кавычках"
    assert ",,\n" not in text and not text.rstrip("\n").endswith(","), "пустых ячеек быть не должно"


def test_validate_submission_detects_errors(tmp_path) -> None:
    template = tmp_path / "sample_submission.csv"
    template.write_text("chip_id,class_id,rle\naf_1,1,\nbs_1,1,\nbs_1,2,\n", encoding="utf-8")
    submission = write_submission(
        [
            {"chip_id": "af_1", "class_id": 1, "rle": "1 1 2 1"},  # серии соприкасаются
            {"chip_id": "bs_1", "class_id": 1, "rle": "1 700000"},  # выход за границы чипа
            {"chip_id": "bs_1", "class_id": 2, "rle": "3 1 1 1"},  # порядок нарушен
        ],
        tmp_path / "submission.csv",
    )
    shapes = {"af_1": (4, 4), "bs_1": (512, 512)}
    report = validate_submission(submission, template, shapes)
    assert not report["ok"]
    text = " ".join(report["errors"])
    assert "не соприкасаться" in text
    assert "за пределы" in text
    assert "по возрастанию" in text
    assert read_template(template) == [("af_1", 1), ("bs_1", 1), ("bs_1", 2)]


def test_validate_submission_ok(tmp_path) -> None:
    template = tmp_path / "sample_submission.csv"
    template.write_text("chip_id,class_id,rle\naf_1,1,\nbs_1,1,\nbs_1,2\n", encoding="utf-8")
    submission = write_submission(
        [
            {"chip_id": "af_1", "class_id": 1, "rle": encode_rle(np.array([[True, False, True, False]]))},
            {"chip_id": "bs_1", "class_id": 1, "rle": ""},
            {"chip_id": "bs_1", "class_id": 2, "rle": ""},
        ],
        tmp_path / "submission.csv",
    )
    report = validate_submission(submission, template, {"af_1": (1, 4), "bs_1": (512, 512)})
    assert report["ok"], report["errors"]
    assert report["got_rows"] == 3
