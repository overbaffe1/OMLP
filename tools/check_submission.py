#!/usr/bin/env python3
"""Проверка `submission.csv` без эталона: формат, полнота, правдоподобие.

    python tools/check_submission.py --data-dir ~/firesdata/test --submission ~/firesdata/test/submission.csv
    python tools/check_submission.py --data-dir /tmp/firesdata/real/train --submission /tmp/train_submission.csv

Что проверяется:

1. **Формат** — ровно шапка ``chip_id,class_id,rle``, поле RLE в двойных кавычках, числовой ``class_id``.
2. **Полнота** — набор пар ``(chip_id, class_id)`` совпадает с шаблоном (для теста — ``sample_submission.csv``,
   для набора без шаблона он собирается из чипов так же, как это делает ``inference.py``).
3. **Корректность RLE** — пары по возрастанию, серии не пересекаются и не соприкасаются, выход за границы чипа.
4. **Непересечение классов** — внутри одного чипа пиксель принадлежит ровно одному классу.
5. **Правдоподобие** — сколько пикселей горения на AF-чип и какая доля площади горит на BS-чип по сравнению с
   прогоном на обучающей части (эталонные значения ниже). Это не метрика, а детектор грубых сбоев: например,
   «пустой» ответ или ответ, где горит половина каждого чипа.

Итоговый Score на тесте локально посчитать нельзя — для него нужны скрытые маски организаторов. Точную оценку
даёт только платформа после загрузки файла.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fires.chips import Dataset  # noqa: E402
from fires.submission import decode_rle, read_template  # noqa: E402

#: Эталонные значения с обучающей части (644 чипа, прогон inference.py на weights/weights.json):
#: AF — 9 478 предсказанных пикселей горения на 420 чипов; BS — 7 749 993 px (13.2 % площади) на 224 чипа.
REFERENCE = {
    "af_px_per_chip": 22.6,
    "bs_burn_fraction": 0.132,
}
#: допустимое отклонение от эталона, при котором выдаётся предупреждение (не ошибка)
TOLERANCE = 2.0

CSV_HEADER = ["chip_id", "class_id", "rle"]


def _kind(chip_id: str) -> str:
    return "af" if chip_id.upper().startswith("AF") else "bs"


def _check_raw(path: Path) -> list[str]:
    """Проверяет «сырой» текст файла: шапку, кавычки, пустые ячейки."""

    errors: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if number == 1:
            if line.strip() != ",".join(CSV_HEADER):
                errors.append(f"строка 1: ожидалась шапка {','.join(CSV_HEADER)}, получено {line.strip()!r}")
            continue
        parts = line.split(",")
        if len(parts) != 3:
            errors.append(f"строка {number}: ожидалось 3 поля, получено {len(parts)}")
            continue
        if not re.fullmatch(r'"[^"]*"', parts[2]):
            errors.append(f"строка {number}: поле rle должно быть строкой в двойных кавычках, получено {parts[2]!r}")
        if not re.fullmatch(r"-?\d+", parts[1]):
            errors.append(f"строка {number}: class_id должен быть целым, получено {parts[1]!r}")
    return errors


def _check_pairs(rows: list[dict], expected: list[tuple[str, int]]) -> tuple[list[str], list[str]]:
    """Сверяет набор пар с шаблоном; возвращает (ошибки, предупреждения)."""

    errors: list[str] = []
    warnings: list[str] = []
    got = [(row["chip_id"], int(row["class_id"])) for row in rows]
    if len(got) != len(set(got)):
        duplicates = sorted({p for p in got if got.count(p) > 1})
        errors.append(f"повторяющиеся пары (chip_id, class_id): {duplicates[:5]}")
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    if missing:
        errors.append(f"нет строк для {len(missing)} пар, например {missing[:3]}")
    if extra:
        errors.append(f"лишние пары: {extra[:3]}")
    if errors:
        return errors, warnings
    if [p for p in got] != list(expected):
        warnings.append("порядок строк отличается от шаблона (постановка это допускает)")
    return errors, warnings


def _decode_checked(row: dict, shape: tuple[int, int]) -> tuple[np.ndarray | None, list[str]]:
    """Декодирует RLE строки в маску чипа (``decode_rle`` сам ловит пересечения и выход за границы)."""

    try:
        mask = decode_rle(row["rle"], shape)
    except Exception as exc:
        return None, [f"{row['chip_id']} класс {row['class_id']}: RLE не читается ({exc})"]
    return mask, []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Проверка submission.csv без эталона")
    ap.add_argument("--data-dir", required=True, help="каталог набора (там же ищется sample_submission.csv)")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--template", default=None, help="шаблон ответа, если он лежит не рядом с данными")
    ap.add_argument("--json", action="store_true", help="машинный вывод в конце")
    args = ap.parse_args(argv)

    submission = Path(args.submission)
    if not submission.exists():
        print(f"нет файла {submission}", file=sys.stderr)
        return 1

    dataset = Dataset(args.data_dir)
    shapes: dict[str, tuple[int, int]] = {}
    for chip_id in dataset.chip_ids("af") + dataset.chip_ids("bs"):
        row = dataset.meta.get(chip_id, {})
        height, width = int(row.get("height") or 0), int(row.get("width") or 0)
        if height and width:
            shapes[chip_id] = (height, width)

    template_path = Path(args.template) if args.template else Path(args.data_dir) / "sample_submission.csv"
    if template_path.exists():
        expected = read_template(template_path)
        source = str(template_path)
    else:
        expected = [(chip_id, 1) for chip_id in dataset.chip_ids("af")] + [
            (chip_id, cls) for chip_id in dataset.chip_ids("bs") for cls in (1, 2, 3)
        ]
        source = "собран из чипов (шаблона рядом нет)"

    rows = list(csv.DictReader(submission.open(newline="", encoding="utf-8")))
    errors = _check_raw(submission)
    pair_errors, warnings = _check_pairs(rows, expected)
    errors += pair_errors

    # --- декодирование, границы и непересечение классов
    stats = {
        "af": {"chips": set(), "px": 0, "empty": 0, "rows": 0},
        "bs": {"chips": set(), "px": 0, "empty": 0, "rows": 0, "by_class": {}},
    }
    occupied: dict[str, np.ndarray] = {}
    for row in rows:
        chip_id = row["chip_id"]
        kind = _kind(chip_id)
        shape = shapes.get(chip_id)
        stats[kind]["rows"] += 1
        if shape is None:
            warnings.append(f"{chip_id}: нет размеров чипа в meta.csv — RLE не проверяется")
            continue
        stats[kind]["chips"].add(chip_id)
        mask, decode_errors = _decode_checked(row, shape)
        errors += decode_errors
        if mask is None:
            continue
        count = int(mask.sum())
        if count == 0:
            stats[kind]["empty"] += 1
            continue
        stats[kind]["px"] += count
        if kind == "bs":
            stats["bs"]["by_class"][row["class_id"]] = stats["bs"]["by_class"].get(row["class_id"], 0) + count
        canvas = occupied.setdefault(chip_id, np.zeros((shape[0], shape[1]), dtype=np.uint8))
        if np.any((canvas != 0) & mask):
            errors.append(f"{chip_id} класс {row['class_id']}: маски классов пересекаются")
        canvas[mask] = int(row["class_id"])

    # --- правдоподобие
    verdicts: list[str] = []
    af_chips = len(stats["af"]["chips"]) or 1
    bs_chips = len(stats["bs"]["chips"]) or 1
    af_per_chip = stats["af"]["px"] / af_chips
    bs_px_per_chip = stats["bs"]["px"] / bs_chips
    bs_pixel_area = float(np.mean([shapes[c][0] * shapes[c][1] for c in stats["bs"]["chips"]]) if stats["bs"]["chips"] else 262144)
    bs_fraction = bs_px_per_chip / bs_pixel_area

    for label, value, reference in (
        ("пикселей горения на AF-чип", af_per_chip, REFERENCE["af_px_per_chip"]),
        ("доля горящей площади на BS-чип", bs_fraction, REFERENCE["bs_burn_fraction"]),
    ):
        ratio = value / reference if reference else 0.0
        mark = "ok"
        if ratio < 1 / TOLERANCE or ratio > TOLERANCE:
            mark = "ВНИМАНИЕ"
            warnings.append(
                f"{label}: {value:.3g} против {reference:.3g} на обучающей части — отличается в {ratio:.2f} раза"
            )
        verdicts.append(f"{label}: {value:.3g} (на train {reference:.3g}, отношение {ratio:.2f}) — {mark}")

    if not stats["bs"]["by_class"]:
        warnings.append("в ответе нет ни одного пикселя гари — проверьте, что --data-dir указывает на тестовый набор")

    report = {
        "submission": str(submission),
        "template": source,
        "rows": len(rows),
        "expected_rows": len(expected),
        "af": {
            "chips": len(stats["af"]["chips"]),
            "fire_px": stats["af"]["px"],
            "px_per_chip": round(af_per_chip, 2),
            "empty_rows": stats["af"]["empty"],
        },
        "bs": {
            "chips": len(stats["bs"]["chips"]),
            "burned_px": stats["bs"]["px"],
            "px_per_chip": round(bs_px_per_chip, 1),
            "burn_fraction": round(bs_fraction, 4),
            "empty_rows": stats["bs"]["empty"],
            "by_class": stats["bs"]["by_class"],
        },
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }

    print(f"файл: {submission}")
    print(f"шаблон: {source} — строк {len(rows)} из {len(expected)}")
    print(
        f"AF: {len(stats['af']['chips'])} чипов, {stats['af']['px']} px горения "
        f"({af_per_chip:.2f} на чип), пустых строк {stats['af']['empty']}"
    )
    print(
        f"BS: {len(stats['bs']['chips'])} чипов, {stats['bs']['px']} px гари ({bs_fraction * 100:.2f} % площади), "
        f"пустых строк {stats['bs']['empty']}"
    )
    if stats["bs"]["by_class"]:
        total = sum(stats["bs"]["by_class"].values()) or 1
        shares = ", ".join(f"класс {k}: {v / total * 100:.1f} %" for k, v in sorted(stats["bs"]["by_class"].items()))
        print(f"доли по степеням: {shares}")
    print("правдоподобие (не метрика, а проверка на грубые сбои):")
    for line in verdicts:
        print(f"  - {line}")
    for line in warnings:
        print(f"[warn] {line}")
    for line in errors:
        print(f"[err]  {line}")
    print("ИТОГ:", "формат и полнота в порядке" if not errors else f"найдено ошибок: {len(errors)}")
    print("(Score локально не считается: нужны скрытые маски организаторов — оценку даёт платформа после загрузки)")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
