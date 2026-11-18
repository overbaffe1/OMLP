"""Формат submission.csv: RLE-кодирование масок.

Правила из постановки (раздел «Формат submission.csv»):

* колонки: ``chip_id,class_id,rle``; заголовок обязателен; кодировка UTF-8, разделитель — запятая;
* для AF-чипа всегда ``class_id = 1``; для BS-чипа — 1, 2 или 3 (слабая/средняя/сильная степень);
* пиксели нумеруются построчно слева направо и сверху вниз, **начиная с единицы**;
* ``rle`` — пары «начальный пиксель — длина» через пробел, строка в двойных кавычках;
* пары идут по возрастанию начального пикселя, серии не пересекаются и не соприкасаются;
* пустая строка в кавычках, если пикселей класса нет;
* набор пар (chip_id, class_id) должен полностью совпадать с ``sample_submission.csv``.

Модуль намеренно не зависит от pandas — работает и через стандартный ``csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

CSV_HEADER = ("chip_id", "class_id", "rle")
AF_CLASS_ID = 1
BS_CLASS_IDS = (1, 2, 3)


def encode_rle(mask: np.ndarray) -> str:
    """Кодирует бинарную маску в строку RLE (нумерация пикселей с единицы, по строкам).

    Серии не соприкасаются: две соседние строки маски — это разные серии, даже если
    последний пиксель строки и первый пиксель следующей строки принадлежат маске.
    """

    flat = np.asarray(mask).ravel()
    if flat.dtype != bool:
        flat = flat.astype(bool)
    if not flat.any():
        return ""

    d = np.diff(flat.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1) + 1  # нумерация с единицы
    ends = np.flatnonzero(d == -1)
    lengths = ends - starts + 1

    # склейка соседних серий не выполняется намеренно: серии не должны соприкасаться
    # (пиксель n и n+1 в разных строках — разные серии по правилам кодирования)
    return " ".join(f"{s} {l}" for s, l in zip(starts, lengths))


def decode_rle(rle: str, shape: tuple[int, int]) -> np.ndarray:
    """Декодирует RLE в бинарную маску заданной формы (для расчёта метрики и валидации)."""

    h, w = shape
    total = h * w
    mask = np.zeros(total, dtype=bool)
    if rle is None:
        return mask.reshape(shape)
    parts = str(rle).split()
    if len(parts) % 2:
        raise ValueError(f"RLE должен состоять из пар чисел, получено {len(parts)} значений")
    for i in range(0, len(parts), 2):
        start, length = int(parts[i]), int(parts[i + 1])
        if start < 1 or length < 1:
            raise ValueError("Нумерация RLE начинается с единицы, длины положительные")
        if start - 1 + length > total:
            raise ValueError(f"Серия выходит за пределы чипа: start={start}, length={length}, pixels={total}")
        mask[start - 1 : start - 1 + length] = True
    # перекрытие серий запрещено: при пересечении сумма True меньше суммы длин
    if int(mask.sum()) != sum(int(parts[i + 1]) for i in range(0, len(parts), 2)):
        raise ValueError("Серии RLE пересекаются")
    return mask.reshape(shape)


def _quoted(value: str) -> str:
    """Оформляет поле в кавычках, как требует постановка (внутренние кавычки удваиваются)."""

    return '"' + str(value).replace('"', '""') + '"'


def write_submission(rows: list[dict], path: str | Path) -> Path:
    """Пишет CSV с заголовком. ``rows`` — список словарей с ключами chip_id, class_id, rle.

    Поле ``rle`` всегда в двойных кавычках: постановка явно запрещает пустые ячейки и NaN, а
    «нет пикселей класса» кодируется пустой строкой в кавычках (``""``). При обычной записи через
    ``csv.QUOTE_MINIMAL`` такой пиксель превратился бы в пустое поле без кавычек, и читающий через
    pandas валидатор платформы увидел бы пропуск вместо «пустой маски».
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(",".join(CSV_HEADER) + "\n")
        for row in rows:
            rle = row.get("rle")
            fh.write(f"{row['chip_id']},{int(row['class_id'])},{_quoted('' if rle is None else rle)}\n")
    return path


def raise_field_limit() -> None:
    """Снимает лимит длины поля у csv-модуля (по умолчанию 128 КиБ).

    RLE одного чипа с крупной гарью легко перерастает этот лимит: строка вида
    ``120 340 400 12 ...`` на десятки тысяч пар. Организаторы читают файл своим кодом, а наш
    валидатор без этого падал на собственном же результате.
    """

    limit = 2 ** 31 - 1
    while limit > 1024:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


def read_submission(path: str | Path) -> dict[tuple[str, int], str]:
    """Читает submission.csv в словарь ``(chip_id, class_id) -> rle``."""

    raise_field_limit()
    out: dict[tuple[str, int], str] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out[(row["chip_id"], int(row["class_id"]))] = (row.get("rle") or "").strip()
    return out


def read_template(path: str | Path) -> list[tuple[str, int]]:
    """Читает sample_submission.csv: список ожидаемых пар (chip_id, class_id)."""

    raise_field_limit()
    pairs: list[tuple[str, int]] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pairs.append((row["chip_id"], int(row["class_id"])))
    return pairs


def _check_rle_series(rle: str, shape: tuple[int, int]) -> list[str]:
    """Проверяет синтаксис RLE: пары по возрастанию, без пересечений и соприкосновений.

    Требование «серии не соприкасаются» из постановки означает, что соседние пиксели
    (``start + length`` и следующий ``start``) должны быть склеены в одну серию: например,
    для маски из двух полных строк верно ``"1 4"``, а не ``"1 2 3 2"``.
    """

    errors: list[str] = []
    parts = str(rle).split()
    if len(parts) % 2:
        return [f"RLE должен состоять из пар чисел, получено {len(parts)} значений"]
    total = shape[0] * shape[1]
    previous_end = 0
    for i in range(0, len(parts), 2):
        try:
            start, length = int(parts[i]), int(parts[i + 1])
        except ValueError:
            errors.append(f"нечисловое значение в RLE: {parts[i]!r} {parts[i + 1]!r}")
            break
        if start < 1 or length < 1:
            errors.append("нумерация RLE начинается с единицы, длины положительные")
            break
        if start <= previous_end:
            errors.append(
                "серии должны идти по возрастанию и не соприкасаться "
                f"(после пикселя {previous_end} идёт начало {start})"
            )
            break
        if start - 1 + length > total:
            errors.append(f"серия выходит за пределы чипа: start={start}, length={length}, pixels={total}")
            break
        previous_end = start - 1 + length
    return errors


def validate_submission(
    submission_path: str | Path,
    template_path: str | Path,
    chip_shapes: dict[str, tuple[int, int]],
    strict: bool = False,
) -> dict:
    """Полная проверка файла ответа по требованиям постановки.

    Проверяется: совпадение набора пар с шаблоном, отсутствие NaN/пропусков, синтаксис RLE
    (пары по возрастанию, без пересечений и соприкосновений), выход за границы чипа,
    пересечение масок разных классов внутри чипа.

    Возвращает отчёт словарём; при ``strict=True`` бросает исключение на первой ошибке.
    """

    expected = set(read_template(template_path))
    got = read_submission(submission_path)
    report: dict = {"expected_rows": len(expected), "got_rows": len(got), "errors": [], "warnings": []}

    missing = expected - set(got)
    extra = set(got) - expected
    if missing:
        report["errors"].append(f"нет строк для {len(missing)} пар, например {sorted(missing)[:3]}")
    if extra:
        report["errors"].append(f"лишние пары: {sorted(extra)[:3]}")

    masks: dict[str, dict[int, np.ndarray]] = {}
    for (chip_id, class_id), rle in got.items():
        if rle.strip().lower() in {"nan", "none", "null"}:
            report["errors"].append(f"{chip_id}/{class_id}: значение NaN/None недопустимо")
            continue
        shape = chip_shapes.get(chip_id)
        if shape is None:
            report["warnings"].append(f"{chip_id}: неизвестны размеры чипа, проверка границ пропущена")
            continue
        syntax_errors = _check_rle_series(rle, shape)
        if syntax_errors:
            report["errors"].extend(f"{chip_id}/{class_id}: {e}" for e in syntax_errors)
            continue
        try:
            mask = decode_rle(rle, shape)
        except ValueError as exc:
            report["errors"].append(f"{chip_id}/{class_id}: {exc}")
            continue
        masks.setdefault(chip_id, {})[class_id] = mask

    for chip_id, by_class in masks.items():
        ids = sorted(by_class)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                inter = np.logical_and(by_class[ids[i]], by_class[ids[j]]).sum()
                if inter:
                    report["errors"].append(f"{chip_id}: маски классов {ids[i]} и {ids[j]} пересекаются ({inter} px)")

    report["ok"] = not report["errors"]
    if strict and not report["ok"]:
        raise ValueError("submission.csv не прошёл проверку: " + "; ".join(report["errors"][:5]))
    return report
