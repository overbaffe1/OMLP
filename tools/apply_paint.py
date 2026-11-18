#!/usr/bin/env python3
"""Мерж кисти с карты (paint_overlay.json) в submission + точный скор.

    py -3 tools/apply_paint.py --overlay paint_overlay.json \
        --submission submission_probe_35.csv [--out submission_test.csv] \
        [--score] [--score-dir <путь>/train]

Оверлей формируется кнопкой «csv (тест)» на карте:
``{"overlay": {"BS_te_000040": [[u, v, deg], ...]}}`` — координаты пикселя (0-based,
0..511) и степень EFFIS 0/1/2/3, где 0 — стёрка: удалить пиксель из ответа чемпиона.

* Точки на чипах, которые ЕСТЬ в сабмите (test), ЗАМЕНЯЮТ ответ чемпионки на этих
  пикселях (в т.ч. перекрашивают степень); остальные строки не трогаются, AF нетронут.
* Точки на чипах, которых в сабмите нет (train-вкладка: BS_tr_*), в CSV не попадают
  (платформа принимает только test), но учитываются в ``--score``: кисть сравнивается
  с mask_sev напрямую — это песочница, чтобы прикинуть скор своей разметки.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fires.chips import Dataset, find_data_dir  # noqa: E402
from fires.metrics import score_from_pairs  # noqa: E402
from fires.submission import (  # noqa: E402
    decode_rle,
    encode_rle,
    raise_field_limit,
    read_submission,
    write_submission,
)

SIZE = (512, 512)


def parse_overlay(path: str) -> dict[str, list[list[int]]]:
    parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    overlay = parsed.get("overlay") if isinstance(parsed, dict) and "overlay" in parsed else parsed
    return overlay or {}


def brush_mask(dots: list[list[int]], cid: str) -> np.ndarray:
    """Растр кисти (uint8, 512x512): 0 — пусто, 1..3 — степень, 4 — стёрка пикселя чемпиона."""
    seen: dict[int, int] = {}
    for u, v, deg in dots:
        if deg not in (0, 1, 2, 3):
            raise ValueError(f"{cid}: неизвестная степень {deg}")
        if not (0 <= u <= 511 and 0 <= v <= 511):
            raise ValueError(f"{cid}: точка ({u},{v}) вне чипа")
        idx = v * 512 + u
        if idx in seen and seen[idx] != deg:
            xy = (idx % 512, idx // 512)
            raise ValueError(f"{cid}: точка {xy} нарисована дважды с разными степенями")
        seen[idx] = deg
    out = np.zeros(SIZE[0] * SIZE[1], dtype=np.uint8)
    for idx, deg in seen.items():
        out[idx] = 4 if deg == 0 else deg
    return out


def apply_overlay(base: dict[tuple[str, int], str],
                  overlay: dict[str, list[list[int]]]) -> tuple[dict[tuple[str, int], str], dict[str, np.ndarray]]:
    """Мержит кисть в ответ; возвращает (обновлённый словарь, растры кисти по чипам)."""
    result = dict(base)
    brushes: dict[str, np.ndarray] = {}
    for cid, dots in overlay.items():
        brush = brush_mask(dots, cid)
        brushes[cid] = brush
        if not any((cid, deg) in result for deg in (1, 2, 3)):
            continue  # train-вкладка и т.п.: в CSV не попадает, идёт только в скор
        brush_2d = brush.reshape(SIZE)
        for deg in (1, 2, 3):
            if (cid, deg) not in result:
                continue
            own = decode_rle(result[(cid, deg)], SIZE)
            kept = own & (brush_2d == 0)
            result[(cid, deg)] = encode_rle(kept | (brush_2d == deg))
    return result, brushes


def sandbox_score(brushes: dict[str, np.ndarray], truth_dir: Path) -> dict:
    """Кисть против mask_sev для train-чипов (BS_tr_*), затронутых кистью."""
    truth_dir = find_data_dir(truth_dir, prefer="masks")
    ds = Dataset(truth_dir)
    affected = sorted(cid for cid in brushes if cid.startswith("BS_tr_"))
    bs_pairs = {}
    for cid in affected:
        chip = ds.load(cid)
        if chip.mask_sev is None:
            continue
        truth = np.asarray(chip.mask_sev).astype(np.uint8)
        pred = brushes[cid].reshape(truth.shape).copy()
        pred[pred == 4] = 0
        bs_pairs[cid] = (truth, pred)
    if not bs_pairs:
        return {}
    return score_from_pairs({}, bs_pairs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Мерж кисти в submission + точный скор")
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--submission", default="submission.csv")
    ap.add_argument("--out", default="submission_test.csv")
    ap.add_argument("--score", action="store_true", help="скор кисти по mask_sev train-чипов")
    ap.add_argument("--score-dir", default=None, help="путь к train-набору с mask_sev (напр. <данные>/train)")
    args = ap.parse_args(argv)

    raise_field_limit()
    overlay = parse_overlay(args.overlay)
    if not overlay:
        print("оверлей пуст — нарисуй точки кистью и нажми «csv (тест)»")
        return 1

    base = read_submission(args.submission)
    rows, brushes = apply_overlay(base, overlay)

    touched_csv = sorted(cid for cid in brushes if any((cid, deg) in rows for deg in (1, 2, 3)))
    skipped = sorted(set(brushes) - set(touched_csv))
    out = Path(args.out)
    rows_list = [{"chip_id": cid, "class_id": deg, "rle": rle} for (cid, deg), rle in rows.items()]
    write_submission(rows_list, out)
    if touched_csv:
        print(f"готово: {out} ({len(rows_list)} строк; внесено в CSV: {touched_csv})")
    else:
        print(f"готово: {out} ({len(rows_list)} строк; сабмит не изменён — кисть была на train)")
    if skipped:
        print(f"песочница (в CSV не попали): {skipped} — скор кисти по истине: см. ниже")

    if args.score:
        if not args.score_dir:
            print("скор пропущен: укажи --score-dir <данные>/train")
            return 0
        res = sandbox_score(brushes, Path(args.score_dir))
        if res:
            print(f"скор кисти против истины (только train-чипы с маской): score={res['score']:.5f} "
                  f"iou_burn={res['iou_burn']:.5f} miou_sev={res['miou_sev']:.5f}")
        else:
            print("train-чипов с маской среди затронутых нет (кисть была на test?) — скор не считается")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
