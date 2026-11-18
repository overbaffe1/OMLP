#!/usr/bin/env python3
"""Подставляет числа из ``docs/numbers.json`` в ``docs/report.md`` и ``README.md``.

    python tools/collect_numbers.py --data-dir <train> --out docs/numbers.json
    python tools/update_report.py            # report.md и README.md

Скрипт переписывает только участки между маркерами ``<!-- auto:... -->``, поэтому текст отчёта
можно свободно править руками — числа обновляются без риска затереть формулировки.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_block(text: str, marker: str, body: str) -> str:
    """Меняет тело между ``<!-- auto:имя -->`` и ``<!-- /auto:имя -->``.

    Закрывающий маркер именованный: общий ``<!-- /auto -->`` слишком легко «съедает» соседний
    блок при нежадном поиске (первый же подходящий конец строки — уже конец следующего блока).
    """

    # тело — целые строки (возможно, ни одной), поэтому «пустой» блок тоже валиден
    pattern = re.compile(
        rf"(<!-- auto:{re.escape(marker)} -->\n)(?:.*\n)*?(<!-- /auto:{re.escape(marker)} -->)"
    )
    if not pattern.search(text):
        raise SystemExit(f"в тексте нет маркеров auto:{marker}")
    replacement = rf"\1{body}\n\2"
    return pattern.sub(replacement, text, count=1)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _fmt_int(value: int | float) -> str:
    return f"{int(value):,}".replace(",", " ")


def main_table(numbers: dict) -> str:
    final = numbers["final"]
    af = final.get("af_counts") or {}
    af_detail = ""
    if af:
        af_detail = f" (tp {_fmt_int(af['tp'])}, fp {_fmt_int(af['fp'])}, fn {_fmt_int(af['fn'])})"
    classes = final.get("iou_per_class")
    miou_detail = ""
    if classes:
        miou_detail = " (класс 1 — {}, класс 2 — {}, класс 3 — {})".format(
            _fmt(classes["1"]), _fmt(classes["2"]), _fmt(classes["3"])
        )
    burn = final.get("burn_counts")
    burn_detail = ""
    if burn:
        burn_detail = f" (tp {_fmt_int(burn['tp'])}, fp {_fmt_int(burn['fp'])}, fn {_fmt_int(burn['fn'])})"
    return (
        "| Компонента | Значение |\n|---|---|\n"
        f"| Score | **{_fmt(final['score'])}** |\n"
        f"| F1_af | {_fmt(final['f1_af'])}{af_detail} |\n"
        f"| IoU_burn | {_fmt(final['iou_burn'])}{burn_detail} |\n"
        f"| mIoU_sev | {_fmt(final['miou_sev'])}{miou_detail} |\n"
        f"| Чипов в оценке | {final.get('n_af', '—')} AF + {final.get('n_bs', '—')} BS |"
    )


def ablation_table(numbers: dict) -> str:
    final = numbers["final"]
    lines = [
        "| Вариант | Score | Δ Score | F1_af | IoU_burn | mIoU_sev |",
        "|---|---|---|---|---|---|",
        f"| Итоговая конфигурация | {_fmt(final['score'])} | — | {_fmt(final['f1_af'])} | "
        f"{_fmt(final['iou_burn'])} | {_fmt(final['miou_sev'])} |",
    ]
    for row in sorted(numbers.get("ablations", []), key=lambda r: r.get("delta_score", 0.0)):
        lines.append(
            f"| {row['name']} | {_fmt(row['score'])} | {row['delta_score']:+.4f} | "
            f"{_fmt(row['f1_af'])} | {_fmt(row['iou_burn'])} | {_fmt(row['miou_sev'])} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Подстановка чисел в отчёт и README")
    ap.add_argument("--numbers", default=str(ROOT / "docs" / "numbers.json"))
    args = ap.parse_args(argv)

    numbers = _load(Path(args.numbers))

    report_path = ROOT / "docs" / "report.md"
    report = report_path.read_text(encoding="utf-8")
    report = _replace_block(report, "main-table", main_table(numbers))
    report = _replace_block(report, "ablation-table", ablation_table(numbers))
    report_path.write_text(report, encoding="utf-8")
    print(f"обновлён {report_path}")

    readme_path = ROOT / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        if "<!-- auto:readme-score -->" in readme:
            readme = _replace_block(readme, "readme-score", readme_score(numbers))
            readme = _replace_block(readme, "readme-ablation", readme_ablation(numbers))
            readme_path.write_text(readme, encoding="utf-8")
            print(f"обновлён {readme_path}")
    return 0


def readme_ablation(numbers: dict) -> str:
    final = numbers["final"]
    lines = [
        f"| Итоговая конфигурация | {_fmt(final['score'])} | — | {_fmt(final['f1_af'])} | "
        f"{_fmt(final['iou_burn'])} | {_fmt(final['miou_sev'])} |",
    ]
    ordered = sorted(numbers.get("ablations", []), key=lambda r: r.get("delta_score", 0.0))[:5]
    for row in ordered:
        lines.append(
            f"| {row['name']} | {_fmt(row['score'])} | {row['delta_score']:+.4f} | "
            f"{_fmt(row['f1_af'])} | {_fmt(row['iou_burn'])} | {_fmt(row['miou_sev'])} |"
        )
    return "\n".join(lines)


def readme_score(numbers: dict) -> str:
    final = numbers["final"]
    speed = numbers.get("inference") or {}
    return (
        f"| Score на реальной обучающей части (644 чипа) | **{_fmt(final['score'])}** |\n"
        f"| F1_af / IoU_burn / mIoU_sev | {_fmt(final['f1_af'])} / {_fmt(final['iou_burn'])} / "
        f"{_fmt(final['miou_sev'])} |\n"
        f"| Инференс всей обучающей части | {speed.get('seconds', '—')} с на {speed.get('chips', '—')} чипах, "
        f"{speed.get('jobs', '—')} процесса |"
    )


if __name__ == "__main__":
    raise SystemExit(main())
