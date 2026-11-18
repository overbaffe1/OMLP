#!/usr/bin/env python3
"""Собирает макет тестовой части из обучающей: чипы без масок и без геопривязки.

    python tools/make_mock_test.py --data-dir /tmp/firesdata/real/train --out /tmp/firesdata/mock_test/test \
                                   --n-af 4 --n-bs 2

Зачем: структура теста по постановке — те же каналы, но **без масок** и (в версии организаторов)
без геопривязки, плюс `sample_submission.csv`. Такой макет позволяет прогнать `inference.py --validate`
на своей машине до открытия реального теста и убедиться, что отсутствие масок ничего не ломает.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fires.chips import Dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Макет тестовой части: чипы без масок + sample_submission.csv")
    ap.add_argument("--data-dir", required=True, help="обучающая часть организаторов")
    ap.add_argument("--out", required=True, help="каталог теста (внутрь копируются af/, bs/, meta.csv)")
    ap.add_argument("--n-af", type=int, default=4)
    ap.add_argument("--n-bs", type=int, default=2)
    args = ap.parse_args(argv)

    source = Dataset(args.data_dir)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rows = ["chip_id,class_id,rle"]
    copied = {"af": 0, "bs": 0}
    for kind, limit in (("af", args.n_af), ("bs", args.n_bs)):
        for chip_id in source.chip_ids(kind)[:limit]:
            for sub in sorted((Path(args.data_dir) / kind).iterdir()):
                if not sub.is_dir() or sub.name == "masks":
                    continue
                target = out / kind / sub.name
                target.mkdir(parents=True, exist_ok=True)
                for file in sub.glob(f"{chip_id}_*"):
                    shutil.copy2(file, target / file.name)
            # формат ответа из постановки: AF — одна строка с class_id=1,
            # BS — три строки (1/2/3) на чип
            for class_id in ((1,) if kind == "af" else (1, 2, 3)):
                rows.append(f"{chip_id},{class_id},")
            copied[kind] += 1

    shutil.copy2(Path(args.data_dir) / "af" / "meta.csv", out / "af" / "meta.csv")
    shutil.copy2(Path(args.data_dir) / "bs" / "meta.csv", out / "bs" / "meta.csv")
    header = (Path(args.data_dir) / "af" / "meta.csv").read_text(encoding="utf-8").splitlines()[0]
    lines = [header]
    for kind in ("af", "bs"):
        for line in (Path(args.data_dir) / kind / "meta.csv").read_text(encoding="utf-8").splitlines()[1:]:
            if line.split(",")[0] in {row.split(",")[0] for row in rows[1:]}:
                lines.append(line)
    (out / "meta.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "sample_submission.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"макет теста готов: {out} (AF {copied['af']}, BS {copied['bs']}), масок нет, шаблон из {len(rows) - 1} строк")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
