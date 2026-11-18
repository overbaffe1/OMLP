#!/usr/bin/env python3
"""Единая точка входа инференса (требование постановки).

    python inference.py --data-dir /path/to/test --output /path/to/submission.csv

Скрипт:

* читает ``sample_submission.csv`` из каталога данных и обрабатывает **все** чипы из шаблона,
  включая сильно заоблачные (они не должны приводить к падению);
* формирует файл строго в требуемом формате (RLE, пары ``chip_id,class_id`` как в шаблоне);
* не требует интерактивного ввода, печатает тайминг и завершается с кодом 0;
* детерминирован: пороги берутся из ``weights.json`` (или значения по умолчанию), случайных
  операций нет — повторный запуск даёт тот же файл.

Время работы — жёсткое условие кейса (30 с на тестовый набор), а чипы BS распаковываются из
сжатых TIFF заметно дольше, чем считаются: на один BS-чип уходит ≈ 100 мс на чтение против
≈ 35 мс на расчёт. Поэтому чипы обрабатываются пулом процессов (``--jobs``, по умолчанию —
число ядер процессора, не больше восьми). В режиме обученной модели (``bs_mode: model``) стоимость
BS-чипа растёт до ≈0.7 с на ядро, и именно этот режим задаёт бюджет времени: на 12 потоках тест
считается ≈10 с, на двух ядрах — ≈20–25 с (лимит кейса 30 с на прогон). Порядок строк при этом не меняется: результаты
собираются по исходному списку шаблона, а при любой ошибке пула происходит откат к
последовательной обработке.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from fires import af as af_module
from fires import bs as bs_module
from fires.chips import Dataset
from fires.submission import read_template, validate_submission, write_submission
from fires.weights import load_weights

#: Глобальные объекты воркера (заполняются один раз на процесс при инициализации пула).
_WORKER: dict = {}


def _load_solution(weights_path: Path | None, device: str | None = None):
    weights = load_weights(weights_path)
    af_params = af_module.to_params(weights.get("af", {}))
    bs_params = bs_module.to_params(weights.get("bs", {}))
    base = Path(weights_path).parent if weights_path else Path(__file__).resolve().parent / "weights"
    blob_filter = None
    if weights.get("af_mode") == "blobs" and weights.get("blob_filter"):
        from fires.blobfilter import BlobFilter

        blob_filter = BlobFilter.load(base / weights["blob_filter"])
    severity_model = None
    if weights.get("bs_mode") == "model" and not weights.get("severity_model"):
        # Так выглядит испорченный weights.json: режим «модель», а имени файла нет — решение
        # молча уходит на пороги dNBR и теряет ~0.05 Score. Лучше сказать об этом громко.
        print("[!] weights.json в режиме «model», но поле severity_model пустое — это признак "
              "перезаписанных весов.\n    Решение сейчас считает по порогам dNBR (хуже на "
              "несколько сотых Score).\n    Верните файл weights/weights.json из репозитория.",
              file=sys.stderr)
    if weights.get("bs_mode") == "model" and weights.get("severity_model"):
        from fires.severity import SeverityModel

        path = base / weights["severity_model"]
        try:
            severity_model = SeverityModel.load(path)
            if severity_model is not None and weights.get("severity_class_bias"):
                # Смещения вероятностей классов перед argmax: подобраны на отложенных
                # событиях пожара (три сида сплита дают +0.002..+0.007 Score по BS-части).
                severity_model.class_bias = tuple(float(x) for x in weights["severity_class_bias"])
            if device:
                from fires.severity import resolve_device

                severity_model.device = resolve_device(device)
        except Exception as exc:
            # Файл модели мог быть собран другой версией scikit-learn: без него решение
            # продолжает работать на порогах dNBR, а не падает с ошибкой.
            print(f"[warn] модель степени поражения {path} не загрузилась ({exc}); "
                  f"работаю на порогах dNBR", file=sys.stderr)
    return af_params, bs_params, blob_filter, severity_model


def _init_worker(data_dir: str, weights_path: str | None, device: str | None = None,
                 threads: int = 1) -> None:
    """Инициализация воркера: датасет и пороги читаются один раз на процесс.

    Параллелизм внутри воркера ограничивается долей ядер (``threads``): иначе каждый процесс
    тянет за собой все ядра BLAS/OpenMP, и на 6-ядерной машине двенадцать воркеров дерутся за
    двенадцать потоков, теряя на переключениях больше, чем выигрывают. Явно заданные переменные
    окружения не перетираются — их можно выставить снаружи.
    """

    if threads:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(name, str(threads))

    af_params, bs_params, blob_filter, severity_model = _load_solution(
        Path(weights_path) if weights_path else None, device)
    _WORKER.update(
        dataset=Dataset(Path(data_dir)),
        af_params=af_params,
        bs_params=bs_params,
        blob_filter=blob_filter,
        severity_model=severity_model,
    )


def _process_chip(chip_id: str) -> dict:
    """Обрабатывает один чип и возвращает готовые строки ответа (без записи файла)."""

    dataset: Dataset = _WORKER["dataset"]
    af_params = _WORKER["af_params"]
    bs_params = _WORKER["bs_params"]
    blob_filter = _WORKER.get("blob_filter")

    result = {"chip_id": chip_id, "rows": [], "af_px": 0, "bs_px": 0, "details": None, "kind": None, "error": None}
    try:
        chip = dataset.load(chip_id)
    except Exception as exc:  # чип не должен «ронять» инференс
        result["error"] = str(exc)
        return result

    kind = dataset.kind_of(chip_id)
    result["kind"] = kind
    shape = chip.shape
    result["shape"] = shape

    if kind == "af":
        try:
            mask, details = af_module.detect_chip(chip, af_params, return_details=True, blob_filter=blob_filter)
            result["details"] = details
        except Exception as exc:
            result["error"] = f"AF не посчитан ({exc}), записана пустая маска"
            mask = np.zeros(shape, dtype=bool)
        result["af_px"] = int(mask.sum())
        result["rows"].append((1, mask))
    else:
        try:
            severity, details = bs_module.classify_chip(
                chip, bs_params, return_details=True, severity_model=_WORKER.get("severity_model")
            )
            result["details"] = details
        except Exception as exc:
            result["error"] = f"BS не посчитан ({exc}), записаны пустые маски"
            severity = np.zeros(shape, dtype=np.uint8)
        for class_id in (1, 2, 3):
            mask = severity == class_id
            result["bs_px"] += int(mask.sum())
            result["rows"].append((class_id, mask))
    return result


def process(
    data_dir: Path,
    output: Path,
    weights_path: Path | None = None,
    limit: int | None = None,
    verbose: bool = True,
    jobs: int = 0,
    device: str | None = None,
) -> dict:
    t0 = time.perf_counter()
    af_params, bs_params, blob_filter, severity_model = _load_solution(weights_path, device)

    dataset = Dataset(data_dir)
    template_path = data_dir / "sample_submission.csv"
    if template_path.exists():
        template = read_template(template_path)
    else:
        # На обучающей части шаблона нет: собираем его из самих чипов (AF — класс 1,
        # BS — классы 1/2/3), чтобы пайплайн можно было прогнать и на train. На тестовой
        # части шаблон организаторов обязателен и имеет приоритет.
        template = [(chip_id, 1) for chip_id in dataset.chip_ids("af")] + [
            (chip_id, cls) for chip_id in dataset.chip_ids("bs") for cls in (1, 2, 3)
        ]
        if verbose:
            print(f"[warn] {template_path} не найден — шаблон собран из чипов ({len(template)} строк)", file=sys.stderr)
    template_ids = sorted({chip_id for chip_id, _ in template})
    if limit:
        template_ids = template_ids[:limit]
    if not template_ids:
        # пустой ответ хуже понятной ошибки: молча записанный файл с одним заголовком
        # участник загрузит на платформу и получит 0 баллов
        raise SystemExit(
            f"в каталоге {data_dir} не нашлось ни чипов, ни строк в sample_submission.csv — "
            f"проверьте путь. Что видит решение: py -3 -m fires.chips --data-dir \"{data_dir}\"")

    n_jobs = jobs if jobs > 0 else min(12, os.cpu_count() or 1)
    results: list[dict] = []
    used_jobs = 1
    if n_jobs > 1 and len(template_ids) > 4:
        try:
            from concurrent.futures import ProcessPoolExecutor

            ctx = __import__("multiprocessing").get_context("fork" if sys.platform != "win32" else "spawn")
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(str(data_dir), str(weights_path) if weights_path else None, device,
                          max(1, (os.cpu_count() or 1) // n_jobs)),
            ) as pool:
                results = list(pool.map(_process_chip, template_ids, chunksize=1))
            used_jobs = n_jobs
        except Exception as exc:  # пул недоступен (ограничения песочницы) — работаем последовательно
            if verbose:
                print(f"[warn] параллельная обработка недоступна ({exc}), перехожу к последовательной", file=sys.stderr)
            results = []
    if not results:
        _init_worker(str(data_dir), str(weights_path) if weights_path else None, device)
        # пороги и модель берём из уже загруженного решения, чтобы не читать файлы дважды
        _WORKER["af_params"], _WORKER["bs_params"] = af_params, bs_params
        _WORKER["blob_filter"], _WORKER["severity_model"] = blob_filter, severity_model
        results = [_process_chip(chip_id) for chip_id in template_ids]

    rows: list[dict] = []
    empty_by_class: dict[str, int] = {"1": 0, "2": 0, "3": 0}
    shapes: dict[str, tuple[int, int]] = {}
    stats: dict[str, dict] = {"af": {}, "bs": {}}
    af_px_total = bs_px_total = 0
    processed = skipped = 0

    for result in results:
        chip_id = result["chip_id"]
        if result["error"]:
            print(f"[warn] {chip_id}: {result['error']}", file=sys.stderr)
        if "shape" not in result:
            skipped += 1
            continue
        shapes[chip_id] = result["shape"]
        if result["details"]:
            stats[result["kind"]] = result["details"]
        af_px_total += result["af_px"]
        bs_px_total += result["bs_px"]
        for class_id, mask in result["rows"]:
            rle = _rle(mask)
            if not rle:
                empty_by_class[str(class_id)] = empty_by_class.get(str(class_id), 0) + 1
            rows.append({"chip_id": chip_id, "class_id": class_id, "rle": rle})
        processed += 1

    write_submission(rows, output)
    elapsed = time.perf_counter() - t0

    report = {
        "chips_processed": processed,
        "chips_skipped": skipped,
        "rows_written": len(rows),
        "rows_without_pixels": empty_by_class,
        "af_fire_px": af_px_total,
        "bs_burned_px": bs_px_total,
        "seconds": round(elapsed, 2),
        "seconds_per_chip": round(elapsed / processed, 3) if processed else None,
        "jobs": used_jobs,
        "threads_per_job": max(1, (os.cpu_count() or 1) // used_jobs),
        "output": str(output),
        "details": stats,
    }
    if verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _rle(mask: np.ndarray) -> str:
    from fires.submission import encode_rle

    return encode_rle(mask)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Инференс решения «КосмоФАЙР-2»")
    ap.add_argument("--data-dir", required=True, help="каталог с тестовыми чипами и sample_submission.csv")
    ap.add_argument("--output", required=True, help="путь к submission.csv")
    ap.add_argument("--weights", default=None, help="путь к weights.json (по умолчанию weights/weights.json)")
    ap.add_argument("--limit", type=int, default=None, help="обработать только первые N чипов (отладка)")
    ap.add_argument("--jobs", type=int, default=0, help="число процессов (0 — авто: ядра, не больше восьми)")
    ap.add_argument("--validate", action="store_true", help="проверить результат по шаблону и границам чипов")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None,
                    help="где считать модель степени поражения (по умолчанию — как обучена)")
    args = ap.parse_args(argv)

    from fires.chips import find_data_dir

    data_dir = find_data_dir(args.data_dir, prefer="submission")
    if data_dir != Path(args.data_dir):
        print(f"[warn] в {args.data_dir} чипов нет; использую найденный рядом каталог {data_dir}",
              file=sys.stderr)

    report = process(
        data_dir,
        Path(args.output),
        Path(args.weights) if args.weights else None,
        limit=args.limit,
        jobs=args.jobs,
        device=args.device,
    )

    if args.validate:
        shapes = {cid: (int(a), int(b)) for cid, (a, b) in report_shapes(data_dir).items()}
        check = validate_submission(args.output, data_dir / "sample_submission.csv", shapes, strict=False)
        print(json.dumps({"validation": check}, ensure_ascii=False, indent=2))
        return 0 if check["ok"] else 1
    return 0


def report_shapes(data_dir: Path) -> dict[str, tuple[int, int]]:
    """Размеры чипов из meta.csv (для валидации границ RLE)."""

    dataset = Dataset(data_dir)
    shapes: dict[str, tuple[int, int]] = {}
    for chip_id in dataset.chip_ids():
        row = dataset.meta.get(chip_id, {})
        try:
            shapes[chip_id] = (int(row.get("height")), int(row.get("width")))
        except (TypeError, ValueError):
            try:
                shapes[chip_id] = dataset.load(chip_id).shape
            except Exception:
                continue
    return shapes


if __name__ == "__main__":
    raise SystemExit(main())
