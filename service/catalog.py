"""Построение каталога сцен для информационно-аналитического сервиса.

Сервису нужны две вещи, которых нет в тестовых чипах: геопривязка и даты. Организаторы
скрывают их в тестовой части (это прямо оговорено в постановке), но разрешают сервису
работать «на заранее подготовленном ограниченном наборе сцен». Поэтому каталог строится
из любой части набора, где привязка известна:

* ``meta.csv`` с координатами чипов (``epsg``, ``x_min``, ``y_min``, ``gsd``) — так поставляется
  обучающая часть кейса: чипы заданы в UTM, а модуль :mod:`fires.geo` переводит их в WGS84 без
  внешних зависимостей. Это реальный режим работы сервиса;
* ``chips_geo.csv`` — файл вида ``chip_id,origin_lon,origin_lat,gsd,date,kind`` в корне набора
  (резервный вариант, если привязка дана как долгота/широта);
* если файла нет, каталог собирается в **демонстрационном** режиме: чипы размещаются на
  регулярной сетке в заданном регионе, чтобы карта, контуры и справка были наглядными.
  Это честно помечается в каталоге полем ``"mode": "demo"`` и не влияет на submission.csv.

Формат каталога (``catalog.json``):

```json
{
  "mode": "geo" | "demo",
  "generated_at": "...",
  "thermopoints": [{"id": "...", "lon": ..., "lat": ..., "date": "...", "chip_id": "...", "delta_t": ...}],
  "contours": [{"id": "...", "severity": 2, "area_ha": 12.4, "chip_id": "...", "date": "...", "ring": [[lon, lat], ...]}],
  "report": {"total_area_ha": ..., "by_severity_ha": {...}}
}
```
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fires import af as af_module
from fires import bs as bs_module
from fires.chips import Dataset
from fires import geo

def _clean_date(meta: dict, kind: str) -> str:
    """Дата сцены без NaN: у BS это дата «после», у AF — время съёмки."""

    candidates = ("date_post", "acq_datetime") if kind == "bs" else ("acq_datetime", "date_post")
    for key in candidates:
        value = meta.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "nat", "none"}:
            return text[:10]
    return ""
from fires.vectorize import contours_report, mask_to_contours, pixel_to_lonlat
from fires.weights import load_weights

#: Демонстрационный регион: юг Красноярского края (там же, где работает наш кейс).
DEMO_ORIGIN = (90.5, 54.5)
DEMO_GRID_STEP = 0.14  # градусов между чипами демо-сетки


def _read_geo(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["chip_id"]] = row
    return out


def build(
    data_dir: str | Path,
    out_path: str | Path,
    weights_path: str | Path | None = None,
    blob_path: str | Path | None = None,
    demo_origin: tuple[float, float] = DEMO_ORIGIN,
    max_chips: int = 0,
    max_bs: int = 0,
    min_contour_ha: float = 0.0,
) -> dict:
    data_dir = Path(data_dir)
    dataset = Dataset(data_dir)
    weights = load_weights(weights_path)
    af_params = af_module.to_params(weights.get("af", {}))
    bs_params = bs_module.to_params(weights.get("bs", {}))

    blob_filter = None
    if weights.get("af_mode") == "blobs" and weights.get("blob_filter"):
        from fires.blobfilter import BlobFilter

        base = Path(weights_path).parent if weights_path else Path(__file__).resolve().parents[1] / "weights"
        blob_filter = BlobFilter.load(base / weights["blob_filter"])
    elif blob_path:
        from fires.blobfilter import BlobFilter

        blob_filter = BlobFilter.load(blob_path)

    severity_model = None
    if weights.get("bs_mode") == "model" and weights.get("severity_model"):
        from fires.severity import SeverityModel

        base = Path(weights_path).parent if weights_path else Path(__file__).resolve().parents[1] / "weights"
        model_path = base / weights["severity_model"]
        if model_path.exists():
            severity_model = SeverityModel.load(model_path)

    chip_ids = dataset.chip_ids()
    if max_chips:
        chip_ids = chip_ids[:max_chips]
    if max_bs:
        # BS-контуры занимают основную часть файла: оставляем крупнейшие по площади гари
        af_ids = [c for c in chip_ids if dataset.kind_of(c) == "af"]
        bs_ids = [c for c in chip_ids if dataset.kind_of(c) == "bs"]
        bs_ids.sort(key=lambda c: float(dataset.meta.get(c, {}).get("burn_area_ha") or 0.0), reverse=True)
        chip_ids = af_ids + bs_ids[:max_bs]
    geo_rows = _read_geo(data_dir / "chips_geo.csv")
    has_utm = any(
        row.get("epsg") is not None and row.get("x_min") is not None for row in dataset.meta.values()
    )
    mode = "geo" if (geo_rows or has_utm) else "demo"

    thermopoints: list[dict] = []
    contours: list[dict] = []
    per_chip_report: list[dict] = []
    demo_index = 0

    for chip_id in chip_ids:
        chip = dataset.load(chip_id)
        row = geo_rows.get(chip_id, {})
        utm = geo.utm_zone_from_epsg(chip.meta.get("epsg"))
        if utm and chip.meta.get("x_min") is not None:
            # Реальные координаты из meta.csv: пиксель (y, x) → UTM → WGS84.
            zone, northern = utm
            gsd = float(chip.meta.get("gsd") or (375.0 if chip.kind == "af" else 20.0))
            x_min = float(chip.meta.get("x_min"))
            y_min = float(chip.meta.get("y_min"))
            height = float(chip.meta.get("height") or chip.shape[0])
            y_max = y_min + height * gsd
            date = _clean_date(chip.meta, chip.kind)

            def transform(px: float, py: float, _x=x_min, _ymax=y_max, _gsd=gsd, _z=zone, _n=northern):
                return geo.utm_to_lonlat(_x + (px + 0.5) * _gsd, _ymax - (py + 0.5) * _gsd, _z, _n)

        elif row:
            origin_lon = float(row["origin_lon"])
            origin_lat = float(row["origin_lat"])
            gsd = float(row.get("gsd") or chip.meta.get("gsd") or 20.0)
            date = str(row.get("date") or _clean_date(chip.meta, chip.kind))
            transform = pixel_to_lonlat(origin_lon, origin_lat, gsd)  # noqa: F841 (используется ниже)
        else:
            gsd = float(chip.meta.get("gsd") or (375.0 if chip.kind == "af" else 20.0))
            # Демо-раскладка: компактная сетка, чипы BS — сверху, AF — отдельным блоком ниже
            step = DEMO_GRID_STEP
            columns = 8 if chip.kind == "bs" else 10
            if chip.kind == "af":
                origin_lon = demo_origin[0] + (demo_index % columns) * step
                origin_lat = demo_origin[1] - 1.35 - (demo_index // columns) * step
            else:
                origin_lon = demo_origin[0] + (demo_index % columns) * step
                origin_lat = demo_origin[1] - (demo_index // columns) * step
            demo_index += 1
            date = str(chip.meta.get("acq_datetime") or chip.meta.get("date_post") or "2024-08-01")  # демо-режим
            transform = pixel_to_lonlat(origin_lon, origin_lat, gsd)

        if chip.kind == "af":
            mask = af_module.detect_chip(chip, af_params, blob_filter=blob_filter)
            i4 = np.asarray(chip.layers["I4"], dtype=np.float32)
            i5 = np.asarray(chip.layers["I5"], dtype=np.float32)
            ys, xs = np.nonzero(mask)
            for n, (y, x) in enumerate(zip(ys, xs)):
                lon, lat = transform(float(x), float(y))
                thermopoints.append(
                    {
                        "id": f"{chip_id}_tp{n}",
                        "lon": round(lon, 6),
                        "lat": round(lat, 6),
                        "date": date[:10],
                        "chip_id": chip_id,
                        "delta_t": round(float(i4[y, x] - i5[y, x]), 1),
                        "i4": round(float(i4[y, x]), 1),
                    }
                )
            per_chip_report.append({"chip_id": chip_id, "kind": "af", "n_thermopoints": int(mask.sum())})
        else:
            severity = bs_module.classify_chip(chip, bs_params, severity_model=severity_model)
            chip_contours = mask_to_contours(
                severity > 0,
                chip_id=chip_id,
                severity_map=severity,
                min_area_px=max(4, int(bs_params.min_patch_px)),
                extra={"date": date[:10]},
            )
            px_ha = (gsd**2) / 10_000.0
            kept = 0
            for contour in chip_contours:
                area_ha = contour.area_px * px_ha
                if min_contour_ha and area_ha < min_contour_ha:
                    continue  # мелкие контуры не попадают в каталог: их десятки тысяч
                ring = contour.to_ring(transform)
                n = kept
                kept += 1
                contours.append(
                    {
                        "id": f"{chip_id}_c{n}",
                        "severity": int(contour.severity),
                        "area_ha": round(contour.area_px * px_ha, 3),
                        "chip_id": chip_id,
                        "date": date[:10],
                        "ring": [[round(lon, 6), round(lat, 6)] for lon, lat in ring],
                    }
                )
            report = contours_report(chip_contours, gsd)
            per_chip_report.append({"chip_id": chip_id, "kind": "bs", **report})

    total_ha = round(sum(c["area_ha"] for c in contours), 2)
    by_sev = {f"severity_{k}": round(sum(c["area_ha"] for c in contours if c["severity"] == k), 2) for k in (1, 2, 3)}
    catalog = {
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "region": {"lon": demo_origin[0], "lat": demo_origin[1]},
        "thermopoints": thermopoints,
        "contours": contours,
        "report": {
            "total_area_ha": total_ha,
            "by_severity_ha": by_sev,
            "n_thermopoints": len(thermopoints),
            "n_contours": len(contours),
            "chips": per_chip_report,
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    return {
        "catalog": str(out_path),
        "mode": mode,
        "thermopoints": len(thermopoints),
        "contours": len(contours),
        "total_area_ha": total_ha,
        "by_severity_ha": by_sev,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Сборка каталога сцен для сервиса")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="service/catalog.json")
    ap.add_argument("--weights", default="weights/weights.json")
    ap.add_argument("--max-chips", type=int, default=0, help="ограничить число чипов (для компактного каталога)")
    ap.add_argument("--max-bs", type=int, default=0, help="оставить N BS-чипов с наибольшей площадью гари")
    ap.add_argument("--min-contour-ha", type=float, default=0.0, help="не включать контуры меньше этой площади")
    args = ap.parse_args(argv)
    print(json.dumps(
        build(
            args.data_dir,
            args.out,
            args.weights,
            max_chips=args.max_chips,
            max_bs=args.max_bs,
            min_contour_ha=args.min_contour_ha,
        ),
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
