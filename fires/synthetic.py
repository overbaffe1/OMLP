"""Синтетический набор в формате соревнования — для разработки без доступа к выдаче.

Зачем нужен: данные организаторов лежат на Яндекс.Диске (2,2 ГБ train + 878 МБ test) и
недоступны из оффлайн-окружения. Синтетика повторяет структуру набора (AF-чипы 256×256 @375 м
с каналами I1–I5 и вспомогательными слоями; BS-чипы 512×512 @20 м с каналами B2–B12, SCL,
Sentinel-1 VV/VH, рельефом и типом покрова), чтобы отладить весь пайплайн, метрики и формат
submission.csv ещё до появления реальных данных.

Физика синтетики намеренно «честная»:
* канал I4 считается как средняя яркостная температура по правилу смешения
  (субпиксельный очаг поднимает I4 на сотни кельвинов, I5 — на десятки, saturation 367 K);
* гари генерируются с учётом типа покрова: в степи и на пашне dNBR физически меньше,
  чем в лесу (то же ограничение, что описано в постановке).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

#: Коды ESA WorldCover v200, используемые в кейсе.
LANDCOVER = {
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "cropland",
    50: "built",
    60: "bare",
    70: "snow",
    80: "water",
    90: "wetland",
    95: "mangrove",
    100: "moss",
}

#: Каналы AF-чипа по порядку из постановки.
AF_CHANNELS = ("I1", "I2", "I3", "I4", "I5")
BS_CHANNELS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12", "SCL")


@dataclass
class SyntheticConfig:
    n_af: int = 24
    n_bs: int = 12
    n_af_fire: int | None = None          # по умолчанию — 70 % чипов с горением
    seed: int = 20260918
    af_size: int = 256
    bs_size: int = 512
    af_gsd: float = 375.0
    bs_gsd: float = 20.0
    #: Пространственная неоднородность фона и погоды внутри чипа.
    background_sigma: float = 3.0


def _fire_pixels(rng, size: int, n: int) -> np.ndarray:
    """Собирает пятно горения: несколько перекрывающихся эллипсов."""

    mask = np.zeros((size, size), dtype=bool)
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(max(1, int(rng.integers(1, 4)))):
        cy, cx = rng.uniform(0.2 * size, 0.8 * size, 2)
        ry, rx = rng.uniform(0.01 * size, 0.06 * size), rng.uniform(0.01 * size, 0.06 * size)
        angle = rng.uniform(0, math.pi)
        xr = (xx - cx) * math.cos(angle) + (yy - cy) * math.sin(angle)
        yr = -(xx - cx) * math.sin(angle) + (yy - cy) * math.cos(angle)
        mask |= (xr / rx) ** 2 + (yr / ry) ** 2 <= 1
    # подгоняем число пикселей под медианные 21 px из описания набора
    if n > 0 and mask.sum() > 0:
        coords = np.argwhere(mask)
        if len(coords) > n:
            keep = coords[rng.choice(len(coords), size=n, replace=False)]
            mask = np.zeros_like(mask)
            mask[keep[:, 0], keep[:, 1]] = True
    return mask


def _context_noise(rng, base: np.ndarray, radius: int = 26) -> np.ndarray:
    """Пространственно-коррелированный шум (погода/фон): свертка белого шума."""

    noise = rng.standard_normal(base.shape).astype(np.float32)
    k = radius // 2
    kernel = np.exp(-0.5 * (np.linspace(-2, 2, k) ** 2))
    kernel /= kernel.sum()
    noise = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, noise)
    noise = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, noise)
    return base + noise * 0.6


def generate_af_chip(rng, size: int = 256, with_fire: bool = True, tech_source: bool = False) -> dict:
    """AF-чип: каналы I1–I5 + вспомогательные слои + эталонная маска горения."""

    # фон: температуры земли и отражения
    t5_bg = _context_noise(rng, np.full((size, size), 300.0, dtype=np.float32))    # 11,45 мкм
    # Реальные дневные чипы: фон I4 − I5 ≈ +8 K (канал 3,74 мкм теплее из-за отражённой
    # солнечной радиации), ночью разница близка к нулю. Сцена генератора — дневная.
    t4_bg = t5_bg + rng.uniform(6.0, 10.0)
    i1 = np.clip(_context_noise(rng, np.full((size, size), 0.10, dtype=np.float32), 12), 0, 1)
    i2 = np.clip(_context_noise(rng, np.full((size, size), 0.22, dtype=np.float32), 12), 0, 1)
    i3 = np.clip(_context_noise(rng, np.full((size, size), 0.18, dtype=np.float32), 12), 0, 1)

    landcover = np.full((size, size), 30, dtype=np.uint8)  # по умолчанию степь
    valid = np.ones((size, size), dtype=np.uint8)

    # солнечный блик на воде/кровлях: яркий I1/I3 и ложные «горячие» пиксели
    glint = np.zeros((size, size), dtype=bool)
    if rng.random() < 0.25:
        yy, xx = np.mgrid[0:size, 0:size]
        cy, cx = rng.uniform(0.2, 0.8, 2) * size
        glint = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (rng.uniform(0.15, 0.3) * size) ** 2
        i1[glint] += rng.uniform(0.35, 0.6)
        i3[glint] += rng.uniform(0.25, 0.5)
        t4_bg[glint] += rng.uniform(4.0, 12.0)
        t5_bg[glint] += rng.uniform(1.0, 4.0)   # блик греет оба ИК-канала, но I4 сильнее
        landcover[glint] = rng.choice([50, 80])

    target = np.zeros((size, size), dtype=bool)
    if with_fire:
        n = int(np.clip(rng.lognormal(mean=math.log(21), sigma=0.7), 3, 400))
        target = _fire_pixels(rng, size, n)
        # субпиксельное смешение по правилу из постановки: I4 растёт на ~237 K,
        # I5 — на ~52 K при 3,6 % горящей площади пикселя.
        # Доля горящей площади откалибрована по реальным чипам: у размеченных пикселей
        # ΔT(I4) днём 22–49 K, ночью 12–52 K, что соответствует доле 0.05–0.30.
        frac = np.clip(rng.uniform(0.05, 0.30, size=(size, size)), 0, 1) * target
        t4 = t4_bg + 237.0 * frac * rng.uniform(0.6, 1.4, (size, size))
        t5 = t5_bg + 52.0 * frac
        t4 = np.clip(t4, None, 367.0)  # насыщение канала I4 по постановке
        i2[target] *= rng.uniform(0.75, 0.95)
        landcover[target] = rng.choice([10, 20, 30, 40, 90])
    else:
        t4, t5 = t4_bg, t5_bg

    if tech_source:  # газовый факел/промплощадка: горит, но техногенное
        yy, xx = np.mgrid[0:size, 0:size]
        cy, cx = rng.uniform(0.3, 0.7, 2) * size
        r_spot = rng.uniform(0.008, 0.02) * size
        spot = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r_spot**2
        t4[spot] = np.clip(t4[spot] + rng.uniform(120, 230), None, 367.0)
        t5[spot] += rng.uniform(15, 35)
        # вокруг факела — промплощадка: тип покрова «застройка» шире самого факела
        site = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r_spot * rng.uniform(2.5, 6.0)) ** 2
        landcover[site] = 50
        target = np.zeros((size, size), dtype=bool)

    # облака: низкая T5 и высокая яркость видимого
    cloud = (t5_bg < 285) & (i1 > 0.3)
    valid[cloud] = 0

    return {
        "I1": i1.astype(np.float32),
        "I2": i2.astype(np.float32),
        "I3": i3.astype(np.float32),
        "I4": t4.astype(np.float32),
        "I5": t5.astype(np.float32),
        "landcover": landcover,
        "valid": valid,
        "sun_zenith": np.full((size, size), rng.uniform(35, 60), dtype=np.float32),
        "sensor_zenith": np.full((size, size), rng.uniform(0, 25), dtype=np.float32),
        "temperature_2m": np.full((size, size), rng.uniform(15, 30), dtype=np.float32),
        "wind_speed": np.full((size, size), rng.uniform(1, 8), dtype=np.float32),
        "rel_humidity": np.full((size, size), rng.uniform(20, 60), dtype=np.float32),
        "target": target,
    }


def generate_bs_chip(rng, size: int = 512) -> dict:
    """BS-чип: пара снимков Sentinel-2 «до/после», SCL, Sentinel-1, рельеф, тип покрова."""

    yy, xx = np.mgrid[0:size, 0:size]
    # мозаика типов покрова: лес, степь, пашня, вода
    landcover = np.full((size, size), 30, dtype=np.uint8)
    landcover[:, : size // 2] = 10  # лесная половина
    patch = rng.integers(0, 4)
    if patch >= 2:
        landcover[size // 2 :, : size // 2] = 40  # пашня
    if patch == 3:
        landcover[: size // 4, : size // 4] = 80  # озеро

    # NBR «до» зависит от биомассы: лес выше, степь/пашня ниже
    nbr_pre = np.where(landcover == 10, 0.72, np.where(landcover == 40, 0.35, 0.42))
    nbr_pre = nbr_pre + rng.normal(0, 0.03, (size, size))
    nbr_pre[landcover == 80] = -0.4

    # гари: несколько контуров, степень поражения задаётся типом покрова
    sev = np.zeros((size, size), dtype=np.uint8)
    for _ in range(int(rng.integers(2, 6))):
        cy, cx = rng.uniform(0.15, 0.85, 2) * size
        r = rng.uniform(0.05, 0.18) * size
        blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2
        level = int(rng.integers(1, 4))
        sev[blob] = np.maximum(sev[blob], level)

    # лес: полный диапазон dNBR; степь/пашня: масштаб вниз (биомассы меньше)
    scale = np.where(landcover == 10, 1.0, np.where(landcover == 80, 0.0, 0.45))
    level_to_dnbr = np.array([0.0, 0.16, 0.36, 0.78])
    dnbr_true = level_to_dnbr[sev] * scale
    nbr_post = nbr_pre - dnbr_true + rng.normal(0, 0.02, (size, size))

    def to_reflectance(nbr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Переводит NBR в пару каналов B8A/B12 так, чтобы индекс сошёлся."""

        denom = 1.0 + nbr
        b12 = np.clip(rng.uniform(0.12, 0.22), 0, 1) * (1.0 - nbr) / np.maximum(denom, 1e-3) * 2
        b8a = np.clip(b12 * (1.0 + nbr) / np.maximum(1.0 - nbr, 1e-3), 0.01, 0.6)
        return b8a, b12

    b8a_pre, b12_pre = to_reflectance(nbr_pre)
    b8a_post, b12_post = to_reflectance(nbr_post)

    def band(value, sigma=0.02):
        return np.clip(value + rng.normal(0, sigma, (size, size)), 0.001, 0.99)

    pre = {
        "B2": band(0.05), "B3": band(0.08), "B4": band(0.07), "B5": band(0.15),
        "B6": band(0.22), "B7": band(0.25), "B8A": b8a_pre, "B11": band(0.18), "B12": b12_pre,
    }
    post = {
        "B2": band(0.06), "B3": band(0.09), "B4": band(0.08), "B5": band(0.14),
        "B6": band(0.19), "B7": band(0.22), "B8A": b8a_post, "B11": band(0.2), "B12": b12_post,
    }

    scl = np.full((size, size), 4, dtype=np.uint8)   # vegetation
    scl[landcover == 80] = 6                          # water
    scl[landcover == 40] = 5                          # bare (пашня)
    if rng.random() < 0.5:  # облако и тень облака
        cy, cx = rng.uniform(0.2, 0.8, 2) * size
        cloud = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (rng.uniform(0.08, 0.2) * size) ** 2
        scl[cloud] = 9
        scl[np.roll(cloud, 12, axis=0)] = 3           # тень смещена

    # Sentinel-1: в лесу после пожара падает VH (объёмное рассеяние крон)
    # Значения в дБ, как в постановке (см. таблицу каналов), поэтому отдельная функция
    def db_band(value: float, sigma: float = 0.35) -> np.ndarray:
        # σ ≈ 0.35 дБ — типичная точность калибровки Sentinel-1 в VH для L2A
        return (value + rng.normal(0, sigma, (size, size))).astype(np.float64)

    vh_drop = np.where(landcover == 10, -dnbr_true * 4.0, -dnbr_true * 1.2)
    s1_pre = {"VV": db_band(-8.0), "VH": db_band(-14.0)}
    s1_post = {"VV": db_band(-8.0) + vh_drop * 0.4, "VH": db_band(-14.0, 0.7) + vh_drop}

    dem = _context_noise(rng, np.full((size, size), 180.0, dtype=np.float32), 40).astype(np.float32)
    slope = np.clip(np.abs(np.gradient(dem)[0]) * 10, 0, 30).astype(np.float32)

    valid = np.ones((size, size), dtype=np.uint8)
    valid[~np.isfinite(pre["B8A"])] = 0

    return {
        "pre": pre, "post": post, "scl": scl,
        "s1_pre": s1_pre, "s1_post": s1_post,
        "dem": dem, "slope": slope, "landcover": landcover, "valid": valid,
        "target_sev": sev,
        "cloud_frac": float((scl == 9).mean()),
    }


def _chip_id(kind: str, split: str, idx: int) -> str:
    return f"{kind}_{split}_{idx:04d}"


def generate_dataset(root: str | Path, cfg: SyntheticConfig | None = None, splits: tuple[str, ...] = ("train", "test")) -> dict:
    """Создаёт каталог с чипами, ``meta.csv`` и ``sample_submission.csv``.

    Раскладка файлов совпадает с ожидаемой от организаторов (автоопределение в ``fires.chips``):
    отдельный файл на канал/слой плюс маски для обучающей части.
    """

    cfg = cfg or SyntheticConfig()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    meta_rows: list[dict] = []
    template_rows: list[tuple[str, int]] = []

    for split in splits:
        for kind, count, size, gsd in (
            ("af", cfg.n_af, cfg.af_size, cfg.af_gsd),
            ("bs", cfg.n_bs, cfg.bs_size, cfg.bs_gsd),
        ):
            sub = root / split / kind
            sub.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                cid = _chip_id(kind, split, i)

                if kind == "af":
                    with_fire = bool(rng.random() < 0.7)
                    tech = bool(not with_fire and rng.random() < 0.4)
                    chip = generate_af_chip(rng, size, with_fire=with_fire, tech_source=tech)
                    for name in AF_CHANNELS:
                        tifffile.imwrite(sub / f"{cid}_{name}.tif", chip[name])
                    tifffile.imwrite(sub / f"{cid}_landcover.tif", chip["landcover"])
                    tifffile.imwrite(sub / f"{cid}_valid.tif", chip["valid"])
                    for extra in ("sun_zenith", "sensor_zenith", "temperature_2m", "wind_speed", "rel_humidity"):
                        tifffile.imwrite(sub / f"{cid}_{extra}.tif", chip[extra].astype(np.float32))
                    if split == "train":
                        tifffile.imwrite(sub / f"{cid}_mask.tif", chip["target"].astype(np.uint8))
                    meta_rows.append(
                        {
                            "chip_id": cid, "kind": "af", "split": split, "width": size, "height": size, "gsd": gsd,
                            "acq_datetime": f"2024-0{rng.integers(4, 10)}-{rng.integers(1, 28):02d}T0{rng.integers(4, 20)}:00:00",
                            "satellite": str(rng.choice(["Suomi NPP", "NOAA-20", "NOAA-21"])),
                            "valid_frac": round(float(chip["valid"].mean()), 4),
                            "cloud_frac": None,
                            "n_fire_px": int(chip["target"].sum()) if split == "train" else None,
                            "fire_event_id": f"ev_{rng.integers(0, 40):03d}",
                        }
                    )
                else:
                    chip = generate_bs_chip(rng, size)
                    for name, arr in chip["pre"].items():
                        tifffile.imwrite(sub / f"{cid}_pre_{name}.tif", (arr * 10000).astype(np.uint16))
                    for name, arr in chip["post"].items():
                        tifffile.imwrite(sub / f"{cid}_post_{name}.tif", (arr * 10000).astype(np.uint16))
                    for name, arr in chip["s1_pre"].items():
                        tifffile.imwrite(sub / f"{cid}_pre_{name}.tif", (arr * 100).astype(np.int16))
                    for name, arr in chip["s1_post"].items():
                        tifffile.imwrite(sub / f"{cid}_post_{name}.tif", (arr * 100).astype(np.int16))
                    tifffile.imwrite(sub / f"{cid}_scl.tif", chip["scl"])
                    tifffile.imwrite(sub / f"{cid}_dem.tif", chip["dem"])
                    tifffile.imwrite(sub / f"{cid}_slope.tif", chip["slope"])
                    tifffile.imwrite(sub / f"{cid}_landcover.tif", chip["landcover"])
                    tifffile.imwrite(sub / f"{cid}_valid.tif", chip["valid"])
                    if split == "train":
                        tifffile.imwrite(sub / f"{cid}_mask_sev.tif", chip["target_sev"].astype(np.uint8))
                    px_ha = (gsd ** 2) / 10_000.0
                    counts = [int((chip["target_sev"] == k).sum()) for k in (1, 2, 3)]
                    meta_rows.append(
                        {
                            "chip_id": cid, "kind": "bs", "split": split, "width": size, "height": size, "gsd": gsd,
                            "date_pre": f"2024-0{rng.integers(4, 8)}-{rng.integers(1, 28):02d}",
                            "date_post": f"2024-0{rng.integers(8, 10)}-{rng.integers(1, 28):02d}",
                            "s1_date_pre": f"2024-0{rng.integers(4, 8)}-{rng.integers(1, 28):02d}",
                            "s1_date_post": f"2024-0{rng.integers(8, 10)}-{rng.integers(1, 28):02d}",
                            "valid_frac": round(float(chip["valid"].mean()), 4),
                            "cloud_frac": round(chip["cloud_frac"], 4),
                            "burn_area_ha": round(float((chip["target_sev"] > 0).sum() * px_ha), 2),
                            "sev1_px": counts[0], "sev2_px": counts[1], "sev3_px": counts[2],
                            "fire_event_id": f"ev_{rng.integers(0, 40):03d}",
                        }
                    )

                if split != "train":
                    if kind == "af":
                        template_rows.append((cid, 1))
                    else:
                        template_rows += [(cid, 1), (cid, 2), (cid, 3)]

    with (root / "meta.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted({k for row in meta_rows for k in row}), lineterminator="\n")
        writer.writeheader()
        writer.writerows(meta_rows)
    with (root / "sample_submission.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(("chip_id", "class_id", "rle"))
        for cid, class_id in template_rows:
            writer.writerow((cid, class_id, ""))
    return {"root": str(root), "chips": len(meta_rows), "template_rows": len(template_rows)}


def main(argv: list[str] | None = None) -> int:
    """CLI генератора: ``python -m fires.synthetic <каталог> [--n-af N] [--n-bs N] ...``"""

    import argparse
    import json
    import shutil

    ap = argparse.ArgumentParser(description="Генератор синтетического набора чипов для отладки")
    ap.add_argument("root", nargs="?", default="data/synthetic", help="каталог набора")
    ap.add_argument("--n-af", type=int, default=24, help="число AF-чипов на сплит")
    ap.add_argument("--n-bs", type=int, default=12, help="число BS-чипов на сплит")
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--af-size", type=int, default=256)
    ap.add_argument("--bs-size", type=int, default=512)
    ap.add_argument("--splits", default="train,test", help="сплиты через запятую")
    ap.add_argument("--clean", action="store_true", help="удалить каталог перед генерацией")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if args.clean and root.exists():
        shutil.rmtree(root)
    cfg = SyntheticConfig(
        n_af=args.n_af,
        n_bs=args.n_bs,
        seed=args.seed,
        af_size=args.af_size,
        bs_size=args.bs_size,
    )
    splits = tuple(part for part in args.splits.split(",") if part)
    print(json.dumps(generate_dataset(root, cfg, splits=splits), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
