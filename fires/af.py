"""Модуль детекции активного горения (AF).

Алгоритм повторяет логику оперативного продукта VIIRS 375 м (Schroeder et al., 2014),
адаптированную под чипы соревнования:

1. **Отбор валидных пикселей.** Исключаются вода (WorldCover 80), невалидные пиксели,
   облака (по слою valid/SCL, если он есть) и области солнечного блика.
2. **Абсолютные тесты.** ``I4 > 367 K`` — насыщение канала, безусловное горение;
   ``I4 > 360 K`` — очень яркий очаг.
3. **Контекстный тест.** Пиксель — кандидат, если одновременно
   ``I4 > μ_bg + k·σ_bg`` и ``I4 − I5 > ΔT`` (день 15 K, ночь 10 K) и ``I4 > 320 K``.
   Фон оценивается по окну 71×71 пиксель с исключением потенциально горящих пикселей —
   ровно так это сделано в операционном алгоритме.
4. **Отсев техногенных термоаномалий.** Промплощадки и застройка (WorldCover 50),
   отражающие поверхности (блик) и одиночные «горячие» пиксели на пашне убираются.
5. **Постобработка.** Удаляются изолированные пиксели и слишком крупные «пожары»
   аномальной формы (артефакты блика), если это разрешено параметрами.

Все пороги вынесены в :class:`AFParams` и подбираются на обучающей части
(``train.py``) либо берутся из ``weights.json``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from fires.morphology import (
    box_count,
    box_mean,
    box_std,
    label,
    remove_large_objects,
    remove_small_objects,
)

#: Код ESA WorldCover для воды и застройки.
LC_WATER = 80
LC_BUILT = 50
LC_CROPLAND = 40
LC_BARE = 60


@dataclass
class AFParams:
    """Пороги детекции; значения по умолчанию — операционные для VIIRS 375 м."""

    sat_t4: float = 367.0            # насыщение канала I4 (день)
    absolute_t4: float = 360.0       # абсолютный порог «очень яркого» очага (день)
    candidate_t4: float = 330.0      # минимум для контекстного теста (день)
    #: Ночные пороги. Ночью солнечный нагрев выключен: фон в I4 падает примерно до 283 K,
    #: поэтому «дневная» планка 320 K отсекала бы почти все ночные очаги.
    sat_t4_night: float = 340.0
    absolute_t4_night: float = 310.0
    candidate_t4_night: float = 295.0
    candidate_delta: float = 5.0     # мягкий порог I4 − I5 для отбора кандидатов в блобы
    delta_day: float = 25.0          # I4 − I5 днём (на реальных чипах фон τ4 днём прогревается до 310–315 K)
    delta_night: float = 10.0        # I4 − I5 ночью
    k_sigma: float = 2.0             # множитель σ фона в контекстном тесте
    bg_window: int = 71              # окно оценки фона, пикселей
    exclude_water: bool = True       # маскировать воду (WorldCover 80) по SCL-эквиваленту
    exclude_built: bool = True       # отсев застройки/промплощадок
    built_t4_margin: float = 8.0     # насколько ΔT выше порога должен быть пиксель в застройке
    built_min_cluster: int = 4       # минимальная площадь очага в застройке
    glint_i1: float = 0.32           # порог видимого канала для блика
    glint_i3: float = 0.28           # порог SWIR для блика
    glint_margin: float = 9.0        # в зоне блика требуем заметно больший ΔT (I4 − I5)
    glint_t4_min: float = 335.0      # либо очень высокую I4: блик столько не даёт
    isolated_guard: bool = True      # убирать одиночные пиксели со слабым ΔT
    marginal_rescue: bool = False    # спасать thermal-пиксели пашни/голой почвы у кромки
    min_cluster_px: int = 1          # минимальный размер очага
    max_cluster_px: int = 20000      # максимум (артефакты блика)
    night_sun_zenith: float = 95.0   # порог зенитного угла Солнца «ночь»
    sat_always_fire: bool = True     # насыщение трактуется как горение всегда


def _background_stats(i4: np.ndarray, candidate: np.ndarray, valid: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Среднее, σ и число фоновых пикселей в окне (без кандидатов и невалидных)."""

    radius = max(1, window // 2)
    bg_mask = valid & ~candidate
    values = np.where(bg_mask, i4, 0.0).astype(np.float32)
    counts = box_count(bg_mask, radius)
    sums = box_mean(values, radius) * ((2 * radius + 1) ** 2)
    mean = np.where(counts > 0, sums / np.maximum(counts, 1), i4)
    sq = np.where(bg_mask, i4**2, 0.0).astype(np.float32)
    mean_sq = np.where(counts > 0, (box_mean(sq, radius) * ((2 * radius + 1) ** 2)) / np.maximum(counts, 1), i4**2)
    std = np.sqrt(np.clip(mean_sq - mean**2, 0, None))
    return mean.astype(np.float32), std.astype(np.float32), counts


def detect(
    i4: np.ndarray,
    i5: np.ndarray,
    i1: np.ndarray | None = None,
    i3: np.ndarray | None = None,
    landcover: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    sun_zenith: np.ndarray | None = None,
    params: AFParams | None = None,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Возвращает бинарную маску природного горения (1 — пожар, 0 — всё остальное)."""

    p = params or AFParams()
    i4 = np.asarray(i4, dtype=np.float32)
    i5 = np.asarray(i5, dtype=np.float32)
    shape = i4.shape

    if valid is None:
        valid = np.ones(shape, dtype=bool)
    else:
        valid = np.asarray(valid).astype(bool)
    if landcover is None:
        landcover = np.full(shape, 30, dtype=np.uint8)
    else:
        landcover = np.asarray(landcover)

    # 1. экранирование
    # В реальных данных I4/I5 содержат NaN там, где наблюдения нет (край полосы обзора,
    # маскирование): такие пиксели не валидны по определению.
    valid = valid & np.isfinite(i4) & np.isfinite(i5)
    screen = valid & ((landcover != LC_WATER) if p.exclude_water else True)
    glint = np.zeros(shape, dtype=bool)
    if i1 is not None and i3 is not None:
        i1 = np.asarray(i1, dtype=np.float32)
        i3 = np.asarray(i3, dtype=np.float32)
        # блик — это яркий видимый/SWIR при повёрнутых к Солнцу поверхностях (вода, кровли, пески).
        # Полностью выкидывать такие пиксели нельзя: очаг может оказаться внутри пятна блика,
        # поэтому зона блика не исключается из анализа, а переводится в «подозрительный» режим,
        # где требуются более сильные признаки горения (см. ниже).
        glint = (i1 >= p.glint_i1) & (i3 >= p.glint_i3)  # NaN (ночь) даёт False — это корректно

    # 2. абсолютные тесты; пороги зависят от времени суток
    night = bool(sun_zenith is not None and float(np.nanmedian(np.asarray(sun_zenith))) >= p.night_sun_zenith)
    sat_threshold = p.sat_t4_night if night else p.sat_t4
    absolute_threshold = p.absolute_t4_night if night else p.absolute_t4
    candidate_threshold = p.candidate_t4_night if night else p.candidate_t4
    saturated = i4 >= sat_threshold
    absolute = i4 >= absolute_threshold
    candidate = screen & ((i4 >= candidate_threshold) | absolute | saturated)

    # 3. контекстный тест
    mean_bg, std_bg, counts = _background_stats(i4, candidate, screen, p.bg_window)
    mean_bg = np.where(counts > 0, mean_bg, np.nanmean(i4[screen]) if screen.any() else i4)
    delta = i4 - i5
    delta_threshold = p.delta_night if night else p.delta_day
    contextual = candidate & (i4 - mean_bg > p.k_sigma * std_bg.clip(0.5, None)) & (delta > delta_threshold)
    if glint.any():
        # в зоне возможного блика принимаем только уверенные признаки горения
        contextual &= ~glint | (delta > delta_threshold + p.glint_margin) | (i4 >= p.glint_t4_min)

    fire = (absolute & screen) | (saturated & screen & p.sat_always_fire) | contextual

    # 4. отсев техногенных источников и отражающих объектов
    if p.exclude_built:
        built = landcover == LC_BUILT
        # В застройке и на промплощадках горение есть почти всегда (факелы, печи), но эталон
        # такие пиксели помечает как «не пожар». Оставляем только очаги, которые заметно
        # сильнее и крупнее типичного техногенного источника.
        fire_built = fire & built
        if fire_built.any():
            labels, n = label(fire_built)
            if n:
                sizes = np.bincount(labels.ravel())
                keep_ids = np.flatnonzero(sizes >= max(p.built_min_cluster, p.min_cluster_px))
                keep_ids = keep_ids[keep_ids != 0]
                strong = (delta > delta_threshold + p.built_t4_margin) | (i4 >= sat_threshold)
                keep_mask = np.isin(labels, keep_ids) & strong
                fire = (fire & ~built) | (fire & built & keep_mask)
        # на пашне и открытой почве одиночные пиксели без устойчивого контекста — шум
    if p.exclude_built or p.marginal_rescue:
        marginal = (landcover == LC_CROPLAND) | (landcover == LC_BARE)
        # правка пашни/голой почвы: без устойчивого контекста одиночные пиксели — шум;
        # с marginal_rescue окно отсева сужается до delta < порога: пиксели у самой
        # кромки реального пожара имеют delta уже порога и dense истины
        cutoff = delta_threshold if p.marginal_rescue else delta_threshold + 2.0
        fire = fire & ~(marginal & (delta < cutoff) & ~absolute)

    fire &= screen

    # 4b. одиночные слабые пиксели: у горящего пикселя рядом почти всегда есть сосед
    if p.isolated_guard:
        neighbors = box_count(fire, 1) - fire.astype(np.float32)
        weak = fire & (neighbors == 0) & (delta < delta_threshold + 3.0) & (i4 < absolute_threshold)
        fire = fire & ~weak

    # 5. постобработка
    if p.min_cluster_px > 1:
        fire = remove_small_objects(fire, p.min_cluster_px)
    if p.max_cluster_px > 0:
        fire = remove_large_objects(fire, p.max_cluster_px)

    if return_details:
        return fire, {
            "night": bool(night),
            "delta_threshold": float(delta_threshold),
            "screened_px": int((~screen).sum()),
            "glint_px": int(glint.sum()),
            "absolute_px": int((absolute & screen).sum()),
            "contextual_px": int(contextual.sum()),
            "fire_px": int(fire.sum()),
            "background_mean_t4": float(np.nanmean(mean_bg)),
            "night": bool(night),
            "thresholds": {"sat": sat_threshold, "absolute": absolute_threshold, "candidate": candidate_threshold},
        }
    return fire


def detect_chip(chip, params: AFParams | None = None, return_details: bool = False, blob_filter=None):
    """Обёртка над :func:`detect` для объекта :class:`fires.chips.Chip`.

    Если передан обученный ``blob_filter`` (:class:`fires.blobfilter.BlobFilter`), кандидаты
    сначала собираются в блобы, а затем фильтруются моделью — это отсекает газовые факелы
    и блик, которые по признакам одного пикселя от пожара не отличаются.
    """

    if blob_filter is not None and getattr(blob_filter, "ready", False):
        mask = blob_filter.apply(chip, params or AFParams())
        if return_details:
            return mask, {"mode": "blob_filter", "threshold": blob_filter.threshold, "fire_px": int(mask.sum())}
        return mask

    layers = chip.layers
    return detect(
        i4=layers["I4"],
        i5=layers["I5"],
        i1=layers.get("I1"),
        i3=layers.get("I3"),
        landcover=layers.get("landcover"),
        valid=layers.get("valid"),
        sun_zenith=layers.get("sun_zenith"),
        params=params,
        return_details=return_details,
    )


def to_params(values: dict) -> AFParams:
    """Собирает параметры из словаря (например, прочитанного из weights.json)."""

    allowed = {f for f in asdict(AFParams())}
    return AFParams(**{k: v for k, v in values.items() if k in allowed})
