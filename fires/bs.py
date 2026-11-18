"""Модуль картирования гарей и классификации степени поражения (BS).

Схема:

1. **Индексы.** ``NBR = (B8A − B12) / (B8A + B12)`` для дат «до» и «после»;
   ``dNBR = NBR_pre − NBR_post``; ``rdNBR = dNBR / sqrt(|NBR_pre|)``.
2. **Маскировка.** SCL (облака, тень облаков, вода) и слой валидности; отдельно
   отбрасываются пиксели, где индекс физически не определён.
3. **Пороги по типу покрова.** В лесу и кустарнике сохраняется шкала USGS
   (0,10 / 0,27 / 0,66), для степи, пашни, поймы и открытой почвы пороги масштабируются
   вниз: при малом запасе биомассы dNBR физически не достигает лесных значений.
   Коэффициенты подбираются на обучающей части (``train.py``).
4. **Проверка по Sentinel-1.** Радиолокация не зависит от облачности и дыма; падение
   обратного рассеяния в VH (и, слабее, в VV) подтверждает гарь в лесу. Используется
   как подтверждающий признак для пикселей, где оптика сомнительна (облачность/дым),
   и как способ усилить слабые классы.
5. **Постобработка.** Удаление мелких контуров, заполнение внутренних «дыр»,
   сглаживание границ.

Тип покрова (ESA WorldCover v200): 10 дерево, 20 кустарник, 30 травостой, 40 пашня,
50 застройка, 60 открытая почва, 70 снег, 80 вода, 90 пойма, 95 мангры, 100 мхи.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from fires.morphology import fill_holes, remove_small_objects

#: Группы типов покрова для порогов (ключ — имя группы, значение — коды WorldCover).
LANDCOVER_GROUPS: dict[str, tuple[int, ...]] = {
    "forest": (10, 20, 95, 100),
    "semi": (30, 90),      # травостой и пойма: мало биомассы
    "cropland": (40, 60),  # пашня и открытая почва: частые ложные контуры
    "other": (50, 70),     # застройка, снег
}

#: Код WorldCover для воды.
LC_WATER = 80

#: Маскировка по SCL: 3 — тень облака, 8/9 — облака, 11 — снег, 6 — вода.
#: Класс 10 (тонкие перистые облака) НЕ маскируется: cirrus почти прозрачен в NIR/SWIR,
#: dNBR сквозь него работает (медиана 0.22 в гари против −0.03 в фоне, см. tools/diag_scl.py),
#: cloud_frac организаторов класс 10 тоже игнорирует. До исправления маска по классу 10
#: запрещала предсказывать 17.8% истины гари (1.05M px, десятки чипов с IoU=0.000).
SCL_BAD = (3, 8, 9, 11)
SCL_WATER = 6


@dataclass
class BSParams:
    """Пороги dNBR по группам покрова и параметры постобработки."""

    #: Лес/кустарник: границы классов выше, чем в степи — плотная биомасса даёт больший dNBR
    #: при той же степени поражения (подобрано на реальной обучающей части).
    forest: tuple[float, float, float] = (0.20, 0.27, 0.55)
    #: Степь и пойма.
    semi: tuple[float, float, float] = (0.14, 0.22, 0.40)
    #: Пашня/открытая почта: низкие значения dNBR и высокая фенологическая изменчивость,
    #: поэтому порог «горело» самый низкий, а границы классов сдвинуты вниз.
    cropland: tuple[float, float, float] = (0.12, 0.19, 0.37)
    #: Застройка и снег: гарь не картируем (0 — отключено).
    other: tuple[float, float, float] = (1.0, 1.0, 1.0)
    min_patch_px: int = 6            # минимальный размер контура гари
    fill_holes_mask: bool = True     # заполнять недогоревшие «острова» внутри контура
    use_sar: bool = True             # подтверждение по Sentinel-1
    sar_vh_drop_db: float = 1.5      # падение VH подтверждает гарь в лесу
    sar_upgrade: bool = True         # пиксели с сильным падением VH усиливают класс
    cloud_gate: bool = True          # не переклассифицировать пиксели под облаком
    mask_scl: bool = True            # маскировать облака/тень/снег/воду по слою SCL
    scl_weight: float = 1.0          # запас: вес слоя SCL в решении
    #: Восстановление гари под облаком/тенью по Sentinel-1.
    #: На реальном наборе отключено: класс, собранный из ΔVH, грубее оптического, и на
    #: отложенной части восстановление стоит ≈ −0.14 IoU (train.py подтверждает это
    #: перебором вариантов и записывает результат в weights.json).
    sar_recovery: bool = False
    #: Калибровка «дБ падения VH на единицу dNBR»: лес и остальные группы покрова.
    #: Радиолокация откликается на потерю биомассы, поэтому dNBR восстанавливается
    #: из ΔVH делением на эту чувствительность (подбирается в train.py).
    sar_scale: tuple[float, float] = (3.0, 1.5)
    #: Минимальное падение VH (дБ), ниже которого восстановление не запускается:
    #: отсекает шум калибровки радиолокатора (σ ≈ 0.3–0.4 дБ на дату).
    sar_min_drop_db: float = 1.0
    #: Пороги классов для радиолокационной оценки. Сравнение идёт со универсальной
    #: шкалой USGS: чувствительность sar_scale уже учитывает разницу в биомассе,
    #: поэтому повторно масштабировать пороги под тип покрова не нужно.
    sar_class_thresholds: tuple[float, float, float] = (0.10, 0.27, 0.66)
    #: Размер окна медианного сглаживания dNBR перед порогами (0 — выключено).
    #: Соседние пиксели горят вместе, а одиночные выбросы dNBR (тени облаков, границы полей,
    #: сдвиг сцен) дают россыпь ложных пикселей: медиана 3×3 убирает выбросы, не размывая контуры.
    #: Размер выбирается в train.py по двум фолдам (0.2932 против 0.2850 у варианта без сглаживания).
    smooth_dnbr: int = 0

    def groups(self) -> dict[str, tuple[float, float, float]]:
        return {"forest": self.forest, "semi": self.semi, "cropland": self.cropland, "other": self.other}


def nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """NBR с защитой от нулей в знаменателе."""

    nir = np.asarray(nir, dtype=np.float32)
    swir = np.asarray(swir, dtype=np.float32)
    den = nir + swir
    out = np.where(np.abs(den) < 1e-6, 0.0, (nir - swir) / np.where(np.abs(den) < 1e-6, 1.0, den))
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def scale_reflectance(arr: np.ndarray) -> np.ndarray:
    """Приводит L2A-отражение (uint16, масштаб 0–10000) к долям единицы.

    Значения 0 означают «нет данных» и отбрасываются вызывающим кодом через маску.
    """

    arr = np.asarray(arr).astype(np.float32)
    if arr.max() > 1.5:  # масштабированное отражение
        return arr / 10000.0
    return arr


def group_codes(groups: dict[str, tuple[int, ...]] | None = None) -> dict[int, str]:
    groups = groups or LANDCOVER_GROUPS
    out: dict[int, str] = {}
    for name, codes in groups.items():
        for code in codes:
            out[int(code)] = name
    return out


def classify(
    nbr_pre: np.ndarray,
    nbr_post: np.ndarray,
    landcover: np.ndarray | None = None,
    scl: np.ndarray | None = None,
    scl_pre: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    s1_vh_pre: np.ndarray | None = None,
    s1_vh_post: np.ndarray | None = None,
    params: BSParams | None = None,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Возвращает маску 0–3: 0 — не горело, 1/2/3 — слабая/средняя/сильная степень."""

    p = params or BSParams()
    nbr_pre = np.asarray(nbr_pre, dtype=np.float32)
    nbr_post = np.asarray(nbr_post, dtype=np.float32)
    shape = nbr_pre.shape
    landcover = np.full(shape, 30, dtype=np.uint8) if landcover is None else np.asarray(landcover)

    dnbr = nbr_pre - nbr_post
    if p.smooth_dnbr and p.smooth_dnbr >= 3:
        # Невалидные пиксели заменяются на 0 до фильтрации и восстанавливаются после: медиана
        # по соседям не должна «вытягивать» значения из облаков и воды.
        from scipy import ndimage

        filled = np.where(np.isfinite(dnbr), dnbr, 0.0)
        smoothed = ndimage.median_filter(filled, size=int(p.smooth_dnbr))
        dnbr = np.where(np.isfinite(dnbr), smoothed, dnbr).astype(np.float32)
    rd = dnbr / np.sqrt(np.abs(nbr_pre) + 1e-3)

    # 1. маски
    ok = np.ones(shape, dtype=bool)
    if valid is not None:
        ok &= np.asarray(valid).astype(bool)
    ok &= landcover != LC_WATER
    dnbr_valid = dnbr.copy()
    if scl is not None and p.mask_scl:
        scl = np.asarray(scl)
        ok &= ~np.isin(scl, SCL_BAD)
        ok &= scl != SCL_WATER
    if scl_pre is not None and p.mask_scl:
        # облако или тень в дате «до» портит dNBR так же, как в дате «после»
        scl_pre = np.asarray(scl_pre)
        ok &= ~np.isin(scl_pre, SCL_BAD)
        ok &= scl_pre != SCL_WATER
    ok &= np.isfinite(nbr_pre) & np.isfinite(nbr_post)
    ok &= (nbr_pre != 0.0) | (nbr_post != 0.0)  # обе даты с нулевым отражением — нет наблюдения
    dnbr_valid[~ok] = np.nan

    # 2. пороги по типу покрова
    codes = group_codes()
    severity = np.zeros(shape, dtype=np.uint8)
    for group, thresholds in p.groups().items():
        if group == "other" and thresholds[0] >= 1.0:
            continue
        t1, t2, t3 = thresholds
        mask_group = np.zeros(shape, dtype=bool)
        for code, name in codes.items():
            if name == group:
                mask_group |= landcover == code
        if not mask_group.any():
            continue
        d = np.where(np.isnan(dnbr_valid) | ~mask_group, -np.inf, dnbr_valid)
        severity = np.where((d >= t1) & (d < t2), 1, severity).astype(np.uint8)
        severity = np.where((d >= t2) & (d < t3), 2, severity).astype(np.uint8)
        severity = np.where(d >= t3, 3, severity).astype(np.uint8)

    # 3. подтверждение и восстановление по радиолокации
    sar_info = {"s1_used": False}
    if p.use_sar and s1_vh_pre is not None and s1_vh_post is not None:
        vh_pre = np.asarray(s1_vh_pre, dtype=np.float32)
        vh_post = np.asarray(s1_vh_post, dtype=np.float32)
        # Постановка допускает оба формата: дБ как есть и дБ ×100 (int16).
        # Масштаб проверяется по обеим датам: после пожара отдельные пиксели могут
        # быть нулевыми, и решение по одной дате было бы ошибочным.
        magnitude = max(
            float(np.nanmax(np.abs(vh_pre))) if np.isfinite(vh_pre).any() else 0.0,
            float(np.nanmax(np.abs(vh_post))) if np.isfinite(vh_post).any() else 0.0,
        )
        if magnitude > 100:
            vh_pre, vh_post = vh_pre / 100.0, vh_post / 100.0
        d_vh = vh_post - vh_pre
        forest = np.isin(landcover, LANDCOVER_GROUPS["forest"])
        # чувствительность: в лесу потеря биомассы даёт заметно большее падение VH
        scale = np.where(forest, p.sar_scale[0], p.sar_scale[1]).astype(np.float32)
        dnbr_sar = -d_vh / np.maximum(scale, 1e-3)
        dnbr_sar = np.where(np.isfinite(dnbr_sar), dnbr_sar, 0.0).astype(np.float32)
        sar_info = {
            "s1_used": True,
            "median_dvh": float(np.nanmedian(d_vh)),
            "sar_recovered_px": 0,
        }

        if p.sar_upgrade:
            # слабый оптический класс + сильное падение VH → средняя степень (в лесу)
            weak = (severity == 1) & forest & (dnbr_sar >= p.groups()["forest"][1])
            severity = np.where(weak, np.uint8(2), severity).astype(np.uint8)

        if p.cloud_gate and p.sar_recovery:
            # Пиксель заблокирован оптикой (облако, тень, SCL) либо dNBR недостоверен,
            # но радиолокация видит потерю биомассы: восстанавливаем класс по ΔVH.
            # Требуется собственная валидность (SAR не зависит от облачности, но зависит
            # от наличия наблюдения) и не водный тип покрова.
            sar_ok = (landcover != LC_WATER) & np.isfinite(d_vh)
            if valid is not None:
                sar_ok &= np.asarray(valid).astype(bool)
            drop_db = -d_vh
            recover = (~ok) & sar_ok & (drop_db >= p.sar_min_drop_db)
            if recover.any():
                t1, t2, t3 = p.sar_class_thresholds
                values = np.where(recover, dnbr_sar, -np.inf)
                severity = np.where((values >= t1) & (values < t2), np.uint8(1), severity).astype(np.uint8)
                severity = np.where((values >= t2) & (values < t3), np.uint8(2), severity).astype(np.uint8)
                severity = np.where(values >= t3, np.uint8(3), severity).astype(np.uint8)
                sar_info["sar_recovered_px"] = int((recover & (severity > 0)).sum())

    # 4. постобработка
    severity = postprocess(severity, p)

    if return_details:
        return severity, {
            "valid_frac": float(ok.mean()),
            "dnbr_median": float(np.nanmedian(dnbr_valid)) if np.isfinite(dnbr_valid).any() else None,
            "dnbr_p95": float(np.nanpercentile(dnbr_valid, 95)) if np.isfinite(dnbr_valid).any() else None,
            "rdnbr_median": float(np.nanmedian(rd)) if np.isfinite(rd).any() else None,
            "areas_px": {int(k): int((severity == k).sum()) for k in (0, 1, 2, 3)},
            **sar_info,
        }
    return severity


def postprocess(severity: np.ndarray, params: BSParams | None = None) -> np.ndarray:
    """Морфология поверх решения о степени поражения: заполнение дыр и чистка мелких контуров.

    Общая для обоих режимов BS (порогов и обучаемой модели) — иначе сравнение режимов было бы
    нечестным: часть разницы давала бы именно постобработка.
    """

    p = params or BSParams()
    burned = severity > 0
    if p.fill_holes_mask:
        burned = fill_holes(burned)
    if p.min_patch_px > 1:
        burned = remove_small_objects(burned, p.min_patch_px)
    severity = np.where(burned, severity, 0).astype(np.uint8)
    # после заполнения дыр могли появиться пиксели класса 0 внутри контура — присваиваем им слабую степень
    holes = burned & (severity == 0)
    return np.where(holes, np.uint8(1), severity).astype(np.uint8)


def classify_chip(chip, params: BSParams | None = None, return_details: bool = False, severity_model=None):
    """Обёртка над :func:`classify` для :class:`fires.chips.Chip`.

    Если передан ``severity_model`` (обученная модель степени поражения), решение принимает она,
    а пороги используются только как запасной вариант. Постобработка в обоих случаях одна и та же.
    """

    if severity_model is not None:
        severity = severity_model.predict_chip(chip)
        severity = postprocess(severity, params)
        if return_details:
            return severity, {
                "mode": "model",
                "areas_px": {int(k): int((severity == k).sum()) for k in (0, 1, 2, 3)},
            }
        return severity

    pre, post = chip.pre, chip.post
    layers = chip.layers
    if post and "SCL" in post:
        layers.setdefault("SCL", post["SCL"])        # маска облаков на дату «после»
    if pre and "SCL" in pre:
        layers.setdefault("SCL_pre", pre["SCL"])     # и на дату «до»
    if not pre or not post:
        # единый стек: B8A/B12 лежат в layers, даты не разделены — тогда функции нужны явные данные
        pre = {k: v for k, v in layers.items() if k in {"B8A", "B12"}}
        post = pre
        raise ValueError(
            f"{chip.chip_id}: не найдены слои pre/post. Запустите `python -m fires.chips --data-dir <путь>` "
            "и при необходимости дополните правила распознавания имён в fires/chips.py"
        )
    if "B8A" not in pre or "B12" not in pre:
        raise ValueError(f"{chip.chip_id}: в слоях pre нет B8A/B12: {sorted(pre)}")
    out = classify(
        nbr_pre=nbr(scale_reflectance(pre["B8A"]), scale_reflectance(pre["B12"])),
        nbr_post=nbr(scale_reflectance(post["B8A"]), scale_reflectance(post["B12"])),
        landcover=layers.get("landcover"),
        scl=layers.get("SCL"),
        scl_pre=layers.get("SCL_pre"),
        valid=layers.get("valid"),
        s1_vh_pre=chip.s1_pre.get("VH"),
        s1_vh_post=chip.s1_post.get("VH"),
        params=params,
        return_details=return_details,
    )
    return out


def to_params(values: dict) -> BSParams:
    allowed = {f for f in asdict(BSParams())}
    return BSParams(**{k: v for k, v in values.items() if k in allowed})
