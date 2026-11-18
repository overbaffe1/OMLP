"""Обучаемая модель степени поражения гари (классы 0–3) — второй режим модуля BS.

Зачем она нужна. Пороги dNBR — это то, как построен эталон организаторов, и на реальных данных они
работают: единая шкала USGS проигрывает подобранным по покрову порогам сразу 0.12 Score. Но порог — это
одно число на группу покрова, а решение о степени поражения зависит от многих факторов: запаса биомассы
(NDVI до пожара), сезона (что видно по состоянию пашни), влажности (NDMI, B11/B12), текстуры пятна
(огонь выжигает пятном, а фенология даёт россыпь), радиолокационного отклика (ΔVH не зависит от
облачности). Модель на градиентном бустинге учитывает всё это сразу.

Как устроено:

* признаки считаются по тем же формулам, что и в :mod:`fires.bs` (NBR, dNBR, rdNBR, NDVI, NDMI, каналы
  до/после, тип покрова, DEM, VV/VH и их разности, а также контекст dNBR в окне 15×15) — базовый набор
  :data:`FEATURE_NAMES_BASE` из 23 столбцов; расширенный :data:`FEATURE_NAMES_EXT` добавляет каналы
  видимого диапазона, индексы влажности, рельеф, локальную текстуру dNBR и сезон;
* обучение — бустинг по гистограммам: :class:`sklearn.ensemble.HistGradientBoostingClassifier` на CPU
  либо :class:`xgboost.XGBClassifier` (``device="cuda"``) на GPU, если он есть; оба варианта
  складываются в один и тот же объект :class:`SeverityModel`;
* предсказание — только внутри области интереса: пиксели, где dNBR даже близко не поднялся к порогу
  горения, заведомо «не горели», модель им не нужна (ROI-гейт ускоряет инференс в ~2 раза и на
  валидации ничего не ухудшает);
* при желании к вероятностям классов добавляются смещения (``class_bias``): это позволяет подтянуть
  слабый класс 1 (низкая степень), не переобучая модель;
* несколько моделей можно объединить в ансамбль (``members``) — предсказания усредняются по
  вероятностям;
* режим выбирается на отложенной части по событиям пожара: если модель не выигрывает у порогов,
  ``train.py`` оставляет пороги (``bs_mode: thresholds``).

Файл модели (``weights/severity_model.pkl``) — это обычный pickle с оценщиком и метаданными; он
собирается из кода в репозитории, никаких внешних загрузок в инференсе не происходит.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from fires import bs as bs_module

#: Базовые имена признаков (порядок фиксирован — он же порядок столбцов матрицы).
FEATURE_NAMES_BASE: tuple[str, ...] = (
    "dnbr", "rdnbr", "nbr_pre", "nbr_post", "ndvi_pre", "ndvi_post", "dndvi",
    "ndmi_pre", "ndmi_post", "b12_post", "b11_post", "b8a_pre", "b8a_post",
    "landcover", "dem", "vh_pre", "vh_post", "dvh", "vv_pre", "vv_post", "dvv",
    "dnbr_ctx15", "dnbr_max15",
)

#: Расширенный набор: то же плюс каналы видимого/красного края, влажность, рельеф, текстура и сезон.
#: Смысл добавки: 65–68 % ошибок контура приходится на пашню, а пашня отличается от гари не глубиной
#: dNBR, а сопутствующим состоянием поверхности (влажность B11/B12, зелёность B3/B4/B5/B7, рельеф
#: ровный) и сезоном — этих осей у базового набора не было.
FEATURE_NAMES_EXT: tuple[str, ...] = FEATURE_NAMES_BASE + (
    "b3_pre", "b3_post", "b4_pre", "b4_post", "b2_post", "b5_post", "b7_post",
    "ndwi_pre", "ndwi_post", "dndmi", "b12_b11_post",
    "dem_slope", "dem_rel", "dnbr_std15", "dnbr_contrast",
    "doy", "doy_cos",
)

#: Наборы признаков по имени (используются в ``train.py`` и ``tools/train_loop.py``).
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "base": FEATURE_NAMES_BASE,
    "ext": FEATURE_NAMES_EXT,
}

#: Имена признаков по умолчанию — базовый набор (обратная совместимость с первой версией модели).
FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES_BASE

#: Порог dNBR, при котором пиксель попадает в область интереса.
ROI_CUT = 0.05
#: Размер окна контекстных признаков (средний и максимальный dNBR вокруг пикселя).
CONTEXT_SIZE = 15
#: Размер окна расширения области интереса (пикселей); 0 — без расширения.
#: Расширение проверялось, но не оправдалось: при окне 15×15 в модель уходит вдвое больше
#: пикселей (0.55 против 0.34 от чипа) при том же качестве — дыры внутри контура всё равно
#: заполняет морфология.
ROI_SIZE = 0

#: Доступные устройства и бэкенды обучения.
DEVICE_CHOICES = ("auto", "cpu", "cuda")
BACKEND_CHOICES = ("auto", "sklearn", "xgboost")

_DEVICE_CACHE: dict[str, str] = {}


def _decibels(arr: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """Приводит канал Sentinel-1 к дБ (в поставке встречается и «дБ ×100»)."""

    if arr is None:
        return np.zeros(shape, dtype=np.float32)
    data = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(data)
    if finite.any() and float(np.nanmax(np.abs(data[finite]))) > 100:
        data = data / 100.0
    return np.nan_to_num(data, nan=0.0)


# ---------------------------------------------------------------- устройства и бэкенды

#: Покровы, на которых dNBR чаще всего путает пашню и степь с гарью (ESA WorldCover).
CONFUSABLE_LANDCOVER: tuple[int, ...] = (30, 40, 60)


def smooth_probabilities(proba: np.ndarray, indices: np.ndarray, shape: tuple[int, int],
                         radius: int) -> np.ndarray:
    """Усредняет вероятности классов по окну ``(2·radius+1)²`` внутри области интереса.

    Решение по пикселю принимается не только по его спектру, но и по окружению: одиночные
    «выбросы» класса после такого усреднения теряют перевес. Пиксели вне области интереса в
    среднее не входят (иначе край ROI тянуло бы к фону), поэтому знаменатель — доля пикселей
    ROI в окне.
    """

    radius = int(radius)
    if radius <= 0 or indices.size == 0:
        return proba
    h, w = int(shape[0]), int(shape[1])
    size = 2 * radius + 1
    valid = np.zeros(h * w, dtype=np.float32)
    valid[indices] = 1.0
    denom = ndimage.uniform_filter(valid.reshape(h, w), size=size, mode="constant", cval=0.0)
    denom = np.maximum(denom, 1e-6)
    out = np.empty_like(proba, dtype=np.float32)
    for cls in range(proba.shape[1]):
        canvas = np.zeros(h * w, dtype=np.float32)
        canvas[indices] = proba[:, cls]
        num = ndimage.uniform_filter(canvas.reshape(h, w), size=size, mode="constant", cval=0.0)
        out[:, cls] = (num / denom).reshape(-1)[indices]
    total = np.maximum(out.sum(axis=1, keepdims=True), 1e-6)
    return out / total


def majority_classes(classes: np.ndarray, indices: np.ndarray, shape: tuple[int, int],
                     radius: int, iterations: int = 1) -> np.ndarray:
    """Фильтр большинства по карте классов: класс пикселя выбирает окно ``(2·radius+1)²``.

    Отличие от :func:`smooth_probabilities`: тот усредняет уверенность модели, а этот голосует
    уже за классы, поэтому убирает «соль и перец» — одиночные пиксели другого класса внутри
    однородного пятна гари.
    """

    radius = int(radius)
    if radius <= 0 or indices.size == 0:
        return classes
    h, w = int(shape[0]), int(shape[1])
    size = 2 * radius + 1
    valid = np.zeros(h * w, dtype=np.float32)
    valid[indices] = 1.0
    valid_map = valid.reshape(h, w)
    current = np.zeros(h * w, dtype=np.uint8)
    current[indices] = classes
    for _ in range(max(1, int(iterations))):
        votes = np.stack([
            ndimage.uniform_filter((current.reshape(h, w) == cls).astype(np.float32),
                                   size=size, mode="constant", cval=0.0)
            for cls in range(4)
        ], axis=0)
        chosen = np.argmax(votes, axis=0).astype(np.uint8)
        current = np.where(valid_map > 0, chosen, 0).astype(np.uint8).reshape(-1)
    return current[indices]


def fill_degree_holes(classes: np.ndarray, indices: np.ndarray, shape: tuple[int, int],
                      max_px: int) -> np.ndarray:
    """Доливает чёрные точки: дыры внутри однородного пятна степени ``cls``.

    Дырой считается связная область (в том числе класс 0), окружённая пятном степени
    ``cls`` и не касающаяся края полотна. Области крупнее ``max_px`` не трогаются —
    это осмысленные пропуски (ROI-гейт, пропуски в данных), а не соль.
    """

    max_px = int(max_px)
    if max_px <= 0 or indices.size == 0:
        return classes
    h, w = int(shape[0]), int(shape[1])
    full = np.zeros(h * w, dtype=np.uint8)
    full[indices] = classes
    grid = full.reshape(h, w).copy()
    for cls in (3, 2, 1):                     # старшие степени приоритетнее
        zone = (grid == cls)
        if not zone.any():
            continue
        filled = ndimage.binary_fill_holes(zone)
        candidates = filled & (grid < cls)
        if not candidates.any():
            continue
        labels, count = ndimage.label(candidates)
        if not count:
            continue
        sizes = np.bincount(labels.reshape(-1))
        small = np.zeros(count + 1, dtype=bool)
        small[1:] = sizes[1:] <= max_px
        fill_mask = small[labels] & candidates
        grid[fill_mask] = cls
    return grid.reshape(-1)[indices]


def demote_small_patches(classes: np.ndarray, indices: np.ndarray, shape: tuple[int, int],
                         min_px: int, classes_to_demote: tuple[int, ...] = (3,)) -> np.ndarray:
    """Понижает класс у слишком мелких пятен высокой степени поражения.

    Пятно из нескольких пикселей не бывает «сильно выгоревшим»: в разметке это чаще всего край
    настоящего контура или пиксель, где dNBR сложился случайно. Такие пятна понижаются на один
    класс (3 → 2, 2 → 1), что улучшает и IoU_burn, и mIoU_sev.
    """

    min_px = int(min_px)
    if min_px <= 1 or indices.size == 0:
        return classes
    h, w = int(shape[0]), int(shape[1])
    out = classes.copy()
    for cls in classes_to_demote:
        binary = np.zeros(h * w, dtype=bool)
        binary[indices] = out == cls
        grid = binary.reshape(h, w)
        labels, count = ndimage.label(grid)
        if not count:
            continue
        sizes = np.bincount(labels.reshape(-1))
        small = np.zeros(count + 1, dtype=bool)
        small[1:] = sizes[1:] < min_px
        demote = small[labels.reshape(-1)] & grid.reshape(-1)
        out[demote[indices]] = np.uint8(cls - 1)
    return out


def classify_from_proba(proba: np.ndarray, indices: np.ndarray, shape: tuple[int, int],
                        bias=None, smooth: int = 0, majority: int = 0,
                        majority_iters: int = 1, demote_min_px: int = 0,
                        fill_holes_px: int = 0) -> np.ndarray:
    """Вероятности ROI → классы 0–3 (смещения, сглаживание вероятностей, фильтр большинства).

    Один и тот же путь используется моделью при инференсе и циклом обучения при переборе
    постобработки — иначе метрика на отложенной части не соответствовала бы ответу.
    """

    if indices.size == 0:
        return np.zeros(0, dtype=np.uint8)
    scored = proba
    if int(smooth) > 0:
        scored = smooth_probabilities(np.asarray(scored, dtype=np.float32),
                                      indices, shape, int(smooth))
    if bias is not None:
        scored = np.asarray(scored, dtype=np.float32) + np.asarray(bias, dtype=np.float32)
    classes = np.argmax(scored, axis=1).astype(np.uint8)
    if int(majority) > 0:
        classes = majority_classes(classes, indices, shape, int(majority), int(majority_iters))
    if int(demote_min_px) > 1:
        classes = demote_small_patches(classes, indices, shape, int(demote_min_px))
    if int(fill_holes_px) > 0:
        classes = fill_degree_holes(classes, indices, shape, int(fill_holes_px))
    return classes


def _cuda_dll_dirs() -> list[str]:
    """Каталоги с CUDA-библиотеками внутри Python (пакет ``nvidia-cuda-runtime-cu12``).

    На Windows xgboost не находит ``cudart64_12.dll``, пока её каталог не добавлен в PATH —
    типичная причина ответа «сборка с CUDA, но устройства нет» при исправном драйвере.
    """

    import glob
    import site
    import sys

    roots: list[str] = []
    for base in [sys.prefix, getattr(site, "getusersitepackages", lambda: "")()]:
        if base:
            roots.append(base)
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass
    dirs: list[str] = []
    for base in roots:
        for pattern in ("nvidia/*/bin", "nvidia/*/lib", "nvidia/*/lib/x64"):
            dirs.extend(glob.glob(os.path.join(base, pattern)))
    # нормpath: glob по маске с «/» на Windows отдаёт смешанные разделители, а add_dll_directory
    # такие пути принимает не всегда
    return sorted({os.path.normpath(d) for d in dirs if os.path.isdir(d)})


def _add_cuda_dll_dirs() -> list[str]:
    """Добавляет найденные каталоги CUDA-библиотек к путям поиска DLL. Возвращает список."""

    dirs = _cuda_dll_dirs()
    for directory in dirs:
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(directory)
            except Exception:
                pass
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
    return dirs


#: Сообщения, по которым xgboost сообщает, что GPU не задействован.
_CUDA_FALLBACK_MARKS = (
    "Device is changed from GPU to CPU",
    "No visible GPU is found",
    "no CUDA-capable device",
    "GPU is not available",
    "CUDA is not available",
)


def _nvidia_smi() -> dict:
    """Что говорит драйвер: есть ли ``nvidia-smi``, какая карта и версия драйвера.

    Это отдельный от xgboost источник правды: если карта видна здесь, проблема в CUDA-рантайме,
    а не в драйвере — и наоборот.
    """

    import shutil
    import subprocess

    path = shutil.which("nvidia-smi")
    info: dict = {"nvidia_smi": path, "gpu": None, "driver": None, "error": None}
    if not path:
        info["error"] = "nvidia-smi не найден: драйвер NVIDIA не установлен или не в PATH"
        return info
    try:
        out = subprocess.run(
            [path, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    text = (out.stdout or "").strip()
    if text:
        name, _, driver = text.splitlines()[0].partition(",")
        info["gpu"], info["driver"] = name.strip(), driver.strip()
    else:
        info["error"] = (out.stderr or "").strip()[:200] or "nvidia-smi не вернул данные о карте"
    return info


def _cuda_libraries() -> dict:
    """Какие CUDA-библиотеки реально загружаются, а какие нет.

    Отделяет три разные причины «CUDA не работает»: нет драйвера (не грузится ``nvcuda``),
    нет рантайма (не грузится ``cudart``) или всё на месте, но xgboost всё равно уходит на CPU.
    """

    import ctypes

    _add_cuda_dll_dirs()      # иначе на рабочей карте библиотеки видны как «нет»: каталоги не в PATH
    names = (["nvcuda.dll", "cudart64_12.dll", "cudart64_13.dll", "cublas64_12.dll",
              "cublasLt64_12.dll", "nvrtc64_120_0.dll"]
             if os.name == "nt" else
             ["libcuda.so.1", "libcudart.so.12", "libcudart.so.13", "libcublas.so.12"])
    out: dict[str, str] = {}
    for name in names:
        try:
            ctypes.CDLL(name)
            out[name] = "загружается"
        except OSError as exc:
            text = str(exc).splitlines()[0][:80] if str(exc) else exc.__class__.__name__
            out[name] = f"нет ({text})"
    return out


def _probe_cuda(xgb, verbose: bool = True) -> tuple[bool, str]:
    """Пробное обучение на ``device="cuda"``: работает ли CUDA здесь и сейчас.

    Признак отката берём у самого xgboost — он печатает «No visible GPU is found, setting device to
    CPU» и «Device is changed from GPU to CPU». Проверка только по ``save_config`` давала ложный
    минус на исправном GPU (в сборке 3.4 поле устройства в конфиге не отражает фактическое
    устройство), поэтому смотрим и на конфиг, и на сообщения.
    """

    import contextlib
    import io
    import re
    import warnings as _warnings

    rng = np.random.default_rng(0)
    x = rng.random((64, 3), dtype=np.float32)
    y = (x[:, 0] > 0.5).astype(np.int32)
    out, err = io.StringIO(), io.StringIO()
    try:
        # verbosity=2 всегда: при тихом режиме xgboost молча уходит на CPU, и отличить
        # настоящий GPU от отката невозможно (проверено на машине без видеокарты)
        probe = xgb.XGBClassifier(n_estimators=1, tree_method="hist", device="cuda", verbosity=2)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            probe.fit(x, y)
        raw = "\n".join([out.getvalue(), err.getvalue()]
                        + [str(item.message) for item in caught])
        text = " ".join(raw.split())
        _DEVICE_CACHE["xgboost_warning"] = text[:400]
        config = probe.get_booster().save_config()
        found = re.search(r'"device"\s*:\s*"([^"]+)"', config)
        in_config = bool(found) and found.group(1).lower().startswith("cuda")
        fell_back = any(mark.lower() in text.lower() for mark in _CUDA_FALLBACK_MARKS)
        if fell_back:
            return False, _fallback_reason(raw)
        # прямое подтверждение: GPU-структура данных xgboost (ellpack) или device=cuda в конфиге
        if in_config or "ellpack" in text.lower():
            return True, ""
        return False, ("не удалось подтвердить, что обучение шло на GPU (xgboost не сказал об этом "
                       "прямо) — считаем, что GPU нет")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _fallback_reason(text: str) -> str:
    """Самая содержательная строка из сообщений xgboost про откат на CPU."""

    lines = [part.strip() for part in text.replace("UserWarning:", "\n").splitlines() if part.strip()]
    hits = [line for line in lines
            if any(mark.lower() in line.lower() for mark in _CUDA_FALLBACK_MARKS)]
    for line in hits:
        if "WARNING" in line.upper():
            return line[:300]
    if hits:
        return hits[0][:300]
    for mark in _CUDA_FALLBACK_MARKS:      # текст одной строкой: берём фрагмент вокруг маркера
        position = text.lower().find(mark.lower())
        if position >= 0:
            return text[max(0, position - 60):position + 200].strip()
    return "xgboost не включил CUDA-устройство"


def gpu_report() -> dict:
    """Что известно про GPU: версия xgboost, собран ли он с CUDA и работает ли она здесь.

    Проверка «работает» — не только флаг сборки: ``xgboost`` может быть собран с CUDA, но не найти
    подходящий ``libcudart`` (типичная ситуация на чужой машине). Поэтому один раз выполняется
    микрообучение на одну итерацию, результат кэшируется.
    """

    info: dict = {"xgboost": None, "build_use_cuda": False, "runtime_ok": False, "reason": "",
                  "cuda_dll_dirs": [], "nvidia_smi": None, "gpu": None, "driver": None}
    try:
        import xgboost as xgb  # noqa: F401  (нужен только факт наличия)
    except Exception as exc:
        info["reason"] = f"xgboost не установлен ({exc.__class__.__name__})"
        return info
    info["xgboost"] = getattr(xgb, "__version__", "?")
    try:
        info["build_use_cuda"] = bool(xgb.build_info().get("USE_CUDA", False))
    except Exception:
        info["build_use_cuda"] = False
    if not info["build_use_cuda"]:
        info["reason"] = "сборка xgboost без CUDA (pip install xgboost — вил с поддержкой CUDA)"
        return info
    if "runtime" in _DEVICE_CACHE:
        info["runtime_ok"] = _DEVICE_CACHE["runtime"] == "cuda"
        info["reason"] = "" if info["runtime_ok"] else _DEVICE_CACHE.get("reason", "GPU недоступен")
        return info
    # Тонкость: xgboost не падает, если GPU не найден, а молча продолжает на CPU. Поэтому
    # после пробного обучения смотрим на устройство, которое реально записано в конфиг бустера.
    info["runtime_ok"], reason = _probe_cuda(xgb)
    if not info["runtime_ok"]:
        # вторая попытка — с каталогами CUDA-библиотек из pip-пакета nvidia-* в путях поиска DLL
        added = _add_cuda_dll_dirs()
        if added:
            again, why = _probe_cuda(xgb, verbose=True)
            if again:
                info["runtime_ok"], reason = True, ""
                info["cuda_dll_dirs"] = added
            else:
                reason = (f"{reason}; после добавления каталогов "
                          f"{', '.join(added[:2])} — {why}")
    if not info["runtime_ok"]:
        hint = ("проверьте драйвер: nvidia-smi должен показать карту; если команды нет — "
                "поставьте драйвер NVIDIA, если есть — обновите его")
        reason = f"{reason} ({hint})"
        # подробная проба: xgboost с verbosity=2 сам называет причину (нет DLL, нет устройства,
        # несовпадение версий) — это точнее любых догадок
        _, detailed = _probe_cuda(xgb, verbose=True)
        if detailed:
            reason += f". Подробный ответ xgboost: {detailed}"
        warning = _DEVICE_CACHE.get("xgboost_warning")
        if warning:
            reason += f". Сообщение xgboost: {warning}"
    info["cuda_dll_dirs"] = info.get("cuda_dll_dirs") or _cuda_dll_dirs()
    info["cuda_libraries"] = _cuda_libraries()
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        info["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
    info.update({k: v for k, v in _nvidia_smi().items() if k not in info or info[k] is None})
    _DEVICE_CACHE["runtime"] = "cuda" if info["runtime_ok"] else "cpu"
    _DEVICE_CACHE["reason"] = reason
    info["reason"] = reason
    return info


def resolve_device(spec: str = "auto") -> str:
    """Превращает ``auto|cpu|cuda`` в конкретное устройство, молча откатываясь на CPU."""

    spec = (spec or "auto").lower()
    if spec in ("gpu", "cuda"):
        spec = "cuda"
    if spec == "cpu":
        return "cpu"
    if spec == "auto" and os.environ.get("COSMOH_DEVICE"):
        spec = os.environ["COSMOH_DEVICE"].lower()
    if spec == "cuda":
        return "cuda" if gpu_report()["runtime_ok"] else "cpu"
    if spec == "auto":
        return "cuda" if gpu_report()["runtime_ok"] else "cpu"
    raise ValueError(f"неизвестное устройство: {spec!r}")


def pick_backend(device: str, backend: str = "auto") -> str:
    """Бэкенд обучения: на GPU — xgboost (sklearn GPU не умеет), на CPU — тоже xgboost, если просили."""

    backend = (backend or "auto").lower()
    if backend not in BACKEND_CHOICES:
        raise ValueError(f"неизвестный бэкенд: {backend!r}")
    if backend == "auto":
        return "xgboost" if device == "cuda" else "sklearn"
    if backend == "xgboost" and device == "cuda" and not gpu_report()["runtime_ok"]:
        return "sklearn"
    return backend


def make_estimator(backend: str, device: str, params: dict, seed: int):
    """Создаёт оценщик нужного бэкенда с похожими гиперпараметрами.

    ``params`` — общий словарь (``iterations``, ``learning_rate``, ``leaves``, ``min_samples_leaf``,
    ``l2``, ``subsample``, ``colsample``); для xgboost он переводится в его собственные имена.
    """

    iterations = int(params.get("iterations", 60))
    lr = float(params.get("learning_rate", 0.2))
    leaves = int(params.get("leaves", 24))
    min_leaf = int(params.get("min_samples_leaf", 80))
    l2 = float(params.get("l2", 1.0))
    kind = str(params.get("kind", "classifier"))
    if backend == "xgboost":
        import xgboost as xgb

        if kind == "regressor":
            return xgb.XGBRegressor(
                n_estimators=iterations,
                learning_rate=lr,
                max_depth=int(params.get("max_depth", 0)) or max(3, int(np.log2(max(leaves, 2)))),
                max_leaves=leaves,
                grow_policy="lossguide",
                min_child_weight=float(params.get("min_child_weight", max(1.0, min_leaf / 20.0))),
                reg_lambda=l2,
                subsample=float(params.get("subsample", 0.8)),
                colsample_bytree=float(params.get("colsample", 0.9)),
                tree_method="hist",
                device=device,
                objective="reg:squarederror",
                n_jobs=int(params.get("n_jobs", 0)) or os.cpu_count() or 4,
                random_state=seed,
                verbosity=0,
            )

        return xgb.XGBClassifier(
            n_estimators=iterations,
            learning_rate=lr,
            max_depth=int(params.get("max_depth", 0)) or max(3, int(np.log2(max(leaves, 2)))),
            max_leaves=leaves,
            grow_policy="lossguide",
            min_child_weight=float(params.get("min_child_weight", max(1.0, min_leaf / 20.0))),
            reg_lambda=l2,
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample", 0.9)),
            tree_method="hist",
            device=device,
            objective="multi:softprob",
            num_class=4,
            n_jobs=int(params.get("n_jobs", 0)) or os.cpu_count() or 4,
            random_state=seed,
            verbosity=0,
        )
    if kind == "regressor":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=iterations,
            learning_rate=lr,
            max_leaf_nodes=leaves,
            min_samples_leaf=min_leaf,
            l2_regularization=l2,
            random_state=seed,
            early_stopping=False,
        )
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=iterations,
        learning_rate=lr,
        max_leaf_nodes=leaves,
        min_samples_leaf=min_leaf,
        l2_regularization=l2,
        random_state=seed,
        early_stopping=False,
    )


def set_predict_device(estimator, device: str) -> None:
    """Переключает устройство предсказания у xgboost-модели (у sklearn устройства нет)."""

    if type(estimator).__module__.startswith("xgboost"):
        try:
            estimator.get_booster().set_param({"device": device})
        except Exception:
            pass


# ---------------------------------------------------------------- признаки

def feature_names_for(spec: str | tuple[str, ...]) -> tuple[str, ...]:
    """``"base"``/``"ext"`` (или готовый кортеж имён) → кортеж имён признаков."""

    if isinstance(spec, str):
        if spec not in FEATURE_SETS:
            raise ValueError(f"неизвестный набор признаков: {spec!r} (есть {sorted(FEATURE_SETS)})")
        return FEATURE_SETS[spec]
    return tuple(spec)


def _day_of_year(chip) -> float:
    """День года по метаданным чипа (0, если даты нет)."""

    for key in ("acq_datetime", "datetime", "date", "date_pre", "acq_time"):
        value = str(chip.meta.get(key, "") or "")
        if len(value) >= 10 and value[4] == "-" and value[7] == "-":
            try:
                import datetime as _dt

                date = _dt.date(int(value[0:4]), int(value[5:7]), int(value[8:10]))
                return float(date.timetuple().tm_yday)
            except Exception:
                continue
    return 0.0


def _feature_dict(chip) -> dict[str, np.ndarray]:
    """Считает все признаки чипа (словарь имя → массив ``H×W``)."""

    pre, post = chip.pre, chip.post
    scale = bs_module.scale_reflectance
    nbr_pre = bs_module.nbr(scale(pre["B8A"]), scale(pre["B12"]))
    nbr_post = bs_module.nbr(scale(post["B8A"]), scale(post["B12"]))
    dnbr = nbr_pre - nbr_post
    rd = dnbr / np.sqrt(np.abs(nbr_pre) + 1e-3)

    def index(a, b) -> np.ndarray:
        a, b = scale(a), scale(b)
        return (a - b) / (a + b + 1e-6)

    def band(date: str, name: str) -> np.ndarray:
        stack = pre if date == "pre" else post
        if name not in stack:
            return np.zeros(np.asarray(nbr_pre).shape, dtype=np.float32)
        return scale(stack[name])

    ndvi_pre = index(pre["B8A"], pre["B4"])
    ndvi_post = index(post["B8A"], post["B4"])
    landcover = chip.layers.get("landcover")
    if landcover is None:
        landcover = np.full(dnbr.shape, 30, dtype=np.int16)
    dem = chip.layers.get("DEM")
    if dem is None:
        dem = np.zeros(dnbr.shape, dtype=np.float32)
    dem = np.asarray(dem, dtype=np.float32)

    vh_pre = _decibels(chip.s1_pre.get("VH"), dnbr.shape)
    vh_post = _decibels(chip.s1_post.get("VH"), dnbr.shape)
    vv_pre = _decibels(chip.s1_pre.get("VV"), dnbr.shape)
    vv_post = _decibels(chip.s1_post.get("VV"), dnbr.shape)

    finite_dnbr = np.nan_to_num(dnbr, nan=0.0)
    ctx = ndimage.uniform_filter(finite_dnbr, size=CONTEXT_SIZE)
    ctx_max = ndimage.maximum_filter(np.nan_to_num(dnbr, nan=-1.0), size=CONTEXT_SIZE)
    ctx_sq = ndimage.uniform_filter(finite_dnbr ** 2, size=CONTEXT_SIZE)
    dnbr_std = np.sqrt(np.clip(ctx_sq - ctx ** 2, 0.0, None))

    ndwi_pre = index(pre["B3"], pre["B8A"]) if "B3" in pre else np.zeros(dnbr.shape, dtype=np.float32)
    ndwi_post = index(post["B3"], post["B8A"]) if "B3" in post else np.zeros(dnbr.shape, dtype=np.float32)
    ndmi_pre = index(pre["B8A"], pre["B11"])
    ndmi_post = index(post["B8A"], post["B11"])
    b12_post = band("post", "B12")
    b11_post = band("post", "B11")
    doy = _day_of_year(chip)

    return {
        "dnbr": dnbr, "rdnbr": rd, "nbr_pre": nbr_pre, "nbr_post": nbr_post,
        "ndvi_pre": ndvi_pre, "ndvi_post": ndvi_post, "dndvi": ndvi_post - ndvi_pre,
        "ndmi_pre": ndmi_pre, "ndmi_post": ndmi_post,
        "b12_post": b12_post, "b11_post": b11_post,
        "b8a_pre": scale(pre["B8A"]), "b8a_post": scale(post["B8A"]),
        "landcover": np.asarray(landcover, dtype=np.float32), "dem": dem,
        "vh_pre": vh_pre, "vh_post": vh_post, "dvh": vh_post - vh_pre,
        "vv_pre": vv_pre, "vv_post": vv_post, "dvv": vv_post - vv_pre,
        "dnbr_ctx15": ctx, "dnbr_max15": ctx_max,
        # расширенный набор
        "b3_pre": band("pre", "B3"), "b3_post": band("post", "B3"),
        "b4_pre": band("pre", "B4"), "b4_post": band("post", "B4"),
        "b2_post": band("post", "B2"), "b5_post": band("post", "B5"),
        "b7_post": band("post", "B7"),
        "ndwi_pre": ndwi_pre, "ndwi_post": ndwi_post, "dndmi": ndmi_post - ndmi_pre,
        "b12_b11_post": b12_post / (np.abs(b11_post) + 1e-6),
        "dem_slope": np.hypot(*np.gradient(dem)),
        "dem_rel": dem - float(np.median(dem)),
        "dnbr_std15": dnbr_std, "dnbr_contrast": finite_dnbr - ctx,
        "doy": np.full(dnbr.shape, doy, dtype=np.float32),
        "doy_cos": np.full(dnbr.shape, float(np.cos(2 * np.pi * doy / 365.0)) if doy else 0.0,
                           dtype=np.float32),
    }


def chip_features(chip, feature_names: str | tuple[str, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Считает матрицу признаков ``(H*W, F)`` и маску валидных пикселей для одного BS-чипа.

    ``feature_names`` — имя набора (``"base"``/``"ext"``) или готовый кортеж имён; по умолчанию
    базовый набор из :data:`FEATURE_NAMES_BASE`. Порядок столбцов всегда соответствует этому кортежу.
    """

    names = feature_names_for(feature_names or FEATURE_NAMES)
    values = _feature_dict(chip)
    matrix = np.stack([np.asarray(values[name], dtype=np.float32) for name in names], axis=-1)

    ok = np.ones(matrix.shape[:2], dtype=bool)
    ok &= np.asarray(values["landcover"]) != bs_module.LC_WATER
    for key in ("SCL", "SCL_pre"):
        scl = chip.layers.get(key)
        if scl is not None:
            scl = np.asarray(scl)
            ok &= ~np.isin(scl, bs_module.SCL_BAD)
            ok &= scl != bs_module.SCL_WATER
    ok &= (values["b8a_pre"] != 0) | (values["b8a_post"] != 0)
    return matrix.reshape(-1, matrix.shape[-1]), ok.reshape(-1)


def roi_mask(chip, cut: float = ROI_CUT, size: int = ROI_SIZE) -> np.ndarray:
    """Пиксели, где модель вообще вызывается: dNBR выше порога шума.

    Пиксели вне области интереса заведомо не горели (dNBR ниже самого мягкого порога степени
    для пашни), поэтому модель их не считает — это и ускоряет инференс, и не портит ответ:
    проверка на валидации показала, что качество от гейта не падает (0.5332 против 0.5288 по
    IoU_burn при заведомо меньшем числе строк).
    """

    pre, post = chip.pre, chip.post
    scale = bs_module.scale_reflectance
    dnbr = (bs_module.nbr(scale(pre["B8A"]), scale(pre["B12"]))
            - bs_module.nbr(scale(post["B8A"]), scale(post["B12"])))
    hot = np.nan_to_num(dnbr, nan=-1.0) > cut
    return ndimage.maximum_filter(hot, size=size) if size else hot


# ---------------------------------------------------------------- модель

@dataclass
class SeverityModel:
    """Обёртка над оценщиком: предсказание степени поражения для чипа.

    :param estimator: обученный оценщик с методами ``predict`` и ``predict_proba``;
    :param kind: ``"classifier"`` (многоклассовый 0–3) или ``"regressor"`` (непрерывная степень,
        переводится в классы порогами 0.5/1.5/2.5);
    :param cuts: пороги перевода регрессии в классы;
    :param metrics: метрики, полученные на отложенной части (для отчёта);
    :param feature_names: в каком порядке и составе считать признаки чипа;
    :param backend: ``"sklearn"`` или ``"xgboost"`` (нужно для отчёта и для переключения устройства);
    :param device: где модель обучалась и где предсказывает (``"cpu"``/``"cuda"``);
    :param class_bias: смещения вероятностей классов перед ``argmax`` (подтягивают слабые классы);
    :param members: список оценщиков ансамбля — вероятности усредняются.
    """

    estimator: object = None
    kind: str = "classifier"
    cuts: tuple[float, float, float] = (0.5, 1.5, 2.5)
    roi_cut: float = ROI_CUT
    roi_size: int = ROI_SIZE
    metrics: dict = field(default_factory=dict)
    feature_names: tuple[str, ...] = FEATURE_NAMES_BASE
    backend: str = "sklearn"
    device: str = "cpu"
    class_bias: tuple[float, float, float, float] | None = None
    members: list = field(default_factory=list)
    gate: object = None          # бинарная голова «гарь / не гарь» поверх класса 0
    gate_members: list = field(default_factory=list)
    proba_smooth: int = 0        # радиус усреднения вероятностей по окружению (0 — выключено)
    majority_filter: int = 0     # радиус фильтра большинства по карте классов (0 — выключено)
    majority_iters: int = 1
    demote_min_px: int = 0       # мелкие пятна класса 2/3 понижаются на класс (0 — выключено)
    fill_holes_px: int = 0       # дыры в степенях до N px доливаются окружением (0 — выключено)

    # --- предсказание

    def _estimators(self) -> list:
        return list(self.members) if self.members else [self.estimator]

    def proba_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Вероятности классов 0–3 для строк матрицы признаков (усредняются по ансамблю)."""

        estimators = [e for e in self._estimators() if e is not None]
        if not estimators:
            raise ValueError("в модели нет ни одного оценщика")
        total = None
        for estimator in estimators:
            if self.backend != "regressor":
                set_predict_device(estimator, resolve_device(self.device))
            part = np.asarray(estimator.predict_proba(matrix) if hasattr(estimator, "predict_proba")
                              else estimator.predict(matrix), dtype=np.float32)
            if part.ndim == 2 and part.shape[1] != 4:
                # Модель, обученная не на всех четырёх классах (бывает на синтетике и на
                # маленьких подвыборках), должна давать полный вектор вероятностей.
                full = np.zeros((part.shape[0], 4), dtype=np.float32)
                classes = np.asarray(getattr(estimator, "classes_", np.arange(part.shape[1])), dtype=int)
                for column, cls in enumerate(classes):
                    if 0 <= int(cls) < 4 and column < part.shape[1]:
                        full[:, int(cls)] = part[:, column]
                part = full
            total = part if total is None else total + part
        proba = total / len(estimators)
        burn = self._burn_probability(matrix)
        if burn is not None:
            proba = _apply_burn_gate(proba, burn)
        return proba

    def _burn_probability(self, matrix: np.ndarray) -> np.ndarray | None:
        """Вероятность «гарь» по бинарной голове (усредняется по ансамблю голов), иначе None."""

        gates = [g for g in (self.gate_members or [self.gate]) if g is not None]
        if not gates:
            return None
        total = None
        for gate in gates:
            set_predict_device(gate, resolve_device(self.device))
            part = np.asarray(gate.predict_proba(matrix), dtype=np.float32)
            column = 1 if part.ndim == 2 and part.shape[1] > 1 else 0
            values = part[:, column]
            total = values if total is None else total + values
        return total / len(gates)

    def _classes(self, raw: np.ndarray) -> np.ndarray:
        if self.kind == "regressor":
            out = np.zeros(raw.shape, dtype=np.uint8)
            for cls, cut in zip((1, 2, 3), self.cuts):
                out[raw >= cut] = cls
            return out
        return raw.astype(np.uint8)

    def _classes_from_proba(self, proba: np.ndarray) -> np.ndarray:
        if self.class_bias is not None:
            proba = proba + np.asarray(self.class_bias, dtype=np.float32)
        return np.argmax(proba, axis=1).astype(np.uint8)

    def predict_chip(self, chip) -> np.ndarray:
        """Степень поражения 0–3 для чипа (пиксели вне области интереса — 0)."""

        matrix, ok = chip_features(chip, self.feature_names)
        shape = np.asarray(chip.pre["B8A"]).shape
        hot = roi_mask(chip, self.roi_cut, self.roi_size).reshape(-1)
        indices = np.flatnonzero(np.asarray(ok).reshape(-1) & hot)
        out = np.zeros(matrix.shape[0], dtype=np.uint8)
        if indices.size:
            if self.kind == "regressor":
                out[indices] = self._classes(np.asarray(self.estimator.predict(matrix[indices])))
            else:
                out[indices] = classify_from_proba(
                    self.proba_matrix(matrix[indices]), indices, (int(shape[0]), int(shape[1])),
                    bias=self.class_bias, smooth=self.proba_smooth,
                    majority=self.majority_filter, majority_iters=self.majority_iters,
                    demote_min_px=self.demote_min_px, fill_holes_px=self.fill_holes_px)
        return out.reshape(shape)

    def predict_matrix(self, matrix: np.ndarray, ok: np.ndarray, hot: np.ndarray) -> np.ndarray:
        """То же, но по готовой матрице признаков (используется при обучении и оценке)."""

        indices = np.flatnonzero(np.asarray(ok).reshape(-1) & np.asarray(hot).reshape(-1))
        out = np.zeros(matrix.shape[0], dtype=np.uint8)
        if indices.size:
            if self.kind == "regressor":
                out[indices] = self._classes(np.asarray(self.estimator.predict(matrix[indices])))
            else:
                out[indices] = self._classes_from_proba(self.proba_matrix(matrix[indices]))
        return out

    def describe(self) -> str:
        """Короткая строка для отчётов: бэкенд, устройство, число моделей, набор признаков."""

        n = len(self.members) if self.members else 1
        bias = "" if self.class_bias is None else " + смещения классов"
        if self.proba_smooth:
            side = 2 * int(self.proba_smooth) + 1
            bias += f" + сглаживание вероятностей {side}×{side}"
        if self.majority_filter:
            side = 2 * int(self.majority_filter) + 1
            bias += f" + фильтр большинства {side}×{side}"
        if self.demote_min_px:
            bias += f" + мелкие пятна класса 3 < {int(self.demote_min_px)} px понижаются"
        if self.fill_holes_px:
            bias += f" + дыры в степенях до {int(self.fill_holes_px)} px доливаются"
        if self.gate is not None or self.gate_members:
            bias += " + бинарная голова «гарь/не гарь»"
        return (f"{self.backend}/{self.device}, признаков {len(self.feature_names)}, "
                f"моделей {n}{bias}")

    # --- сериализация

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=5)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SeverityModel":
        with Path(path).open("rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, cls):
            raise TypeError(f"{path}: ожидался SeverityModel, получено {type(model)!r}")
        # модели, сохранённые до появления устройств/наборов признаков, читаются как есть
        for name, default in (("feature_names", FEATURE_NAMES_BASE), ("backend", "sklearn"),
                              ("device", "cpu"), ("class_bias", None), ("members", []),
                              ("proba_smooth", 0), ("majority_filter", 0), ("majority_iters", 1),
                              ("demote_min_px", 0), ("fill_holes_px", 0), ("gate", None), ("gate_members", [])):
            if not hasattr(model, name) or getattr(model, name) is None and name == "members":
                setattr(model, name, default)
        return model


# ---------------------------------------------------------------- обучение

def collect_training_pixels(dataset, chip_ids: list[str], pixels_per_chip: int,
                            feature_names: str | tuple[str, ...] = "base",
                            seed: int = 20260918, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Собирает обучающую матрицу: разреженная подвыборка пикселей по каждому чипу.

    Пиксели берутся разреженно, потому что количество положительных пикселей на чипах различается
    на порядки; ``pixels_per_chip=0`` — брать все валидные пиксели чипа.
    """

    names = feature_names_for(feature_names)
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for chip_id in chip_ids:
        chip = dataset.load(chip_id)
        if chip.mask_sev is None:
            continue
        matrix, ok = chip_features(chip, names)
        truth = np.asarray(chip.mask_sev, dtype=np.uint8).reshape(-1)
        indices = np.flatnonzero(ok)
        if pixels_per_chip and indices.size > pixels_per_chip:
            indices = rng.choice(indices, size=pixels_per_chip, replace=False)
        features.append(matrix[indices])
        labels.append(truth[indices])
    if not features:
        raise ValueError("не нашлось ни одного чипа с маской степени поражения")
    X = np.concatenate(features)
    y = np.concatenate(labels)
    if verbose:
        print(f"  обучающих пикселей: {X.shape[0]}, признаков: {X.shape[1]}, "
              f"классы: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def class_weights(y: np.ndarray, mode: str = "sqrt", X: np.ndarray | None = None,
                  hard_boost: float = 0.0,
                  landcover_column: int | None = None) -> np.ndarray:
    """Веса пикселей по частоте классов: ``sqrt``, ``inv``, ``none``, ``hardneg``.

    ``hardneg`` — тот же ``sqrt`` плюс надбавка пикселям класса 0 (не гарь) на пашне, степи и
    открытой почве: по разбору ошибок 66 % ложных пикселей BS приходится именно на эти покровы
    (фенологический dNBR там неотличим от слабого горения), и модели выгодно ошибаться на них
    реже. Для этого нужен столбец ``landcover`` в матрице признаков.
    """

    counts = np.bincount(y, minlength=4).astype(float)
    if mode == "none":
        weights = np.ones(4)
    elif mode == "inv":
        weights = 1.0 / np.maximum(counts, 1)
    else:
        weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    sample_weight = weights[y]
    if mode == "hardneg" and X is not None and hard_boost > 0:
        column = -1 if landcover_column is None else int(landcover_column)
        if 0 <= column < X.shape[1]:
            codes = np.rint(np.asarray(X[:, column], dtype=np.float32)).astype(int)
            confusable = np.isin(codes, CONFUSABLE_LANDCOVER) & (y == 0)
            sample_weight = sample_weight * np.where(confusable, 1.0 + float(hard_boost), 1.0)
    return sample_weight * (len(sample_weight) / sample_weight.sum())


def train_model(X: np.ndarray, y: np.ndarray, *, params: dict | None = None,
                backend: str = "auto", device: str = "auto", feature_names="base",
                weight_mode: str = "sqrt", seed: int = 20260918, verbose: bool = True) -> SeverityModel:
    """Обучает одну модель степени поражения и упаковывает её в :class:`SeverityModel`."""

    params = dict(params or {})
    device = resolve_device(device)
    backend = pick_backend(device, backend)
    estimator = make_estimator(backend, device, params, seed)
    names = feature_names_for(feature_names)
    landcover_column = names.index("landcover") if "landcover" in names else None
    sample_weight = class_weights(y, weight_mode, X=X,
                                  hard_boost=float(params.get("hard_boost", 0.0)),
                                  landcover_column=landcover_column)
    estimator.fit(X, y, sample_weight=sample_weight)
    gate = None
    if bool(params.get("gate")):
        # бинарная голова: «гарь (1–3) против не гари (0)» на тех же признаках и весах
        binary = (y > 0).astype(np.int32)
        gate_params = dict(params)
        gate_params["kind"] = "classifier"
        gate = make_estimator(backend, device, gate_params, seed + 1)
        gate.fit(X, binary, sample_weight=sample_weight)
    model = SeverityModel(
        gate=gate,
        estimator=estimator,
        kind=str(params.get("kind", "classifier")),
        feature_names=feature_names_for(feature_names),
        backend=backend,
        device=device,
        metrics={"backend": backend, "device": device, "params": params, "weight_mode": weight_mode,
                 "features": list(feature_names_for(feature_names)), "seed": seed},
    )
    if verbose:
        print(f"  модель обучена: {backend}/{device}, итераций {params.get('iterations')}, "
              f"{X.shape[0]} пикселей, {X.shape[1]} признаков")
    return model


def _apply_burn_gate(proba: np.ndarray, burn: np.ndarray) -> np.ndarray:
    """Пересобирает вероятности по бинарной голове: P(0) = 1 − P(гарь), классы 1–3 — внутри гари.

    Так слабое разделение «гарь / не гарь» (главный источник ошибок на пашне и в степи) уходит
    отдельной голове, а голова степени отвечает только за силу выгорания внутри уже найденной гари.
    """

    burn = np.clip(np.asarray(burn, dtype=np.float32).reshape(-1, 1), 1e-6, 1.0 - 1e-6)
    proba = np.asarray(proba, dtype=np.float32)
    inside = proba[:, 1:].sum(axis=1, keepdims=True)
    out = np.zeros_like(proba, dtype=np.float32)
    out[:, 0] = 1.0 - burn.reshape(-1)
    scale = np.where(inside > 1e-6, burn / np.maximum(inside, 1e-6), 0.0)
    out[:, 1:] = proba[:, 1:] * scale
    # если голова степени не дала ни одного горящего класса, делим вероятность гари поровну
    empty = inside.reshape(-1) <= 1e-6
    if np.any(empty):
        out[empty, 1:] = np.repeat(burn.reshape(-1)[empty] / 3.0, 3).reshape(-1, 3)
    return out


def ensemble_model(models: list[SeverityModel], metrics: dict | None = None) -> SeverityModel:
    """Собирает ансамбль: усреднение вероятностей моделей одного набора признаков.

    Модели должны быть обучены на одних и тех же столбцах (проверяется по ``feature_names``).
    """

    models = [m for m in models if m is not None]
    if not models:
        raise ValueError("ансамбль пуст")
    names = models[0].feature_names
    for model in models:
        if tuple(model.feature_names) != tuple(names):
            raise ValueError("в ансамбле модели с разными наборами признаков")
    result = SeverityModel(
        estimator=None,
        kind=models[0].kind,
        feature_names=tuple(names),
        backend=models[0].backend,
        device=models[0].device,
        members=[m.estimator for m in models],
        gate_members=[m.gate for m in models if m.gate is not None],
        metrics=dict(metrics or {}),
    )
    result.metrics.setdefault("ensemble_size", len(models))
    if result.gate_members:
        result.metrics.setdefault("ensemble_gates", len(result.gate_members))
    return result
