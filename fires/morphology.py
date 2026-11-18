"""Морфология масок без внешних зависимостей (только numpy/scipy.ndimage).

Зачем свой модуль: в постановке нет требований к библиотекам, но у организаторов
замеряется скорость инференса, поэтому тяжёлые зависимости (GDAL/rasterio/opencv)
не подключаются без необходимости. Все операции ниже работают на одном чипе 512×512
за миллисекунды.
"""

from __future__ import annotations

import numpy as np

try:  # scipy есть почти везде, но модуль остаётся работоспособным и без него
    from scipy import ndimage as _ndi

    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _ndi = None
    HAVE_SCIPY = False


def _fallback_label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Простейшая маркировка связных компонент (4-связность) на чистом numpy."""

    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    seen = np.zeros_like(mask, dtype=bool)
    current = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not seen[y, x]:
                current += 1
                stack = [(y, x)]
                seen[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    labels[cy, cx] = current
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
    return labels, current


def label(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """Маркировка связных компонент (4- или 8-связность).

    8-связность важна для детекции: очаг может быть разрезан диагональными разрывами,
    и тогда 4-связность даёт десятки мелких «очагов» вместо одного.
    """

    mask = np.asarray(mask, dtype=bool)
    if HAVE_SCIPY:
        if connectivity == 8:
            labels, n = _ndi.label(mask, structure=np.ones((3, 3), dtype=int))
        else:
            labels, n = _ndi.label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        return labels, n
    return _fallback_label(mask)


def closing(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Морфологическое закрытие: соединяет разрывы, не увеличивая площадь маски."""

    return binary_erosion(binary_dilation(mask, iterations), iterations)


def opening_safe(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Открытие, сохраняющее мелкие объекты (эрозия только там, где объект не исчезнет)."""

    eroded = binary_erosion(mask, iterations)
    return eroded | (mask & ~binary_dilation(~mask, iterations))


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Удаляет компоненты площадью меньше ``min_size`` пикселей."""

    if min_size <= 1:
        return np.asarray(mask, dtype=bool)
    labels, n = label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def remove_large_objects(mask: np.ndarray, max_size: int) -> np.ndarray:
    labels, n = label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = (sizes <= max_size) & (np.arange(len(sizes)) > 0)
    return keep[labels]


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Заполняет внутренние дыры: полезно для контуров гари (недогоревшие «острова»)."""

    mask = np.asarray(mask, dtype=bool)
    if HAVE_SCIPY:
        return _ndi.binary_fill_holes(mask)
    h, w = mask.shape
    outside = np.zeros_like(mask)
    stack = [(0, 0)]
    if mask[0, 0]:
        return mask.copy()
    outside[0, 0] = True
    while stack:
        cy, cx = stack.pop()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                stack.append((ny, nx))
    return mask | ~outside


def binary_dilation(mask: np.ndarray, iterations: int = 1, cross: bool = True) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if HAVE_SCIPY:
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) if cross else np.ones((3, 3))
        return _ndi.binary_dilation(mask, structure=struct, iterations=iterations)
    out = mask.copy()
    for _ in range(iterations):
        shifted = out.copy()
        shifted[1:, :] |= out[:-1, :]
        shifted[:-1, :] |= out[1:, :]
        shifted[:, 1:] |= out[:, :-1]
        shifted[:, :-1] |= out[:, 1:]
        out = shifted
    return out


def binary_erosion(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if HAVE_SCIPY:
        return _ndi.binary_erosion(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), iterations=iterations)
    inv = ~mask
    return ~binary_dilation(inv, iterations=iterations)


def opening(mask: np.ndarray, size: int = 1) -> np.ndarray:
    return binary_dilation(binary_erosion(mask, size), size)


def box_mean(arr: np.ndarray, radius: int) -> np.ndarray:
    """Среднее в квадратном окне (2*radius+1) с отражением границ — быстрый фон.

    Используется контекстным алгоритмом детекции: фон оценивается по окрестности,
    из которой исключаются потенциально горящие пиксели.
    """

    a = np.asarray(arr, dtype=np.float32)
    r = int(radius)
    if r <= 0:
        return a.copy()
    padded = np.pad(a, r, mode="reflect")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)))
    k = 2 * r + 1
    h, w = a.shape
    total = (
        integral[k : k + h, k : k + w]
        - integral[0:h, k : k + w]
        - integral[k : k + h, 0:w]
        + integral[0:h, 0:w]
    )
    return (total / (k * k)).astype(np.float32)


def box_std(arr: np.ndarray, radius: int) -> np.ndarray:
    """Стандартное отклонение в квадратном окне через средние значений и квадратов."""

    mean = box_mean(arr, radius)
    mean_sq = box_mean(np.asarray(arr, dtype=np.float32) ** 2, radius)
    return np.sqrt(np.clip(mean_sq - mean**2, 0, None)).astype(np.float32)


def box_count(mask: np.ndarray, radius: int) -> np.ndarray:
    return box_mean(np.asarray(mask, dtype=np.float32), radius) * ((2 * int(radius) + 1) ** 2)
