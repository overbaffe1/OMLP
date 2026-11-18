"""Векторизация растровых масок: контуры пожаров с атрибутами для сервиса.

Реализация без GDAL: обход границы компонента (Moore-neighbor tracing) + упрощение
линии алгоритмом Дугласа—Пекара. Этого достаточно для картографического вывода и
экспорта в GeoJSON: контуры получаются замкнутыми, с корректной топологией по маске.

Для каждого контура считаются атрибуты, требуемые критериями сервиса:
``id``, ``severity`` (класс степени поражения), ``area_ha``, ``chip_id``, ``date``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fires.morphology import label

#: Восемь направлений обхода (по часовой стрелке), начиная с восточного.
NEIGHBOURS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


@dataclass
class Contour:
    """Контур в пиксельных координатах (x = столбец, y = строка)."""

    points: list[tuple[float, float]]
    area_px: int
    severity: int
    chip_id: str
    extra: dict

    def to_ring(self, transform=None) -> list[list[float]]:
        """Переводит пиксельные координаты в географические (если задано преобразование)."""

        ring = []
        for x, y in self.points:
            if transform is None:
                ring.append([float(x), float(y)])
            else:
                lon, lat = transform(x, y)
                ring.append([float(lon), float(lat)])
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        return ring


def trace_boundary(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    """Обход внешней границы компонента от стартового пикселя (8-связность)."""

    h, w = mask.shape
    boundary = [start]
    current = start
    prev_dir = 4  # пришли «снизу»: начинаем обход против часовой стрелки
    limit = 8 * int(mask.sum()) + 8
    for _ in range(limit):
        for step in range(8):
            direction = (prev_dir + 1 + step) % 8
            dy, dx = NEIGHBOURS[direction]
            ny, nx = current[0] + dy, current[1] + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                current = (ny, nx)
                prev_dir = (direction + 4) % 8  # откуда пришли — на 180°
                boundary.append(current)
                break
        else:
            break
        if current == start and len(boundary) > 2:
            break
    return boundary


def simplify(points: list[tuple[float, float]], tolerance: float = 0.6) -> list[tuple[float, float]]:
    """Упрощение линии алгоритмом Дугласа—Пекара (итеративно, без рекурсии)."""

    if len(points) < 3:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    pts = np.asarray(points, dtype=float)
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        norm = np.hypot(*ab)
        offsets = pts[i + 1 : j] - a
        if norm < 1e-9:
            d = np.hypot(offsets[:, 0], offsets[:, 1])
        else:
            # псевдоскалярное произведение в 2D (np.cross для 2D объявлен устаревшим)
            d = np.abs(ab[0] * offsets[:, 1] - ab[1] * offsets[:, 0]) / norm
        k = int(np.argmax(d))
        if d[k] > tolerance:
            idx = i + 1 + k
            keep[idx] = True
            stack.append((i, idx))
            stack.append((idx, j))
    return [tuple(p) for p, k in zip(points, keep) if k]


def mask_to_contours(
    mask: np.ndarray,
    chip_id: str = "chip",
    severity_map: np.ndarray | None = None,
    min_area_px: int = 4,
    simplify_tolerance: float = 0.6,
    min_severity: int = 0,
    extra: dict | None = None,
) -> list[Contour]:
    """Превращает бинарную маску в список контуров с атрибутами.

    :param severity_map: маска 0–3 (для BS-чипов); в контуре берётся доминирующий класс > 0;
    :param min_severity: минимальный класс, который попадает в выдачу.
    """

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    labels, count = label(mask, connectivity=8)
    contours: list[Contour] = []
    for blob_id in range(1, count + 1):
        blob = labels == blob_id
        area = int(blob.sum())
        if area < min_area_px:
            continue
        if severity_map is not None:
            values = np.asarray(severity_map)[blob]
            values = values[values > 0]
            severity = int(np.bincount(values).argmax()) if values.size else 0
        else:
            severity = 1
        if severity_map is not None and severity == 0:
            continue  # в контур гари не попал ни один пиксель класса > 0
        if severity < min_severity:
            continue
        ys, xs = np.nonzero(blob)
        start = (int(ys[0]), int(xs[0]))
        boundary = trace_boundary(blob, start)
        points = [(float(x), float(y)) for y, x in boundary]
        contours.append(
            Contour(
                points=simplify(points, simplify_tolerance),
                area_px=area,
                severity=severity,
                chip_id=chip_id,
                extra=dict(extra or {}),
            )
        )
    return contours


def pixel_to_lonlat(origin_lon: float, origin_lat: float, gsd: float):
    """Функция преобразования «пиксель → координаты» для чипов с известной привязкой.

    Пиксель (0, 0) — северо-западный угол: координаты считаются по центру пикселя,
    долгота масштабируется на косинус широты. Точность такой аппроксимации для
    чипа 10×10 км — доли пикселя, что для отображения контуров достаточно.
    """

    import math

    def transform(x: float, y: float) -> tuple[float, float]:
        lat = origin_lat - (y + 0.5) * gsd / 111_320.0
        lon = origin_lon + (x + 0.5) * gsd / (111_320.0 * max(0.2, math.cos(math.radians(lat))))
        return lon, lat

    return transform


def polygon_area_ha(ring: list[list[float]]) -> float:
    """Площадь полигона (в градусах) в гектарах: формула Гаусса с поправкой на широту.

    Используется для произвольных полигонов (запросы сервиса); для растровых масок
    площадь точнее считать по числу пикселей (см. :func:`contours_report`).
    """

    pts = np.asarray(ring, dtype=float)
    if len(pts) < 4:
        return 0.0
    lat0 = float(np.mean(pts[:, 1]))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(0.2, np.cos(np.radians(lat0)))
    x = pts[:, 0] * m_per_deg_lon
    y = pts[:, 1] * m_per_deg_lat
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 10_000.0)


def contours_report(contours: list[Contour], gsd: float, pixel_ha: float | None = None) -> dict:
    """Сводка по контурам: суммарная площадь и распределение по степеням поражения (в га)."""

    px_ha = pixel_ha if pixel_ha is not None else (gsd ** 2) / 10_000.0
    by_severity = {1: 0.0, 2: 0.0, 3: 0.0}
    for contour in contours:
        if contour.severity < 1:
            continue
        by_severity[contour.severity] += contour.area_px * px_ha
    total = sum(by_severity.values())
    return {
        "total_area_ha": round(total, 2),
        "by_severity_ha": {f"severity_{k}": round(v, 2) for k, v in sorted(by_severity.items())},
        "n_contours": len(contours),
        "min_contour_ha": round(min((c.area_px * px_ha for c in contours), default=0.0), 3),
        "max_contour_ha": round(max((c.area_px * px_ha for c in contours), default=0.0), 3),
    }
