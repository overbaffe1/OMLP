"""Геодезические преобразования без внешних зависимостей.

В наборе кейса чипы заданы в UTM (зоны 37N и 38N, WGS84), а сервису нужны географические
координаты. Пакеты ``pyproj``/``GDAL`` на площадке могут отсутствовать, поэтому здесь
реализованы обратная проекция UTM → WGS84 и прямая проекция WGS84 → UTM по стандартным
формулам Крагера (точность порядка сантиметров на территории AOI, чего с запасом хватает
для 20-метровых пикселей Sentinel-2).

Формулы: Snyder, J. P. (1987), «Map Projections — A Working Manual», USGS Professional
Paper 1395, раздел 8 (Transverse Mercator, ряды для эллипсоида).
"""

from __future__ import annotations

import math

#: Эллипсоид WGS84.
WGS84_A = 6378137.0                 # большая полуось, м
WGS84_F = 1.0 / 298.257223563       # сжатие

#: Параметры проекции UTM.
UTM_K0 = 0.9996
UTM_FALSE_EASTING = 500000.0
UTM_FALSE_NORTHING_SOUTH = 10000000.0


def utm_to_lonlat(easting: float, northing: float, zone: int, northern: bool = True) -> tuple[float, float]:
    """Переводит координаты UTM в долготу и широту WGS84 (градусы)."""

    a, f = WGS84_A, WGS84_F
    e2 = f * (2.0 - f)
    e1 = (1.0 - math.sqrt(1.0 - e2)) / (1.0 + math.sqrt(1.0 - e2))

    x = float(easting) - UTM_FALSE_EASTING
    y = float(northing) - (0.0 if northern else UTM_FALSE_NORTHING_SOUTH)
    m = y / UTM_K0
    mu = m / (a * (1.0 - e2 / 4.0 - 3.0 * e2**2 / 64.0 - 5.0 * e2**3 / 256.0))

    phi1 = (
        mu
        + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2.0 * mu)
        + (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4.0 * mu)
        + (151.0 * e1**3 / 96.0) * math.sin(6.0 * mu)
        + (1097.0 * e1**4 / 512.0) * math.sin(8.0 * mu)
    )

    sin_phi1, cos_phi1, tan_phi1 = math.sin(phi1), math.cos(phi1), math.tan(phi1)
    c1 = e2 * cos_phi1**2 / (1.0 - e2)
    t1 = tan_phi1**2
    n1 = a / math.sqrt(1.0 - e2 * sin_phi1**2)
    r1 = a * (1.0 - e2) / (1.0 - e2 * sin_phi1**2) ** 1.5
    d = x / (n1 * UTM_K0)

    # Ряды до d^6 (формулы Snyder 8-17)
    lat = phi1 - (n1 * tan_phi1 / r1) * (
        d**2 / 2.0
        - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1**2 - 9.0 * e2 * cos_phi1**2 / (1.0 - e2) ** 2) * d**4 / 24.0
        + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1**2 - 252.0 * e2 * cos_phi1**2 / (1.0 - e2) - 3.0 * c1**2)
        * d**6
        / 720.0
    )
    lon = (
        d
        - (1.0 + 2.0 * t1 + c1) * d**3 / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1**2 + 8.0 * e2 * cos_phi1**2 / (1.0 - e2) + 24.0 * t1**2) * d**5 / 120.0
    ) / cos_phi1

    lon_deg = math.degrees(lon) + (zone * 6.0 - 183.0)
    return lon_deg, math.degrees(lat)


def lonlat_to_utm(lon: float, lat: float, zone: int | None = None, northern: bool | None = None) -> tuple[float, float, int, bool]:
    """Прямая проекция WGS84 → UTM. Возвращает ``(easting, northing, зона, северное_полушарие)``."""

    a, f = WGS84_A, WGS84_F
    e2 = f * (2.0 - f)
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    if zone is None:
        zone = int((lon + 180.0) // 6.0) + 1
    if northern is None:
        northern = lat >= 0.0

    lon0 = math.radians(zone * 6.0 - 183.0)
    n = a / math.sqrt(1.0 - e2 * math.sin(lat_r) ** 2)
    t = math.tan(lat_r) ** 2
    c = e2 * math.cos(lat_r) ** 2 / (1.0 - e2)
    aa = math.cos(lat_r) * (lon_r - lon0)
    m = a * (
        (1.0 - e2 / 4.0 - 3.0 * e2**2 / 64.0 - 5.0 * e2**3 / 256.0) * lat_r
        - (3.0 * e2 / 8.0 + 3.0 * e2**2 / 32.0 + 45.0 * e2**3 / 1024.0) * math.sin(2.0 * lat_r)
        + (15.0 * e2**2 / 256.0 + 45.0 * e2**3 / 1024.0) * math.sin(4.0 * lat_r)
        - (35.0 * e2**3 / 3072.0) * math.sin(6.0 * lat_r)
    )
    easting = UTM_K0 * n * (
        aa
        + (1.0 - t + c) * aa**3 / 6.0
        + (5.0 - 18.0 * t + t**2 + 72.0 * c - 58.0 * e2 / (1.0 - e2)) * aa**5 / 120.0
    ) + UTM_FALSE_EASTING
    northing = UTM_K0 * (
        m
        + n
        * math.tan(lat_r)
        * (aa**2 / 2.0 + (5.0 - t + 9.0 * c + 4.0 * c**2) * aa**4 / 24.0 + (61.0 - 58.0 * t + t**2 + 600.0 * c - 330.0 * e2 / (1.0 - e2)) * aa**6 / 720.0)
    )
    if not northern:
        northing += UTM_FALSE_NORTHING_SOUTH
    return easting, northing, zone, northern


def utm_zone_from_epsg(epsg: int | str | None) -> tuple[int, bool] | None:
    """Возвращает ``(зона, северное_полушарие)`` для кодов EPSG 326xx/327xx, иначе ``None``."""

    try:
        code = int(float(str(epsg)))
    except (TypeError, ValueError):
        return None
    if 32601 <= code <= 32660:
        return code - 32600, True
    if 32701 <= code <= 32760:
        return code - 32700, False
    return None
