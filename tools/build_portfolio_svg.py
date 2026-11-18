#!/usr/bin/env python3
"""Генератор векторных иллюстраций (SVG) и схемы Mermaid для портфолио КосмоФАЙР-2.

Работает на стандартной библиотеке Python 3 (без внешних зависимостей, без сети) —
по тому же принципу, что и генераторы мастерской gitEdit.

Собирает в assets/:
* hero-dark.svg, hero-light.svg         — обложка репозитория (дневная и ночная палитры);
* card-01-wipe.svg, card-02-stl.svg,
  card-03-map.svg                       — три карточки интерактивной витрины экспонатов;
* architecture-dark.svg,
  architecture-light.svg,
  architecture.mmd                      — схема двухэтапного конвейера и геосервиса;
* experiments-dark.svg,
  experiments-light.svg                 — находка SCL=10 и траектория Score до 0.7455;
* footer-dark.svg, footer-light.svg     — финальная топографическая подпись-разделитель.

Запуск из корня репозитория:
    python3 tools/build_portfolio_svg.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

FONT_SANS = "system-ui, -apple-system, 'Segoe UI', Roboto, 'DejaVu Sans', sans-serif"
FONT_MONO = "ui-monospace, SFMono-Regular, 'JetBrains Mono', 'DejaVu Sans Mono', monospace"

PALETTES = {
    "dark": {
        "bg0": "#091410",
        "bg1": "#0F211B",
        "surface": "#122720",
        "surface2": "#163027",
        "grid": "#183329",
        "border": "#244A3C",
        "border_strong": "#336B56",
        "ink": "#F0F7F4",
        "ink2": "#BED3C9",
        "muted": "#82A395",
        "mint": "#6EE7B7",
        "mint_bg": "#113529",
        "mint_border": "#2A7A5B",
        "amber": "#FBBF24",
        "amber_bg": "#2B2412",
        "amber_border": "#8A6518",
        "ember": "#FB7185",
        "orange": "#FB923C",
        "orange_bg": "#2D1C12",
        "orange_border": "#8F4A22",
        "red": "#F87171",
        "sky": "#7DD3FC",
        "sky_bg": "#112836",
        "sky_border": "#276285",
        "aoi_fill": "#14382B",
        "aoi_stroke": "#34D399",
    },
    "light": {
        "bg0": "#F5F7F4",
        "bg1": "#EAEFE9",
        "surface": "#FFFFFF",
        "surface2": "#F0F5F1",
        "grid": "#DFE8E1",
        "border": "#C5D6CB",
        "border_strong": "#8FB39E",
        "ink": "#0E221B",
        "ink2": "#2E4A3F",
        "muted": "#567367",
        "mint": "#047857",
        "mint_bg": "#E6F6EF",
        "mint_border": "#6CC4A1",
        "amber": "#B45309",
        "amber_bg": "#FEF6E4",
        "amber_border": "#E2AE55",
        "ember": "#BE123C",
        "orange": "#C2410C",
        "orange_bg": "#FFF1E8",
        "orange_border": "#E89B6E",
        "red": "#DC2626",
        "sky": "#0369A1",
        "sky_bg": "#E8F4FB",
        "sky_border": "#7ABCE0",
        "aoi_fill": "#DCF2E7",
        "aoi_stroke": "#059669",
    },
}


def _aoi_svg_points(cx: float, cy: float, scale_w: float, scale_h: float) -> tuple[str, list[tuple[float, float]]]:
    """Проецирует реальный полигон AOI из data/aoi/fire_monitoring_aoi.geojson в координаты SVG."""
    p = ROOT / "data" / "aoi" / "fire_monitoring_aoi.geojson"
    data = json.loads(p.read_text(encoding="utf-8"))
    ring = data["features"][0]["geometry"]["coordinates"][0]
    lons = [pt[0] for pt in ring]
    lats = [pt[1] for pt in ring]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    def proj(lon: float, lat: float) -> tuple[float, float]:
        x = cx - scale_w / 2 + (lon - min_lon) / (max_lon - min_lon) * scale_w
        y = cy + scale_h / 2 - (lat - min_lat) / (max_lat - min_lat) * scale_h
        return round(x, 1), round(y, 1)

    pts = " ".join(f"{x},{y}" for x, y in (proj(lo, la) for lo, la in ring))
    # Несколько репрезентативных точек реального каталога внутри полигона
    sample_ll = [
        (40.15, 48.20), (41.65, 47.72), (42.05, 46.42), (43.42, 49.12),
        (45.85, 48.95), (46.22, 49.55), (44.68, 45.82), (41.05, 49.45),
        (39.48, 47.18), (47.12, 48.42), (42.85, 50.10), (40.55, 49.12),
    ]
    spots = [proj(lo, la) for lo, la in sample_ll]
    return pts, spots


def build_hero(theme: str) -> str:
    c = PALETTES[theme]
    W, H = 1200, 392
    aoi_pts, spots = _aoi_svg_points(992, 184, 304, 240)

    grid_lines = []
    for x in range(0, W + 1, 48):
        grid_lines.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{H}"/>')
    for y in range(0, H + 1, 48):
        grid_lines.append(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}"/>')

    hotspot_svg = []
    for i, (sx, sy) in enumerate(spots):
        col = c["red"] if i % 3 == 0 else (c["orange"] if i % 3 == 1 else c["amber"])
        r = 4.2 if i % 3 == 0 else 3.2
        if i in (0, 3, 5):
            delay = f"{i * 0.6:.1f}s"
            hotspot_svg.append(
                f'<circle class="pulse" style="animation-delay:{delay}" cx="{sx}" cy="{sy}" r="{r + 6}" '
                f'fill="none" stroke="{col}" stroke-width="1.4" opacity="0.55"/>'
            )
        hotspot_svg.append(f'<circle cx="{sx}" cy="{sy}" r="{r}" fill="{col}"/>')

    cards = [
        ("83,67 / 100", "19-е место из 47 · итог жюри", c["amber"], c["amber_bg"], c["amber_border"]),
        ("0.7455", "лучший Score · probe 35", c["mint"], c["mint_bg"], c["mint_border"]),
        ("82 / 82", "pytest · RLE 447 строк", c["sky"], c["sky_bg"], c["sky_border"]),
        ("256 520 га", "4 892 контура · 9 478 точек", c["orange"], c["orange_bg"], c["orange_border"]),
    ]
    card_svg = []
    cw, ch, gap = 186, 86, 12
    x0, y0 = 44, 272
    for idx, (val, lbl, col, bg, brd) in enumerate(cards):
        bx = x0 + idx * (cw + gap)
        card_svg.append(
            f'<g transform="translate({bx},{y0})">'
            f'<rect width="{cw}" height="{ch}" rx="11" fill="{bg}" stroke="{brd}" stroke-width="1.3"/>'
            f'<text x="14" y="38" font-family="{FONT_SANS}" font-size="25" font-weight="800" fill="{col}">{val}</text>'
            f'<text x="14" y="64" font-family="{FONT_SANS}" font-size="11.5" font-weight="500" fill="{c["ink2"]}">{lbl}</text>'
            f'</g>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="КосмоФАЙР-2 — Двухэтапный мониторинг природных пожаров по данным ДЗЗ. КосмоХакатон 2026: 83,67 из 100, 19-е место из 47, Score 0.7455.">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{c["bg0"]}"/>
      <stop offset="100%" stop-color="{c["bg1"]}"/>
    </linearGradient>
    <linearGradient id="topBar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#F97316"/>
      <stop offset="35%" stop-color="#F59E0B"/>
      <stop offset="70%" stop-color="#10B981"/>
      <stop offset="100%" stop-color="#0EA5E9"/>
    </linearGradient>
    <style>
      @keyframes pulseRing {{
        0%, 100% {{ opacity: 0.18; stroke-width: 1.0; }}
        50% {{ opacity: 0.78; stroke-width: 2.0; }}
      }}
      .pulse {{ animation: pulseRing 3.4s ease-in-out infinite; }}
      @media (prefers-reduced-motion: reduce) {{
        .pulse {{ animation: none; opacity: 0.45; }}
      }}
    </style>
  </defs>

  <rect x="1" y="1" width="{W - 2}" height="{H - 2}" rx="16" fill="{c["bg0"]}" stroke="{c["border"]}" stroke-width="1.5"/>
  <g stroke="{c["grid"]}" stroke-width="1" opacity="0.7">{"".join(grid_lines)}</g>
  <rect x="16" y="0" width="{W - 32}" height="4" rx="2" fill="#10B981"/>
  <rect x="16" y="0" width="380" height="4" rx="2" fill="#F97316"/>

  <!-- Левая колонка: типографика и метрики -->
  <rect x="44" y="34" width="462" height="28" rx="14" fill="{c["mint_bg"]}" stroke="{c["mint_border"]}" stroke-width="1.1"/>
  <circle cx="61" cy="48" r="4" fill="{c["orange"]}"/>
  <text x="73" y="52.5" font-family="{FONT_MONO}" font-size="11.5" font-weight="700" fill="{c["mint"]}">КОСМОХАКАТОН 2026 · КРАСНОЯРСК · КЕЙС ДЗЗ / ML</text>

  <text x="44" y="122" font-family="{FONT_SANS}" font-size="54" font-weight="800" letter-spacing="-0.8" fill="{c["ink"]}">КосмоФАЙР-2</text>
  <text x="44" y="160" font-family="{FONT_SANS}" font-size="20" font-weight="700" fill="{c["mint"]}">Огонь найти. Последствия измерить.</text>

  <text x="44" y="196" font-family="{FONT_SANS}" font-size="15" fill="{c["ink2"]}">Двухэтапный мониторинг природных пожаров по спутниковым снимкам:</text>
  <text x="44" y="219" font-family="{FONT_SANS}" font-size="15" fill="{c["ink2"]}">активное горение VIIRS (375 м) → контур гари и степень поражения Sentinel-2 (20 м)</text>
  <text x="44" y="246" font-family="{FONT_MONO}" font-size="12" fill="{c["muted"]}">Score = 0.35·F1_AF + 0.35·IoU_burn + 0.30·mIoU_sev  |  CPU · FastAPI · GeoJSON / SHP · Three.js</text>

  {"".join(card_svg)}

  <!-- Правая колонка: карточка полигона AOI Нижнего Поволжья и Подонья -->
  <rect x="828" y="34" width="332" height="324" rx="14" fill="{c["surface"]}" stroke="{c["border_strong"]}" stroke-width="1.3"/>
  <text x="848" y="58" font-family="{FONT_MONO}" font-size="10.5" font-weight="700" fill="{c["muted"]}">AOI · EPSG:4326 · UTM 37N / 38N</text>
  <line x1="992" y1="68" x2="992" y2="306" stroke="{c["border"]}" stroke-width="1" stroke-dasharray="4 4"/>
  <text x="998" y="80" font-family="{FONT_MONO}" font-size="9.5" fill="{c["muted"]}">42°E</text>

  <polygon points="{aoi_pts}" fill="{c["aoi_fill"]}" stroke="{c["aoi_stroke"]}" stroke-width="2" stroke-linejoin="round"/>
  <!-- Рамки масштабов чипов: AF 96 км и BS 10 км -->
  <rect x="922" y="150" width="46" height="46" rx="3" fill="none" stroke="{c["orange"]}" stroke-width="1.3" stroke-dasharray="3 2"/>
  <rect x="939" y="167" width="12" height="12" rx="1.5" fill="none" stroke="{c["mint"]}" stroke-width="1.4"/>
  {"".join(hotspot_svg)}

  <!-- Легенда внизу карты -->
  <line x1="844" y1="312" x2="1144" y2="312" stroke="{c["border"]}" stroke-width="1"/>
  <text x="848" y="331" font-family="{FONT_SANS}" font-size="12" font-weight="700" fill="{c["ink"]}">Нижнее Поволжье и Подонье · 435 тыс. км²</text>
  <circle cx="853" cy="346" r="3.5" fill="{c["amber"]}"/>
  <text x="861" y="349.5" font-family="{FONT_SANS}" font-size="10.5" fill="{c["muted"]}">1 слаб.</text>
  <circle cx="912" cy="346" r="3.5" fill="{c["orange"]}"/>
  <text x="920" y="349.5" font-family="{FONT_SANS}" font-size="10.5" fill="{c["muted"]}">2 сред.</text>
  <circle cx="972" cy="346" r="3.5" fill="{c["red"]}"/>
  <text x="980" y="349.5" font-family="{FONT_SANS}" font-size="10.5" fill="{c["muted"]}">3 сильн.</text>
  <text x="1144" y="349.5" text-anchor="end" font-family="{FONT_MONO}" font-size="10" fill="{c["mint"]}">644 / 269 чипов</text>
</svg>
'''


def build_vitrine_cards() -> dict[str, str]:
    """Три SVG-карточки витрины с явными атрибутами fill/stroke (работают и без CSS) + CSS для тёмной темы."""
    common_style = """
    <style>
      @media (prefers-color-scheme: dark) {
        .card { fill: #0F211B !important; stroke: #275243 !important; }
        .well { fill: #0A1612 !important; stroke: #1E4034 !important; }
        .ink  { fill: #EFF7F3 !important; }
        .sub  { fill: #9AB7AA !important; }
        .tag  { fill: #6EE7B7 !important; }
        .grid { stroke: #163027 !important; }
      }
    </style>"""

    # Карточка 01: шторка dNBR → маска 0–3
    c1 = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 214" width="380" height="214" role="img" aria-label="01 · Шторка dNBR и маска 0–3">
  <defs>{common_style}</defs>
  <rect class="card" x="1" y="1" width="378" height="212" rx="13" fill="#FFFFFF" stroke="#C5D6CB" stroke-width="1.5"/>
  <text class="tag" x="20" y="28" fill="#047857" font-family="{FONT_MONO}" font-size="11" font-weight="700">01 · СРАВНЕНИЕ В КАДРЕ (GIF)</text>
  <rect class="well" x="20" y="40" width="340" height="106" rx="9" fill="#0D1D18" stroke="#285042" stroke-width="1.2"/>
  <!-- Левая половина: спектральный dNBR -->
  <path d="M48,136 Q78,76 135,74 L190,74 L190,136 Z" fill="#9333EA" opacity="0.65"/>
  <path d="M68,136 Q96,88 148,86 L190,86 L190,136 Z" fill="#F97316" opacity="0.80"/>
  <path d="M92,136 Q118,102 162,100 L190,100 L190,136 Z" fill="#FDE047" opacity="0.86"/>
  <!-- Правая половина: дискретные классы 1/2/3 -->
  <path d="M190,74 L268,74 Q314,82 326,136 L190,136 Z" fill="#FB923C"/>
  <path d="M214,92 L262,90 Q288,104 294,136 L208,136 Z" fill="#EF4444"/>
  <ellipse cx="238" cy="58" rx="34" ry="9" fill="#EAB308"/>
  <!-- Вертикальная шторка -->
  <line x1="190" y1="40" x2="190" y2="146" stroke="#E6FFF5" stroke-width="2"/>
  <circle cx="190" cy="93" r="10" fill="#0F211B" stroke="#6EE7B7" stroke-width="1.8"/>
  <polygon points="185,93 188,90 188,96" fill="#6EE7B7"/>
  <polygon points="195,93 192,90 192,96" fill="#6EE7B7"/>
  <text class="ink" x="20" y="172" fill="#0E221B" font-family="{FONT_SANS}" font-size="15" font-weight="700">dNBR → маска степеней 0–3</text>
  <text class="sub" x="20" y="193" fill="#4E6B5F" font-family="{FONT_SANS}" font-size="12">Sentinel-2 · BS_tr_000095 (20 м) · открыть GIF →</text>
</svg>
'''

    # Карточка 02: изометрический 3D-рельеф dNBR (STL)
    iso_lines = []
    for i in range(7):
        t = i / 6.0
        pts = []
        for j in range(13):
            u = j / 12.0
            cx, cy = 0.52, 0.58
            d2 = (u - cx) ** 2 + (t - cy) ** 2
            elev = 26.0 * math.exp(-d2 / 0.11) + (8.0 * math.exp(-((u - 0.5) ** 2 + (t - 0.22) ** 2) / 0.03))
            px = 190 + (u - t) * 112
            py = 72 + (u + t) * 33 - elev
            pts.append(f"{px:.1f},{py:.1f}")
        col = "#34D399" if i < 2 else ("#FBBF24" if i < 4 else "#FB923C")
        iso_lines.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="1.8" stroke-linecap="round"/>')

    c2 = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 214" width="380" height="214" role="img" aria-label="02 · Покрутить рельеф dNBR в 3D (STL)">
  <defs>{common_style}</defs>
  <rect class="card" x="1" y="1" width="378" height="212" rx="13" fill="#FFFFFF" stroke="#C5D6CB" stroke-width="1.5"/>
  <text class="tag" x="20" y="28" fill="#047857" font-family="{FONT_MONO}" font-size="11" font-weight="700">02 · 3D-ПРОСМОТРЩИК GITHUB (STL)</text>
  <rect class="well" x="20" y="40" width="340" height="106" rx="9" fill="#0D1D18" stroke="#285042" stroke-width="1.2"/>
  <polygon points="190,139 78,105 190,71 302,105" fill="none" stroke="#1F4437" stroke-width="1.2"/>
  {"".join(iso_lines)}
  <text class="ink" x="20" y="172" fill="#0E221B" font-family="{FONT_SANS}" font-size="15" font-weight="700">Слепок гари dNBR в 3D</text>
  <text class="sub" x="20" y="193" fill="#4E6B5F" font-family="{FONT_SANS}" font-size="12">56 × 56 граней · покрутить модель на GitHub →</text>
</svg>
'''

    # Карточка 03: интерактивная GeoJSON-карта AOI + 50 крупнейших гарей
    aoi_mini, mini_spots = _aoi_svg_points(190, 93, 180, 84)
    dots = "".join(
        f'<circle cx="{x}" cy="{y}" r="{3.8 if idx % 3 == 0 else 2.8}" '
        f'fill="{"#EF4444" if idx % 3 == 0 else ("#FB923C" if idx % 3 == 1 else "#FBBF24")}"/>'
        for idx, (x, y) in enumerate(mini_spots[:9])
    )
    c3 = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 214" width="380" height="214" role="img" aria-label="03 · Интерактивная карта контуров гари (GeoJSON)">
  <defs>{common_style}</defs>
  <rect class="card" x="1" y="1" width="378" height="212" rx="13" fill="#FFFFFF" stroke="#C5D6CB" stroke-width="1.5"/>
  <text class="tag" x="20" y="28" fill="#047857" font-family="{FONT_MONO}" font-size="11" font-weight="700">03 · ИНТЕРАКТИВНАЯ КАРТА (GEOJSON)</text>
  <rect class="well" x="20" y="40" width="340" height="106" rx="9" fill="#0D1D18" stroke="#285042" stroke-width="1.2"/>
  <line x1="20" y1="75" x2="360" y2="75" stroke="#163027" stroke-width="1"/>
  <line x1="20" y1="111" x2="360" y2="111" stroke="#163027" stroke-width="1"/>
  <line x1="130" y1="40" x2="130" y2="146" stroke="#163027" stroke-width="1"/>
  <line x1="250" y1="40" x2="250" y2="146" stroke="#163027" stroke-width="1"/>
  <polygon points="{aoi_mini}" fill="#10B981" fill-opacity="0.18" stroke="#34D399" stroke-width="1.8" stroke-linejoin="round"/>
  {dots}
  <text class="ink" x="20" y="172" fill="#0E221B" font-family="{FONT_SANS}" font-size="15" font-weight="700">Полигон и 50 крупнейших гарей</text>
  <text class="sub" x="20" y="193" fill="#4E6B5F" font-family="{FONT_SANS}" font-size="12">WGS84 · 435 тыс. км² · открыть карту GitHub →</text>
</svg>
'''
    return {
        "card-01-wipe.svg": c1,
        "card-02-stl.svg": c2,
        "card-03-map.svg": c3,
    }


def build_experiments(theme: str) -> str:
    """Двухпанельная инфографика: находка SCL=10 (слева) и траектория Score 0.6979 → 0.7455 (справа)."""
    c = PALETTES[theme]
    W, H = 1200, 438

    # Правая панель: шкала Score от 0.690 до 0.752
    px0, px1 = 626, 1142
    py0, py1 = 96, 336
    s_min, s_max = 0.690, 0.752

    def sy(score: float) -> float:
        return round(py1 - (score - s_min) / (s_max - s_min) * (py1 - py0), 1)

    # Сетка по оси Y
    grid_svg = []
    for tick in (0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75):
        yy = sy(tick)
        grid_svg.append(
            f'<line x1="{px0}" y1="{yy}" x2="{px1}" y2="{yy}" stroke="{c["grid"]}" stroke-width="1"/>'
            f'<text x="{px0 - 10}" y="{yy + 4}" text-anchor="end" font-family="{FONT_MONO}" font-size="11" fill="{c["muted"]}">{tick:.2f}</text>'
        )

    # Принятая траектория (4 опорные точки платформы)
    accepted = [
        (656, 0.6979, "Ранний", "до правки SCL", -14),
        (778, 0.7391, "v1 · цикл 17", "SCL≠10 + HGB 23", -16),
        (904, 0.7393, "probe 24", "patch 96 + k=2.5", -16),
        (1028, 0.7455, "probe 35 ★", "majority 3×3", -16),
    ]
    line_pts = " ".join(f"{x},{sy(sc)}" for x, sc, _, _, _ in accepted)

    acc_svg = [
        f'<polyline points="{line_pts}" fill="none" stroke="{c["mint"]}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
    ]
    for i, (x, sc, title, sub, dy) in enumerate(accepted):
        y = sy(sc)
        is_best = i == len(accepted) - 1
        r = 7.5 if is_best else 5.5
        if is_best:
            acc_svg.append(f'<circle cx="{x}" cy="{y}" r="13" fill="{c["mint"]}" opacity="0.22"/>')
        acc_svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c["mint"]}" stroke="{c["surface"]}" stroke-width="2"/>')
        acc_svg.append(
            f'<text x="{x}" y="{y + dy}" text-anchor="middle" font-family="{FONT_MONO}" '
            f'font-size="{"14" if is_best else "12.5"}" font-weight="800" fill="{c["mint"] if is_best else c["ink"]}">{sc:.4f}</text>'
        )
        # Подпись под осью X
        acc_svg.append(
            f'<text x="{x}" y="{py1 + 22}" text-anchor="middle" font-family="{FONT_SANS}" font-size="11.5" font-weight="700" fill="{c["ink"]}">{title}</text>'
            f'<text x="{x}" y="{py1 + 37}" text-anchor="middle" font-family="{FONT_SANS}" font-size="10.5" fill="{c["muted"]}">{sub}</text>'
        )

    # Отклонённые гипотезы (пунктир от родительской точки + полый маркер)
    rejected = [
        (778, 0.7391, 846, 0.7357, "v2 бэггинг: 0.7357", +19, "middle"),
        (1028, 0.7455, 1118, 0.7409, "3×3×2: 0.7409", -11, "middle"),
        (1028, 0.7455, 1118, 0.7313, "5×5: 0.7313", +19, "end"),
    ]
    rej_svg = []
    for x0, s0, x1, s1, lbl, dy, anchor in rejected:
        y0, y1 = sy(s0), sy(s1)
        tx = x1 + 14 if anchor == "end" else x1
        rej_svg.append(
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{c["orange"]}" stroke-width="1.6" stroke-dasharray="4 3"/>'
            f'<circle cx="{x1}" cy="{y1}" r="5" fill="{c["surface"]}" stroke="{c["orange"]}" stroke-width="2"/>'
            f'<text x="{tx}" y="{y1 + dy}" text-anchor="{anchor}" font-family="{FONT_MONO}" font-size="10.5" font-weight="700" fill="{c["orange"]}">{lbl}</text>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Инженерные находки КосмоФАЙР-2: размаскировка SCL=10 и рост Score платформы с 0.6979 до 0.7455 с откатом неудачных гипотез.">
  <rect x="1" y="1" width="{W - 2}" height="{H - 2}" rx="16" fill="{c["bg0"]}" stroke="{c["border"]}" stroke-width="1.5"/>

  <!-- Левая панель: EDA и маска SCL=10 -->
  <rect x="24" y="22" width="528" height="394" rx="13" fill="{c["surface"]}" stroke="{c["border"]}" stroke-width="1.2"/>
  <text x="46" y="52" font-family="{FONT_MONO}" font-size="11" font-weight="700" fill="{c["mint"]}">01 · ДИАГНОСТИКА ДАННЫХ (TRAIN: 224 ЧИПА BS)</text>
  <text x="46" y="80" font-family="{FONT_SANS}" font-size="20" font-weight="800" fill="{c["ink"]}">Маска облачности SCL скрывала 17,8% гари</text>

  <rect x="46" y="98" width="224" height="92" rx="10" fill="{c["mint_bg"]}" stroke="{c["mint_border"]}" stroke-width="1.2"/>
  <text x="62" y="138" font-family="{FONT_SANS}" font-size="34" font-weight="800" fill="{c["mint"]}">17.8%</text>
  <text x="62" y="160" font-family="{FONT_SANS}" font-size="12" font-weight="600" fill="{c["ink"]}">1 051 349 px эталона</text>
  <text x="62" y="178" font-family="{FONT_SANS}" font-size="11" fill="{c["muted"]}">попало под жёсткую маску SCL</text>

  <rect x="284" y="98" width="246" height="92" rx="10" fill="{c["surface2"]}" stroke="{c["border"]}" stroke-width="1.2"/>
  <text x="300" y="122" font-family="{FONT_SANS}" font-size="12.5" font-weight="700" fill="{c["ink"]}">Исключили SCL = 10 (cirrus)</text>
  <text x="300" y="143" font-family="{FONT_MONO}" font-size="11" fill="{c["muted"]}">SCL_BAD = (3, 8, 9, 11)</text>
  <text x="300" y="164" font-family="{FONT_SANS}" font-size="11.5" fill="{c["ink2"]}">Тонкая перистая прозрачна в SWIR;</text>
  <text x="300" y="180" font-family="{FONT_SANS}" font-size="11.5" fill="{c["ink2"]}">плотные облака 8/9 и тени закрыты.</text>

  <!-- Полосы компромисса FN vs FP -->
  <text x="46" y="218" font-family="{FONT_SANS}" font-size="12.5" font-weight="700" fill="{c["ink"]}">Пропущенная гарь (FN): 1,46 млн → 0,78 млн px  (−47%)</text>
  <rect x="46" y="227" width="350" height="12" rx="4" fill="{c["border"]}"/>
  <text x="404" y="237" font-family="{FONT_MONO}" font-size="10.5" fill="{c["muted"]}">до: 1.46M</text>
  <rect x="46" y="244" width="187" height="12" rx="4" fill="{c["mint"]}"/>
  <text x="241" y="254" font-family="{FONT_MONO}" font-size="10.5" font-weight="700" fill="{c["mint"]}">после: 0.78M</text>

  <text x="46" y="284" font-family="{FONT_SANS}" font-size="12.5" font-weight="700" fill="{c["ink"]}">Ложная гарь (FP): 1,97 млн → 2,44 млн px  (+24% — цена полноты)</text>
  <rect x="46" y="293" width="290" height="12" rx="4" fill="{c["border"]}"/>
  <text x="344" y="303" font-family="{FONT_MONO}" font-size="10.5" fill="{c["muted"]}">до: 1.97M</text>
  <rect x="46" y="310" width="359" height="12" rx="4" fill="{c["orange"]}"/>
  <text x="413" y="320" font-family="{FONT_MONO}" font-size="10.5" font-weight="700" fill="{c["orange"]}">после: 2.44M</text>

  <rect x="46" y="342" width="484" height="54" rx="9" fill="{c["surface2"]}" stroke="{c["border"]}" stroke-width="1"/>
  <text x="62" y="364" font-family="{FONT_SANS}" font-size="12" font-weight="700" fill="{c["ink"]}">Эффект на train (644 чипа): Score 0.6811 → 0.7128 (+0.0317)</text>
  <text x="62" y="383" font-family="{FONT_MONO}" font-size="10.2" fill="{c["muted"]}">IoU_burn 0.565→0.615 · mIoU_sev 0.551→0.598 · чип 111: 0.00→0.77</text>

  <!-- Правая панель: траектория Score на платформе -->
  <rect x="572" y="22" width="604" height="394" rx="13" fill="{c["surface"]}" stroke="{c["border"]}" stroke-width="1.2"/>
  <text x="594" y="52" font-family="{FONT_MONO}" font-size="11" font-weight="700" fill="{c["amber"]}">02 · КОНКУРСНАЯ ПЛАТФОРМА (TEST: 180 AF + 89 BS)</text>
  <text x="594" y="80" font-family="{FONT_SANS}" font-size="20" font-weight="800" fill="{c["ink"]}">Рост +0.0476 и осознанные откаты</text>

  {"".join(grid_svg)}
  {"".join(rej_svg)}
  {"".join(acc_svg)}

  <!-- Легенда графика -->
  <circle cx="604" cy="396" r="5" fill="{c["mint"]}"/>
  <text x="615" y="400" font-family="{FONT_SANS}" font-size="11.5" fill="{c["ink2"]}">принятые шаги (чемпион: probe 35 = 0.7455, RLE −42%)</text>
  <circle cx="964" cy="396" r="4.5" fill="{c["surface"]}" stroke="{c["orange"]}" stroke-width="2"/>
  <text x="975" y="400" font-family="{FONT_SANS}" font-size="11.5" fill="{c["ink2"]}">проверены и отклонены</text>
</svg>
'''


def build_architecture(theme: str) -> str:
    """Схема двухэтапного конвейера КосмоФАЙР-2 и аналитического сервиса."""
    c = PALETTES[theme]
    W, H = 1200, 444

    def box(x: int, y: int, w: int, h: int, title: str, l1: str, l2: str, accent: str, bg: str, brd: str) -> str:
        return (
            f'<g transform="translate({x},{y})">'
            f'<rect width="{w}" height="{h}" rx="10" fill="{bg}" stroke="{brd}" stroke-width="1.3"/>'
            f'<rect x="0" y="12" width="4" height="{h - 24}" rx="2" fill="{accent}"/>'
            f'<text x="15" y="27" font-family="{FONT_SANS}" font-size="13" font-weight="800" fill="{c["ink"]}">{title}</text>'
            f'<text x="15" y="47" font-family="{FONT_SANS}" font-size="11.5" fill="{c["ink2"]}">{l1}</text>'
            f'<text x="15" y="64" font-family="{FONT_MONO}" font-size="10.5" fill="{c["muted"]}">{l2}</text>'
            f'</g>'
        )

    def arrow(x1: int, y1: int, x2: int, y2: int, col: str) -> str:
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2 - 6}" y2="{y2}" stroke="{col}" stroke-width="2"/>'
            f'<polygon points="{x2},{y2} {x2 - 8},{y2 - 4.5} {x2 - 8},{y2 + 4.5}" fill="{col}"/>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Архитектура КосмоФАЙР-2: конвейер AF (VIIRS 375 м), конвейер BS (Sentinel-2 20 м + HistGradientBoosting + majority 3×3), RLE-ответ и геосервис FastAPI.">
  <rect x="1" y="1" width="{W - 2}" height="{H - 2}" rx="16" fill="{c["bg0"]}" stroke="{c["border"]}" stroke-width="1.5"/>

  <text x="32" y="38" font-family="{FONT_MONO}" font-size="11" font-weight="700" fill="{c["mint"]}">АРХИТЕКТУРА РЕШЕНИЯ · ДВА КОНТУРА ДЗЗ → ЕДИНЫЙ RLE-ОТВЕТ И ГЕОСЕРВИС</text>

  <!-- Полоса 1: AF (Active Fire) -->
  <rect x="24" y="54" width="820" height="122" rx="12" fill="{c["surface"]}" stroke="{c["border"]}" stroke-width="1.1"/>
  <text x="40" y="76" font-family="{FONT_MONO}" font-size="11" font-weight="700" fill="{c["orange"]}">ЭТАП 1 · ACTIVE FIRE (fires/af.py · вес 0.35 · F1_AF = 0.9093 на train)</text>
  {box(40, 88, 240, 76, "Чип VIIRS 375 м", "I1–I5, углы, WorldCover, ERA5", "256 × 256 px · 96 × 96 км", c["orange"], c["surface2"], c["border"])}
  {arrow(280, 126, 304, 126, c["orange"])}
  {box(304, 88, 252, 76, "Скрининг и шкалы день/ночь", "Вода, NaN, режим блика; SZA 95°", "T4 ≥ 330/295 K · ΔT ≥ 25/10 K", c["orange"], c["surface2"], c["border"])}
  {arrow(556, 126, 580, 126, c["orange"])}
  {box(580, 88, 248, 76, "Контекстный тест 71 × 71", "I4 > μ_bg + k·σ_bg (Schroeder)", "k_sigma = 2.5 (p35) · маска 0/1", c["orange"], c["orange_bg"], c["orange_border"])}

  <!-- Полоса 2: BS (Burn Scar + Severity) -->
  <rect x="24" y="192" width="820" height="122" rx="12" fill="{c["surface"]}" stroke="{c["border"]}" stroke-width="1.1"/>
  <text x="40" y="214" font-family="{FONT_MONO}" font-size="11" font-weight="700" fill="{c["mint"]}">ЭТАП 2 · BURN SCAR &amp; SEVERITY (fires/bs.py + severity.py · вес 0.35 + 0.30)</text>
  {box(40, 226, 240, 76, "Sentinel-2 до/после + S1", "B2–B12, SCL, VV/VH, DEM, LC", "512 × 512 px · 20 м · 10,2 км", c["mint"], c["surface2"], c["border"])}
  {arrow(280, 264, 304, 264, c["mint"])}
  {box(304, 226, 252, 76, "Индексы + HGB-бустинг", "dNBR, NDVI, NDMI, контекст (23)", "SCL_BAD=(3,8,9,11) · CPU sklearn", c["mint"], c["surface2"], c["border"])}
  {arrow(556, 264, 580, 264, c["mint"])}
  {box(580, 226, 248, 76, "Постобработка (probe 35)", "Голосование 3×3 + чистка пятен", "majority 3×3 · min_patch 96 px", c["mint"], c["mint_bg"], c["mint_border"])}

  <!-- Правая колонка: Выходы (CLI RLE + FastAPI Service) -->
  {arrow(844, 126, 876, 126, c["amber"])}
  {arrow(844, 264, 876, 264, c["sky"])}
  <line x1="858" y1="126" x2="858" y2="264" stroke="{c["border_strong"]}" stroke-width="1.6" stroke-dasharray="4 3"/>

  <rect x="876" y="54" width="300" height="122" rx="12" fill="{c["amber_bg"]}" stroke="{c["amber_border"]}" stroke-width="1.3"/>
  <text x="894" y="78" font-family="{FONT_MONO}" font-size="11" font-weight="700" fill="{c["amber"]}">КОНКУРСНЫЙ ОТВЕТ (CLI)</text>
  <text x="894" y="103" font-family="{FONT_SANS}" font-size="15" font-weight="800" fill="{c["ink"]}">inference.py --validate</text>
  <text x="894" y="125" font-family="{FONT_SANS}" font-size="12" fill="{c["ink2"]}">Пул CPU-процессов · детерминизм</text>
  <text x="894" y="145" font-family="{FONT_MONO}" font-size="11" fill="{c["ink2"]}">447 строк RLE (180 AF + 3×89 BS)</text>
  <text x="894" y="164" font-family="{FONT_MONO}" font-size="11" font-weight="700" fill="{c["amber"]}">Score 0.7455 · 83,67 / 100 жюри</text>

  <rect x="876" y="192" width="300" height="122" rx="12" fill="{c["sky_bg"]}" stroke="{c["sky_border"]}" stroke-width="1.3"/>
  <text x="894" y="216" font-family="{FONT_MONO}" font-size="11" font-weight="700" fill="{c["sky"]}">ПРОДУКТ И ВИЗУАЛИЗАЦИЯ</text>
  <text x="894" y="241" font-family="{FONT_SANS}" font-size="15" font-weight="800" fill="{c["ink"]}">FastAPI + Three.js 3D</text>
  <text x="894" y="263" font-family="{FONT_SANS}" font-size="12" fill="{c["ink2"]}">bbox / polygon + даты → GeoJSON, SHP</text>
  <text x="894" y="283" font-family="{FONT_SANS}" font-size="12" fill="{c["ink2"]}">Справка по площадям (га, JSON / CSV)</text>
  <text x="894" y="302" font-family="{FONT_MONO}" font-size="11" font-weight="700" fill="{c["sky"]}">UTM→WGS84 без GDAL (fires/geo.py)</text>

  <!-- Нижняя плашка: сквозной фундамент -->
  <rect x="24" y="332" width="1152" height="88" rx="12" fill="{c["surface"]}" stroke="{c["border"]}" stroke-width="1.1"/>
  <text x="44" y="357" font-family="{FONT_MONO}" font-size="11" font-weight="700" fill="{c["muted"]}">СКВОЗНОЙ ИНЖЕНЕРНЫЙ КОНТУР</text>
  <text x="44" y="382" font-family="{FONT_SANS}" font-size="12.5" fill="{c["ink"]}">
    <tspan font-weight="700">fires/chips.py</tspan> — автоопределение раскладки TIFF и каналов   ·
    <tspan font-weight="700">train.py</tspan> — сплит по fire_event_id, SEED=20260918   ·
    <tspan font-weight="700">evaluate.py</tspan> — точная микро-метрика и 12 абляций
  </text>
  <text x="44" y="404" font-family="{FONT_SANS}" font-size="12.5" fill="{c["ink2"]}">
    <tspan font-weight="700">fires/synthetic.py</tspan> — офлайн smoke-стенд без 2,2 ГБ данных   ·
    <tspan font-weight="700">tests/</tspan> — 82 автотеста (RLE, детекторы, бустинг, API)   ·
    <tspan font-weight="700">Dockerfile</tspan> — non-root образ с healthcheck
  </text>
</svg>
'''


def build_footer(theme: str) -> str:
    c = PALETTES[theme]
    W, H = 1200, 82
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="КосмоФАЙР-2 · КосмоХакатон 2026 · Красноярск · 83,67 / 100 · 19 из 47">
  <rect x="1" y="1" width="{W - 2}" height="{H - 2}" rx="12" fill="{c["bg0"]}" stroke="{c["border"]}" stroke-width="1.2"/>
  <path d="M0,58 Q200,34 420,52 T840,46 T1200,56 L1200,82 L0,82 Z" fill="{c["surface"]}" opacity="0.85"/>
  <path d="M0,58 Q200,34 420,52 T840,46 T1200,56" fill="none" stroke="{c["mint_border"]}" stroke-width="1.3"/>
  <circle cx="36" cy="36" r="5" fill="{c["orange"]}"/>
  <text x="50" y="40" font-family="{FONT_SANS}" font-size="13" font-weight="700" fill="{c["ink"]}">КосмоФАЙР-2</text>
  <text x="156" y="40" font-family="{FONT_MONO}" font-size="11.5" fill="{c["muted"]}">КосмоХакатон 2026 · Красноярск · 83,67 / 100 (19 из 47) · Score 0.7455</text>
  <text x="1168" y="40" text-anchor="end" font-family="{FONT_MONO}" font-size="11.5" fill="{c["mint"]}">overbaffe1 / OMLP</text>
</svg>
'''


ARCHITECTURE_MMD = """flowchart LR
    subgraph AF["Этап 1 · Active Fire (VIIRS 375 м, 256×256)"]
        AF_IN["I1–I5 + углы + WorldCover + ERA5"] --> AF_SCR["Скрининг воды/NaN/блика\\nШкалы день/ночь (SZA 95°)"]
        AF_SCR --> AF_CTX["Контекстный тест 71×71\\nk_sigma = 2.5 → маска 0/1"]
    end

    subgraph BS["Этап 2 · Burn Scar & Severity (Sentinel-2 20 м + S1, 512×512)"]
        BS_IN["S2 pre/post + SCL + S1 VV/VH + LC"] --> BS_HGB["dNBR/NDVI/NDMI (23 признака)\\nSCL_BAD=(3,8,9,11) + HistGradientBoosting"]
        BS_HGB --> BS_POST["Majority 3×3 + min_patch 96 px\\nСтепени поражения 0–3"]
    end

    AF_CTX --> SUB["inference.py --validate\\nsubmission.csv (447 строк RLE)\\nScore 0.7455"]
    BS_POST --> SUB

    AF_CTX --> SRV["service/app.py (FastAPI)\\nGeoJSON · Shapefile · CSV/JSON\\nviewer/map.html (Three.js 3D)"]
    BS_POST --> SRV
"""


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for theme in ("dark", "light"):
        (ASSETS / f"hero-{theme}.svg").write_text(build_hero(theme), encoding="utf-8")
        (ASSETS / f"experiments-{theme}.svg").write_text(build_experiments(theme), encoding="utf-8")
        (ASSETS / f"architecture-{theme}.svg").write_text(build_architecture(theme), encoding="utf-8")
        (ASSETS / f"footer-{theme}.svg").write_text(build_footer(theme), encoding="utf-8")

    for name, content in build_vitrine_cards().items():
        (ASSETS / name).write_text(content, encoding="utf-8")

    (ASSETS / "architecture.mmd").write_text(ARCHITECTURE_MMD, encoding="utf-8")
    print("SVG-иллюстрации и architecture.mmd собраны в assets/.")


if __name__ == "__main__":
    main()
