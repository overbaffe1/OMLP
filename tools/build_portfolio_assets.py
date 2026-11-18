#!/usr/bin/env python3
"""Генератор визуальных экспонатов портфолио для README и витрины GitHub.

Строит из артефактов репозитория (docs/figures/severity.png, service/catalog.json,
data/aoi/fire_monitoring_aoi.geojson, docs/presentation.pdf, viewer/map.html):

* assets/burnscar-wipe.gif и assets/burnscar-wipe.png — анимированная шторка
  «dNBR Sentinel-2 (20 м) → ответ модели (степени 0–3)» на реальном чипе BS_tr_000095;
* assets/chip-000117-layers.png — четыре слоя чипа BS_tr_000117 (RGB, SWIR, dNBR, эталон);
* assets/dnbr-relief.stl — замкнутая 3D-модель рельефа dNBR чипа BS_tr_000095
  для встроенного 3D-просмотрщика GitHub;
* assets/burn-contours-top50.geojson — граница AOI (435 тыс. км²) + 50 крупнейших
  векторных контуров гари из каталога сервиса для интерактивной карты GitHub;
* assets/presentation-strip.png — витрина из 4 слайдов защиты (docs/presentation.pdf);
* assets/viewer-3d.png — снимок 3D-интерфейса просмотра чипа с рельефом dNBR;
* assets/social-preview.png — карточка ссылки GitHub 1280 × 640 (< 500 КБ).

Запуск из корня репозитория:
    python3 tools/build_portfolio_assets.py
"""

from __future__ import annotations

import base64
import io
import json
import re
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

FONT_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
FONT_REG_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
FONT_MONO_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def _font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _find_panels_in_severity_png() -> tuple[Image.Image, Image.Image, Image.Image]:
    """Вырезает ровно три квадратные панели 512×512 из docs/figures/severity.png."""
    img = Image.open(ROOT / "docs" / "figures" / "severity.png").convert("RGB")
    arr = np.asarray(img)
    # Ищем тёмные рамки панелей (чёрный бордюр #000000 вокруг каждой карты)
    is_black = (arr[:, :, 0] < 4) & (arr[:, :, 1] < 4) & (arr[:, :, 2] < 4)

    # Левая панель dNBR, средняя Эталон, правая Ответ
    h, w, _ = arr.shape
    boxes: list[tuple[int, int, int, int]] = []
    # Ищем вертикальные чёрные линии длиной > 400 px
    col_counts = is_black.sum(axis=0)
    border_cols = np.where(col_counts > int(h * 0.65))[0]
    # Группируем соседние колонки
    groups: list[int] = []
    for c in border_cols:
        if not groups or c - groups[-1] > 8:
            groups.append(int(c))
    # Пары границ шириной > 400 px (исключая узкую шкалу colorbar)
    for i in range(len(groups) - 1):
        x0, x1 = groups[i], groups[i + 1]
        if x1 - x0 > 400:
            row_counts = is_black[:, x0 + 5 : x1 - 5].sum(axis=1)
            border_rows = np.where(row_counts > int((x1 - x0 - 10) * 0.8))[0]
            y0, y1 = int(border_rows[0]), int(border_rows[-1])
            boxes.append((x0 + 2, y0 + 2, x1 - 1, y1 - 1))

    if len(boxes) != 3:
        raise RuntimeError(f"Ожидались 3 панели в severity.png, найдено {len(boxes)}: {boxes}")

    size = 520
    panels = [img.crop(b).resize((size, size), Image.Resampling.LANCZOS) for b in boxes]
    return panels[0], panels[1], panels[2]


def build_burnscar_wipe(dnbr_img: Image.Image, gt_img: Image.Image, pred_img: Image.Image) -> None:
    """Строит анимированную шторку assets/burnscar-wipe.gif и стоп-кадр .png."""
    W, H = 680, 560
    map_size = 468
    mx = (W - map_size) // 2
    my = 50

    left_map = dnbr_img.resize((map_size, map_size), Image.Resampling.LANCZOS)
    # На правой половине даём тонкую подложку dNBR в негоревших пикселях, чтобы был виден ландшафт
    dnbr_arr = np.asarray(left_map, dtype=np.float32)
    pred_arr = np.asarray(pred_img.resize((map_size, map_size), Image.Resampling.LANCZOS), dtype=np.float32)
    is_bg = (pred_arr[:, :, 0] < 24) & (pred_arr[:, :, 1] < 28) & (pred_arr[:, :, 2] < 34)
    blended = pred_arr.copy()
    # Приглушённый монохромный фон снимка там, где класс 0
    lum = 0.25 * dnbr_arr[:, :, 0] + 0.35 * dnbr_arr[:, :, 1] + 0.40 * dnbr_arr[:, :, 2]
    blended[is_bg, 0] = np.clip(9 + lum[is_bg] * 0.28, 0, 255)
    blended[is_bg, 1] = np.clip(18 + lum[is_bg] * 0.34, 0, 255)
    blended[is_bg, 2] = np.clip(16 + lum[is_bg] * 0.30, 0, 255)
    right_map = Image.fromarray(blended.astype(np.uint8))

    f_title = _font(FONT_BOLD_CANDIDATES, 14)
    f_sub = _font(FONT_REG_CANDIDATES, 12)
    f_pill = _font(FONT_BOLD_CANDIDATES, 11)
    f_mono = _font(FONT_MONO_CANDIDATES, 11)

    def render_frame(split_frac: float) -> Image.Image:
        canvas = Image.new("RGB", (W, H), (10, 20, 17))
        draw = ImageDraw.Draw(canvas)

        # Тонкая рамка карточки
        draw.rounded_rectangle([1, 1, W - 2, H - 2], radius=14, outline=(31, 56, 47), width=1)

        # Шапка
        draw.text((mx, 14), "BS_tr_000095 · Sentinel-2 (20 м, 10,2 × 10,2 км)", fill=(232, 241, 236), font=f_title)
        draw.text((mx, 31), "Слева — спектральный индекс dNBR, справа — ответ модели (классы 0–3)", fill=(142, 168, 156), font=f_sub)

        # Композит со шторкой
        sx = int(round(map_size * split_frac))
        sx = max(24, min(map_size - 24, sx))
        comp = right_map.copy()
        comp.paste(left_map.crop((0, 0, sx, map_size)), (0, 0))
        canvas.paste(comp, (mx, my))

        # Рамка карты и линия шторки
        draw.rectangle([mx - 1, my - 1, mx + map_size, my + map_size], outline=(46, 82, 68), width=1)
        lx = mx + sx
        draw.line([(lx, my), (lx, my + map_size)], fill=(236, 253, 245), width=2)

        # Ручка ползунка по центру
        cy = my + map_size // 2
        draw.ellipse([lx - 13, cy - 13, lx + 13, cy + 13], fill=(12, 26, 22), outline=(140, 233, 181), width=2)
        draw.polygon([(lx - 7, cy), (lx - 3, cy - 4), (lx - 3, cy + 4)], fill=(140, 233, 181))
        draw.polygon([(lx + 7, cy), (lx + 3, cy - 4), (lx + 3, cy + 4)], fill=(140, 233, 181))

        # Плашки поверх углов карты (ширина по реальной метрике шрифта)
        left_txt = "dNBR (до − после)"
        lw = int(draw.textlength(left_txt, font=f_pill)) + 20
        draw.rounded_rectangle([mx + 10, my + 10, mx + 10 + lw, my + 32], radius=6, fill=(10, 20, 17), outline=(66, 98, 86))
        draw.text((mx + 20, my + 14), left_txt, fill=(248, 212, 154), font=f_pill)

        right_txt = "Маска модели 0–3"
        rw = int(draw.textlength(right_txt, font=f_pill)) + 20
        draw.rounded_rectangle([mx + map_size - 10 - rw, my + 10, mx + map_size - 10, my + 32], radius=6, fill=(10, 20, 17), outline=(66, 98, 86))
        draw.text((mx + map_size - rw, my + 14), right_txt, fill=(140, 233, 181), font=f_pill)

        # Легенда снизу
        ly = my + map_size + 13
        legend = [
            ((216, 162, 54), "1 слабая"),
            ((255, 129, 66), "2 средняя"),
            ((244, 72, 58), "3 сильная"),
        ]
        cx = mx
        draw.text((cx, ly), "Степень поражения:", fill=(148, 173, 162), font=f_sub)
        cx += 136
        for color, label in legend:
            draw.rounded_rectangle([cx, ly + 2, cx + 12, ly + 14], radius=3, fill=color)
            draw.text((cx + 17, ly), label, fill=(224, 236, 230), font=f_sub)
            cx += 96
        draw.text((mx + map_size - 48, ly), "GSD 20m", fill=(116, 146, 134), font=f_mono)
        return canvas

    # Стоп-кадр для prefers-reduced-motion
    still = render_frame(0.48)
    still.save(ASSETS / "burnscar-wipe.png", optimize=True)

    # Плавная синусоидальная петля (32 кадра, ~3.8 с) с единой адаптивной палитрой
    n_frames = 32
    ts = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    fracs = 0.50 - 0.36 * np.cos(ts)  # 0.14 → 0.86 → 0.14
    rgb_frames = [render_frame(float(f)) for f in fracs]

    # Общая палитра по ключевому кадру 50/50 (в нём видны оба слоя целиком)
    master_pal = still.quantize(colors=192, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal_frames = [fr.quantize(palette=master_pal, dither=Image.Dither.FLOYDSTEINBERG) for fr in rgb_frames]
    durations = [110] * n_frames
    durations[0] = 420
    durations[n_frames // 2] = 420
    pal_frames[0].save(
        ASSETS / "burnscar-wipe.gif",
        save_all=True,
        append_images=pal_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


def build_dnbr_stl(dnbr_img: Image.Image) -> None:
    """Строит замкнутую ASCII STL-модель рельефа dNBR (BS_tr_000095) на постаменте."""
    grid = 56
    small = dnbr_img.resize((grid, grid), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    # В палитре magma/inferno яркость монотонно растёт с dNBR (от чёрного через пурпурный/оранжевый к жёлтому)
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    # Лёгкое сглаживание 3×3 для аккуратных граней без ступенчатого шума
    pad = np.pad(lum, 1, mode="edge")
    smooth = (
        pad[:-2, :-2] + 2 * pad[:-2, 1:-1] + pad[:-2, 2:]
        + 2 * pad[1:-1, :-2] + 4 * pad[1:-1, 1:-1] + 2 * pad[1:-1, 2:]
        + pad[2:, :-2] + 2 * pad[2:, 1:-1] + pad[2:, 2:]
    ) / 16.0

    # Координаты в мм: площадка 80 × 80 мм, постамент 3.5 мм, подъём гари до +18 мм
    span = 80.0
    base_z = 0.0
    plinth_z = 3.5
    relief_scale = 18.5
    xs = np.linspace(-span / 2, span / 2, grid)
    ys = np.linspace(span / 2, -span / 2, grid)  # север вверху (+Y)
    z_top = plinth_z + np.power(np.clip(smooth, 0.0, 1.0), 1.15) * relief_scale

    tris: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add_tri(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        tris.append((a, b, c))

    # Верхняя поверхность (нормаль вверх +Z; т.к. ys убывает при росте r, обход p00→p10→p11 даёт +Z)
    for r in range(grid - 1):
        for c in range(grid - 1):
            p00 = np.array([xs[c], ys[r], z_top[r, c]], dtype=np.float32)
            p10 = np.array([xs[c], ys[r + 1], z_top[r + 1, c]], dtype=np.float32)
            p01 = np.array([xs[c + 1], ys[r], z_top[r, c + 1]], dtype=np.float32)
            p11 = np.array([xs[c + 1], ys[r + 1], z_top[r + 1, c + 1]], dtype=np.float32)
            add_tri(p00, p10, p11)
            add_tri(p00, p11, p01)

    # Четыре боковые стенки (замкнутый периметр)
    # Северный край (r = 0, y = +span/2, нормаль +Y)
    for c in range(grid - 1):
        t0 = np.array([xs[c], ys[0], z_top[0, c]], dtype=np.float32)
        t1 = np.array([xs[c + 1], ys[0], z_top[0, c + 1]], dtype=np.float32)
        b0 = np.array([xs[c], ys[0], base_z], dtype=np.float32)
        b1 = np.array([xs[c + 1], ys[0], base_z], dtype=np.float32)
        add_tri(b0, t0, t1)
        add_tri(b0, t1, b1)

    # Южный край (r = grid-1, y = -span/2, нормаль -Y)
    rl = grid - 1
    for c in range(grid - 1):
        t0 = np.array([xs[c], ys[rl], z_top[rl, c]], dtype=np.float32)
        t1 = np.array([xs[c + 1], ys[rl], z_top[rl, c + 1]], dtype=np.float32)
        b0 = np.array([xs[c], ys[rl], base_z], dtype=np.float32)
        b1 = np.array([xs[c + 1], ys[rl], base_z], dtype=np.float32)
        add_tri(b0, t1, t0)
        add_tri(b0, b1, t1)

    # Западный край (c = 0, x = -span/2, нормаль -X)
    for r in range(grid - 1):
        t0 = np.array([xs[0], ys[r], z_top[r, 0]], dtype=np.float32)
        t1 = np.array([xs[0], ys[r + 1], z_top[r + 1, 0]], dtype=np.float32)
        b0 = np.array([xs[0], ys[r], base_z], dtype=np.float32)
        b1 = np.array([xs[0], ys[r + 1], base_z], dtype=np.float32)
        add_tri(b0, t1, t0)
        add_tri(b0, b1, t1)

    # Восточный край (c = grid-1, x = +span/2, нормаль +X)
    cl = grid - 1
    for r in range(grid - 1):
        t0 = np.array([xs[cl], ys[r], z_top[r, cl]], dtype=np.float32)
        t1 = np.array([xs[cl], ys[r + 1], z_top[r + 1, cl]], dtype=np.float32)
        b0 = np.array([xs[cl], ys[r], base_z], dtype=np.float32)
        b1 = np.array([xs[cl], ys[r + 1], base_z], dtype=np.float32)
        add_tri(b0, t0, t1)
        add_tri(b0, t1, b1)

    # Дно (нормаль -Z)
    sw = np.array([-span / 2, -span / 2, base_z], dtype=np.float32)
    se = np.array([span / 2, -span / 2, base_z], dtype=np.float32)
    ne = np.array([span / 2, span / 2, base_z], dtype=np.float32)
    nw = np.array([-span / 2, span / 2, base_z], dtype=np.float32)
    add_tri(sw, ne, se)
    add_tri(sw, nw, ne)

    # Двоичный STL (компактный — ~315 КБ, нативно открывается в 3D-просмотрщике GitHub)
    out = ASSETS / "dnbr-relief.stl"
    header = b"CosmoFIRE-2 dNBR burn scar relief BS_tr_000095 (Sentinel-2 20m)".ljust(80, b" ")
    with out.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            n = np.cross(b - a, c - a)
            norm = float(np.linalg.norm(n))
            if norm > 1e-9:
                n = n / norm
            f.write(struct.pack("<12fH", n[0], n[1], n[2], a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], 0))


def build_geojson_showcase() -> None:
    """Собирает интерактивную карту AOI + 50 крупнейших контуров гари (WGS84)."""
    aoi_data = json.loads((ROOT / "data" / "aoi" / "fire_monitoring_aoi.geojson").read_text(encoding="utf-8"))
    catalog = json.loads((ROOT / "service" / "catalog.json").read_text(encoding="utf-8"))

    features: list[dict] = []
    for feat in aoi_data["features"]:
        if feat.get("id") == "aoi":
            props = dict(feat["properties"])
            props.update({
                "stroke": "#10B981",
                "stroke-width": 2.5,
                "stroke-opacity": 0.95,
                "fill": "#10B981",
                "fill-opacity": 0.04,
            })
            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": feat["geometry"],
            })

    sev_palette = {
        1: ("#EAB308", "1 — слабая"),
        2: ("#F97316", "2 — средняя"),
        3: ("#EF4444", "3 — сильная"),
    }
    contours = sorted(catalog.get("contours", []), key=lambda c: float(c.get("area_ha", 0.0)), reverse=True)[:50]
    for c in contours:
        ring = c.get("ring") or []
        if len(ring) < 4:
            continue
        # Округляем до 5 знаков (~1 м) и гарантируем замкнутое кольцо RFC 7946
        coords = [[round(float(lon), 5), round(float(lat), 5)] for lon, lat in ring]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        sev = int(c.get("severity", 1))
        color, label = sev_palette.get(sev, ("#F97316", str(sev)))
        features.append({
            "type": "Feature",
            "properties": {
                "id": c["id"],
                "chip_id": c["chip_id"],
                "date": c.get("date", ""),
                "severity": label,
                "area_ha": round(float(c["area_ha"]), 2),
                "stroke": color,
                "stroke-width": 2,
                "stroke-opacity": 0.95,
                "fill": color,
                "fill-opacity": 0.45,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
        })

    fc = {
        "type": "FeatureCollection",
        "features": features,
    }
    (ASSETS / "burn-contours-top50.geojson").write_text(
        json.dumps(fc, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def build_chip_layers_strip() -> Image.Image | None:
    """Извлекает 4 слоя обучающего чипа BS_tr_000117 из viewer/map.html в аккуратную ленту."""
    map_path = ROOT / "viewer" / "map.html"
    if not map_path.exists():
        return None
    txt = map_path.read_text(encoding="utf-8")
    m = re.search(r"var PACK\s*=\s*", txt)
    if not m:
        return None
    end = txt.find(";\n", m.end())
    pack = json.loads(txt[m.end() : end])
    chip = next((x for x in pack.get("singles", []) if x.get("cid") == "BS_tr_000117"), None)
    if not chip:
        return None

    def decode_data_url(url: str) -> Image.Image:
        raw = base64.b64decode(url.split(",", 1)[1])
        return Image.open(io.BytesIO(raw)).convert("RGBA")

    rgb = decode_data_url(chip["true512"]).convert("RGB")
    swir = decode_data_url(chip["false512"]).convert("RGB")
    dnbr = decode_data_url(chip["dnbr512"]).convert("RGB")
    gt_rgba = decode_data_url(chip["gt"])
    # Накладываем маску эталона полупрозрачно поверх приглушённого SWIR
    dark_swir = Image.eval(swir, lambda p: int(p * 0.42)).convert("RGBA")
    gt_comp = Image.alpha_composite(dark_swir, gt_rgba).convert("RGB")

    tile = 240
    gap = 14
    pad = 18
    W = pad * 2 + tile * 4 + gap * 3
    H = pad + 34 + tile + 32
    canvas = Image.new("RGB", (W, H), (10, 20, 17))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle([1, 1, W - 2, H - 2], radius=12, outline=(31, 56, 47), width=1)

    f_head = _font(FONT_BOLD_CANDIDATES, 13)
    f_cap = _font(FONT_REG_CANDIDATES, 12)
    draw.text(
        (pad, 13),
        "Обучающий чип BS_tr_000117 (Sentinel-2, 512 × 512 px, 20 м) — четыре представления одной сцены",
        fill=(228, 239, 233),
        font=f_head,
    )

    items = [
        (rgb, "01 · Естественные цвета (RGB)"),
        (swir, "02 · Ложные цвета SWIR"),
        (dnbr, "03 · Индекс гари dNBR"),
        (gt_comp, "04 · Эталонная маска 1–3"),
    ]
    y0 = pad + 28
    for i, (im, label) in enumerate(items):
        x0 = pad + i * (tile + gap)
        resized = im.resize((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(resized, (x0, y0))
        draw.rectangle([x0 - 1, y0 - 1, x0 + tile, y0 + tile], outline=(46, 82, 68), width=1)
        draw.text((x0, y0 + tile + 7), label, fill=(154, 180, 168), font=f_cap)

    canvas.save(ASSETS / "chip-000117-layers.png", optimize=True)
    return swir


def build_presentation_assets(swir_fallback: Image.Image | None) -> None:
    """Рендерит витрину из 4 слайдов защиты, кадр 3D-интерфейса и Social Preview 1280×640."""
    try:
        import pymupdf
    except ImportError:
        print("pymupdf не установлен — пропуск рендера слайдов presentation.pdf")
        return

    pdf_path = ROOT / "docs" / "presentation.pdf"
    if not pdf_path.exists():
        return
    doc = pymupdf.open(pdf_path)

    # 1. Витрина 2×2 из четырёх ключевых слайдов (1, 4, 5, 6)
    indices = [0, 3, 4, 5]
    tw, th = 480, 270
    gap = 12
    pad = 14
    W = pad * 2 + tw * 2 + gap
    H = pad * 2 + th * 2 + gap
    grid = Image.new("RGB", (W, H), (8, 16, 13))
    gdraw = ImageDraw.Draw(grid)
    for pos, idx in enumerate(indices):
        pix = doc[idx].get_pixmap(matrix=pymupdf.Matrix(1.25, 1.25), alpha=False)
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).resize((tw, th), Image.Resampling.LANCZOS)
        r, c = divmod(pos, 2)
        x0 = pad + c * (tw + gap)
        y0 = pad + r * (th + gap)
        grid.paste(im, (x0, y0))
        gdraw.rectangle([x0 - 1, y0 - 1, x0 + tw, y0 + th], outline=(38, 69, 57), width=1)
    grid.save(ASSETS / "presentation-strip.png", optimize=True)

    # 2. Чистый снимок 3D-интерфейса со слайда 8 (встроенный растр 1440×900)
    imgs_p8 = doc[7].get_images(full=True)
    if imgs_p8:
        xref = imgs_p8[0][0]
        raw = doc.extract_image(xref)["image"]
        v_im = Image.open(io.BytesIO(raw)).convert("RGB")
        v_im.resize((960, 600), Image.Resampling.LANCZOS).save(ASSETS / "viewer-3d.png", quality=88, optimize=True)

    # 3. SWIR-снимок со слайда 1 (или из map.html) для карточки Social Preview 1280 × 640
    swir = swir_fallback
    imgs_p1 = doc[0].get_images(full=True)
    if swir is None and imgs_p1:
        raw = doc.extract_image(imgs_p1[0][0])["image"]
        swir = Image.open(io.BytesIO(raw)).convert("RGB")

    build_social_preview(swir)


def build_social_preview(swir: Image.Image | None) -> None:
    """Строит GitHub Social Preview 1280 × 640 (< 500 КБ)."""
    W, H = 1280, 640
    card = Image.new("RGB", (W, H), (10, 22, 18))
    draw = ImageDraw.Draw(card)

    # Тонкая координатная сетка
    for x in range(0, W, 64):
        draw.line([(x, 0), (x, H)], fill=(16, 33, 27), width=1)
    for y in range(0, H, 64):
        draw.line([(0, y), (W, y)], fill=(16, 33, 27), width=1)

    # Акцентная полоса сверху
    draw.rectangle([0, 0, W, 5], fill=(16, 185, 129))
    draw.rectangle([0, 0, 420, 5], fill=(249, 115, 22))

    # Справа — реальный SWIR-снимок чипа BS_tr_000117 в рамке
    if swir is not None:
        sq = 430
        sx, sy = W - sq - 64, 78
        # Мягкое свечение под рамкой
        glow = Image.new("RGB", (W, H), (0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.rounded_rectangle([sx - 8, sy - 8, sx + sq + 8, sy + sq + 8], radius=16, fill=(24, 64, 50))
        glow = glow.filter(ImageFilter.GaussianBlur(18))
        card = Image.fromarray(np.clip(np.asarray(card, np.int16) + np.asarray(glow, np.int16), 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(card)

        card.paste(swir.resize((sq, sq), Image.Resampling.LANCZOS), (sx, sy))
        draw.rectangle([sx - 1, sy - 1, sx + sq, sy + sq], outline=(62, 112, 92), width=2)
        draw.rectangle([sx, sy + sq, sx + sq, sy + sq + 46], fill=(16, 35, 29), outline=(44, 80, 66))
        f_cap = _font(FONT_REG_CANDIDATES, 14)
        draw.text((sx + 14, sy + sq + 14), "Sentinel-2 SWIR · BS_tr_000117 · 20 м/пикс", fill=(166, 214, 191), font=f_cap)

    f_kick = _font(FONT_BOLD_CANDIDATES, 14)
    f_h1 = _font(FONT_BOLD_CANDIDATES, 62)
    f_sub = _font(FONT_REG_CANDIDATES, 22)
    f_num = _font(FONT_BOLD_CANDIDATES, 28)
    f_lbl = _font(FONT_REG_CANDIDATES, 14)
    f_foot = _font(FONT_MONO_CANDIDATES, 14)

    lx = 64
    kick_txt = "КОСМОХАКАТОН 2026 · КРАСНОЯРСК · ДЗЗ / ML"
    kw = int(draw.textlength(kick_txt, font=f_kick)) + 32
    draw.rounded_rectangle([lx, 62, lx + kw, 96], radius=17, fill=(19, 42, 34), outline=(46, 96, 77))
    draw.text((lx + 16, 71), kick_txt, fill=(140, 233, 181), font=f_kick)

    draw.text((lx, 126), "КосмоФАЙР-2", fill=(244, 250, 247), font=f_h1)
    draw.text((lx, 212), "Двухэтапный мониторинг природных пожаров:", fill=(188, 208, 198), font=f_sub)
    draw.text((lx, 244), "VIIRS 375 м (очаги) + Sentinel-2 20 м (гарь 0–3)", fill=(188, 208, 198), font=f_sub)
    draw.text((lx, 278), "CPU-бустинг · FastAPI GeoJSON/SHP · Three.js 3D", fill=(134, 165, 152), font=f_sub)

    # Три карточки результатов (ширина 214 px, зазор 12 px → помещаются с запасом до снимка)
    metrics = [
        ("83,67 / 100", "19-е место из 47 · жюри", (251, 191, 36), (92, 70, 24)),
        ("0.7455", "лучший Score · probe 35", (140, 233, 181), (34, 88, 66)),
        ("82 / 82", "pytest · 447 строк RLE", (125, 211, 252), (32, 72, 96)),
    ]
    card_w = 214
    for i, (val, lbl, col, border) in enumerate(metrics):
        bx = lx + i * (card_w + 12)
        by = 366
        draw.rounded_rectangle([bx, by, bx + card_w, by + 116], radius=12, fill=(14, 30, 25), outline=border, width=2)
        draw.text((bx + 15, by + 20), val, fill=col, font=f_num)
        draw.text((bx + 15, by + 72), lbl, fill=(176, 198, 188), font=f_lbl)

    draw.text((lx, 568), "github.com/overbaffe1/OMLP  ·  Python · scikit-learn · FastAPI · Docker", fill=(122, 152, 140), font=f_foot)

    # Сохраняем как оптимизированный PNG (256 цветов или RGB, гарантированно < 500 КБ)
    out = ASSETS / "social-preview.png"
    card.save(out, optimize=True)
    if out.stat().st_size > 850_000:
        card.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(out, optimize=True)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    dnbr_img, gt_img, pred_img = _find_panels_in_severity_png()
    build_burnscar_wipe(dnbr_img, gt_img, pred_img)
    build_dnbr_stl(dnbr_img)
    build_geojson_showcase()
    swir = build_chip_layers_strip()
    build_presentation_assets(swir)
    print("Готово. Собранные файлы в assets/:")
    for p in sorted(ASSETS.iterdir()):
        print(f"  {p.name:32s} {p.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
