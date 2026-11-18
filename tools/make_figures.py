"""Фигуры для отчёта и презентации: примеры детекции, гари, карта сервиса и абляции.

Запуск (нужен локальный датасет, например синтетический):

    python tools/make_figures.py --data-dir /tmp/firesdata/dev2 --out docs/figures

Фигуры сознательно строятся по тому же коду, что используется в решении, — это исключает
расхождение между картинкой в презентации и фактическим поведением алгоритма.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # запуск из любого каталога

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import evaluate as evaluate_module  # noqa: E402
from fires import af as af_module  # noqa: E402
from fires import bs as bs_module  # noqa: E402
from fires.blobfilter import BlobFilter  # noqa: E402
from fires.chips import Dataset  # noqa: E402
from fires.metrics import score_from_pairs  # noqa: E402
from fires.weights import load_weights  # noqa: E402

BG = "#0e1116"
FG = "#e6edf3"
ACCENT = "#ffb03a"


def _style(ax, title: str) -> None:
    ax.set_title(title, color=FG, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def figure_detection(dataset: Dataset, af_params, blob_filter, out: Path) -> tuple[Path, float]:
    """Пример детекции активного горения: яркостная температура, эталон, ответ алгоритма."""

    def fire_px(chip_id: str) -> int:
        mask = dataset.load(chip_id).mask
        return 0 if mask is None else int(np.asarray(mask).sum())

    aff_ids = dataset.chip_ids("af")
    candidates = [cid for cid in aff_ids if fire_px(cid) > 40]
    chip_id = candidates[0] if candidates else aff_ids[0]
    chip = dataset.load(chip_id)
    pred = af_module.detect_chip(chip, af_params, blob_filter=blob_filter)
    truth = chip.mask.astype(bool)
    i4 = chip.layers["I4"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)

    im0 = axes[0].imshow(i4, cmap="inferno")
    _style(axes[0], f"I4 (яркостная температура), {chip_id}")
    cb = fig.colorbar(im0, ax=axes[0], fraction=0.046)
    cb.ax.tick_params(colors=FG, labelsize=8)

    axes[1].imshow(truth, cmap="gray_r")
    _style(axes[1], f"Эталон: {int(truth.sum())} px горения")

    # наглядная карта ошибок: зелёный — верно найдено, красный — ложное, серый — пропуск
    overlay = np.zeros((*truth.shape, 3), dtype=float)
    overlay[truth & ~pred] = (0.55, 0.58, 0.62)
    overlay[pred & truth] = (0.30, 0.90, 0.40)
    overlay[pred & ~truth] = (0.95, 0.25, 0.20)
    axes[2].imshow(overlay)
    _style(axes[2], "Ответ: зелёный — верно, красный — ложное, серый — пропуск")

    fig.tight_layout()
    path = out / "detection.png"
    fig.savefig(path, facecolor=BG, dpi=120)
    plt.close(fig)

    score = score_from_pairs({chip_id: (truth, pred)}, {})
    return path, score["f1_af"]


def figure_severity(dataset: Dataset, bs_params, out: Path, severity_model=None) -> tuple[Path, dict]:
    """Пример картирования гари: dNBR, эталонная степень поражения и ответ алгоритма."""

    ids = [cid for cid in dataset.chip_ids("bs") if dataset.load(cid).mask_sev is not None]
    chip_id = max(ids, key=lambda cid: int((dataset.load(cid).mask_sev > 0).sum()))
    chip = dataset.load(chip_id)
    pred = bs_module.classify_chip(chip, bs_params, severity_model=severity_model)
    truth = chip.mask_sev

    nbr_pre = bs_module.nbr(bs_module.scale_reflectance(chip.pre["B8A"]), bs_module.scale_reflectance(chip.pre["B12"]))
    nbr_post = bs_module.nbr(bs_module.scale_reflectance(chip.post["B8A"]), bs_module.scale_reflectance(chip.post["B12"]))
    dnbr = nbr_pre - nbr_post

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor=BG)
    cmap = matplotlib.colors.ListedColormap(["#0b0f14", "#d6a83a", "#ff8f4a", "#ff4a3c"])
    for ax in axes:
        ax.set_facecolor(BG)
    im = axes[0].imshow(dnbr, cmap="magma", vmin=-0.1, vmax=0.8)
    _style(axes[0], f"dNBR (NBR до − после), {chip_id}")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046)
    cb.ax.tick_params(colors=FG, labelsize=8)
    axes[1].imshow(truth, cmap=cmap, vmin=0, vmax=3)
    _style(axes[1], "Эталон: степень поражения 0–3")
    axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=3)
    _style(axes[2], "Ответ алгоритма: 0–3")

    fig.tight_layout()
    path = out / "severity.png"
    fig.savefig(path, facecolor=BG, dpi=120)
    plt.close(fig)

    score = score_from_pairs({}, {chip_id: (truth, pred)})
    return path, {"iou_burn": score["iou_burn"], "miou_sev": score["miou_sev"]}


def figure_service_map(catalog_path: Path, out: Path) -> Path | None:
    """Карта каталога сервиса: контуры гари по степеням и термоточки."""

    if not catalog_path.exists():
        return None
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    contours = catalog.get("contours", [])
    points = catalog.get("thermopoints", [])
    if not contours and not points:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 6.2), facecolor=BG)
    ax.set_facecolor("#0b0f14")
    colors = {1: "#d6a83a", 2: "#ff8f4a", 3: "#ff4a3c"}
    for contour in contours:
        ring = np.asarray(contour["ring"], dtype=float)
        ax.fill(ring[:, 0], ring[:, 1], color=colors.get(int(contour["severity"]), "#888888"), alpha=0.55)
    if points:
        lon = np.asarray([p["lon"] for p in points])
        lat = np.asarray([p["lat"] for p in points])
        ax.scatter(lon, lat, s=6, color="#ff6b4a", linewidths=0, label=f"термоточки ({len(points)})")
        ax.legend(facecolor=BG, edgecolor="#2a3441", labelcolor=FG, fontsize=9, loc="upper right")
    ax.set_title("Каталог сервиса: контуры гари и термоточки", color=FG, fontsize=11)
    ax.tick_params(colors="#93a4b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2a3441")
    fig.tight_layout()
    path = out / "service_map.png"
    fig.savefig(path, facecolor=BG, dpi=120)
    plt.close(fig)
    return path


def figure_ablations(rows: list[tuple[str, float]], out: Path, title: str = "Абляции: вклад механизмов") -> Path | None:
    """Столбчатая диаграмма Score по вариантам абляций."""

    if not rows:
        return None
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(10.5, 4.6), facecolor=BG)
    ax.set_facecolor("#0b0f14")
    order = np.argsort(values)
    y = np.arange(len(rows))
    ax.barh(y, [values[i] for i in order], color=ACCENT, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], color=FG, fontsize=9)
    ax.axvline(values[names.index("Итоговая конфигурация")] if "Итоговая конфигурация" in names else max(values),
               color="#4fc3ff", linestyle="--", linewidth=1.2)
    for i, idx in enumerate(order):
        ax.text(values[idx] + 0.002, i, f"{values[idx]:.4f}", color=FG, fontsize=8.5, va="center")
    ax.set_xlim(min(values) - 0.03, max(values) + 0.02)
    ax.set_xlabel("Score", color=FG, fontsize=9)
    ax.set_title(title, color=FG, fontsize=11)
    ax.tick_params(colors="#93a4b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2a3441")
    fig.tight_layout()
    path = out / "ablations.png"
    fig.savefig(path, facecolor=BG, dpi=120)
    plt.close(fig)
    return path


def figure_pipeline(out: Path, n_ablations: int = 12) -> Path:
    """Схема конвейера решения: одна фигура для README, отчёта и презентации."""

    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13, 5.6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 47)
    ax.axis("off")

    def box(x, y, w, h, title, lines, color="#1b2430", accent=ACCENT):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    linewidth=1.4, edgecolor=accent, facecolor=color))
        ax.text(x + w / 2, y + h - 2.2, title, color=FG, fontsize=10.5, ha="center", va="top", weight="bold")
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 5.8 - i * 2.9, line, color="#b9c7d6", fontsize=8.4, ha="center", va="top")

    def arrow(x1, y1, x2, y2, label="", dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.5, color="#4fc3ff",
                                     linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.2, label, color="#8fb6d8", fontsize=8, ha="center")

    box(1, 27, 20, 17, "Данные", ["VIIRS I1–I5 + AUX (AF)", "Sentinel-2 pre/post + SCL,", "Sentinel-1 VV/VH (BS)",
                                  "раскладка → fires/chips.py"], accent="#7fb2ff")
    box(1, 3, 20, 17, "Разметка (только train)", ["маски 0/1 (AF), 0–3 (BS)",
                                                   "разделение по fire_event_id", "315/105 и 168/56 чипов"], accent="#7fb2ff")

    box(26, 27, 21, 17, "AF: активное горение", ["день/ночь по sun_zenith", "абсолютные + контекстный тест",
                                                "вода, техногенные, блик", "бинарная маска горения"], accent="#ffb03a")
    box(26, 3, 21, 17, "BS: гарь и степень", ["NBR, dNBR, маска SCL по обеим датам", "пороги по покровам · или",
                                              "модель степени (23 признака)", "Sentinel-1 как подтверждение"], accent="#ff8f4a")

    box(52, 27, 20, 17, "Калибровка (train.py)", ["AF: 108 конфигураций +", "классификатор очагов",
                                                  "BS: пороги + SAR + модель", "→ weights/weights.json"], accent="#9be37f")
    box(52, 3, 20, 17, "Оценка (evaluate.py)", ["метрика соревнования", f"{n_ablations} абляций: вклад механизмов",
                                                "числа → docs/numbers.json"], accent="#9be37f")

    box(77, 27, 22, 17, "Ответ", ["submission.csv (RLE)", "chip_id, class_id, rle",
                                  "проверка формата и времени", "inference.py, пул процессов"], accent="#4fc3ff")
    box(77, 3, 22, 17, "Сервис", ["каталог термоточек и контуров", "REST + GeoJSON/Shapefile + CSV",
                                  "веб-карта офлайн", "UTM 37N/38N → WGS84"], accent="#4fc3ff")

    arrow(21, 35.5, 26, 35.5)
    arrow(21, 11.5, 26, 11.5)
    arrow(47, 35.5, 52, 35.5)
    arrow(47, 11.5, 52, 11.5)
    arrow(62, 27, 62, 20, dashed=True)
    arrow(72, 35.5, 77, 35.5)
    arrow(72, 11.5, 77, 11.5)
    arrow(47, 32, 52, 27)

    ax.text(50, 45.6, "Конвейер решения «КосмоФАЙР-2»: два независимых модуля, один формат ответа",
            color=FG, fontsize=12, ha="center", weight="bold")
    fig.tight_layout()
    path = out / "pipeline.png"
    fig.savefig(path, facecolor=BG, dpi=120)
    plt.close(fig)
    return path


def _ablation_rows_from_numbers(path: Path) -> list[tuple[str, float]]:
    """Читает ``docs/numbers.json`` (сбор tools/collect_numbers.py) в список (вариант, Score)."""

    import json

    numbers = json.loads(path.read_text(encoding="utf-8"))
    rows = [("Итоговая конфигурация", float(numbers["final"]["score"]))]
    for row in numbers.get("ablations", []):
        rows.append((str(row["name"]), float(row["score"])))
    return rows


def _parse_ablation_log(path: Path, final_score: float) -> list[tuple[str, float]]:
    """Разбирает лог ``evaluate.py --ablation`` в список (название варианта, Score)."""

    rows: list[tuple[str, float]] = [("Итоговая конфигурация", float(final_score))]
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Score=" not in line:
            continue
        name, _, rest = line.rpartition("  Score=")
        try:
            value = float(rest.split()[0])
        except (IndexError, ValueError):
            continue
        rows.append((name.strip(), value))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Фигуры для отчёта и презентации")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--weights", default="weights/weights.json")
    ap.add_argument("--catalog", default="service/catalog.json")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--ablation-log", default=None,
                    help="лог evaluate.py --ablation: из него строится фигура с реальными числами")
    ap.add_argument("--numbers", default=None,
                    help="docs/numbers.json (tools/collect_numbers.py) — предпочтительный источник абляций")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dataset = Dataset(args.data_dir)
    weights = load_weights(args.weights)
    af_params = af_module.to_params(weights.get("af", {}))
    bs_params = bs_module.to_params(weights.get("bs", {}))
    blob_filter = None
    if weights.get("af_mode") == "blobs" and weights.get("blob_filter"):
        base = Path(args.weights).parent if args.weights else Path("weights")
        blob_filter = BlobFilter.load(base / weights["blob_filter"])
    severity_model = None
    if weights.get("bs_mode") == "model" and weights.get("severity_model"):
        from fires.severity import SeverityModel

        base = Path(args.weights).parent if args.weights else Path("weights")
        path = base / weights["severity_model"]
        if path.exists():
            severity_model = SeverityModel.load(path)

    af_pairs, bs_pairs = {}, {}
    for chip_id in dataset.chip_ids():
        chip = dataset.load(chip_id)
        if chip.kind == "af" and chip.mask is not None:
            af_pairs[chip_id] = (chip.mask.astype(bool), af_module.detect_chip(chip, af_params, blob_filter=blob_filter))
        elif chip.kind == "bs" and chip.mask_sev is not None:
            bs_pairs[chip_id] = (chip.mask_sev.astype(np.uint8),
                                 bs_module.classify_chip(chip, bs_params, severity_model=severity_model))

    detection_path, chip_f1 = figure_detection(dataset, af_params, blob_filter, out)
    severity_path, severity_metrics = figure_severity(dataset, bs_params, out, severity_model=severity_model)
    map_path = figure_service_map(Path(args.catalog), out)
    overall = score_from_pairs(af_pairs, bs_pairs)

    n_ablations = len(evaluate_module.ablation_variants(
        af_params, bs_params, blob_filter, severity_model=severity_model))
    print("схема конвейера:", figure_pipeline(out, n_ablations))

    if args.numbers:
        rows = _ablation_rows_from_numbers(Path(args.numbers))
        figure_ablations(rows, out, title="Абляции на реальной обучающей части (tools/collect_numbers.py)")
    elif args.ablation_log:
        rows = _parse_ablation_log(Path(args.ablation_log), overall["score"])
        figure_ablations(rows, out, title="Абляции на реальной обучающей части (evaluate.py --ablation)")
    else:
        rows = [("Итоговая конфигурация", overall["score"])]
        figure_ablations(rows, out, title="Абляции: полный прогон — evaluate.py --ablation")

    print(json.dumps({
        "detection_png": str(detection_path),
        "f1_chip_example": round(chip_f1, 4),
        "severity_png": str(severity_path),
        **{k: round(v, 4) for k, v in severity_metrics.items()},
        "service_map_png": None if map_path is None else str(map_path),
        "score_dataset": overall["score"],
        "f1_af": overall["f1_af"],
        "iou_burn": overall["iou_burn"],
        "miou_sev": overall["miou_sev"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
