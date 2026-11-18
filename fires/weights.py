"""Загрузка и сохранение весов (порогов) решения.

Формат ``weights/weights.json``:

```json
{
  "version": 2,
  "af_mode": "rules",
  "bs_mode": "model",
  "severity_model": "severity_model.pkl",
  "af": {"k_sigma": 3.0, "delta_day": 15.0, ...},
  "bs": {"forest": [0.1, 0.27, 0.66], "min_patch_px": 48, ...},
  "train_metrics": {"score": 0.61, "f1_af": 0.62, "iou_burn": 0.58, "miou_sev": 0.55},
  "notes": "подобрано на обучающей части, seed фиксирован"
}
```

``bs_mode`` выбирает, как решается задача степени поражения: ``"thresholds"`` — порогами dNBR,
``"model"`` — обученной моделью из ``severity_model`` (файл рядом с ``weights.json``). Если файла
модели нет или он несовместим с установленной версией scikit-learn, решение не падает, а
автоматически считает порогами.

Значения по умолчанию лежат в dataclass-параметрах модулей, поэтому при отсутствии
файла решение всё равно работает — это важно для воспроизводимости и скорости.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "weights" / "weights.json"


def load_weights(path: str | Path | None = None) -> dict:
    """Читает веса; при отсутствии файла возвращает пустой словарь (значения по умолчанию)."""

    path = Path(path) if path else DEFAULT_PATH
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data


def save_weights(weights: dict, path: str | Path | None = None) -> Path:
    path = Path(path) if path else DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(weights, fh, ensure_ascii=False, indent=2)
    return path
