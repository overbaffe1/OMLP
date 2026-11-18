"""Классификатор очагов: отсев техногенных термоаномалий и блика по признакам блоба.

Зачем отдельный модуль. Контекстный алгоритм выдаёт **кандидатов** — пиксели с аномалией
в канале I4. Часть кандидатов — газовые факелы, печи и промплощадки, которые физически
горят и по одному снимку от пожара не отличаются (это прямо указано в постановке: их
выделяет только повторяемость по многолетнему архиву). Геопривязка в тестовой части
скрыта, поэтому историю наблюдений использовать нельзя — остаётся различать источники
по морфологии и сопутствующим слоям:

* факел — компактный, часто насыщенный по I4, без «шлейфа» слабых пикселей, на застройке;
* пожар — пятно с плавным краем (градиент от ядра к периферии), есть умеренные ΔT вокруг ядра,
  чаще на растительности (лес, травостой, пашня, пойма).

Признаки считаются по каждому связному компоненту кандидатов, метка берётся из
эталонной маски обучающей части (IoU блоба с эталоном), а решение принимает
градиентный бустинг. Такой подход объясним (важности признаков), быстр (десятки блобов
на чип) и не требует нейросетей — важно для критерия «скорость работы».
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fires.af import AFParams
from fires.morphology import binary_dilation, closing, label

#: Коды WorldCover, по которым считаются доли площадей блоба.
LC_TREE, LC_SHRUB, LC_GRASS, LC_CROP, LC_BUILT, LC_BARE, LC_WATER, LC_WETLAND = 10, 20, 30, 40, 50, 60, 80, 90

FEATURE_NAMES = [
    "area_px",                  # площадь блоба
    "bbox_fill",                # заполненность рамки (компактность)
    "max_i4",                   # максимум яркостной температуры в I4
    "mean_i4",
    "p90_i4",
    "max_delta",                # максимум I4 − I5 (сила сигнала)
    "mean_delta",
    "p90_delta",
    "delta_spread",             # разброс ΔT внутри блоба: у пожара шире, у факела узкий
    "frac_saturated",           # доля пикселей на потолке канала I4
    "mean_i1",                  # видимый канал: блик на воде/кровлях
    "max_i1",
    "mean_i3",                  # SWIR: блик и открытая почва
    "max_i3",
    "glint_frac",               # доля пикселей в зоне блика
    "frac_built",               # доля застройки (промплощадки)
    "frac_tree",
    "frac_grass",
    "frac_crop",
    "frac_bare",
    "frac_water",
    "edge_ratio",               # доля периметра: рыхлый край против плотного ядра
    "neighbour_weak",           # сколько слабых (не кандидатов) «тёплых» пикселей вокруг
]


@dataclass
class BlobFilterConfig:
    """Настройки фильтра очагов."""

    threshold: float = 0.5        # порог вероятности «природный пожар»
    min_area: int = 1             # блобы меньше — сразу шум
    #: Пиксельные пороги внутри выбранных блобов (второй стадия отбора)
    pixel_t4: float = 315.0
    pixel_delta: float = 6.0
    dilate_weak: int = 2          # радиус поиска слабых «тёплых» пикселей вокруг блоба
    weak_margin: float = 3.0      # насколько ΔT должен превышать порог, чтобы считаться слабым теплом
    #: Добавлять ли к кандидатам результат правил AF (помимо мягких порогов).
    #: Отключение оставляет только мягкие кандидаты — используется в абляциях.
    use_soft_candidates: bool = True


def candidate_mask(chip, params: AFParams, cfg: BlobFilterConfig | None = None) -> np.ndarray:
    """Кандидаты для классификатора очагов — с заведомо мягкими порогами.

    Логика двухстадийная: сначала берём всё, что хоть немного похоже на горение
    (мягкий порог по I4 и по разности I4 − I5), склеиваем соседние пиксели в блобы,
    а затем решает классификатор — он и отделяет природный пожар от факела, блика
    и шума. Мягкие пороги нужны, чтобы очаг не распадался на фрагменты.
    """

    cfg = cfg or BlobFilterConfig()
    layers = chip.layers
    i4 = np.asarray(layers["I4"], dtype=np.float32)
    i5 = np.asarray(layers["I5"], dtype=np.float32)
    # тип покрова может содержать NaN за границей валидной зоны — приводим безопасно
    landcover = np.nan_to_num(np.asarray(layers.get("landcover", np.full(i4.shape, 30)), dtype=np.float32), nan=0.0).astype(
        np.uint8
    )
    valid = np.asarray(layers.get("valid", np.ones(i4.shape)), dtype=bool)
    delta = i4 - i5
    # В реальных данных I4/I5 содержат NaN (нет наблюдения), а I1–I3 — NaN на ночных
    # пролётах: пиксели без термальных каналов в анализ не берём.
    valid = valid & np.isfinite(i4) & np.isfinite(i5)
    soft = valid & (landcover != LC_WATER) & (
        ((i4 >= params.candidate_t4) & (delta >= params.candidate_delta)) | (i4 >= params.absolute_t4)
    )
    if cfg.use_soft_candidates:
        # К мягким кандидатам добавляется результат работы правил: параметры контекстного теста,
        # отсева застройки и блика влияют на состав блобов, а не только на пороги одного пикселя.
        from fires import af as af_module

        soft |= af_module.detect(
            i4=i4,
            i5=i5,
            i1=layers.get("I1"),
            i3=layers.get("I3"),
            landcover=landcover,
            valid=valid,
            sun_zenith=layers.get("sun_zenith"),
            params=params,
        )
    # склейка соседних фрагментов: морфологическое закрытие (площадь не растёт)
    return closing(soft, 1)


def _blob_features(chip, blob: np.ndarray, params: AFParams, cfg: BlobFilterConfig) -> list[float]:
    """Считает вектор признаков по бинарной маске одного блоба."""

    layers = chip.layers
    i4 = np.asarray(layers["I4"], dtype=np.float32)
    i5 = np.asarray(layers["I5"], dtype=np.float32)
    i1 = np.asarray(layers.get("I1", np.zeros_like(i4)), dtype=np.float32)
    i3 = np.asarray(layers.get("I3", np.zeros_like(i4)), dtype=np.float32)
    # тип покрова может содержать NaN за границей валидной зоны — приводим безопасно
    landcover = np.nan_to_num(np.asarray(layers.get("landcover", np.full(i4.shape, 30)), dtype=np.float32), nan=0.0).astype(
        np.uint8
    )
    valid = np.asarray(layers.get("valid", np.ones_like(i4)), dtype=bool)

    delta = i4 - i5

    def safe(values: np.ndarray, fn, default: float = 0.0) -> float:
        """Статистика по конечным значениям: NaN в реальных данных — обычное дело."""

        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(fn(values)) if values.size else default

    # сам блоб считается только по пикселям с корректными термальными каналами
    idx = blob & np.isfinite(i4) & np.isfinite(i5)
    n = int(idx.sum())
    if n == 0:
        idx = blob
        n = int(idx.sum())
    ys, xs = np.nonzero(idx)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

    # периметр: пиксели блоба, у которых есть внешний сосед
    ring = binary_dilation(idx, 1) & ~idx
    edge = idx & binary_dilation(ring, 1)

    # «слабый» тёплый ореол вокруг ядра — отличает пожар с градиентом от компактного факела
    around = binary_dilation(idx, cfg.dilate_weak) & ~idx
    weak_margin = cfg.weak_margin
    threshold_delta = params.delta_night if params.delta_night < params.delta_day else params.delta_day
    weak = around & (delta > threshold_delta - weak_margin) & valid

    lc = landcover[idx]
    def frac(code: int) -> float:
        return float((lc == code).mean()) if n else 0.0

    return [
        float(n),
        float(n / max(1, (y1 - y0) * (x1 - x0))),
        safe(i4[idx], np.max),
        safe(i4[idx], np.mean),
        safe(i4[idx], lambda v: np.percentile(v, 90)),
        safe(delta[idx], np.max),
        safe(delta[idx], np.mean),
        safe(delta[idx], lambda v: np.percentile(v, 90)),
        safe(delta[idx], np.std),
        float(np.mean(np.nan_to_num(i4[idx], nan=-1e3) >= params.sat_t4)),
        safe(i1[idx], np.mean),
        safe(i1[idx], np.max),
        safe(i3[idx], np.mean),
        safe(i3[idx], np.max),
        safe(np.where((i1[idx] >= params.glint_i1) & (i3[idx] >= params.glint_i3), 1.0, np.nan), np.mean),
        frac(LC_BUILT),
        frac(LC_TREE) + frac(LC_SHRUB),
        frac(LC_GRASS),
        frac(LC_CROP),
        frac(LC_BARE),
        frac(LC_WATER),
        float(edge.sum() / max(1, n)),
        float(weak.sum()),
    ]


def extract_features(chip, params: AFParams, cfg: BlobFilterConfig | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Возвращает матрицу признаков блобов и список их масок."""

    cfg = cfg or BlobFilterConfig()
    cand = candidate_mask(chip, params, cfg)
    labels, count = label(cand, connectivity=8)
    feats: list[list[float]] = []
    masks: list[np.ndarray] = []
    if count == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), masks

    for blob_id in range(1, count + 1):
        blob = labels == blob_id
        if blob.sum() < cfg.min_area:
            continue
        feats.append(_blob_features(chip, blob, params, cfg))
        masks.append(blob)
    return np.asarray(feats, dtype=np.float32), masks


def label_blobs(chip, masks: list[np.ndarray], overlap_positive: float = 0.5) -> np.ndarray:
    """Метки блобов по эталонной маске: 1 — природный пожар, 0 — техногенный/блик/шум.

    Положительным считается блоб, у которого не менее ``overlap_positive`` его площади
    попадает в эталонную маску: блоб целиком «про пожар», а не случайно задевший эталон.
    """

    if chip.mask is None:
        raise ValueError(f"{chip.chip_id}: нет эталонной маски")
    truth = np.asarray(chip.mask).astype(bool)
    labels = []
    for blob in masks:
        area = float(blob.sum())
        inter = float(np.logical_and(blob, truth).sum())
        share = inter / area if area else 0.0
        labels.append(1 if share >= overlap_positive else 0)
    return np.asarray(labels, dtype=np.int32)


@dataclass
class BlobFilter:
    """Обёртка над классификатором блобов с сохранением в файл."""

    model: object | None = None
    threshold: float = 0.5
    config: BlobFilterConfig = field(default_factory=BlobFilterConfig)
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    metrics: dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 20260918, **kwargs) -> "BlobFilter":
        """Обучает классификатор блобов.

        Балансировка классов делается через ``sample_weight``, а не
        ``class_weight='balanced'``: в актуальных версиях scikit-learn параметр
        ``class_weight`` у HistGradientBoostingClassifier приводит к вырожденной модели
        (вероятности 0.5 на всех объектах).
        """

        from sklearn.ensemble import HistGradientBoostingClassifier

        weight_mode = kwargs.pop("class_weight", "balanced")
        # При малой обучающей выборке (типичный случай для кейса) стандартный
        # min_samples_leaf=20 не даёт дереву расщепиться, и модель вырождается
        # в константу; поэтому лист ограничивается размером выборки.
        default_leaf = 20 if len(X) >= 400 else max(2, len(X) // 20)
        model = HistGradientBoostingClassifier(
            max_depth=kwargs.pop("max_depth", 6),
            max_iter=kwargs.pop("max_iter", 200),
            learning_rate=kwargs.pop("learning_rate", 0.08),
            l2_regularization=kwargs.pop("l2_regularization", 1.0),
            min_samples_leaf=kwargs.pop("min_samples_leaf", default_leaf),
            random_state=seed,
            **kwargs,
        )
        y = np.asarray(y).astype(int)
        weights = None
        if weight_mode == "balanced" and len(y):
            positives = max(int(y.sum()), 1)
            negatives = max(int(len(y) - y.sum()), 1)
            weights = np.where(y == 1, len(y) / (2.0 * positives), len(y) / (2.0 * negatives)).astype(np.float32)
        model.fit(X, y, sample_weight=weights)
        self.model = model
        self.metrics.update(
            {
                "n_train": int(len(y)),
                "positives": int(y.sum()),
                "negatives": int(len(y) - y.sum()),
            }
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("Фильтр не обучен: вызовите fit() или загрузите веса")
        if len(X) == 0:
            return np.zeros(0, dtype=np.float32)
        return self.model.predict_proba(X)[:, 1]

    def apply(self, chip, params: AFParams) -> np.ndarray:
        """Возвращает итоговую маску горения после фильтрации блобов."""

        feats, masks = extract_features(chip, params, self.config)
        fire = np.zeros(chip.shape, dtype=bool)
        if not masks:
            return fire
        if not self.ready:
            for blob in masks:
                fire |= blob
            return fire
        proba = self.predict_proba(feats)
        layers = chip.layers
        i4 = np.asarray(layers["I4"], dtype=np.float32)
        i5 = np.asarray(layers["I5"], dtype=np.float32)
        pixel_ok = (i4 >= self.config.pixel_t4) & ((i4 - i5) >= self.config.pixel_delta)
        for blob, p in zip(masks, proba):
            if p >= self.threshold:
                fire |= blob & pixel_ok
        return fire

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            with path.open("wb") as fh:
                pickle.dump(self.model, fh)
        meta = {
            "threshold": self.threshold,
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "model_file": path.name,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path | None) -> "BlobFilter":
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            return cls()
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        model = None
        with path.open("rb") as fh:
            model = pickle.load(fh)
        return cls(
            model=model,
            threshold=float(meta.get("threshold", 0.5)),
            config=BlobFilterConfig(**{k: v for k, v in meta.get("config", {}).items() if k in BlobFilterConfig().__dict__}),
            feature_names=meta.get("feature_names", list(FEATURE_NAMES)),
            metrics=meta.get("metrics", {}),
        )
