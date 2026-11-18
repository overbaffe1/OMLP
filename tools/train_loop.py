#!/usr/bin/env python3
"""Цикл обучения модели степени поражения BS: 15 проходов с отчётом «стало лучше или нет».

Зачем отдельный скрипт. ``train.py`` подбирает всю конфигурацию решения (правила AF, пороги
dNBR, режим BS) и занимает ≈20 минут — гонять его после каждой идеи неудобно. Этот скрипт
работает только с моделью степени поражения: за один проход (цикл) обучает одну конфигурацию
на том же разбиении по событиям пожара, что и ``train.py`` (168 чипов обучение / 56 отложенных),
считает метрики соревнования на отложенной части и печатает разницу с лучшим результатом.

    py -3 tools\\train_loop.py --data-dir %USERPROFILE%\\Documents\\cosmo1\\train          # 15 циклов
    py -3 tools\\train_loop.py --data-dir ... --device cuda --save                        # GPU-обучение
    py -3 tools\\train_loop.py --data-dir ... --device cpu                                # без GPU
    py -3 tools\\train_loop.py --data-dir ... --cycles 4 --quick                          # быстрая проверка

Что печатается по каждому циклу: конфигурация, ``IoU_burn`` и ``mIoU_sev`` на отложенных чипах,
проекция Score соревнования (AF берётся из замера — цикл его не меняет) и вердикт
«лучше / не лучше». В конце — таблица «до и после» на всём обучающем наборе и решение, что
записать в веса: ``weights/weights.json`` + ``weights/severity_model.pkl`` (прежняя модель
сохраняется рядом как ``severity_model.prev.pkl``).

Проекция Score: ``0.35·F1_af + 0.35·IoU_burn + 0.30·mIoU_sev`` — ровно формула кейса. Так как
F1_af не зависит от модели степени поражения, берётся один замер на всём AF-наборе (из
``docs/numbers.json``, если он есть, иначе пересчитывается) и подставляется в обе оценки.

Что перебирается: ёмкость модели, число пикселей на чип, наборы признаков (23 базовых против
40 расширенных), веса классов и веса сложных покрытий (пашня и степь как «трудные» отрицательные
примеры), бэггинг по сидам, ансамбль разных моделей, смещения классов, пространственная
постобработка вероятностей (усреднение по окружению и фильтр большинства) и морфология маски.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fires import bs as bs_module  # noqa: E402
from fires import severity as sev  # noqa: E402
from fires.chips import Dataset, find_data_dir  # noqa: E402
from fires.metrics import score_from_pairs  # noqa: E402
from fires.weights import load_weights, save_weights  # noqa: E402
from train import SEED, SEVERITY_TRAIN_PIXELS, _split_ids, evaluate_af, evaluate_bs  # noqa: E402

#: Конфигурации циклов. Каждый цикл — либо обучение новой модели (``train``), либо шаг,
#: использующий лучшую модель на данный момент: ансамбль (``ensemble``), смещения классов
#: (``bias``), подбор морфологии (``postprocess``), переобучение на всём наборе (``all_data``).
#: Подбор идёт от простого к сложному, как в обычном цикле экспериментов: сначала ёмкость модели,
#: затем признаки, затем постобработка и ансамбли.
CYCLE_SCHEDULE: list[dict] = [
    {"name": "стартовая конфигурация из весов", "mode": "train", "features": "base",
     "params": {"iterations": 60, "learning_rate": 0.20, "leaves": 24, "min_samples_leaf": 80,
                "l2": 1.0}},
    {"name": "больше деревьев (300 при lr 0.08)", "mode": "train", "features": "base",
     "params": {"iterations": 300, "learning_rate": 0.08, "leaves": 24, "min_samples_leaf": 80,
                "l2": 1.0}},
    {"name": "крупные деревья (250 ит., 48 листьев)", "mode": "train", "features": "base", "bias": True,
     "params": {"iterations": 250, "learning_rate": 0.10, "leaves": 48, "min_samples_leaf": 40,
                "l2": 1.0}},
    {"name": "xgboost hist: 400 ит., 64 листа", "mode": "train", "features": "base", "bias": True,
     "backend": "xgboost",
     "params": {"iterations": 400, "learning_rate": 0.06, "leaves": 64, "min_samples_leaf": 80,
                "l2": 1.0, "subsample": 0.8, "colsample": 0.8}},
    {"name": "расширенные признаки (40 против 23)", "mode": "train", "features": "ext", "bias": True,
     "params": {"iterations": 250, "learning_rate": 0.10, "leaves": 48, "min_samples_leaf": 40,
                "l2": 1.0}},
    {"name": "больше пикселей на чип (x2)", "mode": "train", "features": "base", "pixels_scale": 2.0,
     "bias": True,
     "params": {"iterations": 250, "learning_rate": 0.10, "leaves": 48, "min_samples_leaf": 40,
                "l2": 1.0}},
    {"name": "жёсткая регуляризация (l2 3, colsample 0.7)", "mode": "train", "features": "ext",
     "bias": True,
     "params": {"iterations": 300, "learning_rate": 0.06, "leaves": 32, "min_samples_leaf": 120,
                "l2": 3.0, "colsample": 0.7}},
    {"name": "веса классов 1/n вместо 1/sqrt(n)", "mode": "train", "features": "base",
     "weight_mode": "inv", "bias": True,
     "params": {"iterations": 300, "learning_rate": 0.08, "leaves": 24, "min_samples_leaf": 80,
                "l2": 1.0}},
    {"name": "ансамбль трёх лучших моделей", "mode": "ensemble", "top": 3, "bias": True},
    {"name": "ещё больше пикселей на чип (x4)", "mode": "train", "features": "base",
     "pixels_scale": 4.0, "bias": True,
     "params": {"iterations": 250, "learning_rate": 0.10, "leaves": 48, "min_samples_leaf": 40,
                "l2": 1.0}},
    {"name": "веса пикселей: пашня и степь как сложные отрицательные", "mode": "train",
     "features": "base", "weight_mode": "hardneg", "bias": True,
     "params": {"iterations": 250, "learning_rate": 0.10, "leaves": 48, "min_samples_leaf": 40,
                "l2": 1.0, "hard_boost": 1.0}},
    {"name": "бэггинг: лучшая конфигурация на трёх сидах", "mode": "seedbag", "seeds": 3,
     "bias": True},
    {"name": "бинарная голова «гарь / не гарь» поверх лучшей конфигурации", "mode": "gate"},
    {"name": "постобработка вероятностей: сглаживание и фильтр большинства", "mode": "spatial"},
    {"name": "перепроверка морфологии на лучшей модели", "mode": "postprocess"},
    {"name": "тюнинг правил активного горения (F1_af — 35 % Score)", "mode": "af"},
    {"name": "переобучение лучшей конфигурации на всех чипах", "mode": "all_data"},
]

#: Сдвиги, которые перебираются при подборе смещений классов.
BIAS_GRID = (-0.08, -0.05, -0.03, -0.015, 0.0, 0.015, 0.03, 0.05, 0.08)

#: Короткая сетка для дешёвого режима: шаг по осям крупнее, проходов меньше.
BIAS_GRID_CHEAP = (-0.08, -0.03, 0.03, 0.08)

#: Сколько комбинаций постобработки доводится подбором смещений (остальные оцениваются как есть).
SPATIAL_FINALISTS = 3

#: Порог «улучшения» F1 активного горения: меньше этой разницы считаем шумом.
AF_EPS = 3e-4

#: Порог «улучшения»: меньше этой разницы объектива считаем шумом.
IMPROVEMENT_EPS = 2e-4

#: Допуск при проверке выигрыша по половинам отложенной части: на каждой половине улучшение
#: должно быть не хуже, чем на эту величину (иначе выигрыш собран из случайных чипов).
HALF_TOLERANCE = 1.5e-3

#: Что перебираем в шаге «тюнинг правил активного горения». Это операционные пороги
#: детектора (правила, не модель): каждый вариант считается только по уже прочитанным чипам.
AF_TUNE_GRID = {
    "k_sigma": (2.0, 3.0, 4.0),
    "delta_day": (20.0, 25.0, 30.0),
    "candidate_t4": (325.0, 330.0, 335.0),
    "delta_night": (8.0, 10.0, 14.0),
    "min_cluster_px": (1, 2, 3),
}

#: Сколько AF-чипов максимум берётся на тюнинг правил (по половине на поиск и на проверку).
#: Правила — не модель: десятков чипов на половину достаточно, а полный набор удорожал бы шаг.
AF_TUNE_MAX_CHIPS = 120

#: Допуск при котором выигрыш считается «подтверждённым частично»: просадка на одной половине
#: заметна, но невелика. Такие варианты не хуже исходного, однако и не надёжнее его.
PARTIAL_TOLERANCE = 5e-3


class ValCache:
    """Признаки отложенных чипов и эталонные степени: считается один раз на набор признаков.

    Отложенная часть маленькая (56 чипов), но каждая новая модель требует её полного прохода.
    Признаки кэшируются в памяти: на 56 чипов это ≈0.8 ГБ, зато пересчёт 12 циклов не читает
    TIFF-файлы заново.
    """

    def __init__(self, dataset: Dataset, chip_ids: list[str], feature_names):
        self.feature_names = tuple(feature_names)
        self.chips: list[dict] = []
        for chip_id in chip_ids:
            chip = dataset.load(chip_id)
            if chip.mask_sev is None:
                continue
            matrix, ok = sev.chip_features(chip, self.feature_names)
            hot = sev.roi_mask(chip, sev.ROI_CUT, sev.ROI_SIZE).reshape(-1)
            indices = np.flatnonzero(ok & hot)
            self.chips.append({
                "chip_id": chip_id,
                "matrix": matrix[indices],
                "indices": indices,
                "truth": np.asarray(chip.mask_sev, dtype=np.uint8).reshape(-1),
                "shape": np.asarray(chip.mask_sev).shape,
            })

    def __len__(self) -> int:
        return len(self.chips)

    def matrix_for(self, item: dict, model: sev.SeverityModel) -> np.ndarray:
        """Столбцы кэша под набор признаков модели (базовый набор — префикс расширенного)."""

        names = tuple(model.feature_names)
        if names == self.feature_names:
            return item["matrix"]
        if tuple(self.feature_names[:len(names)]) == names:
            return item["matrix"][:, :len(names)]
        raise ValueError(f"кэш собран на наборе {self.feature_names}, а модель ждёт {names}")

    def probabilities(self, model: sev.SeverityModel) -> list[np.ndarray]:
        """Вероятности классов на области интереса каждого чипа — считаются один раз на модель.

        Дальше смещения классов и варианты морфологии перебираются уже по этим числам: полный
        проход по вероятностям занимает доли секунды, тогда как повторный обход чипов моделью —
        десятки секунд.
        """

        out = []
        for item in self.chips:
            if not item["indices"].size:
                out.append(np.zeros((0, 4), dtype=np.float32))
            elif model.kind == "regressor":
                value = np.asarray(estimator_predict(model, self.matrix_for(item, model)),
                                   dtype=np.float32)
                out.append(value)
            else:
                out.append(model.proba_matrix(self.matrix_for(item, model)))
        return out

    def score_from_probabilities(self, values: list[np.ndarray], params: bs_module.BSParams,
                                 bias=None, cuts=(0.5, 1.5, 2.5),
                                 subset: set[int] | None = None,
                                 smooth: int = 0, majority: int = 0,
                                 majority_iters: int = 1, demote: int = 0,
                                 morphology: bool = True) -> dict:
        """Метрики по заранее посчитанным вероятностям (или непрерывной степени для регрессии).

        ``subset`` — позиции чипов, которые входят в замер (по умолчанию — все): на нём держится
        проверка смещений классов на второй половине отложенных чипов.
        """

        if not morphology:
            # Морфология (чистка мелких контуров) — самая дорогая часть замера: на поиске
            # смещений она не нужна, важен лишь порядок вариантов; точный замер — в приёмке.
            params = replace(params, min_patch_px=0, fill_holes_mask=False)
        bias_array = None if bias is None else np.asarray(bias, dtype=np.float32)
        pairs = {}
        for position, (item, value) in enumerate(zip(self.chips, values)):
            if subset is not None and position not in subset:
                continue
            full = np.zeros(int(np.prod(item["shape"])), dtype=np.uint8)
            if item["indices"].size and len(value):
                if value.ndim == 1:  # регрессия: непрерывная степень
                    classes = np.zeros(value.shape, dtype=np.uint8)
                    for cls, cut in zip((1, 2, 3), cuts):
                        classes[value >= cut] = cls
                else:
                    classes = sev.classify_from_proba(
                        np.asarray(value, dtype=np.float32), item["indices"],
                        (int(item["shape"][0]), int(item["shape"][1])), bias=bias_array,
                        smooth=smooth, majority=majority, majority_iters=majority_iters,
                        demote_min_px=demote)
                full[item["indices"]] = classes
            pred = bs_module.postprocess(full.reshape(item["shape"]), params)
            pairs[item["chip_id"]] = (item["truth"].reshape(item["shape"]), pred)
        result = score_from_pairs({}, pairs)
        result["n"] = len(pairs)
        return result

    def evaluate(self, model: sev.SeverityModel, params: bs_module.BSParams,
                 bias=None, chiplimit: int | None = None) -> dict:
        """Метрики BS на отложенной части для готовой модели (со смещениями и постобработкой)."""

        model.class_bias = None if bias is None else tuple(float(v) for v in bias)
        chirps = self.chips[:chiplimit] if chiplimit else self.chips
        if chiplimit:
            saved, self.chips = self.chips, chirps
        try:
            values = self.probabilities(model)
        finally:
            if chiplimit:
                self.chips = saved
        cuts = getattr(model, "cuts", (0.5, 1.5, 2.5))
        extra = dict(
            smooth=int(getattr(model, "proba_smooth", 0) or 0),
            majority=int(getattr(model, "majority_filter", 0) or 0),
            majority_iters=int(getattr(model, "majority_iters", 1) or 1),
            demote=int(getattr(model, "demote_min_px", 0) or 0))
        result = self.score_from_probabilities(values, params, bias=bias, cuts=cuts, **extra)
        # половины отложенной части: проверка, что выигрыш не собран из случайных чипов
        fit, check = verification_split(self)
        halves = []
        for subset in (set(fit), set(check)):
            part = self.score_from_probabilities(values, params, bias=bias, cuts=cuts,
                                                 subset=subset, **extra)
            halves.append(objective(part["iou_burn"], part["miou_sev"]))
        result["half_objectives"] = halves
        return result


def estimator_predict(model: sev.SeverityModel, matrix: np.ndarray) -> np.ndarray:
    """Предсказание непрерывной степени (регрессия) — с учётом ансамбля."""

    items = model.members or [model.estimator]
    total = None
    for estimator in items:
        part = np.asarray(estimator.predict(matrix), dtype=np.float32)
        total = part if total is None else total + part
    return total / len(items)


def objective(iou: float | None, miou: float | None) -> float:
    """Часть Score, которую меняет цикл: ``0.35·IoU_burn + 0.30·mIoU_sev``."""

    if iou is None or miou is None:
        return float("-inf")
    return 0.35 * float(iou) + 0.30 * float(miou)


def projected_score(f1_af: float | None, iou: float, miou: float) -> float:
    """Полная формула кейса с зафиксированным F1 активного горения."""

    f1 = 0.0 if f1_af is None else float(f1_af)
    return 0.35 * f1 + 0.35 * float(iou) + 0.30 * float(miou)


def measure_f1_af(dataset: Dataset, weights: dict, numbers_path: Path,
                  quick: bool = False) -> tuple[float | None, str]:
    """F1 активного горения: из уже посчитанного ``docs/numbers.json`` или замером заново.

    Правила AF цикл не меняет, поэтому значение берётся один раз и подставляется в проекцию Score
    до и после. В быстром режиме (``--quick``) замер идёт по части чипов — это только для отладки.
    """

    if numbers_path.exists() and not quick:
        try:
            data = json.loads(numbers_path.read_text(encoding="utf-8"))
            value = float(data["final"]["f1_af"])
            return value, f"из {numbers_path.name}"
        except Exception:
            pass
    from fires import af as af_module

    af_params = af_module.to_params(weights.get("af", {}))
    blob_filter = None
    if weights.get("af_mode") == "blobs" and weights.get("blob_filter"):
        path = numbers_path.parent.parent / "weights" / weights["blob_filter"]
        if path.exists():
            from fires.blobfilter import BlobFilter

            blob_filter = BlobFilter.load(path)
    ids = dataset.chip_ids("af")
    if quick:
        ids = ids[:40]
    result = evaluate_af(dataset, ids, af_params, blob_filter=blob_filter)
    return (None if result["f1_af"] is None else float(result["f1_af"])), f"замер на {len(ids)} AF-чипах"


def _fmt(value) -> str:
    """Число для печати: None — как «н/д», иначе с четырьмя знаками."""

    return "н/д" if value is None else f"{float(value):.4f}"


def load_blob_filter(weights: dict, numbers_path: Path):
    """Классификатор ложных очагов, если решение работает в режиме ``blobs``."""

    if weights.get("af_mode") != "blobs" or not weights.get("blob_filter"):
        return None
    path = numbers_path.parent.parent / "weights" / weights["blob_filter"]
    if not path.exists():
        return None
    from fires.blobfilter import BlobFilter

    return BlobFilter.load(path)


def af_f1(dataset: Dataset, ids: list[str], params, blob_filter=None,
          verbose: bool = False) -> float | None:
    """F1 активного горения по правилам детектора на перечисленных чипах."""

    result = evaluate_af(dataset, ids, params, blob_filter=blob_filter)
    if verbose and result.get("f1_af") is not None:
        print(f"    AF на {len(ids)} чипах: F1={result['f1_af']:.4f}")
    return None if result.get("f1_af") is None else float(result["f1_af"])


def tune_af(dataset: Dataset, ids: list[str], weights: dict, numbers_path: Path,
            verbose: bool = True) -> tuple[object, dict]:
    """Координатный подъём по порогам детектора активного горения.

    Поиск идёт на одной половине отложенных AF-чипов, приём — только если улучшение видно и
    на второй: правила AF переносятся между наборами хуже, чем кажется по одной выборке.
    """

    from fires import af as af_module

    blob_filter = load_blob_filter(weights, numbers_path)
    if len(ids) > AF_TUNE_MAX_CHIPS:      # равномерная подвыборка: сохраняет состав событий
        step = len(ids) / float(AF_TUNE_MAX_CHIPS)
        ids = [ids[int(position * step)] for position in range(AF_TUNE_MAX_CHIPS)]
    fit_ids, check_ids = [], []
    for position, chip_id in enumerate(ids):
        (fit_ids if position % 2 == 0 else check_ids).append(chip_id)
    base = af_module.to_params(weights.get("af", {}))
    if not fit_ids or not check_ids:
        return base, {"accepted": False, "reason": "мало AF-чипов для проверки"}

    base_fit = af_f1(dataset, fit_ids, base, blob_filter)
    if base_fit is None:
        return base, {"accepted": False, "reason": "на отложенных AF-чипах нет эталона"}
    best_fit, chosen_values = base_fit, {}
    for field, variants in AF_TUNE_GRID.items():
        current = getattr(base, field)
        for value in variants:
            if value == current:
                continue
            candidate = replace(base, **{**chosen_values, field: value})
            value_fit = af_f1(dataset, fit_ids, candidate, blob_filter)
            if value_fit is not None and value_fit > best_fit + AF_EPS:
                best_fit, chosen_values = value_fit, {**chosen_values, field: value}
    if not chosen_values:
        return base, {"accepted": False, "reason": "лучше исходных правил ничего не нашлось",
                      "f1_fit_before": base_fit, "f1_fit_after": base_fit}
    tuned = replace(base, **chosen_values)
    check_before = af_f1(dataset, check_ids, base, blob_filter)
    check_after = af_f1(dataset, check_ids, tuned, blob_filter)
    accepted = (check_before is not None and check_after is not None
                and check_after > check_before + AF_EPS)
    info = {"accepted": accepted, "changed": chosen_values,
            "f1_fit_before": base_fit, "f1_fit_after": best_fit,
            "f1_check_before": check_before, "f1_check_after": check_after}
    if not accepted:
        info["reason"] = (f"на второй половине AF-чипов выигрыш не подтвердился "
                          f"({check_before:.4f} → {check_after:.4f})")
        return base, info
    return tuned, info


def train_cycle(dataset: Dataset, train_ids: list[str], config: dict, pixels: int,
                device: str, seed: int, verbose: bool = True,
                pixel_cache: dict | None = None) -> sev.SeverityModel:
    """Обучает модель одного цикла (``mode`` = ``train`` или ``regressor``)."""

    features = config.get("features", "base")
    backend = config.get("backend", "auto")
    params = dict(config.get("params", {}))
    params.setdefault("kind", "regressor" if config.get("mode") == "regressor" else "classifier")
    # ключ кэша включает число пикселей и состав чипов: иначе цикл «вдвое больше пикселей»
    # молча брал подвыборку предыдущего цикла и повторял его метрику до последнего знака
    key = (features, int(pixels), int(seed), tuple(train_ids))
    if pixel_cache is not None and key in pixel_cache:
        X, y = pixel_cache[key]
    else:
        X, y = sev.collect_training_pixels(dataset, train_ids, pixels, features, seed=seed,
                                           verbose=verbose)
        if pixel_cache is not None:
            pixel_cache[key] = (X, y)
    params.setdefault("gate", bool(config.get("gate")))
    return sev.train_model(X, y, params=params, backend=backend, device=device,
                           feature_names=features,
                           weight_mode=config.get("weight_mode", "sqrt"), seed=seed,
                           verbose=verbose)


def verification_split(cache: ValCache) -> tuple[list[int], list[int]]:
    """Индексы чипов для настройки и проверки смещений: чётные настраивают, нечётные проверяют.

    Смещения классов — четыре числа, подобранные на отложенной части; чтобы не подгоняться под
    шум, они выбираются на половине чипов и принимаются только если помогают и на второй половине.
    """

    fit = [i for i in range(len(cache.chips)) if i % 2 == 0]
    check = [i for i in range(len(cache.chips)) if i % 2 == 1]
    if not fit or not check:
        fit = check = list(range(len(cache.chips)))
    return fit, check


def tune_bias(cache: ValCache, model: sev.SeverityModel, params: bs_module.BSParams,
              base_bias=None, smooth: int = 0, majority: int = 0,
              majority_iters: int = 1, demote: int = 0,
              cheap: bool = False) -> tuple[tuple, dict]:
    """Координатный подъём по смещениям классов: максимум объектива на отложенных чипах.

    Вероятности считаются один раз, поэтому каждый вариант смещений — это только пересчёт
    ``argmax`` и морфология (доли секунды вместо полного прохода моделью).
    """

    values = cache.probabilities(model)
    full = cache.score_from_probabilities(values, params, bias=base_bias, smooth=smooth,
                                          majority=majority, majority_iters=majority_iters,
                                          demote=demote)
    bias = list(base_bias or (0.0, 0.0, 0.0, 0.0))
    fit_ids, check_ids = verification_split(cache)

    fit_set, check_set = set(fit_ids), set(check_ids)

    def subset_score(candidate, positions, fast: bool = False) -> float:
        result = cache.score_from_probabilities(values, params, bias=candidate, subset=positions,
                                                smooth=smooth, majority=majority,
                                                majority_iters=majority_iters, demote=demote,
                                                morphology=not fast)
        return objective(result["iou_burn"], result["miou_sev"])

    # в дешёвом режиме подъём идёт по короткой сетке, за один проход и без морфологии:
    # это поиск, а решение всё равно принимается по точному замеру ниже
    grid = BIAS_GRID_CHEAP if cheap else BIAS_GRID
    best_obj = subset_score(bias, fit_set, fast=cheap)
    for _ in range(1 if cheap else 2):
        for cls in range(4):
            current = bias[cls]
            for delta in grid:
                candidate = list(bias)
                candidate[cls] = current + delta
                value = subset_score(candidate, fit_set, fast=cheap)
                if value > best_obj + 1e-5:
                    best_obj, bias = value, candidate
    # принимаем смещения, только если они улучшают и вторую половину отложенных чипов
    if check_ids != fit_ids:
        base_check = subset_score(list(base_bias or (0.0, 0.0, 0.0, 0.0)), check_set)
        tuned_check = subset_score(bias, check_set)
        if tuned_check <= base_check + 1e-5:
            bias = list(base_bias or (0.0, 0.0, 0.0, 0.0))
    result_full = cache.score_from_probabilities(values, params, bias=bias, smooth=smooth,
                                                 majority=majority, majority_iters=majority_iters,
                                                 demote=demote)
    halves = []
    for subset in (fit_set, check_set):
        part = cache.score_from_probabilities(values, params, bias=bias, subset=subset,
                                              smooth=smooth, majority=majority,
                                              majority_iters=majority_iters, demote=demote)
        halves.append(objective(part["iou_burn"], part["miou_sev"]))
    result_full["half_objectives"] = halves
    result_full["bias_verified"] = bool(any(abs(v) > 1e-9 for v in bias))
    result_full["bias_full_objective"] = objective(full["iou_burn"], full["miou_sev"])
    return tuple(round(v, 4) for v in bias), result_full


def tunable_cuts(cache: ValCache, model: sev.SeverityModel, params: bs_module.BSParams) -> dict:
    """Для регрессии — подбор порогов перевода непрерывной степени в классы 1/2/3."""

    best = None
    values = cache.probabilities(model)
    variants = [tuple(model.cuts), (0.4, 1.4, 2.4), (0.6, 1.6, 2.6), (0.5, 1.6, 2.6),
                (0.55, 1.45, 2.45), (0.45, 1.5, 2.55)]
    for cuts in variants:
        result = cache.score_from_probabilities(values, params, cuts=cuts)
        value = objective(result["iou_burn"], result["miou_sev"])
        if best is None or value > best["objective"]:
            best = {"objective": value, "cuts": cuts, "iou_burn": result["iou_burn"],
                    "miou_sev": result["miou_sev"]}
    model.cuts = best["cuts"]
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Цикл обучения модели степени поражения BS")
    ap.add_argument("--data-dir", required=True, help="каталог обучающей части набора")
    ap.add_argument("--out", default=str(ROOT / "weights" / "weights.json"))
    ap.add_argument("--cycles", type=int, default=len(CYCLE_SCHEDULE), help="сколько циклов выполнить")
    # (--no-spatial убирает один шаг, поэтому число шагов может быть меньше len(CYCLE_SCHEDULE))
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                    help="auto — GPU (xgboost), если он есть, иначе CPU")
    ap.add_argument("--pixels", type=int, default=SEVERITY_TRAIN_PIXELS,
                    help="сколько пикселей брать с чипа при обучении")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--f1-af", type=float, default=None, help="зафиксировать F1 активного горения")
    ap.add_argument("--val-chips", type=int, default=0, help="ограничить отложенную часть (отладка)")
    ap.add_argument("--quick", action="store_true",
                    help="быстрый прогон для проверки работоспособности (меньше пикселей и чипов, "
                         "но все 12 циклов на месте; регрессия степени в расписании не участвует — "
                         "она проверена и оказалась хуже классификации)")
    ap.add_argument("--strict", action="store_true",
                    help="принимать только те улучшения, что подтверждаются обеими половинами "
                         "отложенной части (защита от подгонки под валидацию)")
    ap.add_argument("--gpu-report", action="store_true",
                    help="показать, что известно про GPU, и выйти (диагностика CUDA)")
    ap.add_argument("--report", default=None,
                    help="куда записать отчёт о цикле (по умолчанию docs/severity_loop.json)")
    ap.add_argument("--spatial", action=argparse.BooleanOptionalAction, default=True,
                    help="шаг пространственной постобработки вероятностей (сглаживание и фильтр "
                         "большинства); --no-spatial пропускает его целиком")
    ap.add_argument("--af-tune", action=argparse.BooleanOptionalAction, default=True,
                    help="подбирать пороги детектора активного горения (по умолчанию да; "
                         "--no-af-tune выключает, тогда F1_af берётся как есть)")
    ap.add_argument("--save", action="store_true",
                    help="записать лучшую модель в weights (по умолчанию только отчёт)")
    args = ap.parse_args(argv)

    if args.gpu_report:
        info = sev.gpu_report()
        info["устройство"] = sev.resolve_device(args.device)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        print("\nЕсли runtime_ok = false: py -3 -m pip install --upgrade xgboost "
              "nvidia-cuda-runtime-cu12\n"
              "  проверка драйвера: nvidia-smi (должен показать RTX 3060 и версию драйвера)")
        return 0

    if args.quick:
        # быстрый режим: меньше пикселей и чипов отложенной части, но все циклы на месте —
        # так проверяются и ансамбль, и смещения, и морфология, и переобучение на всём наборе
        args.pixels = min(args.pixels, 2000)
        args.val_chips = args.val_chips or 16

    device = sev.resolve_device(args.device)
    backend_note = "xgboost (GPU)" if device == "cuda" else "sklearn/xgboost (CPU)"
    print(f"=== Цикл обучения модели степени поражения: {args.cycles} циклов, устройство {device} "
          f"({backend_note}) ===")
    if args.device == "cuda" and device != "cuda":
        print(f"    GPU недоступен: {sev.gpu_report()['reason']}")

    # каталог набора часто указывают на уровень выше или ниже фактического (архивы
    # распаковываются во вложенную папку) — ищем чипы рядом, но молча не «догадываемся»
    resolved = find_data_dir(args.data_dir, prefer="masks")
    if resolved != Path(args.data_dir):
        print(f"    в {args.data_dir} чипов нет; использую найденный рядом каталог {resolved}")
    dataset = Dataset(resolved)
    all_ids = dataset.chip_ids()
    if not all_ids:
        print(f"\nв каталоге {args.data_dir} не нашлось ни одного чипа — путь или раскладка не те.\n"
              f"    что видит решение: py -3 -m fires.chips --data-dir \"{args.data_dir}\"\n"
              f"    нужен каталог, внутри которого папки af и bs с файлами чипов и meta.csv "
              f"(обучающая часть)")
        return 1
    print(f"    чипов в каталоге: {len(all_ids)} "
          f"(af {len(dataset.chip_ids('af'))}, bs {len(dataset.chip_ids('bs'))})")
    bs_train, bs_val = _split_ids(dataset, "bs", args.val_frac, args.seed)
    # AF-чипы с эталоном ищем по составу файлов: читать все чипы ради проверки маски дорого
    af_ids = [chip_id for chip_id in dataset.chip_ids("af")
              if any("mask" in path.stem.lower() for path in dataset.files.get(chip_id, ()))] \
        if args.af_tune else []
    if args.val_chips:
        bs_val = bs_val[:args.val_chips]
    # --- диагностика утечки в точку отсчёта: веса собирались на каноническом сплите
    # (val_frac=0.25, SEED) — см. раздел split в weights.json. Сплиты с одним сидом вложенные,
    # поэтому при другом --val-frac часть текущего val входила в обучение baseline-модели,
    # её оценка оптимистична, а сравнение «было → стало» смещено в пользу весов.
    leaked_val: list[str] = []
    if args.seed == SEED and abs(args.val_frac - 0.25) > 1e-9:
        canon_train, _canon_val = _split_ids(dataset, "bs", 0.25, SEED)
        canon_set = set(canon_train)
        leaked_val = [c for c in bs_val if c in canon_set]
        if leaked_val:
            print(f"    ВНИМАНИЕ: {len(leaked_val)} из {len(bs_val)} отложенных чипов входят в обучение "
                  f"канонического сплита (val_frac=0.25) — точка отсчёта из весов на них оптимистична, "
                  f"а сравнение «было → стало» смещено в пользу весов")
    if not bs_train:
        print(f"\nBS-чипов для обучения не нашлось (в каталоге {resolved} их "
              f"{len(dataset.chip_ids('bs'))}) — проверьте, что это обучающая часть с масками")
        return 1
    weights = load_weights(args.out)
    bs_params = bs_module.to_params(weights.get("bs", {}))
    print(f"    чипов: обучение {len(bs_train)}, отложено {len(bs_val)}; "
          f"морфология {bs_params.min_patch_px} px, дыры "
          f"{'заполняются' if bs_params.fill_holes_mask else 'нет'}")

    numbers_path = ROOT / "docs" / "numbers.json"
    f1_af, f1_source = (args.f1_af, "аргумент --f1-af") if args.f1_af is not None else \
        measure_f1_af(dataset, weights, numbers_path, quick=args.quick or bool(args.val_chips))
    print(f"    F1 активного горения: {f1_af if f1_af is None else round(f1_af, 5)} ({f1_source}); "
          + ("в цикле есть шаг тюнинга правил AF" if af_ids
             else "тюнинг AF выключен (--no-af-tune)"))

    t_start = time.perf_counter()
    print("\n=== Считаю признаки отложенной части (один раз) ===")
    t0 = time.perf_counter()
    val_cache = ValCache(dataset, bs_val, sev.FEATURE_NAMES_EXT)
    print(f"    готово: {len(val_cache)} чипов за {time.perf_counter() - t0:.1f} с")
    if not val_cache:
        print(f"\nв каталоге {resolved} нет BS-чипов с маской степени поражения — обучать не на чем.\n"
              f"    маска нужна, чтобы считать метрику: без неё это тестовая часть, а не обучающая")
        return 1

    # --- точка отсчёта: модель из весов (то, что залито на платформу)
    base_model = None
    model_name = weights.get("severity_model")
    if model_name and weights.get("bs_mode") == "model":
        path = Path(args.out).parent / model_name
        if path.exists():
            try:
                base_model = sev.SeverityModel.load(path)
            except Exception as exc:
                print(f"    [warn] модель из весов не загрузилась ({exc})")
    print("\n=== Точка отсчёта (модель из весов) ===")
    baseline = None
    if base_model is not None:
        result = val_cache.evaluate(base_model, bs_params)
        baseline = {"iou_burn": result["iou_burn"], "miou_sev": result["miou_sev"],
                    "objective": objective(result["iou_burn"], result["miou_sev"]),
                    "halves": result.get("half_objectives")}
        print(f"    IoU_burn={baseline['iou_burn']:.4f}  mIoU_sev={baseline['miou_sev']:.4f}  "
              f"проекция Score={projected_score(f1_af, baseline['iou_burn'], baseline['miou_sev']):.5f}")
    else:
        print("    модель в весах не найдена — сравнение только между циклами")

    pixel_cache: dict = {}
    results: list[dict] = []
    af_history: list[dict] = []
    best: dict | None = None
    best_train: dict | None = None  # лучшая конфигурация обучения (без ансамблей и морфологии)
    final: dict | None = None       # переобучение на всех чипах — модель для сабмита
    trained_models: list[tuple[float, sev.SeverityModel, dict]] = []

    schedule = [config for config in CYCLE_SCHEDULE
                if args.spatial or config.get("mode") != "spatial"]
    for index, config in enumerate(schedule[:args.cycles], start=1):
        name = config["name"]
        mode = config.get("mode", "train")
        t0 = time.perf_counter()
        note = ""
        model: sev.SeverityModel | None = None
        bias = None
        params = bs_params
        # текущая принятая постобработка вероятностей: с ней оцениваются все последующие шаги,
        # иначе сравнение шагов было бы неоднородным (сглаживание меняет метрику заметно)
        spatial = ((int(best.get("smooth", 0) or 0), int(best.get("majority", 0) or 0),
                    int(best.get("majority_iters", 1) or 1), int(best.get("demote", 0) or 0))
                   if best else (0, 0, 1, 0))

        if mode == "af":
            # Правила активного горения цикл раньше вообще не трогал, а F1_af — это 35 % Score.
            # Пороги подбираются на одной половине AF-чипов, вторая служит проверкой.
            if not af_ids:
                print(f"\nцикл {index:2d}/{args.cycles}: {name} — пропуск (нет AF-чипов с эталоном)")
                continue
            from fires import af as af_module

            blob_filter = load_blob_filter(weights, numbers_path)
            base_params = af_module.to_params(weights.get("af", {}))
            f1_all_before = af_f1(dataset, af_ids, base_params, blob_filter)
            tuned, af_info = tune_af(dataset, af_ids, weights, numbers_path,
                                     verbose=not args.quick)
            f1_all_after = af_f1(dataset, af_ids, tuned, blob_filter) \
                if af_info.get("accepted") else f1_all_before
            iou, miou = best["iou_burn"], best["miou_sev"]
            value = objective(iou, miou)
            score_before = projected_score(f1_all_before, iou, miou)
            improved = (f1_all_after is not None and f1_all_before is not None
                        and f1_all_after > f1_all_before + AF_EPS)
            if improved:
                weights["af"] = asdict(tuned)
                f1_af = f1_all_after
            score = projected_score(f1_af, iou, miou)
            elapsed = time.perf_counter() - t0
            changed = af_info.get("changed") or {}
            note = ("; ".join(f"{field} → {value_!r}" for field, value_ in changed.items())
                    if changed else str(af_info.get("reason", "правила не изменились")))
            print(f"\nцикл {index:2d}/{args.cycles}: {name}")
            print(f"    F1 активного горения на всех {len(af_ids)} AF-чипах: "
                  f"{_fmt(f1_all_before)} → {_fmt(f1_all_after)}; "
                  f"проекция Score {score_before:.5f} → {score:.5f}  [{elapsed:.0f} с]")
            print(f"    {'ПРИНЯТО — новые пороги уходят в weights.json' if improved else 'ОТКЛОНЕНО'}"
                  f"{'; ' + note if note else ''}")
            entry = {"cycle": index, "name": name, "mode": mode, "iou_burn": iou, "miou_sev": miou,
                     "objective": value, "projected_score": score, "seconds": round(elapsed, 1),
                     "improved": improved, "note": note, "device": "правила AF",
                     "halves": [af_info.get("f1_fit_before"), af_info.get("f1_fit_after")],
                     "stable": bool(improved)}
            results.append(entry)
            af_history.append(entry)
            continue

        if mode in ("train", "regressor"):
            pixels = int(args.pixels * float(config.get("pixels_scale", 1.0)))
            model = train_cycle(dataset, bs_train, config, pixels, device, args.seed,
                                pixel_cache=pixel_cache)
            if mode == "regressor":
                cuts = tunable_cuts(val_cache, model, params)
                note = f"пороги классов {tuple(round(c, 2) for c in cuts['cuts'])}"
            if config.get("bias"):
                bias, result = tune_bias(val_cache, model, params, smooth=spatial[0],
                                         majority=spatial[1], majority_iters=spatial[2],
                                         demote=spatial[3])
                if any(abs(v) > 1e-9 for v in bias):
                    note = (note + "; " if note else "") + "смещения " + ", ".join(
                        f"{v:+.3f}" for v in bias)
            if not note or not config.get("bias"):
                model.proba_smooth, model.majority_filter = spatial[0], spatial[1]
                model.majority_iters = spatial[2]
                model.demote_min_px = spatial[3]
                result = val_cache.evaluate(model, params, bias=bias)
        elif mode == "ensemble":
            # в ансамбль берутся модели с тем же набором признаков, что у лучшей: расширенный
            # набор столбцов несравним с базовым, а усреднять вероятности разных моделей можно
            top_k = int(config.get("top", 3))
            ranked = sorted(trained_models, key=lambda item: item[0], reverse=True)
            compatible = [item for item in ranked
                          if tuple(item[1].feature_names) == tuple(ranked[0][1].feature_names)
                          and item[1].kind == ranked[0][1].kind]
            top = compatible[:top_k]
            if not top:
                print(f"\nцикл {index:2d}/{args.cycles}: {name} — пропуск (нет обученных моделей)")
                continue
            model = sev.ensemble_model([item[1] for item in top],
                                       metrics={"members": [item[2]["name"] for item in top]})
            note = " + ".join(item[2]["name"] for item in top)
            if config.get("bias"):
                bias, result = tune_bias(val_cache, model, params, smooth=spatial[0],
                                         majority=spatial[1], majority_iters=spatial[2],
                                         demote=spatial[3])
                note += "; смещения " + ", ".join(f"{v:+.3f}" for v in bias)
            else:
                model.proba_smooth, model.majority_filter = spatial[0], spatial[1]
                model.majority_iters = spatial[2]
                model.demote_min_px = spatial[3]
                result = val_cache.evaluate(model, params)
        elif mode == "seedbag":
            if best is None:
                print(f"\nцикл {index:2d}/{args.cycles}: {name} — пропуск (нет базовой модели)")
                continue
            source = best_train or best
            config_best = source["config"]
            if config_best.get("mode") != "train":
                config_best = next((c for c in CYCLE_SCHEDULE if c.get("mode") == "train"), config_best)
            seeds = [args.seed + 1000 * k for k in range(int(config.get("seeds", 3)))]
            models = [train_cycle(dataset, bs_train, config_best,
                                  int(args.pixels * float(config_best.get("pixels_scale", 1.0))),
                                  device, seed_k, pixel_cache=pixel_cache) for seed_k in seeds]
            model = sev.ensemble_model(models, metrics={"members": [f"сид {v}" for v in seeds],
                                                        "base": config_best.get("name")})
            note = f"{config_best.get('name')} × {len(seeds)} сида"
            if config.get("bias"):
                bias, result = tune_bias(val_cache, model, params, smooth=spatial[0],
                                         majority=spatial[1], majority_iters=spatial[2],
                                         demote=spatial[3])
                note += "; смещения " + ", ".join(f"{v:+.3f}" for v in bias)
            else:
                result = val_cache.evaluate(model, params)
        elif mode == "gate":
            # Бинарная голова «гарь / не гарь» поверх лучшей обучаемой конфигурации: тот же набор
            # признаков и та же подвыборка пикселей (кэш), поэтому сравнение честное.
            source = best_train or best
            if source is None:
                print(f"\nцикл {index:2d}/{args.cycles}: {name} — пропуск (нет базовой модели)")
                continue
            config_best = source["config"]
            if config_best.get("mode") not in ("train", "regressor"):
                config_best = next((c for c in CYCLE_SCHEDULE if c.get("mode") == "train"), config_best)
            config = {**config_best, "mode": "train", "gate": True, "name": name}
            pixels = int(args.pixels * float(config.get("pixels_scale", 1.0)))
            model = train_cycle(dataset, bs_train, config, pixels, device, args.seed,
                                pixel_cache=pixel_cache)
            model.proba_smooth, model.majority_filter = spatial[0], spatial[1]
            model.majority_iters, model.demote_min_px = spatial[2], spatial[3]
            bias, result = tune_bias(val_cache, model, params, smooth=spatial[0],
                                     majority=spatial[1], majority_iters=spatial[2],
                                     demote=spatial[3])
            note = f"поверх «{source['name']}»; голова «гарь / не гарь» на тех же пикселях"
            if any(abs(v) > 1e-9 for v in bias):
                note += "; смещения " + ", ".join(f"{v:+.3f}" for v in bias)
        elif mode == "spatial":
            if best is None:
                print(f"\nцикл {index:2d}/{args.cycles}: {name} — пропуск (нет базовой модели)")
                continue
            model = best["model"]
            values = val_cache.probabilities(model)
            base_bias = best.get("bias")
            combos = [(0, 0, 1, 0), (1, 0, 1, 0), (2, 0, 1, 0), (1, 1, 1, 0), (2, 1, 1, 0),
                      (1, 2, 1, 0), (2, 2, 1, 0), (1, 1, 1, 16), (2, 1, 1, 16), (2, 2, 1, 24)]
            # Ступень 1: все комбинации с текущими смещениями — точный, но недорогой замер.
            ranked = []
            for combo in combos:
                res = val_cache.score_from_probabilities(
                    values, params, bias=base_bias, smooth=combo[0], majority=combo[1],
                    majority_iters=combo[2], demote=combo[3])
                ranked.append((objective(res["iou_burn"], res["miou_sev"]), combo))
            ranked.sort(key=lambda item: item[0], reverse=True)
            # Ступень 2: смещения заново подбираются только финалистам (сглаживание меняет
            # масштаб вероятностей, поэтому прежние смещения для них не оптимальны)
            chosen = None
            for _, combo in ranked[:SPATIAL_FINALISTS]:
                smooth, majority, iters, demote = combo
                cand_bias, res = tune_bias(val_cache, model, params, base_bias=base_bias,
                                           smooth=smooth, majority=majority,
                                           majority_iters=iters, demote=demote, cheap=True)
                value = objective(res["iou_burn"], res["miou_sev"])
                if chosen is None or value > chosen[0]:
                    chosen = (value, smooth, majority, iters, res, cand_bias, demote)
            _, smooth, majority, iters, result, bias, demote = chosen
            spatial = (smooth, majority, iters, demote)
            parts = []
            if smooth:
                parts.append(f"усреднение вероятностей {2 * smooth + 1}×{2 * smooth + 1}")
            if majority:
                parts.append(f"фильтр большинства {2 * majority + 1}×{2 * majority + 1}")
            if demote:
                parts.append(f"мелкие пятна класса 3 < {demote} px понижаются")
            note = "; ".join(parts) if parts else "пространственная постобработка не помогла"
        elif mode == "bias":
            if best is None:
                print(f"\nцикл {index:2d}/{args.cycles}: {name} — пропуск (нет базовой модели)")
                continue
            model = best["model"]
            bias, result = tune_bias(val_cache, model, params, base_bias=model.class_bias,
                                     smooth=spatial[0], majority=spatial[1],
                                     majority_iters=spatial[2], demote=spatial[3])
            note = "смещения " + ", ".join(f"{v:+.3f}" for v in bias)
        elif mode == "postprocess":
            if best is None:
                print(f"\nцикл {index:2d}/{args.cycles}: {name} — пропуск (нет базовой модели)")
                continue
            model = best["model"]
            variants = [replace(params, min_patch_px=24), replace(params, min_patch_px=96),
                        replace(params, min_patch_px=48, fill_holes_mask=True)]
            variants = [params] + variants
            values = val_cache.probabilities(model)
            chosen = None
            for candidate in variants:
                res = val_cache.score_from_probabilities(
                    values, candidate, bias=model.class_bias,
                    smooth=spatial[0], majority=spatial[1], majority_iters=spatial[2],
                    demote=spatial[3])
                value = objective(res["iou_burn"], res["miou_sev"])
                if chosen is None or value > chosen[0]:
                    chosen = (value, candidate, res)
            _, params, result = chosen
            note = (f"чистка контуров < {params.min_patch_px} px, дыры "
                    f"{'заполняются' if params.fill_holes_mask else 'нет'}")
        else:  # all_data — переобучение лучшей конфигурации на всех чипах
            if best is None:
                print(f"\nцикл {index:2d}/{args.cycles}: {name} — пропуск (нет базовой модели)")
                continue
            source = best_train or best
            config_best = source["config"]
            if config_best.get("mode") != "train":
                config_best = next((c for c in CYCLE_SCHEDULE if c.get("mode") == "train"), config_best)
            all_ids = bs_train + bs_val
            model = train_cycle(dataset, all_ids, config_best,
                                int(args.pixels * float(config_best.get("pixels_scale", 1.0))),
                                device, args.seed, pixel_cache=pixel_cache)
            result = val_cache.evaluate(model, params, bias=best.get("bias"))
            note = "обучение на всех чипах: часть оценки оптимистична"

        iou, miou = result["iou_burn"], result["miou_sev"]
        value = objective(iou, miou)
        score = projected_score(f1_af, iou, miou)
        elapsed = time.perf_counter() - t0

        optimistic = mode == "all_data"
        previous = None if optimistic else (best["objective"] if best else
                                            (baseline["objective"] if baseline else None))
        verdict = "модель для сабмита (на отложенной части оценка оптимистична)" if optimistic \
            else "первый замер"
        improved = False
        halves = result.get("half_objectives")
        stable = None
        if halves is not None:
            # улучшение должно быть видно на обеих половинах отложенных чипов: иначе это
            # выигрыш на случайных чипах, который на платформе легко превращается в проигрыш
            base_halves = (best or baseline or {}).get("halves")
            if base_halves:
                diffs = [h - b for h, b in zip(halves, base_halves)]
                if all(d > -HALF_TOLERANCE for d in diffs):
                    stable = True
                elif all(d > -PARTIAL_TOLERANCE for d in diffs):
                    # не хуже исходного варианта, но и не явно лучше: брать такой риск не стоит
                    stable = "partial"
        if previous is not None:
            delta = value - previous
            improved = delta > IMPROVEMENT_EPS
            if index == 1 and abs(delta) < IMPROVEMENT_EPS:
                verdict = "исходная конфигурация — совпадает с точкой отсчёта"
            else:
                mark = ""
                if improved and stable is True:
                    mark = " — выигрыш подтверждён на обеих половинах"
                elif improved and stable == "partial":
                    mark = " — подтверждён частично (на одной половине просадка)"
                elif improved and stable is False:
                    mark = " — НЕ подтверждён на половинах, к загрузке не рекомендуется"
                verdict = f"{'ЛУЧШЕ' if improved else 'не лучше'} ({delta:+.4f} объектива){mark}"
        if args.strict and improved and stable is not True:
            improved = False
            verdict += " [--strict: отклонено, выигрыш неустойчив]"
        print(f"\nцикл {index:2d}/{args.cycles}: {name}")
        print(f"    IoU_burn={iou:.4f}  mIoU_sev={miou:.4f}  проекция Score={score:.5f}  "
              f"[{elapsed:.0f} с]  {verdict}")
        if note:
            print(f"    {note}")
        if halves:
            base_halves = (best or baseline or {}).get("halves")
            tail = ""
            if base_halves:
                diffs = ", ".join(f"{h - b:+.4f}" for h, b in zip(halves, base_halves))
                tail = f"  к точке отсчёта: {diffs} (половины отложенных чипов)"
            print(f"    половины отложенного набора: {halves[0]:.4f} / {halves[1]:.4f}{tail}")

        entry = {"cycle": index, "name": name, "mode": mode, "iou_burn": iou, "miou_sev": miou,
                 "objective": value, "projected_score": score, "seconds": round(elapsed, 1),
                 "improved": improved, "note": note, "device": device,
                 "halves": halves, "stable": stable}
        results.append(entry)
        if mode in ("train", "regressor", "gate") and model is not None:
            trained_models.append((value, model, entry))

        record = {"objective": value, "model": model, "iou_burn": iou, "miou_sev": miou,
                  "score": score, "cycle": index, "name": name, "config": config,
                  "bias": bias, "params": params, "entry": entry,
                  "smooth": spatial[0], "majority": spatial[1], "majority_iters": spatial[2],
                  "demote": spatial[3], "halves": halves, "stable": stable}
        if optimistic:
            # модель для сабмита: к ней применяется лучшая найденная постобработка вероятностей
            final = record
            model.proba_smooth = int(best.get("smooth", 0) or 0)
            model.majority_filter = int(best.get("majority", 0) or 0)
            model.majority_iters = int(best.get("majority_iters", 1) or 1)
            model.demote_min_px = int(best.get("demote", 0) or 0)
        elif best is None or improved:
            if bias is None and mode == "postprocess":
                bias = best.get("bias") if best else None
            if mode == "spatial" and model is not None:
                model.proba_smooth = int(spatial[0])
                model.majority_filter = int(spatial[1])
                model.majority_iters = int(spatial[2])
                model.demote_min_px = int(spatial[3])
            best = record
            if mode == "train":
                best_train = record

    # --- итог: сравнение «до и после» на отложенной части и на всём обучающем наборе
    print("\n=== Итог цикла ===")
    if best is None:
        print("ни один цикл не отработал")
        return 1
    submit = final or best          # на сабмит идёт модель, обученная на всех чипах, если она есть
    # Если ни один шаг не подтвердил улучшения, на сабмит должна идти прежняя модель из весов: иначе
    # --strict мог бы «сохранить» конфигурацию первого шага, которая хуже исходной.
    baseline_wins = (baseline is not None and base_model is not None
                     and best["objective"] <= baseline["objective"] + IMPROVEMENT_EPS)
    best_model: sev.SeverityModel = best["model"]
    report = {"device": device, "f1_af": f1_af, "f1_source": f1_source,
              "split": {"seed": args.seed, "val_frac": args.val_frac,
                        "train_ids": list(bs_train), "val_ids": list(bs_val),
                        "canonical_overlap_ids": list(leaked_val)},
              "af_tune": af_history,
              "baseline": baseline, "best": {"cycle": best["cycle"], "name": best["name"],
                                             "iou_burn": best["iou_burn"], "miou_sev": best["miou_sev"],
                                             "projected_score": best["score"]},
              "baseline_wins": bool(baseline_wins), "cycles": results}
    if final is not None:
        report["final"] = {"cycle": final["cycle"], "name": final["name"],
                           "iou_burn": final["iou_burn"], "miou_sev": final["miou_sev"],
                           "projected_score": final["score"],
                           "note": "переобучение на всех чипах: оценка на отложенной части оптимистична"}
    if baseline is not None:
        base_projection = projected_score(f1_af, baseline["iou_burn"], baseline["miou_sev"])
        if leaked_val:
            honesty = (f"ВНИМАНИЕ: {len(leaked_val)} из {len(bs_val)} чипов val точка отсчёта "
                       f"видела в обучении — сравнение смещено в пользу весов")
        else:
            honesty = "честное сравнение, эти чипы модель не видела"
        print(f"отложенные {len(val_cache)} чипов ({honesty}): "
              f"было IoU={baseline['iou_burn']:.4f} mIoU={baseline['miou_sev']:.4f} "
              f"→ стало IoU={best['iou_burn']:.4f} mIoU={best['miou_sev']:.4f}")
        print(f"проекция Score на отложенной части: {base_projection:.5f} → {best['score']:.5f} "
              f"({best['score'] - base_projection:+.5f}) — это ожидаемый прирост на платформе")
    if final is not None and not baseline_wins:
        print(f"модель для сабмита (обучена на всех {len(bs_train) + len(bs_val)} BS-чипах): "
              f"IoU={final['iou_burn']:.4f} mIoU={final['miou_sev']:.4f} "
              f"проекция Score={final['score']:.5f} (оценка оптимистична: эти чипы она видела)")
    elif baseline_wins and baseline is not None:
        print(f"на сабмит остаётся прежняя модель из весов: IoU={baseline['iou_burn']:.4f} "
              f"mIoU={baseline['miou_sev']:.4f} проекция Score={base_projection:.5f}")

    if not args.quick:
        print("\n=== Контроль на всём обучающем наборе (224 BS-чипа) ===")
        if baseline is not None and base_model is not None:
            full_base = evaluate_bs(dataset, dataset.chip_ids("bs"), bs_params, severity_model=base_model)
            base_score = projected_score(f1_af, full_base["iou_burn"], full_base["miou_sev"])
            print(f"    было (модель из весов): IoU_burn={full_base['iou_burn']:.4f} "
                  f"mIoU_sev={full_base['miou_sev']:.4f}  Score≈{base_score:.5f}")
            report["full_train_before"] = {"iou_burn": full_base["iou_burn"],
                                           "miou_sev": full_base["miou_sev"], "score": base_score}
        full_best = evaluate_bs(dataset, dataset.chip_ids("bs"), submit.get("params", bs_params),
                                severity_model=submit["model"])
        best_score = projected_score(f1_af, full_best["iou_burn"], full_best["miou_sev"])
        print(f"    стало (модель для сабмита): IoU_burn={full_best['iou_burn']:.4f} "
              f"mIoU_sev={full_best['miou_sev']:.4f}  Score≈{best_score:.5f}")
        report["full_train_after"] = {"iou_burn": full_best["iou_burn"],
                                      "miou_sev": full_best["miou_sev"], "score": best_score}

    print("\n=== Циклы построчно ===")
    for entry in results:
        mark = "+" if entry["improved"] else " "
        tail = ""
        if entry["improved"] and entry.get("stable") is False:
            tail = "  ← выигрыш не подтверждён на половинах"
        print(f" {mark} {entry['cycle']:2d}. {entry['name'][:52]:52s} Score≈{entry['projected_score']:.5f} "
              f"({entry['seconds']:.0f} с){tail}")

    if best is not None and best.get("stable") is False:
        print("\nВНИМАНИЕ: лучший по отложенной части вариант не подтверждён обеими половинами чипов.\n"
              "    На 56 чипах разница до ≈0.01 Score — это шум: такой вариант может на платформе\n"
              "    оказаться хуже. Безопаснее запустить с --strict и взять то, что подтвердится.")
    elif best is not None and len(val_cache) < 40:
        print(f"\n    отложенных чипов мало ({len(val_cache)}): разница с точкой отсчёта сравнима\n"
              f"    с шумом выборки, к выбору лучшего варианта стоит относиться осторожно")

    if baseline_wins:
        print("\nНИ ОДИН ШАГ НЕ ПОДТВЕРДИЛ УЛУЧШЕНИЯ по сравнению с моделью из весов.\n"
              "    Это нормальный исход: значит текущая модель уже лучше всего, что нашёл перебор.\n"
              "    На сабмит идёт прежняя модель, файлы в weights не тронуты — пересобирать ничего "
              "не нужно.")
        if leaked_val:
            print(f"    НО: сравнение смещено — {len(leaked_val)} чипов val входят в обучение весов, "
                  f"поэтому вердикт «веса лучше» может быть артефактом утечки. Для честного отбора "
                  f"перезапустите цикл на каноническом сплите (без --val-frac 0.5).")

    # смещения классов должны попасть в файл модели: иначе инференс их не применит
    best_model.class_bias = None if best.get("bias") is None else tuple(best["bias"])
    if args.save and not baseline_wins:
        out_path = Path(args.out)
        prev = out_path.with_name("severity_model.prev.pkl")
        current = out_path.with_name(weights.get("severity_model") or "severity_model.pkl")
        if current.exists():
            current.replace(prev)
            print(f"\nпрежняя модель сохранена: {prev}")
        submit_model: sev.SeverityModel = submit["model"]
        submit_model.class_bias = None if submit.get("bias") is None else tuple(submit["bias"])
        submit_model.proba_smooth = int(best.get("smooth", 0) or 0)
        submit_model.majority_filter = int(best.get("majority", 0) or 0)
        submit_model.majority_iters = int(best.get("majority_iters", 1) or 1)
        submit_model.demote_min_px = int(best.get("demote", 0) or 0)
        saved = submit_model.save(current)
        weights["bs_mode"] = "model"
        weights["severity_model"] = saved.name
        chosen_params = submit.get("params", bs_params)
        weights.setdefault("bs", {})["min_patch_px"] = chosen_params.min_patch_px
        weights.setdefault("bs", {})["fill_holes_mask"] = chosen_params.fill_holes_mask
        weights.setdefault("bs", {})["proba_smooth"] = submit_model.proba_smooth
        weights.setdefault("bs", {})["majority_filter"] = submit_model.majority_filter
        weights.setdefault("bs", {})["demote_min_px"] = submit_model.demote_min_px
        weights.setdefault("train_metrics", {})["bs_severity_model"] = submit_model.metrics
        weights["train_metrics"]["severity_loop"] = {
            "cycle": submit["cycle"], "name": submit["name"], "device": device,
            "iou_burn": submit["iou_burn"], "miou_sev": submit["miou_sev"],
            "projected_score": submit["score"], "smooth": submit_model.proba_smooth,
            "majority": submit_model.majority_filter,
            "demote": submit_model.demote_min_px,
        }
        # прежние веса сохраняем целиком: в них морфология, режим и правила AF, и без копии
        # «вернуть прежнюю конфигурацию» не получится (модель вернём, а пороги уже другие)
        if out_path.exists():
            backup = out_path.with_name(out_path.stem + ".prev.json")
            backup.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"прежние веса сохранены: {backup}")
        # отдельно кладём модель, обученную только на обучающей части: на всех 224 чипах её можно
        # сравнивать с прежней моделью «на равных» (обе учились на одних и тех же 168 чипах)
        best_train_file = out_path.with_name("severity_model.best_train.pkl")
        best_model.save(best_train_file)
        print(f"модель лучшего шага (обучена на {len(bs_train)} чипах) сохранена: {best_train_file}")

        save_weights(weights, out_path)
        print(f"модель записана: {saved}")
        print(f"веса обновлены: {out_path}")
        report["saved"] = str(saved)

    report_path = Path(args.report) if args.report else ROOT / "docs" / "severity_loop.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"отчёт о цикле: {report_path}")
    print(f"всего времени: {time.perf_counter() - t_start:.0f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
