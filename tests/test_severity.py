"""Тесты обучаемой модели степени поражения (``fires.severity``)."""

from __future__ import annotations

import numpy as np
import pytest

from fires import bs as bs_module
from fires.chips import Chip
from fires.severity import FEATURE_NAMES, SeverityModel, chip_features, roi_mask
from fires.synthetic import generate_bs_chip


def bs_chip(rng, size: int = 96) -> Chip:
    """Собирает BS-чип так же, как его вернул бы ``Dataset.load`` (pre/post разделены)."""

    data = generate_bs_chip(rng, size)
    severity = data.pop("target_sev")
    return Chip(
        chip_id="bs_sev",
        kind="bs",
        layers={"SCL": data["scl"], "landcover": data["landcover"], "valid": data["valid"]},
        pre=dict(data["pre"]),
        post=dict(data["post"]),
        s1_pre=dict(data["s1_pre"]),
        s1_post=dict(data["s1_post"]),
        meta={},
        mask_sev=severity.astype(np.uint8),
    )


def tiny_model(chip: Chip) -> SeverityModel:
    """Обучает крошечный бустинг на признаках одного синтетического чипа."""

    from sklearn.ensemble import HistGradientBoostingClassifier

    matrix, ok = chip_features(chip)
    truth = np.asarray(chip.mask_sev, dtype=np.uint8).reshape(-1)
    indices = np.flatnonzero(ok)
    estimator = HistGradientBoostingClassifier(max_iter=5, max_leaf_nodes=4, random_state=0,
                                               early_stopping=False)
    estimator.fit(matrix[indices], truth[indices])
    return SeverityModel(estimator=estimator)


def test_chip_features_shape_and_finiteness():
    chip = bs_chip(np.random.default_rng(1))
    matrix, ok = chip_features(chip)
    height, width = np.asarray(chip.pre["B8A"]).shape
    assert matrix.shape == (height * width, len(FEATURE_NAMES))
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all(), "в признаках не должно быть NaN/inf"
    assert ok.shape == (height * width,)
    assert ok.any()


def test_feature_names_match_matrix():
    chip = bs_chip(np.random.default_rng(2), size=64)
    matrix, _ = chip_features(chip)
    assert len(FEATURE_NAMES) == matrix.shape[1] == 23


def test_roi_mask_includes_burned_pixels():
    chip = bs_chip(np.random.default_rng(3), size=128)
    mask = roi_mask(chip)
    assert mask.shape == np.asarray(chip.pre["B8A"]).shape
    burned = np.asarray(chip.mask_sev) > 0
    # гейт может не покрыть единичные пиксели на краю пятна, но основную часть обязан
    assert (burned & mask).sum() / max(burned.sum(), 1) > 0.9


def test_model_predicts_valid_classes_and_roundtrips(tmp_path):
    chip = bs_chip(np.random.default_rng(4))
    model = tiny_model(chip)
    severity = model.predict_chip(chip)
    assert severity.shape == np.asarray(chip.pre["B8A"]).shape
    assert severity.dtype == np.uint8
    assert set(np.unique(severity)) <= {0, 1, 2, 3}

    path = model.save(tmp_path / "severity_model.pkl")
    restored = SeverityModel.load(path)
    assert np.array_equal(restored.predict_chip(chip), severity)


def test_regressor_model_maps_continuous_to_classes():
    class ConstantRegressor:
        def __init__(self, value):
            self.value = value

        def predict(self, matrix):
            return np.full(matrix.shape[0], self.value)

    chip = bs_chip(np.random.default_rng(5), size=64)
    shapes = (chip.pre["B8A"]).shape
    _, ok = chip_features(chip)
    # модель вызывается только внутри области интереса и только по валидным пикселям
    gate = roi_mask(chip).reshape(-1) & ok
    for value, expected in ((0.1, 0), (0.9, 1), (1.9, 2), (2.9, 3)):
        model = SeverityModel(estimator=ConstantRegressor(value), kind="regressor")
        severity = model.predict_chip(chip).reshape(-1)
        assert severity.shape == gate.shape
        # вне области интереса модель не вызывается, там ноль; внутри — класс по порогам регрессии
        assert set(np.unique(severity[gate])) == {expected}, value
        assert (severity[~gate] == 0).all(), value


def test_classify_chip_uses_model_and_keeps_postprocess():
    chip = bs_chip(np.random.default_rng(6))
    model = tiny_model(chip)
    with_model = bs_module.classify_chip(chip, bs_module.BSParams(), severity_model=model)
    truth = np.asarray(chip.mask_sev, dtype=np.uint8)
    assert with_model.shape == truth.shape
    # постобработка применяется и к модели: дыр внутри контура не остаётся
    from fires.morphology import fill_holes

    burned = with_model > 0
    assert (fill_holes(burned) & ~burned).sum() == 0
    noisy = bs_module.classify_chip(chip, bs_module.BSParams(), severity_model=model, return_details=True)
    assert noisy[1]["mode"] == "model"


def test_postprocess_is_shared_between_modes():
    severity = np.zeros((32, 32), dtype=np.uint8)
    severity[8:24, 8:24] = 2
    severity[14:18, 14:18] = 0  # дыра в контуре
    out = bs_module.postprocess(severity, bs_module.BSParams(min_patch_px=6, fill_holes_mask=True))
    assert out[15, 15] == 1, "дыра внутри контура заполняется слабой степенью"
    speck = np.zeros((32, 32), dtype=np.uint8)
    speck[3, 3] = 3
    speck[20:28, 20:28] = 1
    out = bs_module.postprocess(speck, bs_module.BSParams(min_patch_px=6, fill_holes_mask=True))
    assert out[3, 3] == 0, "мелкие контуры удаляются в обоих режимах"


def test_inference_skips_model_when_mode_is_thresholds(tmp_path):
    from inference import _load_solution

    weights = tmp_path / "weights.json"
    weights.write_text('{"bs_mode": "thresholds", "severity_model": "severity_model.pkl"}', encoding="utf-8")
    _, _, _, severity_model = _load_solution(weights)
    assert severity_model is None


def test_inference_loads_model_when_mode_is_model(tmp_path):
    from inference import _load_solution

    chip = bs_chip(np.random.default_rng(7), size=64)
    model = tiny_model(chip)
    model.save(tmp_path / "severity_model.pkl")
    weights = tmp_path / "weights.json"
    weights.write_text('{"bs_mode": "model", "severity_model": "severity_model.pkl"}', encoding="utf-8")
    _, _, _, severity_model = _load_solution(weights)
    assert isinstance(severity_model, SeverityModel)


def test_inference_falls_back_when_model_file_is_broken(tmp_path, capsys):
    """Битый (или собранный другой версией sklearn) файл модели не должен ронять инференс."""

    from inference import _load_solution

    (tmp_path / "severity_model.pkl").write_bytes(b"not a pickle")
    weights = tmp_path / "weights.json"
    weights.write_text('{"bs_mode": "model", "severity_model": "severity_model.pkl"}', encoding="utf-8")
    _, _, _, severity_model = _load_solution(weights)
    assert severity_model is None
    assert "не загрузилась" in capsys.readouterr().err


@pytest.mark.parametrize("cut", [0.05, 0.2])
def test_roi_cut_is_monotone(cut):
    """Строгий порог ROI не шире мягкого — иначе гейт терял бы пиксели."""
    chip = bs_chip(np.random.default_rng(8), size=96)
    assert (roi_mask(chip, cut=0.2) & ~roi_mask(chip, cut=cut)).sum() == 0


# ---------------------------------------------------------------- устройства и наборы признаков

def test_extended_feature_set_contains_base_as_prefix():
    """Расширенный набор — надстройка: базовые столбцы идут первыми и совпадают по значениям.

    На этом свойстве держится кэш отложенных чипов в цикле обучения: он собран на расширенном
    наборе, а модели на базовом берут из него первые 23 столбца.
    """

    from fires.severity import FEATURE_NAMES_BASE, FEATURE_NAMES_EXT

    chip = bs_chip(np.random.default_rng(11))
    base, ok_base = chip_features(chip, "base")
    ext, ok_ext = chip_features(chip, "ext")
    assert tuple(FEATURE_NAMES_EXT[:len(FEATURE_NAMES_BASE)]) == tuple(FEATURE_NAMES_BASE)
    assert base.shape[1] == 23 and ext.shape[1] == 40
    assert np.array_equal(base, ext[:, :23])
    assert np.array_equal(ok_base, ok_ext)
    assert np.isfinite(ext).all()


def test_feature_set_names_are_unique_and_documented():
    from fires.severity import FEATURE_SETS

    assert set(FEATURE_SETS) == {"base", "ext"}
    for name, names in FEATURE_SETS.items():
        assert len(names) == len(set(names)), f"в наборе {name} продублированы признаки"


def test_resolve_device_handles_all_choices():
    from fires.severity import gpu_report, resolve_device

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}
    # если GPU нет, «cuda» обязана откатиться на CPU, а не упасть
    assert resolve_device("cuda") == ("cuda" if gpu_report()["runtime_ok"] else "cpu")
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_gpu_report_has_reason_when_gpu_unavailable():
    from fires.severity import gpu_report

    report = gpu_report()
    assert set(report) >= {"xgboost", "build_use_cuda", "runtime_ok", "reason"}
    if not report["runtime_ok"]:
        assert report["reason"], "при недоступном GPU должна быть причина"


def test_pick_backend_matches_device():
    from fires.severity import pick_backend

    assert pick_backend("cpu") == "sklearn"
    assert pick_backend("cpu", "xgboost") == "xgboost"
    assert pick_backend("cuda") == "xgboost"  # GPU умеет только xgboost
    with pytest.raises(ValueError):
        pick_backend("cpu", "catboost")


# ---------------------------------------------------------------- бэкенды обучения

def test_train_model_on_synthetic_chip_and_describe():
    from fires.severity import collect_training_pixels, train_model

    chip = bs_chip(np.random.default_rng(12), size=64)
    dataset = _SingleChipDataset(chip)
    X, y = collect_training_pixels(dataset, [chip.chip_id], 0, "ext")
    model = train_model(X, y, params={"iterations": 5, "leaves": 8, "min_samples_leaf": 5},
                        backend="sklearn", device="cpu", feature_names="ext", verbose=False)
    assert model.backend == "sklearn" and model.device == "cpu"
    assert len(model.feature_names) == 40
    assert "sklearn/cpu" in model.describe()
    prediction = model.predict_chip(chip)
    assert set(np.unique(prediction)) <= {0, 1, 2, 3}


def test_class_weights_modes_differ_and_normalize():
    from fires.severity import class_weights

    y = np.array([0] * 100 + [1] * 10 + [2] * 3 + [3])
    for mode in ("sqrt", "inv", "none"):
        weight = class_weights(y, mode)
        assert weight.shape == y.shape
        assert abs(weight.sum() - len(y)) < 1e-6
    assert class_weights(y, "inv").max() > class_weights(y, "sqrt").max()


def test_class_bias_changes_classes_without_retraining():
    chip = bs_chip(np.random.default_rng(13), size=96)
    model = tiny_model(chip)
    plain = model.predict_chip(chip)
    model.class_bias = (-5.0, 1.0, 1.0, 1.0)
    shifted = model.predict_chip(chip)
    assert plain.shape == shifted.shape
    assert (shifted > 0).sum() >= (plain > 0).sum(), "смещение должно подтягивать ненулевые классы"


def test_ensemble_averages_probabilities():
    from fires.severity import ensemble_model

    chip = bs_chip(np.random.default_rng(14), size=96)
    model = tiny_model(chip)
    ensemble = ensemble_model([model, model])
    assert ensemble.describe().endswith("моделей 2")
    plain = model.predict_chip(chip)
    both = ensemble.predict_chip(chip)
    assert np.array_equal(plain, both), "усреднение одинаковых моделей ничего не меняет"


def test_load_fills_new_fields_for_old_pickles(tmp_path):
    """Модель, сохранённая до появления устройств и наборов признаков, читается без изменений."""

    chip = bs_chip(np.random.default_rng(15), size=64)
    model = tiny_model(chip)
    for name in ("feature_names", "backend", "device", "class_bias", "members"):
        delattr(model, name) if name in ("feature_names", "backend", "device") else setattr(model, name, None)
    path = tmp_path / "old.pkl"
    model.save(path)
    loaded = SeverityModel.load(path)
    assert loaded.feature_names == FEATURE_NAMES
    assert loaded.backend == "sklearn" and loaded.device == "cpu"
    assert loaded.predict_chip(chip).shape == np.asarray(chip.pre["B8A"]).shape


def test_smoothing_and_majority_remove_single_pixel_outliers():
    """Одиночный пиксель другого класса — «соль»: усреднение вероятностей и голосование её убирают."""

    from fires.severity import classify_from_proba, majority_classes, smooth_probabilities

    shape = (48, 48)
    indices = np.arange(shape[0] * shape[1])
    proba = np.zeros((indices.size, 4), dtype=np.float32)
    proba[:, 1] = 1.0
    proba[1000] = [0.0, 0.0, 0.0, 1.0]  # одиночный пиксель класса 3 в однородном классе 1

    assert int(classify_from_proba(proba, indices, shape)[1000]) == 3
    assert int(classify_from_proba(proba, indices, shape, smooth=1)[1000]) == 1
    assert int(classify_from_proba(proba, indices, shape, majority=1)[1000]) == 1
    smoothed = smooth_probabilities(proba, indices, shape, 1)
    assert abs(float(smoothed[1000].sum()) - 1.0) < 1e-5, "вероятности остаются нормированными"
    voted = majority_classes(np.full(indices.size, 2, dtype=np.uint8), indices, shape, 1)
    assert set(np.unique(voted)) == {2}, "однородная карта не меняется"


def test_pixels_outside_roi_do_not_bleed_into_smoothing():
    """Пиксели вне области интереса не участвуют в усреднении: край ROI не тянется к фону."""

    from fires.severity import classify_from_proba

    shape = (16, 16)
    indices = np.array([0, 1, 16, 17], dtype=int)  # квадратик 2×2 в углу
    proba = np.zeros((indices.size, 4), dtype=np.float32)
    proba[:, 1] = 1.0
    out = classify_from_proba(proba, indices, shape, smooth=2)
    assert set(np.unique(out)) == {1}


def test_small_high_severity_patches_are_demoted():
    """Пятно класса 3 из нескольких пикселей не бывает «сильным выгоранием» — класс понижается."""

    from fires.severity import demote_small_patches

    shape = (32, 32)
    indices = np.arange(shape[0] * shape[1])
    classes = np.full(indices.size, 1, dtype=np.uint8)
    classes[100] = 3            # одиночный пиксель
    classes[200:203] = 3        # три пикселя
    classes[300:320] = 3        # двадцать — настоящее пятно
    classes[400:406] = 2        # класс 2 не трогаем
    out = demote_small_patches(classes, indices, shape, min_px=8)
    assert int(out[100]) == 2 and int(out[200]) == 2
    assert int(out[300]) == 3, "крупное пятно сохраняет класс"
    assert int(out[400]) == 2, "понижаются только указанные классы (по умолчанию 3)"
    assert int(out[0]) == 1


def test_model_carries_spatial_postprocessing_through_save(tmp_path):
    """Радиусы сглаживания и большинства живут в файле модели — инференс применит их сам."""

    chip = bs_chip(np.random.default_rng(16), size=96)
    model = tiny_model(chip)
    model.proba_smooth = 2
    model.majority_filter = 1
    model.majority_iters = 2
    model.demote_min_px = 12
    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = SeverityModel.load(path)
    assert (loaded.proba_smooth, loaded.majority_filter, loaded.majority_iters) == (2, 1, 2)
    assert loaded.demote_min_px == 12
    assert "сглаживание вероятностей 5×5" in loaded.describe()
    assert loaded.predict_chip(chip).shape == np.asarray(chip.pre["B8A"]).shape


def test_hardneg_weights_favour_confusable_landcover():
    """Веса «трудных» отрицательных примеров: пашня и степь получают больший вес, сумма сохраняется."""

    from fires.severity import CONFUSABLE_LANDCOVER, class_weights

    y = np.array([0] * 80 + [1] * 10 + [2] * 6 + [3] * 4)
    X = np.zeros((len(y), 23), dtype=np.float32)
    X[:, 13] = 10.0            # лес: не путается с гарью
    X[:40, 13] = CONFUSABLE_LANDCOVER[1]   # пашня
    plain = class_weights(y, "hardneg", X=X, hard_boost=0.0, landcover_column=13)
    boosted = class_weights(y, "hardneg", X=X, hard_boost=1.0, landcover_column=13)
    assert abs(float(boosted.sum()) - len(y)) < 1e-6, "сумма весов сохраняется"
    assert boosted[0] > plain[0], "пашня весит больше"
    assert float(boosted[0] / boosted[60]) > float(plain[0] / plain[60]), \
        "пашня поднимается относительно леса внутри того же класса"
    assert abs(float(boosted[96] / boosted[97]) - float(plain[96] / plain[97])) < 1e-6, \
        "внутри класса гари соотношение весов не меняется (изменился только общий масштаб)"


def test_seed_bagging_ensemble_of_three_models():
    """Бэггинг по сидам — это тот же ансамбль: описание и форма ответа сохраняются."""

    from fires.severity import ensemble_model

    chip = bs_chip(np.random.default_rng(17), size=96)
    models = [tiny_model(chip) for _ in range(3)]
    bag = ensemble_model(models, metrics={"members": ["сид 1", "сид 2", "сид 3"]})
    assert "моделей 3" in bag.describe()
    assert bag.metrics["ensemble_size"] == 3
    assert bag.predict_chip(chip).shape == np.asarray(chip.pre["B8A"]).shape


class _SingleChipDataset:
    """Минимальный «датасет» из одного чипа — для тестов сбора пикселей."""

    def __init__(self, chip: Chip):
        self.chip = chip

    def load(self, chip_id: str) -> Chip:
        assert chip_id == self.chip.chip_id
        return self.chip


def test_tune_af_accepts_only_confirmed_rule(monkeypatch, tiny_dataset):
    """Пороги активного горения принимаются только если выигрыш виден на второй половине чипов."""

    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root / "tools") not in sys.path:
        sys.path.insert(0, str(root / "tools"))
    import train_loop as tl

    from fires.chips import Dataset

    dataset = Dataset(tiny_dataset)
    ids = dataset.chip_ids("af") or dataset.chip_ids()
    assert len(ids) >= 4
    fit = [chip_id for position, chip_id in enumerate(ids) if position % 2 == 0]
    check = [chip_id for position, chip_id in enumerate(ids) if position % 2 == 1]
    tuned_value = tl.AF_TUNE_GRID["k_sigma"][-1]  # 4.0

    def fake_af_f1(_dataset, chip_ids, params, blob_filter=None, verbose=False):
        # послушный детектор: новое правило лучше на половине поиска, но проверочная половина
        # выигрыша не подтверждает — принимать нельзя
        if params.k_sigma == tuned_value:
            return 0.95 if list(chip_ids) == fit else 0.80
        return 0.90

    monkeypatch.setattr(tl, "af_f1", fake_af_f1)
    tuned, info = tl.tune_af(dataset, ids, {"af": {}}, Path("docs/numbers.json"), verbose=False)
    assert info["accepted"] is False
    assert tuned.k_sigma != tuned_value          # остались исходные правила
    assert "не подтвердился" in info["reason"]

    # а если выигрыш подтверждается на обеих половинах — правило принимается
    def good_af_f1(_dataset, chip_ids, params, blob_filter=None, verbose=False):
        return 0.95 if params.k_sigma == tuned_value else 0.90

    monkeypatch.setattr(tl, "af_f1", good_af_f1)
    tuned, info = tl.tune_af(dataset, ids, {"af": {}}, Path("docs/numbers.json"), verbose=False)
    assert info["accepted"] is True
    assert tuned.k_sigma == tuned_value
    assert info["f1_fit_after"] > info["f1_fit_before"]
    assert tuned.delta_day == 25.0               # остальные пороги не тронуты


def test_burn_gate_renormalises_probabilities():
    """Бинарная голова пересобирает вероятности: класс 0 — «не гарь», классы 1–3 делят P(гарь)."""

    from fires.severity import _apply_burn_gate

    proba = np.array([[0.6, 0.2, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    burn = np.array([0.25, 0.9], dtype=np.float32)
    out = _apply_burn_gate(proba, burn)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(out[:, 0], [0.75, 0.1], atol=1e-5)
    # внутри гари соотношение классов сохраняется
    inside = proba[0, 1:] / proba[0, 1:].sum()
    assert np.allclose(out[0, 1:] / out[0, 1:].sum(), inside, atol=1e-5)
    # голова говорит «не гарь» — вероятность уходит классу 0 даже там, где модель дала 1–3
    quiet = _apply_burn_gate(proba, np.zeros(2, dtype=np.float32))
    assert float(quiet[:, 0].min()) > 0.99


def test_model_with_burn_gate_trains_and_survives_save(tmp_path):
    """Голова обучается вместе с моделью, участвует в предсказании и сохраняется в файл."""

    from fires.severity import SeverityModel as Model
    from fires.severity import train_model

    rng = np.random.default_rng(3)
    x = rng.random((400, 5), dtype=np.float32)
    y = np.where(x[:, 0] > 0.7, 1, np.where(x[:, 1] > 0.6, 2, np.where(x[:, 2] > 0.5, 3, 0)))
    y = y.astype(np.int32)
    model = train_model(x, y, params={"kind": "classifier", "iterations": 20, "gate": True},
                        backend="sklearn", device="cpu", verbose=False)
    assert model.gate is not None
    assert "голова" in model.describe()
    proba = model.proba_matrix(x[:30])
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    path = model.save(tmp_path / "gate.pkl")
    back = Model.load(path)
    assert back.gate is not None
    assert np.allclose(back.proba_matrix(x[:30]), proba, atol=1e-6)
    # модель без головы читается как раньше
    plain = train_model(x, y, params={"kind": "classifier", "iterations": 20},
                        backend="sklearn", device="cpu", verbose=False)
    assert plain.gate is None and "голова" not in plain.describe()
