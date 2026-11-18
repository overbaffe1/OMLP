"""Тесты модулей детекции: AF, BS и ML-фильтра очагов."""

from __future__ import annotations

import numpy as np
import pytest

from fires import af as af_module
from fires import bs as bs_module
from fires.chips import Chip, Dataset
from fires.morphology import label
from fires.synthetic import generate_af_chip, generate_bs_chip


def af_chip(rng, size: int, **kwargs) -> Chip:
    """Собирает объект :class:`Chip` из генератора синтетики (как его отдаёт Dataset.load)."""

    data = generate_af_chip(rng, size, **kwargs)
    target = data.pop("target")
    return Chip(chip_id="af_test", kind="af", layers=data, meta={}, mask=target.astype(bool))


def bs_chip(rng, size: int) -> Chip:
    """Собирает BS-чип так же, как его вернул бы ``Dataset.load`` (pre/post разделены)."""

    data = generate_bs_chip(rng, size)
    severity = data.pop("target_sev")
    return Chip(
        chip_id="bs_test",
        kind="bs",
        layers={"SCL": data["scl"], "landcover": data["landcover"], "valid": data["valid"]},
        pre=dict(data["pre"]),
        post=dict(data["post"]),
        s1_pre=dict(data["s1_pre"]),
        s1_post=dict(data["s1_post"]),
        meta={},
        mask_sev=severity.astype(np.uint8),
    )


def test_af_detects_planted_fire() -> None:
    rng = np.random.default_rng(7)
    chip = af_chip(rng, size=128, with_fire=True)
    target = chip.mask
    mask = af_module.detect_chip(chip, af_module.AFParams())
    tp = int((mask & target).sum())
    fp = int((mask & ~target).sum())
    assert target.sum() > 0
    assert tp / target.sum() > 0.5, f"низкая полнота: {tp}/{target.sum()}"
    assert fp <= 0.2 * max(1, target.sum()) + 5, f"много ложных: {fp}"


def test_af_ignores_industrial_flare() -> None:
    """Факел промплощадки не должен попадать в ответ.

    На уровне одного пикселя факел неотличим от пожара: I4 насыщается, ΔT даже выше, чем у
    лесного горения. Поэтому в решении его отсекает обученный классификатор очагов
    (:class:`fires.blobfilter.BlobFilter`), который видит форму, площадь и контекст кластера.
    """

    from fires.blobfilter import BlobFilter, extract_features, label_blobs

    rng = np.random.default_rng(101)
    features, labels = [], []
    for i in range(10):
        train_chip = af_chip(rng, size=128, with_fire=i % 3 == 0, tech_source=i % 3 == 1)
        feats, masks = extract_features(train_chip, af_module.AFParams())
        if len(masks):
            features.append(feats)
            labels.append(label_blobs(train_chip, masks))
    model = BlobFilter().fit(np.vstack(features), np.concatenate(labels))

    flare_chip = af_chip(rng, size=128, with_fire=False, tech_source=True)
    flare_mask = af_module.detect_chip(flare_chip, af_module.AFParams(), blob_filter=model)
    assert flare_mask.sum() == 0, f"техногенный источник принят за пожар: {int(flare_mask.sum())} px"

    fire_chip = af_chip(rng, size=128, with_fire=True)
    fire_mask = af_module.detect_chip(fire_chip, af_module.AFParams(), blob_filter=model)
    truth = fire_chip.mask
    assert (fire_mask & truth).sum() / truth.sum() > 0.3, "вместе с факелом подавили и настоящий очаг"


def test_af_respects_validity_and_cloud() -> None:
    rng = np.random.default_rng(21)
    chip = af_chip(rng, size=128, with_fire=True)
    chip.layers["valid"] = np.zeros_like(chip.layers["valid"])
    mask = af_module.detect_chip(chip, af_module.AFParams())
    assert mask.sum() == 0


def test_bs_classifies_severity() -> None:
    """Без облаков оптическая классификация должна восстанавливать маску и классы."""

    rng = np.random.default_rng(5)
    chip = bs_chip(rng, size=256)
    severity = chip.mask_sev
    # убираем облака и тени, чтобы проверить именно пороги dNBR по типам покрова
    scl = chip.layers["SCL"].copy()
    scl[np.isin(scl, (3, 8, 9, 10, 11))] = 4
    chip.layers["SCL"] = scl

    pred = bs_module.classify_chip(chip, bs_module.BSParams(sar_recovery=False))
    burned = severity > 0
    assert burned.sum() > 0
    detected = pred > 0
    recall = (detected & burned).sum() / burned.sum()
    precision = (detected & burned).sum() / max(1, detected.sum())
    assert recall > 0.9, recall
    assert precision > 0.9, precision
    classes = set(np.unique(pred[burned]).tolist()) - {0}
    assert classes, "ни одна степень поражения не предсказана"


def test_bs_cloud_recovery_is_not_harmful() -> None:
    """Восстановление гари под облаком по Sentinel-1 не должно ухудшать IoU.

    Пиксели под облаком/тенью недостоверны для оптики; радиолокация позволяет вернуть
    часть из них, но при консервативных порогах. Тест допускает небольшой проигрыш
    (шум синтетики), но не деградацию.
    """

    from fires.metrics import score_from_pairs

    rng = np.random.default_rng(7)
    chip = bs_chip(rng, size=512)
    scl = chip.layers["SCL"]
    if not np.isin(scl, (3, 9)).any():
        pytest.skip("в этом чипе нет облачности")

    without = bs_module.classify_chip(chip, bs_module.BSParams(sar_recovery=False))
    with_sar = bs_module.classify_chip(chip, bs_module.BSParams(sar_recovery=True))
    iou_without = score_from_pairs({}, {"c": (chip.mask_sev, without)})["iou_burn"]
    iou_with = score_from_pairs({}, {"c": (chip.mask_sev, with_sar)})["iou_burn"]
    assert iou_with >= iou_without - 0.01, (iou_without, iou_with)


def test_bs_without_s1_still_works() -> None:
    rng = np.random.default_rng(3)
    chip = bs_chip(rng, size=256)
    chip.layers.pop("s1_pre", None)
    chip.layers.pop("s1_post", None)
    pred_no_s1 = bs_module.classify_chip(chip, bs_module.BSParams())
    assert (pred_no_s1 > 0).sum() > 0


def test_blob_filter_training_and_application(tmp_path) -> None:
    from fires.blobfilter import BlobFilter, extract_features, label_blobs

    rng = np.random.default_rng(42)
    Xs, ys = [], []
    features_per_chip = []
    for i in range(6):
        obj = af_chip(rng, size=128, with_fire=i % 2 == 0, tech_source=i == 3)
        obj.layers["valid"] = np.ones_like(obj.layers["valid"])
        feats, masks = extract_features(obj, af_module.AFParams())
        features_per_chip.append((obj, masks))
        if len(masks):
            Xs.append(feats)
            ys.append(label_blobs(obj, masks))

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    assert y.sum() > 0, "в синтетике должны быть положительные блобы"
    model = BlobFilter().fit(X, y)
    assert model.ready
    proba = model.predict_proba(X)
    assert proba.shape == (len(y),)
    assert np.all((proba >= 0) & (proba <= 1))

    path = model.save(tmp_path / "blob.pkl")
    loaded = BlobFilter.load(path)
    assert np.allclose(loaded.predict_proba(X), proba)

    obj, _ = features_per_chip[0]
    mask = loaded.apply(obj, af_module.AFParams())
    assert mask.dtype == bool
    assert mask.shape == obj.shape


def test_dataset_roundtrip(tiny_dataset) -> None:
    dataset = Dataset(tiny_dataset)
    ids = dataset.chip_ids()
    assert ids, "датасет не найден"
    chip = dataset.load(ids[0])
    assert chip.shape[0] > 0 and chip.shape[1] > 0
    assert chip.kind in {"af", "bs"}

    masked = [cid for cid in dataset.chip_ids("af") if dataset.load(cid).mask is not None]
    assert masked, "в обучающей части маски должны быть доступны"
    mask = dataset.load(masked[0]).mask
    assert mask is not None and mask.size > 0
    _, n = label(mask, connectivity=8)
    assert n >= 0
