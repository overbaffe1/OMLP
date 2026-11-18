"""Регрессия на раскладку реального набора организаторов.

Проверяется то, что легко сломать при рефакторинге: сопоставление файлов чипу по префиксу,
порядок каналов в многоканальных стеках, попадание SCL в служебные слои (а не в стек дат),
чтение масок по типам чипов и маскировка облаков в BS.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from fires import bs as bs_module
from fires.chips import Dataset, find_data_dir

pytest.importorskip("tifffile")

H = W = 32


def _write_bs_case(root, chip_id: str = "BS_tr_000001") -> None:
    rng = np.random.default_rng(3)
    (root / "bs" / "sentinel2_pre").mkdir(parents=True)
    (root / "bs" / "sentinel2_post").mkdir(parents=True)
    (root / "bs" / "sentinel1_pre").mkdir(parents=True)
    (root / "bs" / "sentinel1_post").mkdir(parents=True)
    (root / "bs" / "aux").mkdir(parents=True)
    (root / "bs" / "masks").mkdir(parents=True)

    def s2(rise: float, cloud: bool) -> np.ndarray:
        stack = np.zeros((H, W, 10), dtype=np.uint16)
        for i in range(9):  # B2…B12 (девять каналов отражения)
            stack[..., i] = rng.integers(500, 4000, size=(H, W), dtype=np.uint16)
        # полоса выгоревшей земли: B8A падает, B12 растёт — это и есть сигнал dNBR
        burned = np.zeros((H, W), dtype=bool)
        burned[10:20, 10:20] = True
        stack[..., 6][burned] = np.uint16(400)                     # B8A
        stack[..., 8][burned] = np.uint16(3000 + 2000 * rise)      # B12
        scl = np.full((H, W), 4, dtype=np.uint16)                  # растительность
        if cloud:
            scl[24:30, 24:30] = 9                                  # облако высокой вероятности
        stack[..., 9] = scl                                        # SCL — последний канал
        return stack

    tifffile.imwrite(root / "bs" / "sentinel2_pre" / f"{chip_id}_Sentinel-2_pre.tif", s2(0.0, cloud=False))
    tifffile.imwrite(root / "bs" / "sentinel2_post" / f"{chip_id}_Sentinel-2_post.tif", s2(1.0, cloud=True))
    s1_pre = rng.normal(-1200, 30, size=(H, W, 2)).astype(np.int16)   # дБ ×100
    s1_post = s1_pre.copy()
    s1_post[10:20, 10:20, 1] -= 500                                   # падение VH в гари
    tifffile.imwrite(root / "bs" / "sentinel1_pre" / f"{chip_id}_Sentinel-1_pre.tif", s1_pre)
    tifffile.imwrite(root / "bs" / "sentinel1_post" / f"{chip_id}_Sentinel-1_post.tif", s1_post)
    aux = np.zeros((H, W, 3), dtype=np.int16)
    aux[..., 0] = 120                                                  # DEM
    aux[..., 1] = 3                                                    # идентификатор сцены
    aux[..., 2] = 40                                                   # landcover: пашня
    tifffile.imwrite(root / "bs" / "aux" / f"{chip_id}_AUX.tif", aux)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[10:20, 10:20] = 2
    tifffile.imwrite(root / "bs" / "masks" / f"{chip_id}_MASK.tif", mask)

    header = (
        "chip_id,kind,fire_event_id,region,epsg,x_min,y_min,x_max,y_max,width,height,gsd,acq_datetime,"
        "satellite,date_pre,date_post,s1_date_pre,s1_date_post,valid_frac,cloud_frac,landcover_top,"
        "n_fire_px,burn_area_ha,sev1_px,sev2_px,sev3_px\n"
    )
    row = (
        f"{chip_id},bs,FE00001,,32637,384000,5280000,384640,5280640,{W},{H},20,,,2022-06-01,2022-06-11,"
        "2022-05-31,2022-06-12,0.99,0.02,,,10.24,0,100,0\n"
    )
    # поля: chip_id, kind, fire_event_id, region, epsg, x_min, y_min, x_max, y_max, width, height, gsd,
    #       acq_datetime, satellite, date_pre, date_post, s1_date_pre, s1_date_post, valid_frac,
    #       cloud_frac, landcover_top, n_fire_px, burn_area_ha, sev1_px, sev2_px, sev3_px
    (root / "bs" / "meta.csv").write_text(header + row, encoding="utf-8")


def test_real_layout_routes_scl_and_channels(tmp_path) -> None:
    """SCL попадает в служебные слои обеих дат, а каналы — в стеки дат."""

    _write_bs_case(tmp_path)
    dataset = Dataset(tmp_path)
    assert dataset.chip_ids("bs") == ["BS_tr_000001"]
    assert dataset.kind_of("BS_tr_000001") == "bs"

    chip = dataset.load("BS_tr_000001")
    assert sorted(chip.pre) == ["B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8A"]
    assert chip.pre["B12"].shape == (H, W), "стек дат не должен содержать SCL"
    assert "SCL" in chip.layers and "SCL_pre" in chip.layers
    assert np.unique(chip.layers["SCL"])[-1] == 9, "облако из даты «после» не потерялось"
    assert sorted(chip.s1_post) == ["VH", "VV"]
    assert int(chip.mask_sev.max()) == 2 and chip.mask.sum() == 100
    assert float(chip.meta["gsd"]) == 20.0


def test_clouds_do_not_produce_burn(tmp_path) -> None:
    """Пиксели под облаком не должны попадать в гарь: их наблюдение недостоверно."""

    _write_bs_case(tmp_path)
    dataset = Dataset(tmp_path)
    chip = dataset.load("BS_tr_000001")
    severity = bs_module.classify_chip(chip, bs_module.BSParams())
    cloudy = np.asarray(chip.layers["SCL"]) == 9
    assert cloudy.any()
    assert int((severity > 0)[cloudy].sum()) == 0
    assert int((severity > 0).sum()) > 0, "гарь на чипе всё же должна находиться"

    no_mask = bs_module.classify_chip(chip, bs_module.BSParams(mask_scl=False))
    assert int((no_mask > 0)[cloudy].sum()) >= int((severity > 0)[cloudy].sum())


def test_find_data_dir_resolves_nested_dataset(tmp_path) -> None:
    """Набор, распакованный во вложенную папку, находится по пути на уровень выше."""

    nested = tmp_path / "cosmo1" / "train"
    _write_bs_case(nested)
    assert find_data_dir(nested) == nested
    assert find_data_dir(tmp_path / "cosmo1") == nested


def test_find_data_dir_keeps_wrong_path(tmp_path) -> None:
    """Если чипов нет нигде рядом, путь не подменяется: решение должно сказать об ошибке."""

    empty = tmp_path / "empty"
    empty.mkdir()
    assert find_data_dir(empty) == empty
    assert find_data_dir(tmp_path / "nope") == tmp_path / "nope"
