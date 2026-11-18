#!/usr/bin/env python3
"""Вскрытие одного BS-чипа: почему гарь не находится (или находится не та).

Для чипа с эталоном считает четыре варианта предсказания (пороги/модель ×
без морфологии/с морфологией), аудит масок (сколько истины под SCL/valid/водой),
сигнал внутри гари (dNBR, NBR до, падение VH, изменение видимого канала) и состав
покрова. По итогам печатает наиболее вероятную причину.

Запуск::

    py -3 tools\\diag_chip.py --data-dir <путь>\\train --chip BS_tr_000024
    py -3 tools\\diag_chip.py --data-dir <путь>\\train --chip BS_tr_000024 --chip BS_tr_000069

Обучения нет — только чтение чипа и предсказания текущей конфигурацией.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import load_severity_model  # noqa: E402
from fires import bs as bsm  # noqa: E402
from fires.chips import Dataset, find_data_dir  # noqa: E402
from fires.weights import load_weights  # noqa: E402
from train import SEED, _split_ids  # noqa: E402


def _q(vals: np.ndarray) -> str:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "нет данных"
    qs = np.percentile(vals, (5, 25, 50, 75, 95))
    return " / ".join(f"{v:.3f}" for v in qs) + f"  (n={vals.size})"


def _iou(truth: np.ndarray, pred: np.ndarray, cls: int | None = None) -> float:
    if cls is None:
        t, p = truth > 0, pred > 0
    else:
        t, p = truth == cls, pred == cls
    tp = int((t & p).sum())
    return tp / max(1, int(t.sum()) + int(p.sum()) - tp)


def autopsy(dataset: Dataset, chip_id: str, bs_params, sev_model, canon_val: set[str]) -> int:
    chip = dataset.load(chip_id)
    if chip.kind != "bs" or chip.mask_sev is None:
        print(f"{chip_id}: не BS-чип с эталоном — вскрывать нечего")
        return 1
    truth = np.asarray(chip.mask_sev, dtype=np.uint8)
    meta = dataset.meta.get(chip_id, {})
    print(f"\n===== {chip_id} (канонический сплит: {'VAL' if chip_id in canon_val else 'train'}) =====")
    print(f"meta: date_pre={meta.get('date_pre')} date_post={meta.get('date_post')} "
          f"cloud_frac={meta.get('cloud_frac')} burn_area_ha={meta.get('burn_area_ha')} "
          f"fire_event={meta.get('fire_event_id')}")
    hist = {int(k): int((truth == k).sum()) for k in range(4)}
    print(f"истина: {hist} (гарь: {hist[1] + hist[2] + hist[3]} px)")

    raw_params = replace(bs_params, min_patch_px=1)
    variants: dict[str, np.ndarray] = {}
    try:
        variants["пороги, без морфологии"] = np.asarray(
            bsm.classify_chip(chip, raw_params), dtype=np.uint8)
        variants["пороги + морфология"] = np.asarray(
            bsm.classify_chip(chip, bs_params), dtype=np.uint8)
    except Exception as exc:
        print(f"пороговый путь упал: {exc}")
        return 1
    if sev_model is not None:
        variants["модель, без морфологии"] = np.asarray(
            bsm.classify_chip(chip, raw_params, severity_model=sev_model), dtype=np.uint8)
        variants["модель + морфология"] = np.asarray(
            bsm.classify_chip(chip, bs_params, severity_model=sev_model), dtype=np.uint8)
    print(f"{'вариант':<26}{'burn_px':>10}{'IoU_burn':>10}{'IoU_1':>8}{'IoU_2':>8}{'IoU_3':>8}")
    for name, pred in variants.items():
        burn = int((pred > 0).sum())
        print(f"{name:<26}{burn:10d}{_iou(truth, pred):10.3f}{_iou(truth, pred, 1):8.3f}"
              f"{_iou(truth, pred, 2):8.3f}{_iou(truth, pred, 3):8.3f}")

    # --- аудит масок: какая доля истины вообще запрещена к предсказанию
    layers = chip.layers
    h, w = truth.shape
    scl = np.asarray(layers.get("SCL")) if layers.get("SCL") is not None else None
    scl_pre = np.asarray(layers.get("SCL_pre")) if layers.get("SCL_pre") is not None else None
    if scl is None and chip.post and "SCL" in chip.post:
        scl = np.asarray(chip.post["SCL"])
    if scl_pre is None and chip.pre and "SCL" in chip.pre:
        scl_pre = np.asarray(chip.pre["SCL"])
    valid = np.asarray(layers.get("valid")).astype(bool) if layers.get("valid") is not None else None
    lc = np.asarray(layers.get("landcover")) if layers.get("landcover") is not None else None
    burn = truth > 0
    print("--- маски (доля истины гари под маской — то, что решение не предскажет) ---")
    if scl is not None and scl.shape == (h, w):
        bad = np.isin(scl, bsm.SCL_BAD) | (scl == bsm.SCL_WATER)
        print(f"    SCL(post) плохие: {bad.mean():6.1%} чипа, истины под ними: "
              f"{(bad & burn).sum() / max(1, burn.sum()):6.1%}")
    else:
        print("    SCL(post): нет слоя")
    if scl_pre is not None and scl_pre.shape == (h, w):
        bad_pre = np.isin(scl_pre, bsm.SCL_BAD) | (scl_pre == bsm.SCL_WATER)
        print(f"    SCL(pre)  плохие: {bad_pre.mean():6.1%} чипа, истины под ними: "
              f"{(bad_pre & burn).sum() / max(1, burn.sum()):6.1%}")
    if valid is not None and valid.shape == (h, w):
        print(f"    valid=0: {((~valid).mean()):6.1%} чипа, истины под ним: "
              f"{((~valid) & burn).sum() / max(1, burn.sum()):6.1%}")
    if lc is not None and lc.shape == (h, w):
        codes = bsm.group_codes()
        comp = {g: int(np.isin(lc[burn], [c for c, gg in codes.items() if gg == g]).sum())
                for g in ("forest", "semi", "cropland", "other")}
        comp["вода/прочее"] = int(burn.sum()) - sum(comp.values())
        print(f"    покров истины гари: {comp}")
        unmapped = (lc == bsm.LC_WATER) | np.isin(lc, bsm.LANDCOVER_GROUPS["other"])
        print(f"    вода/застройка/снег (не картируются): {unmapped.mean():6.1%} чипа, истины: "
              f"{(unmapped & burn).sum() / max(1, burn.sum()):6.1%}")

    # --- сигнал внутри гари: есть ли что находить
    print("--- сигнал (квантили 5/25/50/75/95) ---")
    try:
        scale = bsm.scale_reflectance
        nbr_pre = bsm.nbr(scale(chip.pre["B8A"]), scale(chip.pre["B12"]))
        nbr_post = bsm.nbr(scale(chip.post["B8A"]), scale(chip.post["B12"]))
        dnbr = (nbr_pre - nbr_post).astype(np.float32)
        finite = np.isfinite(dnbr)
        print(f"    dNBR в гари: { _q(dnbr[burn & finite])}")
        print(f"    dNBR в фоне: { _q(dnbr[(truth == 0) & finite])}")
        print(f"    доля гари с dNBR>0.05 (внутри ROI): "
              f"{((dnbr > 0.05) & burn & finite).sum() / max(1, (burn & finite).sum()):.1%}")
        print(f"    NBR(до) в гари: {_q(nbr_pre[burn & finite])}  (низкий = мало биомассы)")
        b4_pre = scale(np.asarray(chip.pre.get("B4", np.nan), dtype=np.float32))
        b4_post = scale(np.asarray(chip.post.get("B4", np.nan), dtype=np.float32))
        if b4_pre.shape == (h, w):
            db4 = b4_post - b4_pre
            print(f"    ΔB4(видимый) в гари: {_q(db4[burn & finite])}  (минус = потемнение)")
            print(f"    ΔB4(видимый) в фоне: {_q(db4[(truth == 0) & finite])}")
    except (KeyError, TypeError) as exc:
        print(f"    оптика недоступна: {exc}")
    vh_pre = chip.s1_pre.get("VH")
    vh_post = chip.s1_post.get("VH")
    if vh_pre is not None and vh_post is not None:
        dvh = (np.asarray(vh_post, dtype=np.float32) - np.asarray(vh_pre, dtype=np.float32))
        if np.nanmax(np.abs(dvh[np.isfinite(dvh)]), initial=0) > 100:
            dvh = dvh / 100.0  # дБ ×100
        print(f"    ΔVH(радар) в гари: {_q(dvh[burn])} дБ  (минус = падение крон/шероховатости)")
        print(f"    ΔVH(радар) в фоне: {_q(dvh[truth == 0])} дБ")

    # --- автодиагноз
    ious = {name: _iou(truth, pred) for name, pred in variants.items()}
    print("--- автодиагноз ---")
    for name, val in ious.items():
        flag = "  <-- лучший" if val == max(ious.values()) and val > 0 else ""
        print(f"    {name}: IoU={val:.3f}{flag}")
    if max(ious.values()) <= 0.01:
        print("    НИ ОДИН вариант не находит гарь: дело не в модели и не в морфологии — "
              "смотрите маски и сигнал выше (нет сигнала или истина под маской).")
    elif ious.get("модель + морфология", 0) < ious.get("пороги + морфология", 0) - 0.05:
        print("    Пороги находят, модель — нет: модель слепа на этом чипе (признаки/домен).")
    elif ious.get("модель + морфология", 0) < ious.get("модель, без морфологии", 0) - 0.05:
        print("    Морфология съедает предсказание модели: предсказание фрагментировано "
              "(соль), чистка <48px удаляет всё.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Вскрытие BS-чипа: почему гарь не находится")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--chip", required=True, action="append", help="ID чипа (можно несколько)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    args = ap.parse_args(argv)

    resolved = find_data_dir(args.data_dir, prefer="masks")
    dataset = Dataset(resolved)
    weights = load_weights(args.weights)
    bs_params = bsm.to_params(weights.get("bs", {}))
    sev_model = load_severity_model(weights, args.weights, args.device)
    print(f"конфигурация: {'модель ' + str(weights.get('severity_model')) if sev_model is not None else 'только пороги'}, "
          f"морфология {bs_params.min_patch_px}px")
    _canon_train, canon_val = _split_ids(dataset, "bs", 0.25, SEED)
    canon_set = set(canon_val)
    code = 0
    for chip_id in args.chip:
        code = max(code, autopsy(dataset, chip_id, bs_params, sev_model, canon_set))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
