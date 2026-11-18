"""Визуальный и числовой разбор одного чипа: почему пиксели отнесены к очагу.

Пример:
    python tools/inspect_chip.py --data-dir /tmp/firesdata/real/train --chip AF_tr_000239 --out /tmp/insp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fires import af as afm  # noqa: E402
from fires.chips import Dataset  # noqa: E402


def dump(name: str, arr: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.asarray(arr, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5, 4.4), dpi=110)
    im = ax.imshow(a, cmap="inferno")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"{name}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--chip", required=True)
    ap.add_argument("--out", default="/tmp/inspect")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = Dataset(args.data_dir)
    chip = ds.load(args.chip)
    params = afm.AFParams()
    pred, details = afm.detect_chip(chip, params, return_details=True)
    truth = np.asarray(chip.mask).astype(bool)
    i4 = np.asarray(chip.layers["I4"], dtype=np.float64)
    i5 = np.asarray(chip.layers["I5"], dtype=np.float64)
    delta = i4 - i5
    fp = pred.astype(bool) & ~truth

    stats = {
        "chip": args.chip,
        "n_pred": int(pred.sum()),
        "n_truth": int(truth.sum()),
        "details": details,
        "i4_pred": [float(np.nanmin(i4[pred.astype(bool)])), float(np.nanmax(i4[pred.astype(bool)]))] if pred.any() else None,
        "delta_median_pred": float(np.nanmedian(delta[pred.astype(bool)])) if pred.any() else None,
        "i4_fp_median": float(np.nanmedian(i4[fp])) if fp.any() else None,
        "delta_fp_median": float(np.nanmedian(delta[fp])) if fp.any() else None,
        "fp_near_truth_frac": float((np.asarray(chip.layers.get("valid", np.ones_like(i4)), dtype=bool)[fp]).mean()) if fp.any() else None,
    }
    # доля ложных пикселей в радиусе 3 пикселей от истинного очага (значит, разметка у́же пятна)
    if fp.any() and truth.any():
        from fires.morphology import binary_dilation

        near = binary_dilation(truth, 3)
        stats["fp_within3px_of_truth"] = float((fp & near).sum() / fp.sum())
    print(stats)

    dump("I4, K", i4, out / f"{args.chip}_i4.png")
    dump("I4 - I5, K", delta, out / f"{args.chip}_delta.png")
    dump("предсказание", pred, out / f"{args.chip}_pred.png")
    dump("истина", truth, out / f"{args.chip}_truth.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
