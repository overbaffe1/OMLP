"""Общие фикстуры тестов: маленький синтетический датасет собирается один раз на сессию."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fires.synthetic import SyntheticConfig, generate_dataset  # noqa: E402


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory) -> Path:
    """Компактный датасет (8 AF + 4 BS) для проверки сквозного конвейера."""

    root = tmp_path_factory.mktemp("data")  # train + test по 4 AF и 2 BS
    cfg = SyntheticConfig(n_af=4, n_bs=2, af_size=128, bs_size=256, seed=1234)
    generate_dataset(root, cfg)
    return root
