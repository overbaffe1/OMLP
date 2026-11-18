"""Оценка и диагностика: строгая загрузка весов, вложенность сплитов, diag_errors."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import load_blob_filter, load_severity_model  # noqa: E402
from fires.chips import Dataset  # noqa: E402
from train import SEED, _split_ids  # noqa: E402


def run(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_severity_model_missing_warns_and_strict_raises(tmp_path, capsys) -> None:
    weights = {"bs_mode": "model", "severity_model": "no_such_model.pkl"}
    assert load_severity_model(weights, tmp_path / "weights.json") is None
    assert "не найдена" in capsys.readouterr().err
    with pytest.raises(RuntimeError):
        load_severity_model(weights, tmp_path / "weights.json", strict=True)


def test_severity_model_thresholds_mode_returns_none(tmp_path) -> None:
    assert load_severity_model({"bs_mode": "thresholds"}, tmp_path / "w.json") is None


def test_blob_filter_modes(tmp_path, capsys) -> None:
    assert load_blob_filter({"af_mode": "rules"}, tmp_path / "w.json") is None
    assert capsys.readouterr().err == ""
    # режим blobs без файла: предупреждение и None, в strict — ошибка
    weights = {"af_mode": "blobs", "blob_filter": "no_filter.pkl"}
    assert load_blob_filter(weights, tmp_path / "w.json") is None
    assert "не найден" in capsys.readouterr().err
    with pytest.raises(RuntimeError):
        load_blob_filter(weights, tmp_path / "w.json", strict=True)


def test_evaluate_cli_reports_config_and_strict_fails(tiny_dataset, tmp_path) -> None:
    bad = tmp_path / "weights.json"
    bad.write_text(json.dumps({"bs_mode": "model", "severity_model": "missing.pkl",
                               "af_mode": "rules", "af": {}, "bs": {}}), encoding="utf-8")
    ok = run("evaluate.py", "--data-dir", str(tiny_dataset / "train"),
             "--weights", str(bad))
    assert ok.returncode == 0, ok.stderr
    assert "Активная конфигурация" in ok.stdout
    assert "НЕ загрузилась" in ok.stdout or "не загрузилась" in ok.stdout or "пороги" in ok.stdout
    strict = run("evaluate.py", "--data-dir", str(tiny_dataset / "train"),
                 "--weights", str(bad), "--strict")
    assert strict.returncode != 0


def test_split_nesting_same_seed(tiny_dataset) -> None:
    """Сплиты с одним сидом вложенные: val(0.25) ⊆ val(0.5) — отсюда утечка в точку отсчёта."""

    dataset = Dataset(tiny_dataset / "train")
    _train25, val25 = _split_ids(dataset, "bs", 0.25, SEED)
    _train50, val50 = _split_ids(dataset, "bs", 0.5, SEED)
    assert set(val25) <= set(val50)


def test_diag_errors_thresholds_runs(tiny_dataset) -> None:
    result = run("tools/diag_errors.py", "--data-dir", str(tiny_dataset / "train"),
                 "--thresholds-only", "--limit", "2")
    assert result.returncode == 0, result.stderr
    assert "Confusion" in result.stdout
    assert "ROI-гейта" in result.stdout


def test_diag_chip_runs(tiny_dataset) -> None:
    dataset = Dataset(tiny_dataset / "train")
    chip_id = dataset.chip_ids("bs")[0]
    result = run("tools/diag_chip.py", "--data-dir", str(tiny_dataset / "train"),
                 "--chip", chip_id)
    assert result.returncode == 0, result.stderr
    assert "автодиагноз" in result.stdout
    assert "истина" in result.stdout


def test_diag_scl_runs(tiny_dataset) -> None:
    result = run("tools/diag_scl.py", "--data-dir", str(tiny_dataset / "train"),
                 "--limit", "2")
    assert result.returncode == 0, result.stderr
    assert "под маской" in result.stdout
