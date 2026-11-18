"""Сквозные тесты: обязательный CLI организаторов, оценка и обучение весов."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def first_json(text: str) -> dict:
    """Достаёт первый JSON-объект из вывода CLI (у inference их два: отчёт и валидация)."""

    start = text.index("{")
    payload, _ = json.JSONDecoder().raw_decode(text[start:])
    return payload


def run(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_inference_cli_produces_valid_submission(tiny_dataset, tmp_path) -> None:
    """Команда из постановки должна завершаться кодом 0 и создавать CSV по шаблону."""

    out = tmp_path / "submission.csv"
    result = run(
        "inference.py",
        "--data-dir", str(tiny_dataset),
        "--output", str(out),
        "--validate",
    )
    assert result.returncode == 0, result.stderr
    report = first_json(result.stdout)
    assert {"chips_processed", "rows_written", "seconds"} <= set(report)
    assert out.exists()

    header, *rows = out.read_text(encoding="utf-8").splitlines()
    assert header == "chip_id,class_id,rle"
    template = (tiny_dataset / "sample_submission.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == len(template) - 1, "число строк должно совпадать с шаблоном"


def test_inference_is_deterministic(tiny_dataset, tmp_path) -> None:
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    run("inference.py", "--data-dir", str(tiny_dataset), "--output", str(first))
    run("inference.py", "--data-dir", str(tiny_dataset), "--output", str(second))
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_inference_validation_report(tiny_dataset, tmp_path) -> None:
    out = tmp_path / "submission.csv"
    result = run("inference.py", "--data-dir", str(tiny_dataset), "--output", str(out), "--validate")
    assert result.returncode == 0, result.stderr
    validation = first_json(result.stdout[result.stdout.index("=== ", result.stdout.index("{") + 1) :]) \
        if False else json.loads(result.stdout[result.stdout.rindex("{\n  \"validation\"") :])
    assert validation["validation"]["ok"], validation["validation"]["errors"]


def test_evaluate_reports_metric_components(tiny_dataset) -> None:
    result = run("evaluate.py", "--data-dir", str(tiny_dataset))
    assert result.returncode == 0, result.stderr
    payload = first_json(result.stdout)
    for key in ("score", "f1_af", "iou_burn", "miou_sev"):
        assert key in payload
        assert 0.0 <= payload[key] <= 1.0


def test_evaluate_ablation_runs(tiny_dataset) -> None:
    result = run("evaluate.py", "--data-dir", str(tiny_dataset), "--ablation")
    assert result.returncode == 0, result.stderr
    assert "Абляции" in result.stdout
    assert "Score=" in result.stdout


def test_train_writes_weights(tiny_dataset, tmp_path) -> None:
    """Обучение пишет weights.json и (при выборе режима blobs) файл классификатора."""

    out = tmp_path / "weights.json"
    result = run("train.py", "--data-dir", str(tiny_dataset), "--out", str(out), "--fast")
    assert result.returncode == 0, result.stderr
    weights = json.loads(out.read_text(encoding="utf-8"))
    assert "af" in weights and "bs" in weights
    assert weights["af_mode"] in {"rules", "blobs"}
    if weights["af_mode"] == "blobs":
        assert (tmp_path / weights["blob_filter"]).exists()


def test_weights_loader_defaults() -> None:
    from fires.weights import load_weights

    assert isinstance(load_weights(None), dict)
    assert isinstance(load_weights("нет-такого-файла.json"), dict)


@pytest.mark.parametrize("script", ["inference.py", "train.py", "evaluate.py"])
def test_scripts_expose_help(script: str) -> None:
    result = run(script, "--help")
    assert result.returncode == 0
    assert "--data-dir" in result.stdout
