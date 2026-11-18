"""Тесты информационно-аналитического сервиса: каталог, REST API, выгрузки."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from service.catalog import build


@pytest.fixture(scope="module")
def catalog_path(tmp_path_factory, tiny_dataset) -> Path:
    out = tmp_path_factory.mktemp("catalog") / "catalog.json"
    summary = build(tiny_dataset, out, weights_path=None)
    assert summary["thermopoints"] > 0 or summary["contours"] > 0
    return out


@pytest.fixture(scope="module")
def client(catalog_path):
    import importlib
    import os

    os.environ["CATALOG_PATH"] = str(catalog_path)
    import service.app as app_module

    importlib.reload(app_module)
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


def test_catalog_structure(catalog_path) -> None:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert catalog["mode"] in {"geo", "demo"}
    assert isinstance(catalog["thermopoints"], list)
    assert isinstance(catalog["contours"], list)
    assert catalog["report"]["n_thermopoints"] == len(catalog["thermopoints"])
    for contour in catalog["contours"][:5]:
        ring = contour["ring"]
        assert len(ring) >= 4
        assert ring[0] == ring[-1], "кольцо GeoJSON должно быть замкнуто"
        assert contour["severity"] in (1, 2, 3)
        assert contour["area_ha"] > 0


def test_health_and_report(client) -> None:
    health = client.get("/api/health").json()
    assert health["status"] == "ok"

    report = client.get("/api/report").json()
    assert report["total_area_ha"] >= 0
    assert set(report["by_severity_ha"]) == {"severity_1", "severity_2", "severity_3"}

    csv_text = client.get("/api/report?format=csv").text
    assert csv_text.splitlines()[0] == "metric,value"


def test_thermopoints_geojson_and_time_filter(client) -> None:
    features = client.get("/api/thermopoints").json()["features"]
    assert features and features[0]["geometry"]["type"] == "Point"
    assert "date" in features[0]["properties"]

    dates = sorted(f["properties"]["date"] for f in features)
    narrowed = client.get(f"/api/thermopoints?date_from={dates[-1]}&date_to={dates[-1]}").json()["features"]
    assert 0 < len(narrowed) <= len(features)


def test_polygon_query(client) -> None:
    ring = [[89.0, 49.0], [95.0, 49.0], [95.0, 56.0], [89.0, 56.0], [89.0, 49.0]]
    payload = client.post("/api/query", json={"polygon": ring}).json()
    assert payload["report"]["n_thermopoints"] > 0
    assert payload["report"]["total_area_ha"] > 0

    empty = client.post("/api/query", json={"polygon": [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]]}).json()
    assert empty["report"]["n_thermopoints"] == 0


def test_severity_filter_applies_to_contours_only(client) -> None:
    all_contours = client.get("/api/burn-contours").json()["features"]
    strong = client.get("/api/burn-contours?severity=3").json()["features"]
    assert len(strong) <= len(all_contours)
    assert all(f["properties"]["severity"] == 3 for f in strong)
    # термоточки фильтром по степени поражения не отсекаются
    assert client.get("/api/thermopoints?severity=3").json()["features"]


def test_exports(client) -> None:
    geojson = client.get("/api/export/contours.geojson")
    assert geojson.status_code == 200
    assert geojson.json()["type"] == "FeatureCollection"
    assert "attachment" in geojson.headers["content-disposition"]

    archive = client.get("/api/export/contours.shp")
    assert archive.status_code == 200, "для выгрузки Shapefile нужен пакет pyshp"
    names = set(zipfile.ZipFile(io.BytesIO(archive.content)).namelist())
    assert {"burn_contours.shp", "burn_contours.shx", "burn_contours.dbf", "burn_contours.prj"} <= names


def test_bad_bbox_returns_422(client) -> None:
    assert client.get("/api/report?bbox=1,2,3").status_code == 422
    assert client.get("/api/report?bbox=a,b,c,d").status_code == 422


def test_web_page_renders(client) -> None:
    html = client.get("/").text
    assert "КосмоФАЙР" in html
    assert "api/burn-contours" in html
