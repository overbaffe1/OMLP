"""Информационно-аналитический сервис «КосмоФАЙР-2».

Запуск:

    uvicorn service.app:app --host 0.0.0.0 --port 8000
    # каталог сцен задаётся переменной окружения CATALOG_PATH (по умолчанию service/catalog.json)

Функциональность по требованиям критериев раздела 4:

* приём пространственно-временного запроса — прямоугольник (``bbox``) или произвольный
  полигон (``polygon`` в GeoJSON-геометрии) плюс интервал дат;
* карта термоточек за период (``/api/thermopoints``), GeoJSON;
* векторные контуры гари с атрибутами: идентификатор, класс степени поражения, площадь
  (``/api/burn-contours``), выгрузка в GeoJSON и Shapefile (``/api/export/...``);
* аналитическая справка: суммарная площадь гари в гектарах и распределение по трём степеням
  поражения, выгрузка в машиночитаемом виде (``/api/report``, ``?format=csv``);
* тот же результат без пользовательского интерфейса — все методы доступны как REST API.

Геометрия и площади считаются по маскам (число пикселей × площадь пикселя), поэтому числа
в справке воспроизводимы по выданным данным, как того требует критерий.
"""

from __future__ import annotations

import csv
import io
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

CATALOG_PATH = Path(os.environ.get("CATALOG_PATH", Path(__file__).resolve().parent / "catalog.json"))
app = FastAPI(title="КосмоФАЙР-2 — сервис мониторинга природных пожаров", version="2.0.0")


def _load_catalog() -> dict:
    if not CATALOG_PATH.exists():
        return {"mode": "empty", "thermopoints": [], "contours": [], "report": {}}
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


CATALOG = _load_catalog()


# --------------------------------------------------------------------------- фильтрация

class Query(BaseModel):
    """Пространственно-временной запрос."""

    bbox: list[float] | None = Field(default=None, description="[min_lon, min_lat, max_lon, max_lat]")
    polygon: list[list[float]] | None = Field(default=None, description="кольцо [[lon, lat], ...]")
    date_from: str | None = None
    date_to: str | None = None
    severity: list[int] | None = None


def _in_bbox(lon: float, lat: float, bbox: list[float] | None) -> bool:
    if not bbox:
        return True
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def _in_polygon(lon: float, lat: float, ring: list[list[float]]) -> bool:
    """Тест «точка внутри полигона» методом трассировки луча (numpy, без shapely)."""

    pts = np.asarray(ring, dtype=float)
    if len(pts) < 3:
        return False
    x, y = pts[:, 0], pts[:, 1]
    inside = False
    n = len(pts)
    for i in range(n):
        j = (i - 1) % n
        if (y[i] > lat) != (y[j] > lat):
            x_cross = (x[j] - x[i]) * (lat - y[i]) / (y[j] - y[i]) + x[i]
            if lon < x_cross:
                inside = not inside
    return inside


def _in_time(value: str, date_from: str | None, date_to: str | None) -> bool:
    if not value:
        return True
    if date_from and value < date_from:
        return False
    if date_to and value > date_to:
        return False
    return True


def _filter(items: list[dict], query: Query, key_lon: str = "lon", key_lat: str = "lat") -> list[dict]:
    out = []
    for item in items:
        lon, lat = item.get(key_lon), item.get(key_lat)
        if lon is None or lat is None:
            continue
        if not _in_bbox(float(lon), float(lat), query.bbox):
            continue
        if query.polygon and not _in_polygon(float(lon), float(lat), query.polygon):
            continue
        if not _in_time(str(item.get("date", "")), query.date_from, query.date_to):
            continue
        # фильтр по степени поражения относится только к контурам (см. _filter_contours)
        out.append(item)
    return out


def _filter_contours(query: Query) -> list[dict]:
    out = []
    for contour in CATALOG.get("contours", []):
        ring = contour.get("ring") or []
        if not ring:
            continue
        lon, lat = ring[0]
        if not _in_bbox(float(lon), float(lat), query.bbox):
            continue
        if query.polygon and not _in_polygon(float(lon), float(lat), query.polygon):
            continue
        if not _in_time(str(contour.get("date", "")), query.date_from, query.date_to):
            continue
        if query.severity and contour.get("severity") not in query.severity:
            continue
        out.append(contour)
    return out


def _report(contours: list[dict], thermopoints: list[dict]) -> dict:
    by_sev = {1: 0.0, 2: 0.0, 3: 0.0}
    for contour in contours:
        by_sev[int(contour["severity"])] += float(contour["area_ha"])
    return {
        "total_area_ha": round(sum(by_sev.values()), 2),
        "by_severity_ha": {f"severity_{k}": round(v, 2) for k, v in by_sev.items()},
        "share_by_severity": {f"severity_{k}": (round(v / sum(by_sev.values()), 4) if sum(by_sev.values()) else 0.0)
                              for k, v in by_sev.items()},
        "n_contours": len(contours),
        "n_thermopoints": len(thermopoints),
        "catalog_mode": CATALOG.get("mode", "unknown"),
    }


# --------------------------------------------------------------------------- API

@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "catalog_mode": CATALOG.get("mode", "empty"),
        "thermopoints": len(CATALOG.get("thermopoints", [])),
        "contours": len(CATALOG.get("contours", [])),
        "generated_at": CATALOG.get("generated_at"),
    }


@app.get("/api/thermopoints")
def thermopoints(
    bbox: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    format: str = "geojson",
) -> Response:
    """Карта термоточек за указанный период (GeoJSON по умолчанию)."""

    query = Query(bbox=_parse_bbox(bbox), date_from=date_from, date_to=date_to)
    points = _filter(CATALOG.get("thermopoints", []), query)
    if format == "json":
        return JSONResponse(points)
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
            "properties": {k: v for k, v in p.items() if k not in {"lon", "lat"}},
        }
        for p in points
    ]
    return JSONResponse({"type": "FeatureCollection", "features": features})


@app.get("/api/burn-contours")
def burn_contours(
    bbox: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    severity: str | None = None,
) -> dict:
    """Векторные контуры пройденной огнём площади с атрибутами."""

    query = Query(bbox=_parse_bbox(bbox), date_from=date_from, date_to=date_to,
                  severity=_parse_severity(severity))
    contours = _filter_contours(query)
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [c["ring"]]},
            "properties": {
                "id": c["id"],
                "severity": c["severity"],
                "area_ha": c["area_ha"],
                "chip_id": c["chip_id"],
                "date": c["date"],
            },
        }
        for c in contours
    ]
    return {"type": "FeatureCollection", "features": features}


@app.get("/api/report")
def report(
    bbox: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    severity: str | None = None,
    format: str = "json",
) -> Response:
    """Аналитическая справка: площадь гари и распределение по степеням поражения."""

    query = Query(bbox=_parse_bbox(bbox), date_from=date_from, date_to=date_to,
                  severity=_parse_severity(severity))
    contours = _filter_contours(query)
    points = _filter(CATALOG.get("thermopoints", []), query)
    data = _report(contours, points)
    data["query"] = {"bbox": query.bbox, "date_from": query.date_from, "date_to": query.date_to}

    if format == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["metric", "value"])
        writer.writerow(["total_area_ha", data["total_area_ha"]])
        for key, value in data["by_severity_ha"].items():
            writer.writerow([key, value])
        writer.writerow(["n_contours", data["n_contours"]])
        writer.writerow(["n_thermopoints", data["n_thermopoints"]])
        return Response(content=buffer.getvalue(), media_type="text/csv; charset=utf-8")
    return JSONResponse(data)


@app.post("/api/query")
def query_post(query: Query) -> dict:
    """Комбинированный запрос: термоточки + контуры + справка одним вызовом."""

    points = _filter(CATALOG.get("thermopoints", []), query)
    contours = _filter_contours(query)
    return {
        "query": query.model_dump(),
        "report": _report(contours, points),
        "thermopoints": points,
        "contours": [
            {k: v for k, v in c.items() if k != "ring"} | {"ring": c["ring"]} for c in contours
        ],
    }


@app.get("/api/export/contours.geojson")
def export_geojson(bbox: str | None = None, date_from: str | None = None, date_to: str | None = None) -> Response:
    """Выгрузка контуров в GeoJSON (файл, а не ответ API)."""

    data = burn_contours(bbox=bbox, date_from=date_from, date_to=date_to)
    payload = json.dumps(data, ensure_ascii=False)
    return Response(
        content=payload,
        media_type="application/geo+json",
        headers={"Content-Disposition": 'attachment; filename="burn_contours.geojson"'},
    )


@app.get("/api/export/contours.shp")
def export_shapefile(bbox: str | None = None, date_from: str | None = None, date_to: str | None = None) -> Response:
    """Выгрузка контуров в Shapefile (архив с .shp/.shx/.dbf/.prj).

    Реализовано через чистый Python-модуль ``pyshp``: он лёгкий и не требует GDAL, что важно
    для воспроизводимости на стороне организаторов.
    """

    try:
        import shapefile  # pyshp
    except ImportError:
        raise HTTPException(status_code=501, detail="Для Shapefile установите pyshp: pip install pyshp")

    import io as _io
    import zipfile

    contours = _filter_contours(Query(bbox=_parse_bbox(bbox), date_from=date_from, date_to=date_to))

    shp, shx, dbf = _io.BytesIO(), _io.BytesIO(), _io.BytesIO()
    writer = shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POLYGON)
    writer.field("id", "C")
    writer.field("severity", "N")
    writer.field("area_ha", "F", decimal=3)
    writer.field("chip_id", "C")
    writer.field("date", "C")
    for contour in contours:
        writer.poly([[[p[0], p[1]] for p in contour["ring"]]])
        writer.record(contour["id"], int(contour["severity"]), float(contour["area_ha"]),
                      contour["chip_id"], contour["date"])
    writer.close()

    prj = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
           'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')

    buffer = _io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("burn_contours.shp", shp.getvalue())
        zf.writestr("burn_contours.shx", shx.getvalue())
        zf.writestr("burn_contours.dbf", dbf.getvalue())
        zf.writestr("burn_contours.prj", prj)
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="burn_contours_shp.zip"'},
    )


def _parse_bbox(bbox: str | None) -> list[float] | None:
    if not bbox:
        return None
    try:
        values = [float(v) for v in bbox.split(",")]
    except ValueError:
        raise HTTPException(status_code=422, detail="bbox задаётся как min_lon,min_lat,max_lon,max_lat")
    if len(values) != 4:
        raise HTTPException(status_code=422, detail="bbox должен содержать 4 числа")
    return values


def _parse_severity(severity: str | None) -> list[int] | None:
    if not severity:
        return None
    try:
        return [int(v) for v in severity.replace(" ", "").split(",") if v]
    except ValueError:
        raise HTTPException(status_code=422, detail="severity задаётся списком классов, например severity=2,3")


# --------------------------------------------------------------------------- веб-карта

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>КосмоФАЙР-2 — карта термоточек и гарей</title>
<style>
 body{font-family:system-ui,'Segoe UI',Roboto,sans-serif;margin:0;background:#0e1116;color:#e6edf3}
 header{padding:12px 18px;background:#151b23;border-bottom:1px solid #2a3441;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 h1{font-size:17px;margin:0}
 .muted{color:#93a4b8;font-size:12.5px}
 main{display:grid;grid-template-columns:1fr 340px;gap:14px;padding:14px}
 .card{background:#151b23;border:1px solid #2a3441;border-radius:10px;padding:12px}
 canvas{width:100%;height:520px;background:#0b0f14;border-radius:8px;display:block}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{text-align:left;padding:5px 6px;border-bottom:1px solid #222c38}
 th{color:#93a4b8;font-weight:600}
 input,button,select{background:#0b0f14;color:#e6edf3;border:1px solid #2a3441;border-radius:6px;padding:5px 7px;font-size:12.5px}
 button{cursor:pointer;background:#1d2f3d}
 .pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;background:#1d2f3d;color:#9cd2ff}
 .s1{color:#d6a83a}.s2{color:#ff8f4a}.s3{color:#ff4a3c}
 a{color:#4fc3ff}
</style></head><body>
<header>
 <h1>🛰 КосмоФАЙР-2</h1>
 <span class="muted" id="mode"></span>
 <span class="muted">bbox</span><input id="bbox" size="26" value="38.5,45.5,47.5,53.5">
 <span class="muted">даты</span><input id="d1" size="10" value="2019-01-01"><input id="d2" size="10" value="2025-12-31">
 <button onclick="load()">Обновить</button>
 <a class="muted" href="api/export/contours.geojson">GeoJSON</a>
 <a class="muted" href="api/export/contours.shp">Shapefile</a>
 <a class="muted" href="api/report?format=csv">Справка CSV</a>
</header>
<main>
 <div class="card"><canvas id="map" width="1200" height="520"></canvas>
   <div class="muted" id="hint" style="margin-top:8px">Контуры гари по степеням поражения, точки — термоточки за период</div></div>
 <div>
  <div class="card" id="report"></div>
  <div class="card" style="margin-top:14px"><div class="muted" style="margin-bottom:6px">Контуры (топ-10 по площади)</div><div id="list"></div></div>
 </div>
</main>
<script>
const cv=document.getElementById('map'), ctx=cv.getContext('2d');
let view={};
function fit(){
  const bb=document.getElementById('bbox').value.split(',').map(Number);
  view={w:cv.width,h:cv.height,bb};
}
function proj(lon,lat){const [a,b,c,d]=view.bb;
  return [(lon-a)/(c-a)*view.w, view.h-(lat-b)/(d-b)*view.h];}
async function load(){
  fit();
  const p=new URLSearchParams({bbox:document.getElementById('bbox').value,
    date_from:document.getElementById('d1').value, date_to:document.getElementById('d2').value});
  const [contours,points,rep]=await Promise.all([
    fetch('api/burn-contours?'+p).then(r=>r.json()),
    fetch('api/thermopoints?'+p).then(r=>r.json()),
    fetch('api/report?'+p).then(r=>r.json())]);
  ctx.fillStyle='#0b0f14'; ctx.fillRect(0,0,cv.width,cv.height);
  ctx.strokeStyle='#141c25';
  for(let i=1;i<10;i++){ctx.beginPath();ctx.moveTo(i*cv.width/10,0);ctx.lineTo(i*cv.width/10,cv.height);ctx.stroke();
    ctx.beginPath();ctx.moveTo(0,i*cv.height/10);ctx.lineTo(cv.width,i*cv.height/10);ctx.stroke();}
  const colors={1:'rgba(214,168,58,0.55)',2:'rgba(255,143,74,0.6)',3:'rgba(255,74,60,0.65)'};
  contours.features.forEach(f=>{
    const ring=f.geometry.coordinates[0];
    ctx.beginPath();
    ring.forEach(([lon,lat],i)=>{const [x,y]=proj(lon,lat); i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
    ctx.closePath(); ctx.fillStyle=colors[f.properties.severity]||'rgba(200,200,200,0.4)'; ctx.fill();
    ctx.strokeStyle='rgba(255,255,255,0.35)'; ctx.lineWidth=1; ctx.stroke();
  });
  points.features.forEach(f=>{const [lon,lat]=f.geometry.coordinates; const [x,y]=proj(lon,lat);
    ctx.beginPath(); ctx.arc(x,y,2.6,0,6.283); ctx.fillStyle='#ff6b4a'; ctx.fill();});
  const sev=rep.by_severity_ha;
  document.getElementById('report').innerHTML=`<div class="muted" style="margin-bottom:8px">Аналитическая справка</div>
   <table>
    <tr><th>Суммарная площадь гари</th><td><b>${rep.total_area_ha} га</b></td></tr>
    <tr><th class="s1">Слабая</th><td>${sev.severity_1} га (${(rep.share_by_severity.severity_1*100).toFixed(1)}%)</td></tr>
    <tr><th class="s2">Средняя</th><td>${sev.severity_2} га (${(rep.share_by_severity.severity_2*100).toFixed(1)}%)</td></tr>
    <tr><th class="s3">Сильная</th><td>${sev.severity_3} га (${(rep.share_by_severity.severity_3*100).toFixed(1)}%)</td></tr>
    <tr><th>Термоточек</th><td>${rep.n_thermopoints}</td></tr>
    <tr><th>Контуров</th><td>${rep.n_contours}</td></tr>
   </table>`;
  const top=contours.features.map(f=>f.properties).sort((a,b)=>b.area_ha-a.area_ha).slice(0,10);
  document.getElementById('list').innerHTML='<table><tr><th>id</th><th>класс</th><th>площадь, га</th></tr>'+
    top.map(c=>`<tr><td>${c.id}</td><td class="s${c.severity}">${c.severity}</td><td>${c.area_ha}</td></tr>`).join('')+'</table>';
  document.getElementById('mode').textContent='каталог: '+(rep.catalog_mode==='geo'?'с геопривязкой':'демонстрационный');
}
fetch('api/health').then(r=>r.json()).then(h=>{ if(h.thermopoints===0){document.getElementById('hint').textContent='Каталог пуст: соберите его командой python -m service.catalog --data-dir <путь>';} });
load();
</script></body></html>"""


@app.get("/api/catalog/reload", include_in_schema=False)
def reload_catalog() -> dict:
    """Перечитывает каталог без перезапуска сервиса (удобно во время демонстрации)."""

    global CATALOG
    CATALOG = _load_catalog()
    return {"status": "reloaded", "catalog_mode": CATALOG.get("mode")}


def _today() -> str:  # pragma: no cover - вспомогательная
    return date.today().isoformat()
