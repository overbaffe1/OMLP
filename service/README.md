# Сервис «КосмоФАЙР-2»

Информационно-аналитический сервис по требованиям критериев раздела 4 постановки: приём пространственно-временного
запроса, карта термоточек, векторные контуры гари с атрибутами, аналитическая справка по площадям и выгрузки.

## Запуск

```bash
# 1) собрать каталог сцен из подготовленного набора (train или test без эталона)
python -m service.catalog --data-dir /path/to/data --out service/catalog.json \
       --weights weights/weights.json
# реальный набор организаторов: каталог собирается в координатах WGS84 прямо из meta.csv
python -m service.catalog --data-dir /tmp/firesdata/real/train --out service/catalog.json \
       --max-bs 30 --min-contour-ha 1.0

# 2) поднять сервис
CATALOG_PATH=service/catalog.json uvicorn service.app:app --host 0.0.0.0 --port 8000
# документация API: http://localhost:8000/docs
```

Каталог в репозитории собран именно по реальному обучающему набору: 9 478 термоточек AF и 4 892 контура гари
(30 чипов BS с наибольшей площадью, контуры от 1 га) в настоящих координатах — Нижнее Поволжье и Подонье.
Итого в справке 256 520 га гари: 92 675 / 107 248 / 56 597 га по степеням 1 / 2 / 3.

## Методы

| Метод | Параметры | Что возвращает |
|---|---|---|
| `GET /` | — | веб-карта (canvas, без CDN и тайлов) + справка |
| `GET /api/health` | — | состояние каталога, число точек и контуров |
| `GET /api/thermopoints` | `bbox`, `date_from`, `date_to`, `format=geojson\|json` | термоточки за период |
| `GET /api/burn-contours` | `bbox`, `date_from`, `date_to`, `severity=1,2,3` | контуры гари (GeoJSON) с id, классом, площадью |
| `GET /api/report` | `bbox`, `date_from`, `date_to`, `severity`, `format=json\|csv` | справка: га всего и по степеням |
| `POST /api/query` | JSON `{bbox | polygon, date_from, date_to, severity}` | точки + контуры + справка одним вызовом |
| `GET /api/export/contours.geojson` | `bbox`, `date_from`, `date_to` | файл GeoJSON |
| `GET /api/export/contours.shp` | то же | zip c .shp/.shx/.dbf/.prj (pyshp, без GDAL) |
| `GET /api/catalog/reload` | — | перечитать каталог без перезапуска |

Примеры:

```bash
# реальный AOI каталога — Нижнее Поволжье и Подонье
curl 'http://localhost:8000/api/report?bbox=39,46,45,51&date_from=2022-06-01&date_to=2022-09-30'
curl -X POST http://localhost:8000/api/query -H 'Content-Type: application/json' \
     -d '{"polygon": [[39,46],[45,46],[45,51],[39,51],[39,46]], "severity": [2,3]}'
curl -OJ 'http://localhost:8000/api/export/contours.shp'
```

## Геопривязка

Каталог сам определяет, есть ли геопривязка, и работает в одном из двух режимов:

* **geo** — если в `meta.csv` есть координаты чипа (`epsg`, `x_min`, `y_min`, `gsd`), как в обучающей части кейса:
  пиксель переводится UTM → WGS84 модулем `fires/geo.py` (реализация рядов Крагера без pyproj и GDAL, точность
  1e-7 градуса). Резервный вариант — файл `chips_geo.csv` (`chip_id,origin_lon,origin_lat,gsd,date,kind`);
* **demo** — если привязка скрыта (тестовая часть), чипы раскладываются на регулярную сетку в демонстрационном
  регионе; режим явно возвращается в `GET /api/report` (`catalog_mode`) и подписан в веб-карте.

Из-за большого числа мелких контуров каталог собирается с ограничениями `--max-bs` (сколько BS-чипов брать) и
`--min-contour-ha` (минимальная площадь контура): полный набор дал бы десятки тысяч объектов и файл в десятки
мегабайт, что для демонстрации избыточно.

Ответ `submission.csv` от режима каталога не зависит: он считается по маскам чипов.

## Площади

Площадь считается по числу пикселей маски и размеру пикселя (для BS — 20 м, 400 м² на пиксель), поэтому числа в
справке воспроизводимы по выданным данным, как требует критерий. Для произвольных полигонов есть расчёт по формуле
Гаусса с поправкой на широту (`fires/vectorize.py:polygon_area_ha`).
