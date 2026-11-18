# Запуск с нуля (машина без ничего)

Полный порядок с таймингами и проверками — [`RUNBOOK.md`](RUNBOOK.md). Кратко:

1. **Python 3.11** (работает и на 3.10): https://www.python.org/downloads/ — при установке отметить *Add to PATH*.
2. **Батник** (двойной клик или из консоли):
   - `LAUNCH.bat` — окружение + зависимости + 82 теста;
   - `LAUNCH.bat "C:\путь\к\test"` — то же + диагностика данных + полный прогон в `submission_fresh.csv`.
3. Данные (`fire-test-renamed.tar` / `fire-train-renamed.tar`) лежат **вне репозитория**; внутри архива каталог `test/`
   (или `train/`) с `af/`, `bs/`, `meta.csv`, `sample_submission.csv` — этот каталог и есть `--data-dir`.
4. Где появляется ответ: `submission_fresh.csv` (или путь из `--output`) — рядом с точкой запуска.

## Что где находится

| Путь | Что это |
|---|---|
| `README.md` | описание решения, команды, архитектура, результаты |
| `inference.py` | обязательный CLI постановки: `python inference.py --data-dir ... --output submission.csv --validate` |
| `train.py` | обучение (пороги + модель степени поражения) |
| `evaluate.py` | метрика соревнования + абляции |
| `weights/` | `weights.json` (пороги, режимы) + `severity_model.pkl` (модель) |
| `fires/` | все модули конвейера (AF, BS, severity, RLE, гео) |
| `service/` | информационно-аналитический сервис (FastAPI) |
| `tools/make_html_map.py` | генератор интерактивной карты `map.html` |
| `map.html` | интерактивная карта теста (89 чипов, кисть 1/2/3, экспорт ответа) |
| `LAUNCH.bat` | старт с нуля одним кликом |
| `RUNBOOK.md` | полный порядок запуска от установки до сабмита |
| `docs/report.md` | отчёт (структура по паспорту кейса) |
| `docs/presentation.pdf|html` | презентация (13 слайдов) |
| `docs/figures/` | фигуры отчёта и презентации (строятся кодом решения) |
| `tests/` | 82 теста (окружение, конвейер, RLE, метрика, сервис) |
| `Dockerfile` | предпочтительный способ поставки по паспорту |
| `submission_probe_35.csv` | лучший конкурсный сабмит (Score 0.7455) |
| `PROBES_UPLOAD.md` | журнал всех сабмитов и их скоров |

## Сервис (паспорт: карта, контуры, справка, REST)

```bash
python -m service.catalog --data-dir /path/to/train --out service/catalog.json
CATALOG_PATH=service/catalog.json uvicorn service.app:app --host 0.0.0.0 --port 8000
# → http://localhost:8000  (карта, справка, GeoJSON/Shapefile, POST /api/query)
```

## Воспроизводимость

Все случайности зафиксированы (`SEED = 20260918`); повторный прогон даёт побайтово одинаковый CSV
(требование паспорта: разброс метрики не более 0.005 — у решения разброс нулевой).
