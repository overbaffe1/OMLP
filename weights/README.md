# Веса и конфигурации (`weights/`)

В каталоге сохранены проверенные конфигурации конвейера, разделённые по ролям:

| Файл | Модель | Роль и метрика |
|---|---|---|
| **`weights_probe_35.json`** | **`severity_model_p35.pkl`** | **Конфигурация-чемпион (`Score 0.7455` на платформе):** `HistGradientBoostingClassifier` на 23 признаках (CPU), один проход фильтра большинства `3 × 3`, `min_patch_px = 96`, `af.k_sigma = 2.5`, `SCL_BAD = (3, 8, 9, 11)`. |
| `weights.json` | `severity_model.pkl` | Базовая конфигурация `v1` (`Score 0.7391` на платформе): та же модель на 23 признаках без `majority 3 × 3`, `min_patch_px = 48`, `af.k_sigma = 3.0`. Используется по умолчанию в исторических таблицах `docs/report.md` и модульных тестах. |
| `weights_probe_24.json` | `severity_model.pkl` | Промежуточный шаг (`Score 0.7393`): `min_patch_px = 96` и `af.k_sigma = 2.5` до включения `majority 3 × 3`. |
| `weights.fullcycle.json` | — (метаданные) | Задокументированный эксперимент `v2` (40 признаков + бэггинг + бинарная голова): вырос на локальном `val`, но дал `0.7357` на платформе и был отклонён (см. [`docs/experiments.md`](../docs/experiments.md)). |
| `blob_filter.json` / `.pkl` | `blob_filter.pkl` | Обученный ML-фильтр кандидатных блобов AF (23 признака); на валидации уступил физическим правилам (`0.8980` против `0.9225`), поэтому в рабочих конфигах задано `"af_mode": "rules"`. |

## Запуск инференса с конфигурацией-чемпионом

```bash
python inference.py \
  --data-dir /path/to/test \
  --weights weights/weights_probe_35.json \
  --output submission.csv \
  --validate
```
