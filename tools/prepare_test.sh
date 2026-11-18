#!/usr/bin/env bash
# Готовит тестовую часть к инференсу: распаковывает архив и приводит раскладку к виду
#
#   test_data/
#     af/ bs/  meta.csv  sample_submission.csv
#
#   bash tools/prepare_test.sh fire-test-renamed.tar              # → ./test_data
#   bash tools/prepare_test.sh /путь/fire-test-renamed.rar /data/test
#
# Скрипт безопасен для повторного запуска: каталог назначения создаётся заново.
# Кладите тест ВНЕ репозитория (или в каталог из .gitignore) — распакованные данные в git не нужны.

set -euo pipefail

ARCHIVE="${1:-}"
DEST="${2:-test_data}"
UNRAR_BIN="${UNRAR_BIN:-/tmp/rar/unrar}"

if [ -z "$ARCHIVE" ] || [ ! -f "$ARCHIVE" ]; then
  echo "Укажите файл архива: bash tools/prepare_test.sh <архив.tar|.rar|.zip> [куда распаковать]" >&2
  exit 1
fi

echo "== 1/3 распаковываю $ARCHIVE → $DEST"
rm -rf "$DEST"
mkdir -p "$DEST"
case "$ARCHIVE" in
  *.tar)  tar xf "$ARCHIVE" -C "$DEST" ;;
  *.tgz|*.tar.gz) tar xzf "$ARCHIVE" -C "$DEST" ;;
  *.zip)  unzip -q "$ARCHIVE" -d "$DEST" ;;
  *.rar)
    if [ ! -x "$UNRAR_BIN" ]; then
      echo "нужен unrar: соберите его (make в исходниках unrar) и повторите, либо используйте tar/unzip" >&2
      exit 1
    fi
    "$UNRAR_BIN" x -o+ -ep "$ARCHIVE" "$DEST"/ >/dev/null
    ;;
  *) echo "неизвестный формат архива: $ARCHIVE" >&2; exit 1 ;;
esac

echo "== 2/3 привожу раскладку к ожидаемой"
# Архивы организаторов кладут данные в подкаталог (test/ или train/) — поднимаем содержимое наверх.
for candidate in "$DEST"/*; do
  [ -d "$candidate" ] || continue
  if [ -f "$candidate/sample_submission.csv" ] || { [ -d "$candidate/af" ] && [ -d "$candidate/bs" ]; }; then
    echo "   подкаталог найден: $(basename "$candidate") — переношу содержимое в $DEST"
    for inner in "$candidate"/*; do mv "$inner" "$DEST"/; done
    rmdir "$candidate"
    break
  fi
done

missing=()
[ -d "$DEST/af" ] || missing+=("af/")
[ -d "$DEST/bs" ] || missing+=("bs/")
[ -f "$DEST/meta.csv" ] || missing+=("meta.csv")
[ -f "$DEST/sample_submission.csv" ] || missing+=("sample_submission.csv")
if [ "${#missing[@]}" -gt 0 ]; then
  echo "   ! в $DEST нет: ${missing[*]}" >&2
  echo "   фактическое содержимое:" >&2
  find "$DEST" -maxdepth 2 | head -20 >&2
  exit 1
fi

echo "== 3/3 проверка"
rows=$(( $(wc -l < "$DEST/sample_submission.csv") - 1 ))
echo "   каталоги каналов: AF $(ls -1 "$DEST/af" | tr '\n' ' ') | BS $(ls -1 "$DEST/bs" | tr '\n' ' ')"
echo "   строк в sample_submission.csv: $rows"
echo "   масок в наборе: $(ls -1 "$DEST"/af/masks 2>/dev/null | wc -l) AF + $(ls -1 "$DEST"/bs/masks 2>/dev/null | wc -l) BS (в тесте их не должно быть)"
echo "   размер: $(du -sh "$DEST" | cut -f1)"
echo
echo "Дальше:"
echo "  python -m fires.chips --data-dir $DEST"
echo "  python inference.py --data-dir $DEST --output submission.csv --validate"
