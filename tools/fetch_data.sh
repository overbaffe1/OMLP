#!/usr/bin/env bash
# Загрузка и распаковка набора организаторов (КРС, кейс «пожары»).
#
#   bash tools/fetch_data.sh                 # скачать части и распаковать в /tmp/firesdata/real/train
#   DEST=/data/fires bash tools/fetch_data.sh
#
# Почему так, а не «просто скачать по ссылке»: официальное зеркало лежит на Яндекс.Диске и
# требует браузера, а те же самые архивы лежат обычными blobs в репозитории команды
# (main, файлы «Мониторинг DATA.partNNN.rar»). Скрипт забирает только их (sparse-checkout),
# поэтому не тянет историю и остальные файлы репозитория.
#
# Требуется: git, tar, make, g++ и ~6 ГБ свободного места (2.2 ГБ части + 2.2 ГБ распаковка).

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/overbaffe1/CosmoH.git}"
WORK="${WORK:-/tmp/cosmoh-data}"
PARTS="${PARTS:-$WORK/parts}"
DEST="${DEST:-/tmp/firesdata/real/train}"
RAW="${RAW:-/tmp/firesdata/raw}"

echo "== 1/4 клонируем только архивы из $REPO_URL"
mkdir -p "$WORK"
if [ ! -d "$PARTS/.git" ]; then
  git clone --filter=blob:none --no-checkout "$REPO_URL" "$PARTS"
fi
cd "$PARTS"
git sparse-checkout init --no-cone >/dev/null 2>&1 || true
git sparse-checkout set '/*.rar' >/dev/null
git checkout main
echo "   файлов частей: $(ls -1 *.rar 2>/dev/null | wc -l), объём: $(du -sh . | cut -f1)"

echo "== 2/4 собираем unrar (в дистрибутивах он часто отсутствует)"
UNRAR_BIN="${UNRAR_BIN:-/tmp/rar/unrar}"
if [ ! -x "$UNRAR_BIN" ]; then
  mkdir -p "$(dirname "$UNRAR_BIN")" /tmp/unrar-src
  if [ ! -f /tmp/unrar-src/unrar.hpp ]; then
    curl -fsSL https://codeload.github.com/pmachapman/unrar/tar.gz/refs/heads/master \
      | tar xz -C /tmp/unrar-src --strip-components=1
  fi
  make -C /tmp/unrar-src -j"$(nproc)" >/dev/null
  cp /tmp/unrar-src/unrar "$UNRAR_BIN"
fi
echo "   unrar: $("$UNRAR_BIN" 2>&1 | head -2 | tr '\n' ' ')"

echo "== 3/4 распаковываем внешний архив (сборка частей идёт автоматически)"
mkdir -p "$RAW"
first_part="$(ls -1 "$PARTS"/*part001.rar "$PARTS"/*part1.rar 2>/dev/null | head -1 || true)"
if [ -z "$first_part" ]; then
  echo "   ! не найдена первая часть, доступные файлы:"; ls -1 "$PARTS" | head -20; exit 1
fi
"$UNRAR_BIN" x -o+ -ep "$first_part" "$RAW"/ >/dev/null
echo "   в $RAW: $(ls -1 "$RAW" | head -5 | tr '\n' ' ')..."

echo "== 4/4 распаковываем внутренние архивы в $DEST"
mkdir -p "$DEST"
cd "$RAW"
for item in *; do
  case "$item" in
    *.tar.gz|*.tgz) tar xzf "$item" -C "$DEST" ;;
    *.tar) tar xf "$item" -C "$DEST" ;;
    *.zip) unzip -q -o "$item" -d "$DEST" ;;
    *.rar) "$UNRAR_BIN" x -o+ -ep "$item" "$DEST"/ >/dev/null ;;
    *) echo "   ? пропускаю $item (не архив или уже распакован)" ;;
  esac
done

# Внутренний tar содержит каталог верхнего уровня (train/…), поэтому при необходимости
# поднимаем af/ и bs/ на уровень вверх — решение ожидает их прямо в --data-dir.
if [ ! -d "$DEST/af" ] && [ -d "$DEST/train/af" ]; then
  echo "   поднимаю af/ и bs/ из $DEST/train"
  mv "$DEST"/train/* "$DEST"/ && rmdir "$DEST/train"
fi

echo "== готово"
find "$DEST" -maxdepth 2 -type d | head -20
echo "   AF-чипов: $(ls -1 "$DEST"/af/masks 2>/dev/null | wc -l), BS-чипов: $(ls -1 "$DEST"/bs/masks 2>/dev/null | wc -l)"
echo "   проверка: python -m fires.chips --data-dir $DEST"
