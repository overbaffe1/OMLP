"""Чтение чипов: автоопределение раскладки файлов и каналов.

Организаторы не публикуют точные имена файлов, поэтому модуль:

1. находит ``meta.csv`` и ``sample_submission.csv`` в корне набора;
2. для каждого ``chip_id`` собирает файлы, начинающиеся с этого идентификатора;
3. распознаёт канал/слой по имени файла (``I1``…``I5``, ``B2``…``B12``, ``B8A``, ``SCL``,
   ``VV``, ``VH``, ``DEM``, ``landcover``, ``valid``, ``slope``, ``mask``, ``mask_sev``,
   ``pre``/``post``, ``t1``/``t2`` …);
4. поддерживает два варианта поставки: отдельный файл на канал и многоканальный стек,
   где порядок каналов соответствует таблицам постановки.

Диагностика раскладки доступна командой ``python -m fires.chips --data-dir <путь>``:
она печатает, какие каналы найдены для каждого типа чипа. Это первое, что нужно
запустить на реальных данных организаторов.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

#: Порядок каналов в многоканальном стеке (таблицы постановки).
AF_STACK_ORDER = ("I1", "I2", "I3", "I4", "I5")
BS_STACK_ORDER = ("B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12", "SCL")

#: Раскладка организаторов (проверена на обучающей части, 18.09.2026).
#: AF: одна папка на слой, но сами слои — многоканальные стеки:
#:   ``AF_tr_000001_VIIRS_I1-I5.tif`` — I1, I2, I3 (отражение), I4, I5 (K),
#:       зенитный угол Солнца, зенитный угол сенсора, маска валидности;
#:   ``AF_tr_000001_AUX.tif`` — тип покрова WorldCover, высота DEM, температура
#:       приземного воздуха (K), относительная влажность (%), скорость ветра (м/с);
#:   ``AF_tr_000001_MASK.tif`` — бинарный эталон (только обучающая часть).
#: BS:
#:   ``BS_tr_000001_Sentinel-2_pre.tif`` / ``_post.tif`` — 10 каналов B2…B12, SCL;
#:   ``BS_tr_000001_Sentinel-1_pre.tif`` / ``_post.tif`` — VV, VH (дБ ×100, int16);
#:   ``BS_tr_000001_AUX.tif`` — DEM, идентификатор исходной сцены, тип покрова;
#:   ``BS_tr_000001_MASK.tif`` — степени поражения 0–3.
#: Плюс ``meta.csv`` в корне и/или в подпапках (af/, bs/).
AF_VIIRS_STACK = ("I1", "I2", "I3", "I4", "I5", "sun_zenith", "sensor_zenith", "valid")
AF_AUX_STACK = ("landcover", "DEM", "temperature_2m", "rel_humidity", "wind_speed")
BS_S1_STACK = ("VV", "VH")
BS_AUX_STACK = ("DEM", "scene_id", "landcover")

#: Идентификатор чипа: ``AF_tr_000001``, ``BS_te_000123``, ``af_train_0000``.
CHIP_ID_RE = re.compile(r"^(?:af|bs)_[a-z]{2,5}_\d+", re.I)

#: Регулярные выражения распознавания слоёв. Порядок важен: более специфичные — раньше.
LAYER_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("B8A", re.compile(r"(?:^|[^a-z0-9])b8a(?:[^a-z0-9]|$)", re.I)),
    ("B12", re.compile(r"(?:^|[^a-z0-9])b12(?:[^a-z0-9]|$)", re.I)),
    ("B11", re.compile(r"(?:^|[^a-z0-9])b11(?:[^a-z0-9]|$)", re.I)),
    ("SCL", re.compile(r"(?:^|[^a-z0-9])scl(?:[^a-z0-9]|$)", re.I)),
    ("B2", re.compile(r"(?:^|[^a-z0-9])b0?2(?:[^a-z0-9]|$)", re.I)),
    ("B3", re.compile(r"(?:^|[^a-z0-9])b0?3(?:[^a-z0-9]|$)", re.I)),
    ("B4", re.compile(r"(?:^|[^a-z0-9])b0?4(?:[^a-z0-9]|$)", re.I)),
    ("B5", re.compile(r"(?:^|[^a-z0-9])b0?5(?:[^a-z0-9]|$)", re.I)),
    ("B6", re.compile(r"(?:^|[^a-z0-9])b0?6(?:[^a-z0-9]|$)", re.I)),
    ("B7", re.compile(r"(?:^|[^a-z0-9])b0?7(?:[^a-z0-9]|$)", re.I)),
    ("I1", re.compile(r"(?:^|[^a-z0-9])i0?1(?:[^a-z0-9]|$)", re.I)),
    ("I2", re.compile(r"(?:^|[^a-z0-9])i0?2(?:[^a-z0-9]|$)", re.I)),
    ("I3", re.compile(r"(?:^|[^a-z0-9])i0?3(?:[^a-z0-9]|$)", re.I)),
    ("I4", re.compile(r"(?:^|[^a-z0-9])i0?4(?:[^a-z0-9]|$)", re.I)),
    ("I5", re.compile(r"(?:^|[^a-z0-9])i0?5(?:[^a-z0-9]|$)", re.I)),
    ("VV", re.compile(r"(?:^|[^a-z0-9])vv(?:[^a-z0-9]|$)", re.I)),
    ("VH", re.compile(r"(?:^|[^a-z0-9])vh(?:[^a-z0-9]|$)", re.I)),
    ("DEM", re.compile(r"(?:^|[^a-z0-9])(dem|elevation|height)(?:[^a-z0-9]|$)", re.I)),
    ("slope", re.compile(r"(?:^|[^a-z0-9])(slope|уклон)(?:[^a-z0-9]|$)", re.I)),
    ("aspect", re.compile(r"(?:^|[^a-z0-9])(aspect|экспозиц)(?:[^a-z0-9]|$)", re.I)),
    ("landcover", re.compile(r"(?:^|[^a-z0-9])(landcover|lulc|worldcover)(?:[^a-z0-9]|$)", re.I)),
    ("valid", re.compile(r"(?:^|[^a-z0-9])(valid|validity|quality)(?:[^a-z0-9]|$)", re.I)),
    ("mask_sev", re.compile(r"(?:^|[^a-z0-9])(mask_sev|sev_mask|severity)(?:[^a-z0-9]|$)", re.I)),
    ("mask", re.compile(r"(?:^|[^a-z0-9])(mask|target|label)(?:[^a-z0-9]|$)", re.I)),
]

#: Токены для колонок meta.csv, где встречается идентификатор канала.
META_ALIASES = {
    "chip_id": ("chip_id", "id", "chip"),
    "kind": ("kind", "type", "task"),
    "width": ("width", "w"),
    "height": ("height", "h"),
    "gsd": ("gsd", "resolution", "pixel_size"),
    "valid_frac": ("valid_frac",),
    "cloud_frac": ("cloud_frac", "cloud"),
    "acq_datetime": ("acq_datetime", "datetime", "acq_time", "date"),
    "date_pre": ("date_pre",),
    "date_post": ("date_post",),
}


def gapless(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff", ".tff"}


def read_raster(path: str | Path) -> np.ndarray:
    """Читает растр; для 3-канальных изображений возвращает первый канал.

    Возвращает 2D-массив (или 3D, если каналов больше одного и это многоканальный стек —
    тогда решение принимает вызывающий код). Геопривязка в тестовой части скрыта, поэтому
    она не требуется: координаты восстанавливаются сервисом из каталога сцен.
    """

    arr = tifffile.imread(str(path))
    # RGB-превью (uint8, каналы последней осью) читаем первым каналом: это картинка, а не данные.
    # Проверка типа важна: трёхканальный вспомогательный стек организаторов — int16, и его
    # терять нельзя (иначе от AUX остаётся только DEM).
    if arr.ndim == 3 and arr.dtype == np.uint8 and arr.shape[-1] in (3, 4) and min(arr.shape[:2]) > 8:
        arr = arr[..., 0]
    return arr


@dataclass
class Chip:
    """Один чип: словари слоёв, метаданные и (для train) эталонные маски."""

    chip_id: str
    kind: str
    layers: dict[str, np.ndarray] = field(default_factory=dict)
    pre: dict[str, np.ndarray] = field(default_factory=dict)
    post: dict[str, np.ndarray] = field(default_factory=dict)
    s1_pre: dict[str, np.ndarray] = field(default_factory=dict)
    s1_post: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    mask: np.ndarray | None = None
    mask_sev: np.ndarray | None = None

    @property
    def height(self) -> int:
        for value in list(self.layers.values()) + list(self.pre.values()) + list(self.post.values()):
            return int(value.shape[0])
        raise ValueError(f"{self.chip_id}: нет ни одного слоя")

    @property
    def width(self) -> int:
        for value in list(self.layers.values()) + list(self.pre.values()) + list(self.post.values()):
            return int(value.shape[1])
        raise ValueError(f"{self.chip_id}: нет ни одного слоя")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


#: Имена папок, по которым узнаётся каталог набора чипов.
_DATA_MARKERS = ("af", "bs")


def _has_chips(path: Path) -> bool:
    """Есть ли в каталоге сами чипы (папки подзадач или файлы чипов).

    Одной ``meta.csv`` мало: таблица состава лежит и в корне набора, где чипов ещё нет
    (архив организаторов разворачивается во вложенную папку ``train/train``).
    """

    if not path.is_dir():
        return False
    if any((path / name).is_dir() for name in _DATA_MARKERS):
        return True
    try:
        return any(child.is_file() and gapless(child) for child in path.iterdir())
    except OSError:
        return False


def _looks_like_dataset(path: Path) -> bool:
    """Слабый признак: в каталоге есть таблицы набора, но самих чипов может и не быть."""

    return path.is_dir() and ((path / "meta.csv").exists()
                              or (path / "sample_submission.csv").exists())


def _has_masks(path: Path) -> bool:
    """Есть ли в каталоге маски (признак обучающей части: по ним считается метрика)."""

    try:
        return next(itertools.islice(path.rglob("*mask*"), 1), None) is not None
    except OSError:
        return False


def _matches(path: Path, prefer: str | None) -> bool:
    """Подходит ли каталог под предпочтение: маски (обучение) или шаблон ответа (тест)."""

    if prefer is None:
        return True
    if prefer == "masks":
        return _has_masks(path)
    if prefer == "submission":
        return (path / "sample_submission.csv").exists()
    raise ValueError(f"неизвестное предпочтение каталога: {prefer}")


def find_data_dir(root: str | Path, max_depth: int = 2, prefer: str | None = None) -> Path:
    """Ищет каталог с чипами рядом с указанным путём.

    Участники и организаторы кладут набор по-разному: ``cosmo1/train``, ``cosmo1/train/train``
    (архив разворачивается во вложенную папку), ``cosmo1`` целиком. Если в указанном каталоге
    чипов нет, функция смотрит на подпапки до ``max_depth`` уровней вглубь и возвращает первую,
    где чипы нашлись. ``prefer`` уточняет выбор, когда подходящих каталогов несколько:
    ``"masks"`` — обучающая часть (для обучения нужны маски), ``"submission"`` — тестовая (по
    ``sample_submission.csv`` собирается шаблон ответа). Если ничего не нашлось, возвращается
    исходный путь: решение обязано сказать об ошибке раскладки, а не молча работать на пустом
    наборе.
    """

    root = Path(root)
    # каталог, который и так похож на набор (есть таблицы организаторов), не «исправляем»:
    # Dataset индексирует файлы рекурсивно, поэтому корень набора подходит и когда чипы лежат
    # в подпапках вида train/ и test/
    if _matches(root, prefer) and (_has_chips(root) or _looks_like_dataset(root)):
        return root
    strong = root if _has_chips(root) else None
    queue = [(root, 0)]
    seen: set[Path] = set()
    weak: Path | None = root if _looks_like_dataset(root) else None
    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth or current in seen:
            continue
        seen.add(current)
        try:
            children = sorted(child for child in current.iterdir() if child.is_dir())
        except OSError:
            continue
        for child in children:
            if child.name.startswith(".") or child.name in {"__pycache__", "node_modules"}:
                continue
            if _has_chips(child):
                if _matches(child, prefer):
                    return child
                if strong is None:
                    strong = child
            elif weak is None and _looks_like_dataset(child):
                weak = child
            queue.append((child, depth + 1))
    # каталог с чипами лучше каталога с одними таблицами, а тот — лучше исходного пути
    return strong or weak or root


class Dataset:
    """Набор чипов: индексация по ``meta.csv`` или по именам файлов."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta: dict[str, dict] = self._read_meta()
        self.files: dict[str, list[Path]] = {}
        self._index_files()

    def _read_meta(self) -> dict[str, dict]:
        """Читает ``meta.csv`` — в корне набора и/или в подпапках (``af/``, ``bs/``).

        В обучающей части организаторов метаданные лежат по одной таблице на подзадачу;
        колонки дополняют друг друга, поэтому таблицы объединяются по ``chip_id``.
        """

        import csv

        out: dict[str, dict] = {}
        for path in sorted(self.root.rglob("meta.csv")):
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    chip_id = row.get("chip_id") or row.get("id") or row.get("chip")
                    if not chip_id:
                        continue
                    chip_id = str(chip_id)
                    out.setdefault(chip_id, {}).update({k: v for k, v in row.items() if v not in (None, "")})
        return out

    def _index_files(self) -> None:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or not gapless(path):
                continue
            name = path.stem
            for meta_id in sorted(self.meta, key=len, reverse=True):
                if name.startswith(meta_id):
                    self.files.setdefault(meta_id, []).append(path)
                    break
            else:
                match = CHIP_ID_RE.match(name)
                if match:
                    self.files.setdefault(match.group(0), []).append(path)
                    continue
                # чип без meta.csv: идентификатор — часть имени до последнего токена слоя
                stem = re.sub(r"_(pre|post|s1|mask|mask_sev|valid|scl|landcover|dem|slope|aspect|i[1-5]|b\d+a?)$", "", name, flags=re.I)
                self.files.setdefault(stem, []).append(path)

    def chip_ids(self, kind: str | None = None) -> list[str]:
        ids = sorted(self.files)
        if kind is None:
            return ids
        out = []
        for chip_id in ids:
            k = self.kind_of(chip_id)
            if k == kind:
                out.append(chip_id)
        return out

    def kind_of(self, chip_id: str) -> str:
        meta = self.meta.get(chip_id, {})
        kind = str(meta.get("kind") or "").lower()
        if kind in {"af", "bs"}:
            return kind
        return "af" if chip_id.lower().split("_")[0].endswith("af") else ("bs" if "bs" in chip_id.lower() else "af")

    def _classify(self, path: Path) -> tuple[str, str | None]:
        """Возвращает (слой, дата) где дата ∈ {None, 'pre', 'post'}."""

        stem = path.stem
        suffix = stem
        for token, date in (("_pre", "pre"), ("_post", "post"), ("_t1", "pre"), ("_t2", "post"),
                            ("_before", "pre"), ("_after", "post")):
            if token in stem.lower():
                suffix = re.sub(token, "", stem, flags=re.I)
                date = date
                break
        else:
            date = None
        for name, pattern in LAYER_PATTERNS:
            if pattern.search(suffix):
                return name, date
        return "unknown", date

    @staticmethod
    def _descriptor(stem: str, chip_id: str) -> str:
        """Часть имени файла после идентификатора чипа: ``AF_tr_1_AUX`` → ``aux``."""

        rest = stem[len(chip_id):] if stem.lower().startswith(chip_id.lower()) else stem
        return rest.strip("_- .").lower()

    @staticmethod
    def _assign_stack(chip: Chip, target: dict, arr: np.ndarray, order: tuple[str, ...]) -> None:
        """Раскладывает многоканальный стек по именам слоёв (без перезаписи)."""

        stack = arr if arr.ndim == 3 else arr[..., None]
        for index, name in enumerate(order):
            if index < stack.shape[2]:
                target.setdefault(name, stack[..., index])

    def load_case_layout(self, chip: Chip, descriptor: str, path: Path) -> bool:
        """Распознаёт раскладку организаторов (многоканальные стеки).

        Возвращает ``True``, если файл обработан здесь; ``False`` — если это обычный
        одноканальный файл, и его должен разобрать общий путь :meth:`load`.
        """

        kind = chip.kind
        if "viirs" in descriptor:
            self._assign_stack(chip, chip.layers, read_raster(path), AF_VIIRS_STACK)
            return True
        if "sentinel-2" in descriptor:
            date = "post" if "post" in descriptor else "pre"
            target = chip.post if date == "post" else chip.pre
            arr = read_raster(path)
            stack = arr if arr.ndim == 3 else arr[..., None]
            index = BS_STACK_ORDER.index("SCL") if "SCL" in BS_STACK_ORDER else -1
            # Стек даты разбирается без SCL: это служебный слой (облака/тень/вода), а не
            # канал отражательной способности. Обе даты хранятся отдельно — облако портит
            # dNBR с любой стороны.
            reflectance = tuple(name for name in BS_STACK_ORDER if name != "SCL")
            self._assign_stack(chip, target, arr, reflectance)
            if 0 <= index < stack.shape[2]:
                chip.layers.setdefault("SCL" if date == "post" else "SCL_pre", stack[..., index])
            return True
        if "sentinel-1" in descriptor:
            date = "post" if "post" in descriptor else "pre"
            target = chip.s1_post if date == "post" else chip.s1_pre
            self._assign_stack(chip, target, read_raster(path), BS_S1_STACK)
            return True
        if "aux" in descriptor:
            order = AF_AUX_STACK if kind == "af" else BS_AUX_STACK
            self._assign_stack(chip, chip.layers, read_raster(path), order)
            return True
        if "mask" in descriptor:
            arr = read_raster(path)
            arr = np.asarray(arr)
            if arr.ndim == 3:
                arr = arr[..., 0]
            if kind == "bs":
                # эталон BS — степени поражения 0–3; бинарная маска «гарь/не гарь» выводится из неё
                chip.mask_sev = arr.astype(np.uint8)
                chip.mask = arr.astype(np.uint8) > 0
            else:
                chip.mask = arr.astype(bool)
            return True
        return False

    def load(self, chip_id: str) -> Chip:
        paths = self.files.get(chip_id, [])
        if not paths:
            raise KeyError(f"Чип {chip_id} не найден в {self.root}")
        kind = self.kind_of(chip_id)
        chip = Chip(chip_id=chip_id, kind=kind, meta=self.meta.get(chip_id, {}))
        stacks: list[np.ndarray] = []

        for path in paths:
            descriptor = self._descriptor(path.stem, chip_id)
            if self.load_case_layout(chip, descriptor, path):
                continue
            layer, date = self._classify(path)
            arr = read_raster(path)
            if arr.ndim == 3 and arr.shape[0] <= 12:  # (C, H, W) — многоканальный стек
                stacks.append(arr)
                continue
            if layer == "unknown":
                continue
            if kind == "af" and date is None:
                chip.layers[layer] = arr
            elif kind == "bs":
                if layer in {"SCL", "landcover", "valid", "slope", "aspect", "DEM"}:
                    chip.layers[layer] = arr
                elif layer in {"VV", "VH"} and date in {"pre", "post"}:
                    (chip.s1_pre if date == "pre" else chip.s1_post)[layer] = arr
                elif date in {"pre", "post"}:
                    (chip.pre if date == "pre" else chip.post)[layer] = arr
                elif date is None and layer in BS_STACK_ORDER:
                    chip.layers[layer] = arr
            if layer == "mask":
                chip.mask = arr
            elif layer == "mask_sev":
                chip.mask_sev = arr

        for stack in stacks:
            if kind == "af":
                for i, name in enumerate(AF_STACK_ORDER):
                    if i < stack.shape[0]:
                        chip.layers.setdefault(name, stack[i])
            else:
                for i, name in enumerate(BS_STACK_ORDER):
                    if i < stack.shape[0]:
                        # В реальной поставке Sentinel-2 приходит одним стеком (B2…B12 + SCL):
                        # SCL — служебный слой, он идёт в layers, а не в стеки дат.
                        if name == "SCL":
                            chip.layers.setdefault("SCL", stack[i])
                        else:
                            chip.layers.setdefault(name, stack[i])

        return chip

    def describe(self, max_chips: int = 2) -> dict:
        """Диагностика раскладки: какие каналы реально найдены."""

        out: dict = {"root": str(self.root), "n_meta": len(self.meta), "n_chips": len(self.files), "kinds": {}}
        for kind in ("af", "bs"):
            ids = self.chip_ids(kind)
            out["kinds"][kind] = {"count": len(ids), "examples": []}
            for chip_id in ids[:max_chips]:
                chip = self.load(chip_id)
                out["kinds"][kind]["examples"].append(
                    {
                        "chip_id": chip_id,
                        "layers": sorted(chip.layers),
                        "pre": sorted(chip.pre),
                        "post": sorted(chip.post),
                        "s1_pre": sorted(chip.s1_pre),
                        "s1_post": sorted(chip.s1_post),
                        "mask": None if chip.mask is None else list(chip.mask.shape),
                        "mask_sev": None if chip.mask_sev is None else list(chip.mask_sev.shape),
                        "shape": list(chip.shape),
                    }
                )
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Диагностика раскладки чипов организаторов")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-chips", type=int, default=2)
    args = ap.parse_args(argv)
    print(json.dumps(Dataset(args.data_dir).describe(args.max_chips), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
