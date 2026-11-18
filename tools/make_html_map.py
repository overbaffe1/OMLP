"""Интерактивная 3D-карта разметки v7: чип крупно (любой) + мозаика ВСЕХ BS-чипов.

Слои: спутник / SWIR / dNBR (как высоту), разметка: наша / истина (train) / обе.
Чип крупно — любой из 89 (список + листание, сортировка по номеру/гарю),
разрешения 128 / 256 / 512 (512 = оригинал датасета), источник: датасет / FIRMS (очаги).

Запуск::

    py -3 tools\\make_html_map.py --out map.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fires.chips import Dataset, find_data_dir  # noqa: E402
from fires import bs as bsm  # noqa: E402

csv.field_size_limit(2**31 - 1)

SEV_COLORS = {1: (255, 242, 0), 2: (255, 158, 27), 3: (227, 26, 28)}  # EFFIS: желт/оранж/красн
PER_ROW = 9
QUALITY = ["слабое", "среднее", "сильное"]


def to_png(arr: np.ndarray, jpeg: bool = False, quality: int = 85) -> str:
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    if jpeg:
        img.save(buf, "JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        img.save(buf, "PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def stretch(arr: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    out = np.zeros(a.shape, dtype=np.float32)
    for c in range(a.shape[2]):
        pl, ph = np.nanpercentile(a[:, :, c], (lo, hi))
        out[:, :, c] = np.clip((a[:, :, c] - pl) / max(ph - pl, 1e-6), 0, 1) * 255
    return out


def stretch_fixed(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.clip((np.asarray(arr, dtype=np.float32) - vmin) / (vmax - vmin), 0, 1) * 255


def block(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    step = max(1, int(round(max(h, w) / size)))
    arr = arr[: step * (h // step), : step * (w // step)]
    if arr.ndim == 3:
        gh, gw = arr.shape[0] // step, arr.shape[1] // step
        return arr.reshape(gh, step, gw, step, arr.shape[2]).mean(axis=(1, 3))
    gh, gw = arr.shape[0] // step, arr.shape[1] // step
    return arr.reshape(gh, step, gw, step).mean(axis=(1, 3))


def answer_rasters(chip_id: str, path: Path, shape: tuple) -> dict | None:
    """Все строки чипа сливаются в одну one-hot растровую маску по степеням."""
    h, w = shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    found = False
    for row in csv.reader(open(path, encoding="utf-8")):
        if not row or row[0] != chip_id or len(row) < 3 or not row[2].strip():
            continue
        if int(row[1]) == 1 and row[0].startswith("AF"):
            return None
        vals = [int(x) for x in row[2].split()]
        mask = np.zeros(h * w, dtype=np.uint8)
        for start, length in zip(vals[0::2], vals[1::2]):
            mask[start:start + length] = int(row[1])
        mask = mask.reshape(h, w)
        for cls in (1, 2, 3):
            out[:, :, cls - 1] = (mask == cls)
        found = True
    return out if found else None


def sev_rgba(soft: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros(soft.shape[:2] + (4,), dtype=np.float32)
    for cls, (r, g, b) in SEV_COLORS.items():
        if cls - 1 < soft.shape[2]:
            out[:, :, 0] += soft[:, :, cls - 1] * r
            out[:, :, 1] += soft[:, :, cls - 1] * g
            out[:, :, 2] += soft[:, :, cls - 1] * b
            out[:, :, 3] += soft[:, :, cls - 1] * 255 * alpha
    return out


def band(stack: dict, name: str) -> np.ndarray:
    return np.asarray(stack[name], dtype=np.float32)


def build_layers(chip) -> dict:
    """Слои чипа в разрешениях 512 (оригинал) / 256 / 128; всё считается в 512 и уменьшается."""
    post = {k: band(chip.post, k) for k in ("B2", "B3", "B4", "B8A", "B12")}
    true_full = np.dstack([post["B4"], post["B3"], post["B2"]])
    false_full = np.dstack([post["B12"], post["B8A"], post["B4"]])
    nbr_pre = bsm.nbr(bsm.scale_reflectance(chip.pre["B8A"]), bsm.scale_reflectance(chip.pre["B12"]))
    nbr_post = bsm.nbr(bsm.scale_reflectance(chip.post["B8A"]), bsm.scale_reflectance(chip.post["B12"]))
    dnbr_s = stretch_fixed(nbr_pre - nbr_post, -0.2, 0.8)
    true_s = stretch(true_full)
    false_s = stretch(false_full)
    out = {}
    for size in (512, 256, 128):
        q = 75 if size >= 256 else 82
        out[f"true{size}"] = to_png(block(true_s, size), jpeg=True, quality=q)
        out[f"false{size}"] = to_png(block(false_s, size), jpeg=True, quality=q)
        out[f"dnbr{size}"] = to_png(block(dnbr_s, size), jpeg=True, quality=q + 7)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/test", help="Каталог тестовых чипов")
    ap.add_argument("--data-dir-train", default="data/train", help="Каталог обучающих чипов")
    ap.add_argument("--submission", default="submissions/submission_probe_35.csv")
    ap.add_argument("--submission-train", default="submission_train.csv")
    ap.add_argument("--out", default="viewer/map.html")
    args = ap.parse_args(argv)

    submission = Path(args.submission)
    ds_test = Dataset(find_data_dir(args.data_dir, prefer="masks"))
    bs_ids = sorted(ds_test.chip_ids("bs"))
    print(f"BS тестовых чипов: {len(bs_ids)}", flush=True)

    tiles = []
    for idx, c in enumerate(bs_ids):
        chip = ds_test.load(c)
        pack = {"cid": c, "col": idx % PER_ROW, "row": idx // PER_ROW}
        pack.update(build_layers(chip))
        ans = answer_rasters(c, submission, chip.post["B8A"].shape)
        if ans is not None:
            pack["ans"] = to_png(sev_rgba(block(ans, 256), alpha=0.95))
            for cls in (1, 2, 3):
                pack[f"b{cls}"] = int((ans[:, :, cls - 1] > 0).sum())
            pack["b"] = int((ans.max(axis=-1) > 0).sum())
        tiles.append(pack)
        if idx % 10 == 0:
            print(f"    чипы: {idx + 1}/{len(bs_ids)}", flush=True)

    # фокус: два самых больших по гарю + один из середины
    order = sorted(((t.get("b", 0), t["cid"]) for t in tiles), reverse=True)
    focus = [c for _, c in order[:2]]
    if len(order) > 4 and order[len(order) // 2][1] not in focus:
        focus.append(order[len(order) // 2][1])
    focus_set = set(focus)

    singles = []
    for t in tiles:
        if t["cid"] not in focus_set:
            continue
        shape = (512, 512)
        ans = answer_rasters(t["cid"], submission, shape)
        pack = {"cid": t["cid"], "truth": False, "stats": {}, "ans": t.get("ans"),
                "true": t["true512"], "false": t["false512"], "dnbr": t["dnbr512"]}
        if ans is not None:
            for cls in (1, 2, 3):
                pack["stats"][f"наша {cls} {QUALITY[cls - 1]}"] = int((ans[:, :, cls - 1] > 0).sum())
            pack["stats"]["наша гарь (1-3)"] = int((ans.max(axis=-1) > 0).sum())
        singles.append(pack)

    if Path(args.data_dir_train).exists() and Path(args.submission_train).exists():
        ds_train = Dataset(find_data_dir(args.data_dir_train, prefer="masks"))
        top, top_px = None, -1
        for row in csv.reader(open(args.submission_train, encoding="utf-8")):
            if row and row[0].startswith("BS") and len(row) >= 3 and row[2].strip() and int(row[1]) >= 1:
                vals = [int(x) for x in row[2].split()]
                total = sum(vals[1::2])
                if total > top_px:
                    top, top_px = row[0], total
        if top:
            chip = ds_train.load(top)
            pack = {"cid": top, "truth": True, "stats": {}, **build_layers(chip)}
            pack["true"] = pack["true512"]
            pack["false"] = pack["false512"]
            pack["dnbr"] = pack["dnbr512"]
            ans = answer_rasters(top, Path(args.submission_train), chip.post["B8A"].shape)
            mask_sev = np.asarray(chip.mask_sev)
            gt_soft = np.stack([(mask_sev == c).astype(np.float32) for c in (1, 2, 3)], axis=-1)
            pack["gt"] = to_png(sev_rgba(gt_soft, alpha=0.7))
            if ans is not None:
                pack["ans"] = to_png(sev_rgba(ans, alpha=0.95))
                m = ans.max(axis=-1) > 0
                i = int((m & (mask_sev >= 1)).sum())
                u = int((m | (mask_sev >= 1)).sum())
                pack["stats"]["IoU гари"] = round(i / max(u, 1), 3)
            for cls in (1, 2, 3):
                pack["stats"][f"истина {cls} {QUALITY[cls - 1]}"] = int((mask_sev == cls).sum())
            pack["stats"]["истина гарь (1-3)"] = int((mask_sev >= 1).sum())
            singles.append(pack)

    data = json.dumps({"singles": singles, "tiles": tiles}, ensure_ascii=False)
    html = HTML.replace("__DATA__", data)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"готово: {args.out} ({Path(args.out).stat().st_size // 1024 // 1024} МБ, "
          f"чипов {len(tiles)}, фокус {[p['cid'] for p in singles]})")


HTML = """
<!doctype html><html><head><meta charset="utf-8">
<style>
body{margin:0;font:14px/1.4 system-ui,Segoe UI,sans-serif;background:#0d1117;color:#e6edf3;overflow:hidden}
#bar,#hint{position:relative;z-index:10}
canvas{position:fixed;left:0;top:0;z-index:0}
#bar{padding:8px 14px;background:#161b22;border-bottom:1px solid #30363d;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:5px 12px;cursor:pointer}
button.on{background:#1f6feb;border-color:#388bfd;font-weight:600}
button:disabled{opacity:.4;cursor:default}
#bar .lbl{color:#8b949e;font-size:12px}
#chipTabs{color:#8b949e;font-size:12px}
#chipTabs button{padding:4px 10px}
#hint{padding:2px 14px 4px;font-size:11px;color:#6e7681}
#stats{position:fixed;left:12px;bottom:12px;background:#161b22ee;border:1px solid #30363d;border-radius:8px;padding:8px 12px;font-size:12px;max-width:360px;z-index:5}
#stats .tt{border-bottom:1px dotted #8b949e;cursor:help}
#tip{position:fixed;left:12px;top:70px;background:#161b22ee;border:1px solid #30363d;border-radius:8px;padding:6px 12px;font-size:13px;display:none;z-index:8}
#prog{position:fixed;left:50%;bottom:14px;transform:translateX(-50%);background:#161b22ee;border:1px solid #30363d;border-radius:8px;padding:6px 18px;font-size:13px;display:none;z-index:9}
#prog .bar{height:6px;background:#21262d;border-radius:3px;margin-top:4px;overflow:hidden;width:260px}
#prog .fill{height:100%;background:#1f6feb;border-radius:3px}
.panel{position:fixed;right:12px;bottom:60px;width:460px;max-height:76vh;overflow:auto;background:#161b22f5;border:1px solid #30363d;border-radius:8px;padding:10px 14px;font-size:12px;display:none;z-index:6;line-height:1.5}
.panel h3{margin:10px 0 6px;font-size:13px}
.panel .x{float:right;cursor:pointer;color:#8b949e;font-weight:700;font-size:16px;margin-left:10px}
.panel .sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.panel code{background:#21262d;border:1px solid #30363d;border-radius:4px;padding:0 5px;font-size:11px}
.panel .row{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:4px 0}
.panel .lbl{font-size:11px;color:#8b949e;min-width:120px}
.panel .hint2{font-size:10px;color:#6e7681;max-width:420px;margin-top:8px}
.panel input[type=text]{background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e6edf3;padding:4px 8px;font-size:12px}
.panel .c1{background:#fff200;color:#111}
.panel .c2{background:#ff9e1b;color:#111}
.panel .c3{background:#e31a1c;color:#fff}
.c0{background:#fff;color:#111}
#inset{position:fixed;left:8px;bottom:30px;z-index:10;background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:4px;display:none;cursor:pointer}
#inset img{width:190px;height:190px;display:block;border-radius:4px;background:#161b22}
#inset span{font-size:10px;color:#8b949e;display:block;text-align:center;padding-top:2px}
#onl canvas{width:512px;height:512px;display:block;background:#11161d;border-radius:4px;margin:6px 0}
.panel .c1.on,.panel .c2.on,.panel .c3.on{outline:2px solid #388bfd}
.fld{margin:10px 0 2px}
.fld label{display:block;font-size:11px;color:#8b949e;margin-bottom:3px}
.fld input{width:100%;box-sizing:border-box}
.btns{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
#listCid2 button{display:block;width:100%;text-align:left;margin:1px 0;padding:3px 10px}
#testout{font-size:11px;color:#79c0ff;max-width:420px;margin-top:8px}
#pingout{margin-top:8px;font-size:11px;color:#79c0ff;min-height:16px}
input[type=range]{width:110px}
kbd{background:#21262d;border:1px solid #30363d;border-radius:4px;padding:0 5px;font-size:11px}
a{color:#79c0ff}
</style></head><body>
<div id="bar">
  <button id="mSingle">чип крупно</button>
  <button id="mAll">ВСЕ сразу</button>
  <span class="lbl">| слой:</span>
  <button id="bLayer1">спутник</button><button id="bLayer2">SWIR</button><button id="bLayer3">dNBR</button>
  <span class="lbl">| разметка:</span>
  <button id="bMarkup1">нет</button><button id="bMarkup2">наша</button><button id="bMarkup3">истина</button><button id="bMarkup4">обе</button>
  <span class="lbl">| FIRMS:</span><button id="bAr0">регион</button><button id="bAr1">Сибирь</button><button id="bAr2">Красноярск+</button><button id="bAr3">Урал 2019</button><button id="bFirUpd">обновить очаги</button><button id="bL0">Снимок</button><button id="bL1">SWIR 721</button><button id="bL2">Aqua</button><button id="bL3">Sentinel-2</button><button id="bFirML">ML-проверка</button><button id="bOnl">онлайн: Красноярск</button>
  <span class="lbl">| высота dNBR:</span><input id="height" type="range" min="0" max="300" value="120">
  <span class="lbl">прозрачность:</span><input id="alpha" type="range" min="0" max="100" value="100">
  <span class="lbl">|</span>
  <button id="bHome">сброс вида (R)</button><button id="bTop">вид сверху (T)</button><button id="bRot">развернуть 90°</button>
  <span class="lbl">|</span>
  <button id="bList">список чипов</button>
  <button id="bDraw">рисовать</button>
  <button id="bDebug">Debug</button>
  <button id="bInfo">что это?</button>
  <span id="chipTabs"></span>
</div>
<div id="tip"></div>
<div id="prog"><span id="progTxt"></span><div class="bar"><div class="fill" id="progFill" style="width:0%"></div></div></div>
<div id="hint">ЛКМ — крутить · колесо — приближать · ПКМ — двигать · <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> — летать (<kbd>Q</kbd>/<kbd>E</kbd> — вниз/вверх) · <kbd>,</kbd>/<kbd>.</kbd> — листать чипы · <kbd>R</kbd> — сброс · <kbd>T</kbd> — вид сверху · даблклик по пазлу — фрагмент · <kbd>Ctrl</kbd>+<kbd>Z</kbd> — отменить мазок · <kbd>Esc</kbd> — назад</div>
<div id="onl" class="panel">
 <span class="x" id="onlX">✕</span>
<h3>Онлайн тест Красноярска</h3>
 <div id="onlLine" class="line"></div>
 <canvas id="onlCv" width="512" height="512"></canvas>
 <div id="onlOut" class="line"></div>
 <div class="btns"><button id="bML">проверить ML</button><button id="bDrape">наложить на чип</button><button id="bUndrape">вернуть чип</button></div>
 <div class="btns"><span style="font-size:10px;color:#8b949e">позитив:</span><button id="prodAuto">авто</button><button id="prodTerra">Terra</button><button id="prodAqua">Aqua</button><button id="prodViirs">VIIRS</button></div>
 <div class="hint2">Точно область Красноярска: bbox 91.9-94.2E, 55.4-56.6N (кадр ~140x140 км, город
  в центре). Свободный NASA GIBS: сначала MODIS Terra (утренний проход, пиксель ~0.4 км — зернистость
  нормальна), если тёмный — MODIS Aqua (дневной), затем VIIRS SNPP. + термоточки FIRMS (VIIRS NRT,
  48 ч): цвет круга по FRP — зелёный низкая, жёлтая средняя, красная высокая. «проверить ML» — ищет
  тёплые зоны прямо на снимке и сверяет с термоточками. «наложить на чип» — драпирует фото на
  открытый чип вместо снимка датасета (координат у чипов нет, поэтому наложение только тут).</div>
</div>
<div id="inset"><img alt=""><span></span></div>
<div id="stats"></div>

<div id="listCid" class="panel">
 <span class="x" id="listX">✕</span>
<h3>Список чипов</h3>
 <div class="row"><span class="lbl">поиск:</span><input type="text" id="qCid" size="14" placeholder="BS_te_00"></div>
 <div class="row"><span class="lbl">сортировка:</span>
  <button id="bSortNum">по номеру</button><button id="bSortBurn">по гарю (убыв.)</button>
  <button id="bApply">применить и к мозаике</button></div>
 <div id="listCid2" style="max-height:52vh;overflow:auto;margin-top:6px"></div>
 <div class="hint2">клик по чипу — открыть его крупно (в разрешении 512 = оригинал датасета).
  «применить и к мозаике» — переставит клетки мозаики по этой сортировке.
  Листать открытый чип: кнопки рядом с названием или <kbd>,</kbd>/<kbd>.</kbd>.</div>
</div>

<div id="drawbar" class="panel">
 <span class="x" id="drawX">✕</span>
<h3>Нарисовать свой уровень / стереть</h3>
 <div class="row"><span class="lbl">кисть (EFFIS):</span>
  <button id="p1" class="c1">1 слабое</button><button id="p2" class="c2">2 среднее</button>
  <button id="p3" class="c3">3 сильное</button><button id="p0" class="c0">0 стёрка</button><button id="pE">ластик</button>
  <button id="pUndo">↩ отменить</button><button id="pClear">очистить чип</button></div>
 <div class="row"><span class="lbl">пороги (справка):</span>
  <button id="th1">1: 0.10-0.27</button><button id="th2">2: 0.27-0.44</button><button id="th3">3: &gt; 0.44</button></div>
 <div class="row"><button id="pTest">скор (тест)</button><button id="pCsv">csv (тест)</button><span id="helperLbl" style="font-size:10px;color:#6e7681"></span></div>
 <div id="testout"></div>
 <div class="hint2">жёлтая 1 = dNBR 0.10-0.27 (слабое), оранжевая 2 = 0.27-0.44 (среднее),
  красная 3 = &gt; 0.44 (сильное). Выбери степень и рисуй прямо на чипе; «ластик» стирает
  нарисованное, «очистить чип» убирает всё, белый «0 стёрка» удаляет разметку чемпиона.
  Рисуешь тем же цветом поверх размеченного — CSV не меняется. «скор (тест)» — твоя разметка против истины
  (чип «· train»), «csv (тест)» — выгрузка разметки в paint_overlay.json;
  полный CSV и точный скор: <code>tools/apply_paint.py</code> (команда в «что это?»).</div>
</div>

<div id="debug" class="panel">
 <span class="x" id="debugX">✕</span>
<h3>Источники снимков (API)</h3>
 <div class="fld"><label>NASA FIRMS MAP_KEY (активные очаги, NRT)</label><input type="text" id="inFirms"></div>
 <div class="fld"><label>Google Earth Engine — проект</label><input type="text" id="inEeProj"></div>
 <div class="fld"><label>Google Earth Engine — ключ (API key)</label><input type="text" id="inEeKey"></div>
 <div class="btns"><button id="inSave">сохранить ключи</button><button id="inReset">по умолчанию</button><button id="inPing">пинг FIRMS</button></div>
 <div id="pingout"></div>
 <div class="hint2"><b>NASA FIRMS</b> — активные очаги VIIRS/MODIS в реальном времени (NRT),
  бесплатно; на один запрос отдают максимум 3 суток. <b>Google Earth Engine</b> — полная
  история Sentinel-2 и композиты; ключ хранится здесь локально (localStorage), но
  EE из браузера требует авторизации: на командной строке <code>earthengine authenticate</code>.
  <b>Copernicus Data Space</b> — вся история Sentinel-2 бесплатно после регистрации.</div>
</div>

<div id="info" class="panel">
 <span class="x" id="infoX">✕</span>
<h3>Что тут показано</h3>
<b>Данные (слой данных):</b><br>
- <b>спутник</b> — Sentinel-2 True Color (B4/B3/B2) на дату «после»;<br>
- <b>SWIR</b> — (B12/B8A/B4): гарь «горит» чёрным (потемнение SWIR);<br>
- <b>dNBR</b> — разница NBR до-после: степень потемнения растительности; показана и цветом, и <b>высотой</b> (горение возвышается, ползунок «высота»).<br><br>
<b>Разметка (шкала EFFIS в dNBR):</b><br>
- <span class="sw" style="background:#fff200"></span><b>степень 1</b> — dNBR 0.10-0.27, слабое поражение (жёлтая);<br>
- <span class="sw" style="background:#ff9e1b"></span><b>степень 2</b> — dNBR 0.27-0.44, среднее (оранжевая);<br>
- <span class="sw" style="background:#e31a1c"></span><b>степень 3</b> — dNBR &gt; 0.44, сильное (красная).<br><br>
<b>Слои разметки:</b><br>
- <b>наша</b> — наш ответ (то, что уйдёт в CSV);<br>
- <b>истина</b> — эталон организаторов (только train);<br>
- <b>обе</b> — наложение: где совпали, где разошлись.<br>
Числа в табличке слева — количество пикселей каждой степени; наведи курсор на число — подробное описание.<br><br>
В тестовом наборе 269 чипов: 89 BS (степень поражения, показаны здесь) и 180 AF
(активный фронт огня по VIIRS — вторая задача, на карте не показаны).<br><br>
<b>Ложные срабатывания / дым:</b> VIIRS ловит и тёплые поверхности, и дымящиеся очаги.
Проверка распространения: очаг подтверждён, если виден в 2 съёмках подряд и не совпадает
с промзонами; дымовое пятно сдвинуто от очага по ветру — очаг ищут в «хвосте» дыма.
<h3>Чип крупно / список</h3>
«список чипов» — поиск, сортировка (по номеру / по гарю в нашем ответе), клик — открыть
чип крупно. Листать открытый чип: <kbd>,</kbd>/<kbd>.</kbd>. Фрагмент из мозаики: даблклик
по пазлу; разрешение фрагмента — 128 / 256 / 512 (512 = оригинал датасета), кнопки
рядом с названием фрагмента (сверху). Источник в одиночном режиме: «датасет» — встроенный
снимок, «FIRMS очаги» — оперативная проба активных очагов за 48 ч (регион).
<h3>Нарисовать свой уровень / стереть</h3>
Кнопка «рисовать» — кисть ставит точки выбранной степени (1/2/3) прямо на чипе,
«ластик» стирает нарисованное, «очистить чип» убирает всё с текущего чипа, «↩ отменить» (Ctrl+Z) отменяет последний мазок, «0 стёрка» (белый) удаляет пиксель из ответа чемпиона — пишет в CSV «нет гари». Рисование тем же цветом поверх уже размеченного чемпионом ничего не меняет в CSV — «скор (тест)» на тест-чипе показывает сверку с чемпионом: сколько пикселей совпадает, сколько будет заменено/добавлено/стёрто.
«скор (тест)» — твоя разметка против истины на чипе «· train».
«csv (тест)» — отправляет разметку локальному хелперу: тот пишет paint_overlay.json прямо в папку и сразу делает submission_map.csv со скором (окно хелпера: <code>py -3 tools/map_helper.py</code>). Если хелпер не запущен — файл скачается, CSV вручную:
<code>py -3 tools/apply_paint.py --overlay paint_overlay.json --submission submission_probe_35.csv --out submission_map.csv --score --score-dir &lt;данные&gt;/train</code>
— создаст submission_map.csv (чемпион не трогается; если tested скор хуже — просто не отправляй его).
<h3>Камера</h3>
ЛКМ — крутить (3D), колесо — приближать, ПКМ — двигать,
<kbd>W</kbd>/<kbd>A</kbd>/<kbd>S</kbd>/<kbd>D</kbd> — полёт вперёд/влево/назад/вправо (как в юнити),
<kbd>Q</kbd>/<kbd>E</kbd> — вниз/вверх, <kbd>T</kbd> — вид сверху, <kbd>R</kbd> — сброс вида,
«развернуть 90°» — развернуть проекцию вокруг центра (4 клика = полный круг),
двойной ЛКМ по пазлу мозаики — фрагмент крупно, <kbd>Esc</kbd> — назад.
<h3>Источники снимков (оперативные API)</h3>
- <b>NASA FIRMS</b> — активные очаги VIIRS/MODIS в реальном времени; ключ уже вписан в Debug
(бесплатно — до 3 суток на запрос, есть пинг с лимитами);<br>
- <b>Google Earth Engine</b> — полная история Sentinel-2 и композиты; нужна авторизация
(<code>earthengine authenticate</code>), проект и ключ — в Debug;<br>
- <b>Copernicus Data Space</b> — вся история Sentinel-2 бесплатно после регистрации.
</div>

<script src="https://unpkg.com/three@0.147.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.147.0/examples/js/controls/OrbitControls.js"></script>
<script>
var PACK = __DATA__;
var scene, camera, renderer, controls, ground, overlay;
var layerMode = "true", overlayMode = "ans", allMode = false, tilesGroup = null;
var loader = new THREE.TextureLoader();
var paint = {mode:null, drawing:false, canvases:{}, tex:null, hist:{}, stroke:null};
var view = {tile:null, src:"ds"};
var hovered = null, tileMeshes = [];
var keys = {};
var curCid = PACK.singles[0].cid;
var fragRes = 512, resortMode = "num";
var riseR = {};
var fir = {pts:null, ts:0, last:"", busy:false};
var firFloor=false,firFloorMesh=null,insetMode=0;
var HOME={pos:[0,-100,80],tgt:[0,0,0]},HOME_ALL={pos:[0,-70,130],tgt:[0,0,0]};
var COLORS={p0:[255,255,255],p1:[255,242,0],p2:[255,158,27],p3:[227,26,28]},RGB={p0:"#ffffff",p1:"#fff200",p2:"#ff9e1b",p3:"#e31a1c"};
var STAT_KEYS={
 "наша 1":"наш ответ: пиксели степени 1 — слабое поражение (жёлтая). Порог dNBR 0.10-0.27. Именно эти пиксели уйдут в CSV как class_id=1",
 "наша 2":"наш ответ: пиксели степени 2 — среднее поражение (оранжевая). Порог dNBR 0.27-0.44. Уйдут в CSV (class_id=2)",
 "наша 3":"наш ответ: пиксели степени 3 — сильное поражение (красная). Порог dNBR > 0.44. Уйдут в CSV (class_id=3)",
 "наша гарь":"наш ответ: всего пикселей гари (степени 1+2+3) — бинарный контур сгоревшего",
 "истина 1":"истина организаторов (эталон, только train): пиксели степени 1 — слабое (жёлтая)",
 "истина 2":"истина организаторов: пиксели степени 2 — среднее (оранжевая)",
 "истина 3":"истина организаторов: пиксели степени 3 — сильное (красная)",
 "истина гарь":"истина организаторов: всего пикселей гари (степени 1+2+3)",
 "IoU гари":"согласие нашего ответа с истиной по контуру гари: 1 = идеально совпало, 0 = ничего общего. Считается только там, где есть эталон"
};

function prog(txt,pct){
 var p=document.getElementById("prog");
 if(txt==null){p.style.display="none";return;}
 document.getElementById("progTxt").textContent=txt;
 document.getElementById("progFill").style.width=(pct==null?0:pct)+"%";
 p.style.display="block";
}
function findTile(cid){for(var i=0;i<PACK.tiles.length;i++){if(PACK.tiles[i].cid===cid)return PACK.tiles[i];}return null;}
var singlesMap={};
PACK.singles.forEach(function(s){singlesMap[s.cid]=s;});
function getSingle(cid){
 if(singlesMap[cid])return singlesMap[cid];
 var t=findTile(cid);
 if(!t)return null;
 var s={cid:cid,truth:false,stats:{},ans:t.ans||null,
  true:t.true512,false:t.false512,dnbr:t.dnbr512};
 if(t.b){s.stats["наша 1 слабое"]=t.b1||0;s.stats["наша 2 среднее"]=t.b2||0;
  s.stats["наша 3 сильное"]=t.b3||0;s.stats["наша гарь (1-3)"]=t.b;}
 singlesMap[cid]=s;return s;
}
function sortedCids(){
 var arr=PACK.tiles.map(function(t){return t.cid;});
 if(resortMode=="burn"){
  var m={};PACK.tiles.forEach(function(t){m[t.cid]=t.b||0;});
  arr.sort(function(a,b){return m[b]-m[a];});
 }else arr.sort();
 return arr;
}
function resortTiles(){
 var arr=sortedCids();
 arr.forEach(function(cid,i){var t=findTile(cid);t.col=i%9;t.row=(i/9)|0;});
}
function tabs(){
 var t=document.getElementById("chipTabs");t.innerHTML="";
 if(view.tile){
  var s=document.createElement("span");s.textContent="фрагмент "+view.tile.cid+" · разрешение:";
  t.appendChild(s);
  [128,256,512].forEach(function(r){
   var b=document.createElement("button");b.textContent=r+(r===512?" (оригинал)":"");
   b.className=fragRes===r?"on":"";
   b.onclick=function(){fragRes=r;build();tabs();};
   t.appendChild(b);});
  var back=document.createElement("button");back.textContent="назад к мозаике (Esc)";
  back.onclick=closeTile;t.appendChild(back);
  return;
 }
 if(allMode){t.appendChild(document.createTextNode("мозаика: "+PACK.tiles.length+" BS-чипов"));return;}
 var prev=document.createElement("button");prev.textContent="◀ (,)";
 prev.onclick=function(){cycle(-1);};t.appendChild(prev);
 var cur=document.createElement("span");cur.textContent=" "+curCid+" ";
 var s=getSingle(curCid);
 if(s&&s.truth)cur.textContent+="· train ";
 t.appendChild(cur);
 var next=document.createElement("button");next.textContent="(.) ▶";
 next.onclick=function(){cycle(1);};t.appendChild(next);
 var b=document.createElement("button");b.textContent="список";
 b.onclick=function(){togglePanel("listCid");};t.appendChild(b);
 var lb=document.createElement("span");lb.textContent="источник:";
 t.appendChild(lb);
 [["датасет","ds"],["FIRMS очаги","fir"]].forEach(function(p){
  var bb=document.createElement("button");bb.textContent=p[0];
  bb.className=view.src==p[1]?"on":"";
  bb.onclick=function(){view.src=p[1];build();tabs();};
  t.appendChild(bb);});
}
function syncSeg(){
 document.getElementById("mSingle").className=(!allMode)?"on":"";
 document.getElementById("mAll").className=(allMode)?"on":"";
 [["bLayer1","true"],["bLayer2","false"],["bLayer3","dnbr"]].forEach(function(p){
  document.getElementById(p[0]).className=(layerMode==p[1])?"on":"";});
 var anyTruth=PACK.singles.some(function(d){return d.truth;});
 [["bMarkup1","none"],["bMarkup2","ans"],["bMarkup3","gt"],["bMarkup4","both"]].forEach(function(p){
  var b=document.getElementById(p[0]);
  b.className=(overlayMode==p[1])?"on":"";
  b.disabled=(p[1]=="gt"||p[1]=="both")&&!anyTruth&&!allMode&&!view.tile;});
}
function ovlTex(){
 var d=view.tile||getSingle(curCid);
 if(overlayMode=="none")return null;
 if(view.src=="fir")return null;
 if(overlayMode=="ans")return d.ans;
 if(overlayMode=="gt")return d.gt;
 if(!d.ans||!d.gt)return d.ans||d.gt||null;
 var c=document.createElement("canvas");c.width=512;c.height=512;var ctx=c.getContext("2d");
 var i1=new Image(),i2=new Image(),done=0;
 return new Promise(function(res){
  function fin(){if(++done==2){ctx.drawImage(i1,0,0,512,512);ctx.drawImage(i2,0,0,512,512);res(c.toDataURL("image/png"));}}
  i1.onload=fin;i2.onload=fin;i1.src=d.ans;i2.src=d.gt;});
}
function clearScene(){
 if(ground){scene.remove(ground);ground.geometry.dispose();ground.material.dispose();ground=null;}
 if(overlay){scene.remove(overlay);overlay.geometry.dispose();overlay.material.dispose();overlay=null;}
 if(paint.plane){scene.remove(paint.plane);paint.plane.geometry.dispose();paint.plane.material.dispose();paint.plane=null;}
 if(tilesGroup){scene.remove(tilesGroup);tilesGroup.traverse(function(o){if(o.geometry)o.geometry.dispose();if(o.material)o.material.dispose();});tilesGroup=null;}
 tileMeshes=[];
}
function stats(){
 var s=document.getElementById("stats");
 if(view.tile){s.innerHTML="<b>"+view.tile.cid+"</b> · фрагмент, разрешение "+fragRes+
  " px"+(fragRes===512?" (оригинал датасета)":"")+" · разрешение меняется кнопками сверху";return;}
 if(allMode){var burn=PACK.tiles.filter(function(t){return t.ans;}).length;
  s.innerHTML="<b>мозаика: "+PACK.tiles.length+" BS-чипов (тест)</b><br>с гарью в нашем ответе: <b>"+burn+
  "</b><br>даблклик по пазлу — фрагмент · WASD — летать<br>в тестовом наборе всего 269 чипов: 89 BS + 180 AF (вторая задача)";return;}
 if(view.src=="fir"){
  var n=fir.pts?fir.pts.length:0;
  s.innerHTML="<b>"+curCid+"</b> · оперативная проба FIRMS (VIIRS SNPP, последние 48 ч)<br>"+
   "очагов в регионе (Вост. Сибирь): <b>"+n+"</b>"+
   (fir.last?"<br>последняя фиксация: "+fir.last:"")+
   "<br><span style='color:#6e7681;font-size:10px'>это весь регион, не кроп чипа — чипы без геопривязки; красное = высокая уверенность, оранжевое = низкая</span>";
  return;}
 var d=getSingle(getCur()),html="<b>"+d.cid+"</b>"+(d.truth?" · эталон организаторов":" · тест");
 Object.keys(d.stats).forEach(function(k){
  var t="";
  Object.keys(STAT_KEYS).some(function(kk){if(k.indexOf(kk)===0){t=STAT_KEYS[kk];return true;}return false;});
  html+="<br>"+(t?"<span class='tt' title='"+t+"'>"+k+"</span>":k)+": <b>"+d.stats[k]+"</b>";
 });
 html+="<div style='color:#6e7681;font-size:10px;margin-top:4px'>наведи курсор на число — что это значит</div>";
 s.innerHTML=html;
}
function getCur(){return curCid;}
function home(){var h=(allMode&&!view.tile)?HOME_ALL:HOME;camera.position.set.apply(camera.position,h.pos);controls.target.set.apply(controls.target,h.tgt);}
function topView(){
 var h=(allMode&&!view.tile)?180:135;
 camera.position.set(0.01,-0.01,h);controls.target.set(0,0,0);
}
function rot90(){
 if(view.src=="fir"&&ground){
  firRot=(firRot+1)%4;
  ground.rotation.z=firRot*Math.PI/2;
  return;
 }
 var p=camera.position,t=controls.target;
 var dx=p.x-t.x,dy=p.y-t.y;
 p.x=t.x-dy;p.y=t.y+dx;
}
function hScale(){return +document.getElementById("height").value/40;}
function hScaleM(){return +document.getElementById("height").value/200;}
function alphaV(){return +document.getElementById("alpha").value/100;}
function ovlFor(dnbrTex,H,mul){
 var o=ovlTex();
 if(o===null)return;
 Promise.resolve(o).then(function(data){
  var g2=new THREE.PlaneGeometry(100,100,64,64);
  var m2=new THREE.MeshStandardMaterial({map:loader.load(data),transparent:true,
   opacity:alphaV(),depthWrite:false,displacementMap:dnbrTex,
   displacementScale:H*mul,side:THREE.DoubleSide,roughness:1,metalness:0});
  overlay=new THREE.Mesh(g2,m2);overlay.position.z=0.14;overlay.renderOrder=2;scene.add(overlay);
 });
}
function build(){clearScene();hovered=null;firInset();
 if(view.tile)return buildTile();
 if(allMode)return buildAll();
 var d=getSingle(curCid),H=hScale();
 if(view.src=="fir")return buildFIR();
 var dnbrTex=loader.load(d.dnbr);
 var geo=new THREE.PlaneGeometry(100,100,64,64);
 var mat=new THREE.MeshStandardMaterial({
  map:loader.load(layerMode=="true"?d.true:layerMode=="false"?d.false:d.dnbr),
  displacementMap:dnbrTex,displacementScale:H,side:THREE.DoubleSide,roughness:1});
 ground=new THREE.Mesh(geo,mat);scene.add(ground);
 ovlFor(dnbrTex,H,1.04);
 attachPaint(d.cid,d.dnbr);
 stats();
}
function texFor(t){
 var base=layerMode=="true"?"true":layerMode=="false"?"false":"dnbr";
 var k=base+fragRes;
 if(!t[k])k=base+(fragRes===128?"128":fragRes===256?"256":"512");
 return t[k];
}
function buildTile(){
 var t=view.tile;
 var dnbrTex=loader.load(texFor(t));
 var geo=new THREE.PlaneGeometry(100,100,64,64);
 var mat=new THREE.MeshStandardMaterial({map:loader.load(texFor(t)),
   displacementMap:dnbrTex,displacementScale:hScale(),side:THREE.DoubleSide,roughness:1});
 ground=new THREE.Mesh(geo,mat);scene.add(ground);
 ovlFor(dnbrTex,hScale(),1.04);
 attachPaint(t.cid,texFor(t));
 stats();
}
function buildAll(){
 tilesGroup=new THREE.Group();firFloorMesh=null;
 var S=10.2,H=hScaleM(),i=0;
 var rows=Math.max.apply(null,PACK.tiles.map(function(t){return t.row;}))+1;
 var cols=Math.max.apply(null,PACK.tiles.map(function(t){return t.col;}))+1;
 var cx=cols*S/2,cy=rows*S/2;
 function step(){
  var done=Math.min(i+12,PACK.tiles.length);
  for(;i<done;i++){
   var t=PACK.tiles[i];
   var mat=new THREE.MeshStandardMaterial({map:loader.load(texFor(t)),
     displacementMap:loader.load(t.dnbr512||t.dnbr),displacementScale:H,
     side:THREE.DoubleSide,roughness:1});
   var p=new THREE.Mesh(new THREE.PlaneGeometry(10,10,16,16),mat);
   p.position.set(t.col*S-cx+S/2,cy-t.row*S-S/2,0);
   p.userData.tile=t;
   t.pos=[p.position.x,p.position.y];
   t.mesh=p;
   tilesGroup.add(p);tileMeshes.push(p);
   if(t.ans&&overlayMode!="none"&&overlayMode!="gt"){
    var m2=new THREE.MeshStandardMaterial({map:loader.load(t.ans),transparent:true,
     opacity:alphaV(),depthWrite:false,displacementMap:loader.load(t.dnbr512||t.dnbr),
     displacementScale:H*1.06,side:THREE.DoubleSide,roughness:1,metalness:0});
    var o2=new THREE.Mesh(new THREE.PlaneGeometry(10,10,16,16),m2);
    o2.position.copy(p.position);o2.position.z=0.07;o2.renderOrder=2;
    tilesGroup.add(o2);
   }
  }
  var pct=Math.round(100*done/PACK.tiles.length);
  prog("строим мозаику "+done+"/"+PACK.tiles.length,pct);
  if(done<PACK.tiles.length){setTimeout(step,0);return;}
  scene.add(tilesGroup);prog(null);stats();
 }
 step();
}
function buildFIR(){
 if(fir.pts&&(Date.now()-fir.ts)<900000){drawFIR();return;}
 renderFIRMS();
}
var FIR_AREAS=[
 [86.0,51.0,114.0,69.0,"регион",6],
 [78.0,48.0,118.0,72.0,"Сибирь",5],
 [89.0,54.0,96.0,57.5,"Красноярск+",9],
 [34.0,43.5,50.5,51.5,"Урал 2019",7]
];
var firArea=0,firRot=0,firLayer=0,regCache={},regBusyA={},regFailA={};
function regY(lat){var r=lat*Math.PI/180;return (1-Math.log(Math.tan(r)+1/Math.cos(r))/Math.PI)/2*64;}
function fetchRegSnap(area){
 var k2=area+"_"+firLayer;
 if(regBusyA[k2]||regCache[k2])return;
 regBusyA[k2]=true;
 var B=FIR_AREAS[area],N=256,z=B[5],nw=Math.pow(2,z),k=nw/64;
 var x0=Math.floor((B[0]+180)/360*nw),x1=Math.floor((B[2]+180)/360*nw);
 var y0=Math.floor(regY(B[3])*k),y1=Math.floor(regY(B[1])*k);
 var cv=document.createElement("canvas");cv.width=(x1-x0+1)*N;cv.height=(y1-y0+1)*N;
 var cx=cv.getContext("2d");
 var TC=["MODIS_Terra_CorrectedReflectance_TrueColor","GoogleMapsCompatible_Level9","MODIS Terra"],B7=["MODIS_Terra_CorrectedReflectance_Bands721","GoogleMapsCompatible_Level9","MODIS 721"],A2=["MODIS_Aqua_CorrectedReflectance_TrueColor","GoogleMapsCompatible_Level9","MODIS Aqua"],S2=["HLS_S30_Nadir_BRDF_Adjusted_Reflectance","GoogleMapsCompatible_Level9","Sentinel-2"];
 var P=firLayer==3?[S2,TC,A2]:firLayer==2?[A2,TC]:firLayer==1?[B7,TC,A2]:[TC,A2];
 function done_ok(){
  var dcv=document.createElement("canvas");dcv.width=2048;dcv.height=2048;
  dcv.getContext("2d").drawImage(cv,0,0,2048,2048);
  regCache[k2]=dcv;regBusyA[k2]=false;
  if(firArea===area&&view.src=="fir")drawFIR();
 }
 function one(pi,di){
  if(pi>=P.length){regFailA[area]=true;regBusyA[area]=false;return;}
  var pr=P[pi];
  var d=area==3?"2019-04-28":new Date(Date.now()-di*86400000).toISOString().slice(0,10);
  cx.clearRect(0,0,cv.width,cv.height);
  var need=(x1-x0+1)*(y1-y0+1),done=0,ok=0;
  for(var xx=x0;xx<=x1;xx++)for(var yy=y0;yy<=y1;yy++){
   (function(xx,yy){
    var img=new Image();img.crossOrigin="anonymous";
    img.onload=function(){done++;ok++;
     cx.drawImage(img,(xx-x0)*N,(yy-y0)*N);
     if(done>=need)done_ok();};
    img.onerror=function(){done++;
     if(done>=need){
      if(ok>0)done_ok();
      else if(di<2)one(pi,di+1);
      else one(pi+1,0);
     }};
    img.src="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"+pr[0]+"/default/"+d+"/"+pr[1]+"/"+z+"/"+yy+"/"+xx+".jpg";
   })(xx,yy);
  }
 }
 one(0,0);
}
var mlCache={canvas:{}};
function fetchMl(){
 if(mlCache.doc||mlCache.busy)return;
 mlCache.busy=true;
 fetch("http://127.0.0.1:8765/catalog",{timeout:60000}).then(function(r){
  if(!r.ok)throw new Error(r.status);
  return r.json();
 }).then(function(d){
  mlCache.doc=d;mlCache.busy=false;
  if(view.src=="fir"&&firArea==3)drawFIR();
 }).catch(function(e){
  mlCache.busy=false;mlCache.err=String(e);
  if(view.src=="fir"&&firArea==3)drawFIR();
 });
}
function makeMlCanvas(){
 var B=FIR_AREAS[firArea],SZ=2048;
 var demo=firArea!=3;
 var dxLon=0,dyLat=0;
 if(demo){dxLon=(B[0]+B[2])/2-42.25;dyLat=(B[1]+B[3])/2-47.5;}
 var cv=document.createElement("canvas");cv.width=SZ;cv.height=SZ;
 var ctx=cv.getContext("2d");
 var xf=function(lon){return (lon+dxLon-B[0])/(B[2]-B[0])*SZ;};
 var yf=function(lat){return (regY(B[3])-regY(lat+dyLat))/(regY(B[3])-regY(B[1]))*SZ;};
 var doc=mlCache.doc;
 if(!doc)return cv;
 var SEVC={1:"#fff200",2:"#ff9e1b",3:"#e31a1c"};
 ctx.globalAlpha=0.55;
 var cols=doc.contours||[];
 for(var i=0;i<cols.length;i++){
  var c=cols[i],ring=c.ring;
  if(!ring||!ring.length)continue;
  ctx.fillStyle=SEVC[c.severity]||"#fff200";
  ctx.beginPath();
  ctx.moveTo(xf(ring[0][0]),yf(ring[0][1]));
  for(var j=1;j<ring.length;j++)ctx.lineTo(xf(ring[j][0]),yf(ring[j][1]));
  ctx.closePath();
  ctx.fill();
 }
 ctx.globalAlpha=0.5;
 ctx.fillStyle="#ffffff";
 var th=doc.thermopoints||[];
 for(var k=0;k<th.length;k++){
  var x=xf(th[k].lon),y=yf(th[k].lat);
  if(x<0||x>=SZ||y<0||y>=SZ)continue;
  ctx.fillRect(x-3,y-3,6,6);
 }
 ctx.globalAlpha=1;
 if(demo){
  ctx.globalAlpha=0.9;
  ctx.fillStyle="#ffffff";ctx.font="40px system-ui";
  ctx.fillText("demo-раскладка: контуры смещены в видимую зону (реальные координаты 36-48E, Урал)",30,1990);
  ctx.globalAlpha=1;
 }
 return cv;
}
function firCanvas(){
 var SZ=2048;
 var c=document.createElement("canvas");c.width=SZ;c.height=SZ;var ctx=c.getContext("2d");
 ctx.fillStyle="#0f1418";ctx.fillRect(0,0,SZ,SZ);
 var fk2=firArea+"_"+firLayer;
 if(regCache[fk2])ctx.drawImage(regCache[fk2],0,0,SZ,SZ);
 fetchMl();
 if(mlCache.doc){
  var mc=mlCache.canvas[firArea];
  if(!mc)mc=mlCache.canvas[firArea]=makeMlCanvas();
  ctx.drawImage(mc,0,0,SZ,SZ);
 }
 if(firArea==3&&!mlCache.doc){
  ctx.fillStyle="#0f1418";
  ctx.fillRect(0,0,SZ,SZ);
  ctx.fillStyle="#8b949e";ctx.font="44px system-ui";
  ctx.fillText("слой нейронки: контуры пройденного огня 2019",40,960);
  ctx.fillText(mlCache.err?("каталог недоступен ("+mlCache.err+") — запусти py -3 tools/map_helper.py"):"грузим каталог сервиса...",40,1030);
 }
 var B=FIR_AREAS[firArea];
 var step=firArea==1?4:(firArea==2?1:2);
 var regXf=function(lon){return (lon-B[0])/(B[2]-B[0])*SZ;};
 var regYp=function(lat){return (regY(B[3])-regY(lat))/(regY(B[3])-regY(B[1]))*SZ;};
 ctx.strokeStyle="#1c232b";
 for(var lon=Math.ceil(B[0]);lon<=B[2];lon+=step){var x=regXf(lon);ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,SZ);ctx.stroke();}
 for(var la=Math.ceil(B[1]);la<=B[3];la+=step){var y=regYp(la);ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(SZ,y);ctx.stroke();}
 if(fir.pts)fir.pts.forEach(function(p){
  var x=regXf(p[0]),y=regYp(p[1]);
  ctx.fillStyle=(p[3]&&p[3]!="l"&&p[3]!="n")?"#ff6a3d":"#ffb03d";
  ctx.beginPath();ctx.arc(x,y,10,0,Math.PI*2);ctx.fill();
 });
 ctx.fillStyle="#566070";ctx.font="44px system-ui";
 var midLo=Math.round((B[0]+B[2])/2),midLa=Math.round((B[1]+B[3])/2);
 ctx.fillText(B[0]+"E",16,2024);ctx.fillText(midLo+"E",regXf(midLo)-80,2024);ctx.fillText(B[2]+"E",1936,2024);
 ctx.fillText(B[3]+"N",16,regYp(B[3])+48);ctx.fillText(midLa+"N",16,regYp(midLa));ctx.fillText(B[1]+"N",16,2024);
 return c;
}
function drawFIR(){
 var fkey=firArea+"_"+firLayer;
 if(!regCache[fkey]&&!regBusyA[fkey]&&!regFailA[fkey])fetchRegSnap(firArea);
 var c=firCanvas();
 if(ground){scene.remove(ground);ground.geometry.dispose();ground.material.dispose();}
 ground=new THREE.Mesh(new THREE.PlaneGeometry(100,100),
  new THREE.MeshBasicMaterial({map:new THREE.CanvasTexture(c),side:THREE.DoubleSide}));
 scene.add(ground);
 ground.rotation.z=firRot*Math.PI/2;
 stats();
 firInset();
}
function firInset(){
 var el=document.getElementById("inset");
 if(el)el.style.display="none";
}
document.getElementById("inset").onclick=function(){insetMode=(insetMode+1)%3;firInset();};
function addFirFloor(){
 if(firFloorMesh){tilesGroup.remove(firFloorMesh);firFloorMesh=null;}
 if(!firFloor||!allMode)return;
 if(!fir.pts){if(!fir.busy)renderFIRMS();return;}
 var tex=new THREE.CanvasTexture(firCanvas());
 firFloorMesh=new THREE.Mesh(new THREE.PlaneGeometry(340,224),
  new THREE.MeshBasicMaterial({map:tex,transparent:true,opacity:0.85,side:THREE.DoubleSide,depthWrite:false}));
 firFloorMesh.position.set(0,0,-45);firFloorMesh.renderOrder=-1;
 tilesGroup.add(firFloorMesh);
}
function tryText(urls,i,cb){
 if(i>=urls.length)return false;
 fetch(urls[i]).then(function(r){return r.text();}).then(cb)
  .catch(function(){tryText(urls,i+1,cb);});
 return true;
}
function renderFIRMS(){
 if(fir.busy)return;
 fir.busy=true;
 var k=Object.assign({},DEF_KEYS,loadKeys()).firms;
 if(firArea==3){fir.pts=null;drawFIR();prog("слой пожаров 2019: контуры пройденного огня (слой нейронки, каталог сервиса); live FIRMS не показывается",100);return;}
 var B=FIR_AREAS[firArea];
 var api="https://firms.modaps.eosdis.nasa.gov/api/area/csv/"+k+"/VIIRS_SNPP_NRT/"+B[0]+","+B[1]+","+B[2]+","+B[3]+"/2";
 var q=encodeURIComponent(api);
 prog("грузим очаги FIRMS (48 ч, "+B[4]+")...",30);
 var done=function(t){
  fir.busy=false;prog(null);
  var lines=(t||"").split("\\n"),pts=[];
  for(var i=1;i<lines.length;i++){
   var c=lines[i].split(",");
   if(c.length>10&&!isNaN(parseFloat(c[0])))pts.push([parseFloat(c[1]),parseFloat(c[0]),c[5]+" "+c[6],c[9]]);
  }
  pts.sort(function(a,b){return a[2]<b[2]?1:-1;});
  fir.pts=pts;fir.ts=Date.now();fir.last=pts.length?pts[0][2]:"";
  if(view.src=="fir")drawFIR();else if(allMode&&firFloor)addFirFloor();
 };
 if(!tryText([api,"https://api.allorigins.win/raw?url="+q,"https://api.codetabs.com/v1/proxy?quest="+q],0,done)){
  fir.busy=false;prog(null);
  document.getElementById("stats").innerHTML="<b>FIRMS недоступен из браузера (CORS)</b><br>открой вручную: <a href='"+api+"'>"+api+"</a>";
 }
}
function attachPaint(cid,dnbrUrl){
 if(!paint.canvases[cid]){
  var c=document.createElement("canvas");c.width=512;c.height=512;
  paint.canvases[cid]=c;
 }
 paint.tex=new THREE.CanvasTexture(paint.canvases[cid]);
 var p=new THREE.Mesh(new THREE.PlaneGeometry(100,100,64,64),
  new THREE.MeshStandardMaterial({map:paint.tex,transparent:true,depthWrite:false,
   displacementMap:loader.load(dnbrUrl),displacementScale:hScale()*1.08,
   side:THREE.DoubleSide,roughness:1,metalness:0}));
 p.renderOrder=4;p.position.z=0.2;scene.add(p);
 paint.plane=p;
}
function setPaint(mode){
 paint.mode=mode;controls.enabled=!mode;
 renderer.domElement.style.cursor=paint.mode?"crosshair":"";
 ["p1","p2","p3","p0","pE"].forEach(function(id){
  var b=document.getElementById(id);
  b.className=b.className.replace(" on","")+(paint.mode==id?" on":"");});
}
function groundHit(e){
 var raycaster=new THREE.Raycaster();
 raycaster.setFromCamera({x:(e.clientX/innerWidth)*2-1,y:-(e.clientY/innerHeight)*2+1},camera);
 var hit=new THREE.Vector3();
 raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0,0,1),0),hit);
 return hit;
}
function paintCid(){return view.tile?view.tile.cid:curCid;}
function paintDot(e,erase){
 var hit=groundHit(e);if(!isFinite(hit.x))return;
 var c=c2dPaint();var ctx=c.getContext("2d");
 var u=(hit.x+50)/100*512,v=(50-hit.y)/100*512;
 if(u<0||u>512||v<0||v>512)return;
 var ru=Math.min(511,Math.max(0,Math.round(u))),rv=Math.min(511,Math.max(0,Math.round(v)));
 if(!paint.stroke)paint.stroke={erase:!!erase,mode:paint.mode,dots:[]};
 var ds=paint.stroke.dots,n=ds.length;
 if(n&&Math.abs(ru-ds[n-1][0])<4&&Math.abs(rv-ds[n-1][1])<4)return;
 ds.push([ru,rv]);
 ctx.save();
 if(erase)ctx.globalCompositeOperation="destination-out";
 ctx.beginPath();ctx.arc(ru+0.5,rv+0.5,7,0,Math.PI*2);
 ctx.fillStyle=erase?"#000":RGB[paint.mode];
 ctx.fill();ctx.restore();
 paint.tex.needsUpdate=true;
}
function c2dPaint(){
 var cid=paintCid();
 if(!paint.canvases[cid]){
  var c=document.createElement("canvas");c.width=512;c.height=512;
  paint.canvases[cid]=c;
 }
 return paint.canvases[cid];
}
function redrawChip(cid){
 var c=paint.canvases[cid];if(!c)return;
 var ctx=c.getContext("2d");ctx.clearRect(0,0,512,512);
 (paint.hist[cid]||[]).forEach(function(st){
  st.dots.forEach(function(p){
   ctx.save();
   if(st.erase)ctx.globalCompositeOperation="destination-out";
   ctx.beginPath();ctx.arc(p[0]+0.5,p[1]+0.5,7,0,Math.PI*2);
   ctx.fillStyle=st.erase?"#000":RGB[st.mode];
   ctx.fill();ctx.restore();
  });
 });
 if(cid==paintCid())paint.tex.needsUpdate=true;
}
function undoStroke(){
 var cid=paintCid(),h=paint.hist[cid];
 var el=document.getElementById("testout");el.style.display="block";
 if(!h||!h.length){el.textContent="нечего отменять";return;}
 h.pop();redrawChip(cid);
 el.textContent="отменён последний мазок (осталось "+h.length+")";
}
function nearestDeg(r,g,b){
 var best=0,bd=1e9;
 [0,1,2,3].forEach(function(d){var c=COLORS["p"+d],d2=(r-c[0])*(r-c[0])+(g-c[1])*(g-c[1])+(b-c[2])*(b-c[2]);
  if(d2<bd){bd=d2;best=d;}});
 return best;
}
function paintedMask384(paintedCanvas){
 var tmp=document.createElement("canvas");tmp.width=384;tmp.height=384;
 var ctx=tmp.getContext("2d");ctx.imageSmoothingEnabled=true;
 ctx.drawImage(paintedCanvas,0,0,384,384);
 return ctx.getImageData(0,0,384,384);
}
function scoreFromPixels(p,g){
 var inter=0,paintBurn=0,gtBurn=0;
 var tpC=[0,0,0],fpC=[0,0,0],fnC=[0,0,0];
 for(var i=0;i<384*384;i++){
  var pa=p[i*4+3]>128, ga=g[i*4+3]>100;
  if(pa)paintBurn++;if(ga)gtBurn++;if(pa&&ga)inter++;
  var pd=pa?nearestDeg(p[i*4],p[i*4+1],p[i*4+2]):0;
  var gd=ga?nearestDeg(g[i*4],g[i*4+1],g[i*4+2]):0;
  if(pd&&pd===gd)tpC[pd-1]++;
  if(pd&&pd!==gd)fpC[pd-1]++;
  if(gd&&gd!==pd)fnC[gd-1]++;
 }
 var iouB=(paintBurn+gtBurn-inter===0)?1:inter/(paintBurn+gtBurn-inter);
 var per=[0,1,2].map(function(k){var den=tpC[k]+fpC[k]+fnC[k];return den===0?1:tpC[k]/den;});
 var miou=(per[0]+per[1]+per[2])/3;
 var p2=0.35*iouB+0.30*miou;
 return {iou:iouB,miou:miou,part:p2,norm:p2/0.65};
}
function scoreChip(cid,canvas,cb){
 var s=getSingle(cid);
 if(!s.truth){cb(null);return;}
 var img=new Image();
 img.onload=function(){
  var c=document.createElement("canvas");c.width=384;c.height=384;
  var ctx=c.getContext("2d");ctx.drawImage(img,0,0);
  cb(scoreFromPixels(paintedMask384(canvas).data,ctx.getImageData(0,0,384,384).data));
 };
 img.src=s.gt;
}
var champCache={};
function champCanvas(cid,cb){
 var d=getSingle(cid);
 if(!d||!d.ans){cb(null);return;}
 if(champCache[cid]){cb(champCache[cid]);return;}
 var img=new Image();
 img.onload=function(){
  var c=document.createElement("canvas");c.width=512;c.height=512;
  var ctx=c.getContext("2d");ctx.drawImage(img,0,0,512,512);
  champCache[cid]=c;cb(c);
 };
 img.src=d.ans;
}
function champDiff(cid,cb){
 champCanvas(cid,function(cc){
  if(!cc){cb(null);return;}
  var pc=paint.canvases[cid];
  if(!pc){cb({same:0,repl:0,add:0,del:0});return;}
  var p=pc.getContext("2d").getImageData(0,0,512,512).data;
  var g=cc.getContext("2d").getImageData(0,0,512,512).data;
  var same=0,repl=0,add=0,del=0;
  for(var i=0;i<512*512;i++){
   var pd=p[i*4+3]>128?nearestDeg(p[i*4],p[i*4+1],p[i*4+2]):-1;
   var gc=g[i*4+3]>128?nearestDeg(g[i*4],g[i*4+1],g[i*4+2]):0;
   if(pd>0){if(gc===0)add++;else if(pd===gc)same++;else repl++;}
   else if(pd===0&&gc>0)del++;
  }
  cb({same:same,repl:repl,add:add,del:del});
 });
}
function runTest(){
 var out=document.getElementById("testout");out.style.display="block";
 var cid=paintCid(),s=getSingle(cid);
 if(!s.truth){
 out.textContent="сверяем с разметкой чемпиона...";
 champDiff(cid,function(r){
  if(!r){out.textContent="нет разметки чемпиона для этого чипа";return;}
  out.innerHTML="тест-чип — локальной истыни нет; сверка с ответом чемпиона (по этому чипу):<br>"+
   "совпадает: <b>"+r.same+"</b> (скор НЕ изменит) · заменит разметку: <b>"+r.repl+"</b> · добавит нового: <b>"+r.add+"</b> · сотрёт разметку: <b>"+r.del+"</b>"+
   "<br><span style='color:#6e7681'>«совпадает» = закрасил тем же цветом поверх уже размеченного чемпионом — в CSV это ничего не меняет, поэтому скор и стоял на месте</span>";
 });
 return;}
 out.textContent="считаем скор...";
 scoreChip(cid,paint.canvases[cid],function(r){
  if(!r){out.textContent="нет истыни для этого чипа";return;}
  var baseIou=s.stats["IoU гари"]||0;
  out.innerHTML="скор (BS-часть): <b>"+r.part.toFixed(4)+"</b> из 0.65 → ≈ платформа: <b>"+r.norm.toFixed(4)+"</b> из 1.0"+
   "<br>(шкала платформы = BS-часть / 0.65; на платформе ещё 35% на задачу AF)"+
   "<br>IoU гари: было "+baseIou.toFixed(3)+" → стало "+r.iou.toFixed(3)+" · mIoU степеней: "+r.miou.toFixed(3)+
   "<br>чемпион на платформе: <b>0.7455</b> · точный пересчёт CSV: tools/apply_paint.py";
 });
}
var HELPER="http://127.0.0.1:8765",HELPER_ON=null;
function setHelperLbl(){
 var el=document.getElementById("helperLbl");
 if(el)el.textContent=HELPER_ON==null?"проверяю хелпер...":(HELPER_ON?"хелпер: работает — писать в папку + авто CSV":"хелпер: не запущен — файл скачается; запусти py -3 tools\\map_helper.py");
}
function pingHelper(){
 try{
  var r=new XMLHttpRequest();r.open("GET",HELPER+"/ping",true);r.timeout=2500;
  r.onload=function(){HELPER_ON=true;setHelperLbl();};
  r.ontimeout=r.onerror=function(){HELPER_ON=false;setHelperLbl();};
  r.send();
 }catch(e){HELPER_ON=false;setHelperLbl();}
}
function fallbackCsv(overlay,out){
 var n=Object.keys(overlay).length;
 var blob=new Blob([JSON.stringify({overlay:overlay})],{type:"application/json"});
 var a=document.createElement("a");a.href=URL.createObjectURL(blob);
 a.download="paint_overlay.json";a.click();
 out.innerHTML="paint_overlay.json: "+n+" чип(ов) — скачан.<br>хелпер не запущен — писать в папку некому; окно: <code>py -3 tools\\map_helper.py</code><br>CSV вручную: <code>py -3 tools\\apply_paint.py --overlay paint_overlay.json --submission submission_probe_35.csv --out submission_map.csv --score --score-dir <путь_к_train></code>";
}
function postHelper(overlay,out,diffCid){
 if(HELPER_ON==null)HELPER_ON=false;
 if(!HELPER_ON){fallbackCsv(overlay,out);return;}
 out.textContent="отправляю разметку хелперу...";
 try{
  var r=new XMLHttpRequest();r.open("POST",HELPER+"/overlay",true);r.timeout=6000;
  r.onload=function(){
   var resp=null;
   try{resp=JSON.parse(r.responseText);}catch(e){}
   if(!resp||!resp.ok){HELPER_ON=false;setHelperLbl();fallbackCsv(overlay,out);return;}
   var short="";
   if(resp.score)short="скор кисти против истины: <b>"+resp.score+"</b> (iou гари "+resp.iou_burn+" · mIoU "+resp.miou_sev+")";
   else if(resp.log)short=String(resp.log).split("\\n").slice(0,2).join(" · ");
   out.innerHTML="<b>прямо в папку: paint_overlay.json + "+(resp.csv||"submission_map.csv")+"</b>"+(short?"<br>"+short:"")+"<br>чемпион не тронут (база: "+(resp.base||"?")+"); если скор хуже — просто не отправляй новый CSV";
   if(diffCid)champDiff(diffCid,function(r){
    if(r)out.innerHTML+="<br>сверка ("+diffCid+"): совпадает "+r.same+" · заменит "+r.repl+" · добавит "+r.add+" · сотрёт "+r.del;
   });
  };
  r.ontimeout=r.onerror=function(){HELPER_ON=false;setHelperLbl();fallbackCsv(overlay,out);};
  r.setRequestHeader("Content-Type","application/json");
  r.send(JSON.stringify({overlay:overlay}));
 }catch(e){HELPER_ON=false;setHelperLbl();fallbackCsv(overlay,out);}
}
function runCsv(){
 var out=document.getElementById("testout");out.style.display="block";
 var cids=Object.keys(paint.canvases),overlay={},idx=0;
 function step(){
  var done=Math.min(idx+40,cids.length);
  for(;idx<done;idx++){
   var cid=cids[idx];
   var data=paint.canvases[cid].getContext("2d").getImageData(0,0,512,512).data;
   var dots=[];
   for(var v=0;v<512;v++)for(var u=0;u<512;u++){
    var i=(v*512+u)*4;
    if(data[i+3]>128)dots.push([u,v,nearestDeg(data[i],data[i+1],data[i+2])]);
   }
   if(dots.length)overlay[cid]=dots;
  }
  var n=Object.keys(overlay).length;
  var pct=Math.round(100*done/cids.length);
  if(done<cids.length){out.textContent="готовим JSON "+pct+"% ("+n+" чип(ов) с выделением)";setTimeout(step,0);return;}
  if(!n){out.textContent="пусто — нарисуй точки кистью сначала";return;}
  postHelper(overlay,out,Object.keys(overlay)[0]);
 }
 if(!cids.length){out.textContent="пусто — нечего выгружать";return;}
 step();
}
function cycle(d){
 var arr=sortedCids();
 var i=arr.indexOf(curCid);if(i<0)i=0;
 i=(i+d+arr.length)%arr.length;
 loadCid(arr[i]);
}
function loadCid(cid){
 view.tile=null;allMode=false;curCid=cid;
 build();tabs();syncSeg();home();
}
var DEF_KEYS={firms:"",eeProj:"",eeKey:""};
function loadKeys(){try{return JSON.parse(localStorage.getItem("cosmo_keys2"))||{}}catch(e){return {}}}
function fillKeys(){var k=Object.assign({},DEF_KEYS,loadKeys());
 document.getElementById("inFirms").value=k.firms;
 document.getElementById("inEeProj").value=k.eeProj;
 document.getElementById("inEeKey").value=k.eeKey;}
function togglePanel(id){
 ["listCid","drawbar","debug","info","onl"].forEach(function(p){
  var el=document.getElementById(p);
  var was=el.style.display=="block";
  el.style.display=(p==id&&!was)?"block":"none";});
 if(id=="debug")fillKeys();
 if(id=="listCid")refreshList();
}
function refreshList(){
 var q=document.getElementById("qCid").value.trim().toLowerCase();
 var box=document.getElementById("listCid2");box.innerHTML="";
 sortedCids().forEach(function(cid){
  if(q&&cid.toLowerCase().indexOf(q)<0)return;
  var t=findTile(cid);
  var b=document.createElement("button");
  b.textContent=cid+(t.b?" · гарь "+t.b+" px":" · пусто");
  b.onclick=function(){loadCid(cid);};
  box.appendChild(b);
 });
}
function decodeProfiles(){
 var i=0;
 function step(){
  var end=Math.min(i+20,PACK.tiles.length);
  for(;i<end;i++){
   (function(t){
    var img=new Image();
    img.onload=function(){
     var c=document.createElement("canvas");c.width=16;c.height=16;
     var ctx=c.getContext("2d");ctx.drawImage(img,0,0,16,16);
     var d=ctx.getImageData(0,0,16,16).data,s=0;
     for(var k=0;k<256;k++)s+=(d[k*4]+d[k*4+1]+d[k*4+2])/3;
     riseR[t.cid]=s/256/255;
    };
    img.src=t.dnbr512;
   })(PACK.tiles[i]);
  }
  if(end<PACK.tiles.length)setTimeout(step,0);
 }
 step();
}
function thOf(t){
 var H=+document.getElementById("height").value/200;
 var r=(riseR[t.cid]||0)*H;
 var v1=new THREE.Vector3(t.pos[0],t.pos[1],r);v1.project(camera);
 var v2=new THREE.Vector3(t.pos[0]+5.1,t.pos[1],r);v2.project(camera);
 var half=Math.abs((v2.x+1)/2*innerWidth-(v1.x+1)/2*innerWidth);
 return Math.max(24,half*1.1);
}
function pickTile(mx,my){
 var H=+document.getElementById("height").value/200;
 var best=null,bd=1e9;
 PACK.tiles.forEach(function(t){
  if(!t.pos)return;
  var r=(riseR[t.cid]||0)*H;
  var v=new THREE.Vector3(t.pos[0],t.pos[1],r);
  v.project(camera);
  if(v.z>1)return;
  var sx=(v.x+1)/2*innerWidth,sy=(1-v.y)/2*innerHeight;
  var d=(sx-mx)*(sx-mx)+(sy-my)*(sy-my);
  if(d<bd){bd=d;best=t;}
 });
 if(best&&bd<=thOf(best))return best;
 return null;
}
function init(){
 scene=new THREE.Scene();scene.background=new THREE.Color(0x0d1117);
 camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.01,4000);camera.position.set(0,-100,80);
 renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setSize(innerWidth,innerHeight);document.body.appendChild(renderer.domElement);
 controls=new THREE.OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
 var grid=new THREE.GridHelper(180,18,0x30363d,0x21262d);grid.position.z=-8;scene.add(grid);
 var sun=new THREE.DirectionalLight(0xffffff,0.9);sun.position.set(60,-40,120);scene.add(sun);
 var fill=new THREE.DirectionalLight(0x8899bb,0.35);fill.position.set(-50,30,80);scene.add(fill);
 scene.add(new THREE.AmbientLight(0xffffff,0.55));
 decodeProfiles();
 var bTimer=null;
 function rebuild(){if(bTimer)clearTimeout(bTimer);bTimer=setTimeout(function(){bTimer=null;build();},100);}
 document.getElementById("height").oninput=rebuild;document.getElementById("alpha").oninput=rebuild;
var statsEl=document.getElementById("stats");
statsEl.addEventListener("mouseover",function(e){
 var t=e.target.closest?e.target.closest(".tt"):null;
 var tip=document.getElementById("tip");
 if(t){tip.textContent=t.getAttribute("title")||"";tip.style.display="block";}
 else tip.style.display="none";
});
statsEl.addEventListener("mouseout",function(){
 document.getElementById("tip").style.display="none";});
 document.getElementById("bHome").onclick=home;
 document.getElementById("bTop").onclick=topView;
 document.getElementById("bRot").onclick=rot90;
 document.getElementById("mSingle").onclick=function(){view.tile=null;allMode=false;build();tabs();syncSeg();home();};
 document.getElementById("mAll").onclick=function(){view.tile=null;allMode=true;build();tabs();syncSeg();home();};
 document.getElementById("bLayer1").onclick=function(){layerMode="true";build();syncSeg();};
 document.getElementById("bLayer2").onclick=function(){layerMode="false";build();syncSeg();};
 document.getElementById("bLayer3").onclick=function(){layerMode="dnbr";build();syncSeg();};
 document.getElementById("bMarkup1").onclick=function(){overlayMode="none";build();syncSeg();};
 document.getElementById("bMarkup2").onclick=function(){overlayMode="ans";build();syncSeg();};
 document.getElementById("bMarkup3").onclick=function(){overlayMode="gt";build();syncSeg();};
 document.getElementById("bMarkup4").onclick=function(){overlayMode="both";build();syncSeg();};
 document.getElementById("bFirUpd").onclick=function(){fir.pts=null;fir.ts=0;fir.busy=false;renderFIRMS();};
 FIR_AREAS.forEach(function(B,i){
  document.getElementById("bAr"+i).onclick=function(){
   firArea=i;fir.pts=null;fir.ts=0;fir.busy=false;
   FIR_AREAS.forEach(function(B2,j){document.getElementById("bAr"+j).className=(j==i)?"on":"";});
   if(view.src=="fir")renderFIRMS();
  };
 });
 FIR_AREAS.forEach(function(B2,j){document.getElementById("bAr"+j).className=(j==firArea)?"on":"";});
 var FIR_L=["Снимок","SWIR 721","Aqua","Sentinel-2"];
 function syncFirL(){FIR_L.forEach(function(n,j){document.getElementById("bL"+j).className=(j==firLayer)?"on":"";});}
 FIR_L.forEach(function(n,j){
  document.getElementById("bL"+j).onclick=function(){firLayer=j;syncFirL();drawFIR();};
 });
 syncFirL();
 document.getElementById("bOnl").onclick=function(){runOnl(0);};
 document.getElementById("onlX").onclick=function(){togglePanel("onl");};
 document.getElementById("bML").onclick=checkML;
 document.getElementById("bDrape").onclick=drapePhoto;
 document.getElementById("bUndrape").onclick=undrapePhoto;
 [["prodAuto",null],["prodTerra",0],["prodAqua",1],["prodViirs",2]].forEach(function(p){
  document.getElementById(p[0]).onclick=function(){prodSel=p[1];syncProd();runOnl(1);};
 });
 function syncProd(){[["prodAuto",null],["prodTerra",0],["prodAqua",1],["prodViirs",2]].forEach(function(p){
  document.getElementById(p[0]).className=(prodSel==p[1])?"on":"";});}
 syncProd();
var BBOX_ONL=[91.9,55.4,94.2,56.6];
var prodSel=null;
var CITIES=[["Красноярск",56.01,92.87],["Берёзовка",56.02,93.09],["Солнечный",56.11,93.24],["Емельяново",56.15,92.71],["Сосновоборск",56.13,93.35],["Дивногорск",55.94,92.37],["Зеленогорск",55.95,94.60],["Канск",56.20,95.71],["Ачинск",56.27,90.50],["Назарово",56.00,90.40]];
function hav(la1,lo1,la2,lo2){
 var R=6371,dLa=(la2-la1)*Math.PI/180,dLo=(lo2-lo1)*Math.PI/180;
 var a=Math.sin(dLa/2)*Math.sin(dLa/2)+Math.cos(la1*Math.PI/180)*Math.cos(la2*Math.PI/180)*Math.sin(dLo/2)*Math.sin(dLo/2);
 return 2*R*Math.asin(Math.sqrt(a));
}
function runOnl(skip){
 if(!skip)togglePanel("onl");
 var line=document.getElementById("onlLine"),out=document.getElementById("onlOut");
 var cv=document.getElementById("onlCv"),ctx=cv.getContext("2d");
 ctx.fillStyle="#0b0f14";ctx.fillRect(0,0,512,512);
 ctx.fillStyle="#8b949e";ctx.font="14px system-ui";
 ctx.fillText("грузим спутник и очаги...",140,220);
 line.textContent="пинг FIRMS + спутник MODIS Terra...";
 out.textContent="";
 var k=Object.assign({},DEF_KEYS,loadKeys());
 var st="https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY="+k.firms;
 var qs=encodeURIComponent(st);
 tryText([st,"https://api.allorigins.win/raw?url="+qs,"https://api.codetabs.com/v1/proxy?quest="+qs],0,function(t){
  try{var j=JSON.parse(t);
   out.textContent="FIRMS: транзакций "+(j.current_transactions!=null?j.current_transactions:"?")+"/"+j.transaction_limit+" за "+(j.transaction_interval||"?");
  }catch(e){out.textContent="пинг: ответ не JSON (ключ или сеть)";}
 });
 var area="https://firms.modaps.eosdis.nasa.gov/api/area/csv/"+k.firms+"/VIIRS_SNPP_NRT/"+BBOX_ONL[0]+","+BBOX_ONL[1]+","+BBOX_ONL[2]+","+BBOX_ONL[3]+"/2";
 var qa=encodeURIComponent(area);
 tryText([area,"https://api.allorigins.win/raw?url="+qa,"https://api.codetabs.com/v1/proxy?quest="+qa],0,function(t){
  var rows=[];
  String(t).split("\\n").slice(1).forEach(function(l){
   var p=l.split(",");
   var la=parseFloat(p[0]),lo=parseFloat(p[1]);
   if(isFinite(la)&&isFinite(lo))rows.push([la,lo,parseFloat(p[12])||1]);
  });
  fetchGibs(function(gcv,extent,day,prName){
   drawOnl(ctx,gcv,extent,day,rows,line,out,prName);
  },line);
 });
}
function fetchGibs(cb,line){
 var z=8,nw=Math.pow(2,z),N=256;
 function la2y(la){var r=la*Math.PI/180;return (1-Math.log(Math.tan(r)+1/Math.cos(r))/Math.PI)/2*nw;}
 function y2la(y){return Math.atan(Math.sinh(Math.PI*(1-2*y/nw)))*180/Math.PI;}
 function lo2x(lo){return (lo+180)/360*nw;}
 var x0=Math.floor(lo2x(BBOX_ONL[0])),x1=Math.floor(lo2x(BBOX_ONL[2]));
 var y0=Math.floor(la2y(BBOX_ONL[3])),y1=Math.floor(la2y(BBOX_ONL[1]));
 var cv=document.createElement("canvas");cv.width=(x1-x0+1)*N;cv.height=(y1-y0+1)*N;
 var cx=cv.getContext("2d");
 var extent=[x0/nw*360-180,y2la(y0),(x1+1)/nw*360-180,y2la(y1+1)];
 var PRODUCTS=[["MODIS_Terra_CorrectedReflectance_TrueColor","GoogleMapsCompatible_Level9","MODIS Terra (утренний)"],["MODIS_Aqua_CorrectedReflectance_TrueColor","GoogleMapsCompatible_Level9","MODIS Aqua (дневной)"],["VIIRS_SNPP_CorrectedReflectance_TrueColor","GoogleMapsCompatible_Level9","VIIRS SNPP"]];
 function bright(){
  try{
   var d=cx.getImageData(0,0,cv.width,cv.height).data,s=0,n=0;
   for(var i=0;i<d.length;i+=4096*4){s+=d[i]*0.3+d[i+1]*0.6+d[i+2]*0.1;n++;}
   return s/Math.max(1,n);
  }catch(e){return 255;}
 }
 function tryOne(products,pi,di){
  if(pi>=products.length){cb(null,extent,null);return;}
  var pr=products[pi];
  var d=new Date(Date.now()-di*86400000).toISOString().slice(0,10);
  line.textContent="грузим "+pr[2]+" за "+d+"...";
  cx.clearRect(0,0,cv.width,cv.height);
  var need=(x1-x0+1)*(y1-y0+1),done=0,ok=0;
  for(var xx=x0;xx<=x1;xx++)for(var yy=y0;yy<=y1;yy++){
   (function(xx,yy){
    var img=new Image();
    img.crossOrigin="anonymous";
    img.onload=function(){done++;ok++;
     cx.drawImage(img,(xx-x0)*N,(yy-y0)*N);
     if(done>=need){
      if(bright()<15&&di<2)tryOne(products,pi,di+1);
      else if(bright()<15)tryOne(products,pi+1,0);
      else cb(cv,extent,d,pr[2]);
     }};
    img.onerror=function(){done++;
     if(done>=need){
      if(ok>0)cb(cv,extent,d,pr[2]);
      else if(di<2)tryOne(products,pi,di+1);
      else tryOne(products,pi+1,0);
     }};
    img.src="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"+pr[0]+"/default/"+d+"/"+pr[1]+"/"+z+"/"+yy+"/"+xx+".jpg";
   })(xx,yy);
  }
 }
 tryOne(prodSel==null?PRODUCTS:[PRODUCTS[prodSel]],0,0);
}
var onlLast={gcv:null,extent:null,day:null,rows:[],cells:[]};
function mapXY(la,lo,extent){
 return [(lo-extent[0])/(extent[2]-extent[0])*512,(extent[1]-la)/(extent[1]-extent[3])*512];
}
function drawOnl(ctx,gcv,extent,day,rows,line,out,prName){
 var W=512,H=512;
 ctx.clearRect(0,0,W,H);ctx.fillStyle="#0b0f14";ctx.fillRect(0,0,W,H);
 if(gcv)ctx.drawImage(gcv,0,0,W,H);
 line.textContent=(prName?prName+(day?" "+day:""):"спутник недоступен")+" · FIRMS VIIRS NRT 2 сут: "+rows.length+" термоточек";
 var cd={};
 rows.forEach(function(r){
  var k=Math.round(r[0]/0.02)+"_"+Math.round(r[1]/0.02);
  if(!cd[k])cd[k]={la:0,lo:0,n:0,frp:0};
  var c=cd[k];c.la+=r[0];c.lo+=r[1];c.n++;c.frp+=r[2];
 });
 var cells=[];
 for(var k in cd)cells.push(cd[k]);
 onlLast={gcv:gcv,extent:extent,day:day,rows:rows,cells:cells};
 if(!rows.length){out.textContent="активных термоточек нет — в самом городе их обычно нет (активные точки сейчас за сотни км); весь регион со снимком и очагами — FIRMS-режим (источник: FIRMS очаги)";return;}
 var warns=[];
 cells.forEach(function(c){
  var p=mapXY(c.la/c.n,c.lo/c.n,extent);
  var col=c.frp<30?"#2ecc71":(c.frp<100?"#f1c40f":"#e74c3c");
  ctx.strokeStyle="rgba(255,255,255,0.9)";ctx.lineWidth=1.5;
  ctx.fillStyle=col;ctx.beginPath();ctx.arc(p[0],p[1],4+Math.min(10,c.n*1.5),0,Math.PI*2);ctx.fill();ctx.stroke();
  var best=null,bd=1e9;
  CITIES.forEach(function(c2){var d=hav(c.la/c.n,c.lo/c.n,c2[1],c2[2]);if(d<bd){bd=d;best=c2;}});
  var lvl=c.frp<30?"НИЗКИЙ":(c.frp<100?"СРЕДНИЙ":"ВЫСОКИЙ");
  if(bd<40)warns.push("⚠ очаг ("+c.n+" точек, FRP "+Math.round(c.frp)+") в "+Math.round(bd)+" км от "+best[0]+" — уровень "+lvl);
 });
 var outTxt=warns.length?warns.join("<br>"):"в радиусе 40 км от населённых пунктов очагов нет";
 var rec=warns.length?"<br><span style='color:#8b949e'>рекомендация: ВЫСОКИЙ — авиаразведка бортом + оповестить администрацию; СРЕДНИЙ — авиаразведка, наземная группа в готовности; НИЗКИЙ — контроль следующим проходом спутника</span>":"";
 out.innerHTML=outTxt+rec;
}
function checkML(){
 var line=document.getElementById("onlLine"),out=document.getElementById("onlOut");
 var cv=document.getElementById("onlCv"),ctx=cv.getContext("2d");
 if(!onlLast.gcv||!onlLast.extent){out.textContent="сначала грузи карту (кнопка «онлайн: Красноярск»)";return;}
 line.textContent="ML-проверка: анализ снимка (тёплые зоны)...";
 var g=onlLast.gcv,gx=g.getContext("2d");
 var img;
 try{img=gx.getImageData(0,0,g.width,g.height).data;}
 catch(e){out.textContent="ML: снимок защищён (CORS) — перезапусти сеть/панель";return;}
 var sx=512/g.width,sy=512/g.height;
 var grid={};
 for(var y=0;y<g.height;y+=2)for(var x=0;x<g.width;x+=2){
  var i=(y*g.width+x)*4;
  var r=img[i],gg=img[i+1],b=img[i+2];
  if(r>150&&r-b>40&&r-gg>15){
   var k=Math.floor(x*sx/8)+"_"+Math.floor(y*sy/8);
   grid[k]=(grid[k]||0)+1;
  }
 }
 var warm=[];
 for(var k in grid){
  var parts=k.split("_");
  if(grid[k]>=3)warm.push([(+parts[0])*8+4,(+parts[1])*8+4]);
 }
 line.textContent="ML-проверка: тёплых зон "+warm.length+" · термоточек "+onlLast.cells.length;
 drawOnl(ctx,onlLast.gcv,onlLast.extent,onlLast.day,onlLast.rows,line,out,null);
 ctx.strokeStyle="#ff4fd8";ctx.lineWidth=2;
 warm.forEach(function(w){ctx.strokeRect(w[0]-7,w[1]-7,14,14);});
 var ext=onlLast.extent;
 var confirmed=0,cand=0;
 warm.forEach(function(w){
  var lo=ext[0]+w[0]/512*(ext[2]-ext[0]);
  var la=ext[1]-w[1]/448*(ext[1]-ext[3]);
  var near=false;
  onlLast.cells.forEach(function(c){
   if(Math.abs(c.la/c.n-la)<0.06&&Math.abs(c.lo/c.n-lo)<0.06)near=true;
  });
  if(near)confirmed++;else cand++;
 });
 var remote=0;
 onlLast.cells.forEach(function(c){
  var p=mapXY(c.la/c.n,c.lo/c.n,ext);
  var isWarm=warm.some(function(w){return Math.abs(w[0]-p[0])<12&&Math.abs(w[1]-p[1])<12;});
  if(!isWarm)remote++;
 });
 var verdict=(confirmed+cand)>0?
  ("очаги "+((confirmed+cand)>2?"множественные":"единичные")+" — за сутки в области "+(confirmed>0?"вероятно АКТИВНОЕ горение":"пожарной активности мало")+""):
  "признаков активного горения на снимке не найдено";
 out.innerHTML="ML за 48 ч: подтверждено (фото+термоточки): <b>"+confirmed+"</b> · кандидатов (только тёплые зоны): <b>"+cand+"</b> · термоточек без тёплых зон (дым/облака): <b>"+remote+"</b>"+
  "<br><span style='color:#8b949e'>розовые рамки — тёплые зоны, найденные ML на снимке; вывод: "+verdict+"</span>";
}
function drapePhoto(){
 var out=document.getElementById("onlOut");
 if(!onlLast.gcv){out.textContent="сначала грузи карту (кнопка «онлайн: Красноярск»)";return;}
 if(allMode||view.src=="fir"||!ground){out.textContent="открой чип крупно (кнопка «чип крупно»), потом жми снова";return;}
 var dcv=document.createElement("canvas");dcv.width=512;dcv.height=512;
 var dx=dcv.getContext("2d");
 dx.drawImage(onlLast.gcv,0,0,512,512);
 var ext=onlLast.extent;
 onlLast.cells.forEach(function(c){
  var la=c.la/c.n,lo=c.lo/c.n;
  var xx=(lo-ext[0])/(ext[2]-ext[0])*512;
  var yy=(ext[1]-la)/(ext[1]-ext[3])*512;
  var col=c.frp<30?"#2ecc71":(c.frp<100?"#f1c40f":"#e74c3c");
  dx.strokeStyle="rgba(255,255,255,0.9)";dx.lineWidth=2;
  dx.fillStyle=col;dx.beginPath();dx.arc(xx,yy,5+Math.min(12,c.n*1.5),0,Math.PI*2);dx.fill();dx.stroke();
 });
 ground.material.map=new THREE.CanvasTexture(dcv);
 ground.material.color.set(0xffffff);ground.material.needsUpdate=true;
 out.innerHTML="фото наложено на чип (вместо снимка датасета) — очаги нарисованы; «вернуть чип» вернёт датасет";
}
function undrapePhoto(){
 build();
 var o=document.getElementById("onlOut");
 if(o)o.innerHTML="чип возвращён (снимок датасета)";
}
function firML(){
 if(view.src!="fir"||!ground)return -1;
 var tex=ground.material.map,img=tex&&tex.image;
 if(!img||!img.getContext)return -1;
 var c2=img.getContext("2d"),W=img.width,H=img.height,data;
 try{data=c2.getImageData(0,0,W,H);}catch(e){return -2;}
 var d=data.data,cell=8,grid={};
 for(var y=0;y<H;y+=2)for(var x=0;x<W;x+=2){
  var i=(y*W+x)*4,r=d[i],g=d[i+1],b=d[i+2];
  if(r>150&&r-b>40&&r-g>15){
   var k=Math.floor(x/cell)+"_"+Math.floor(y/cell);
   grid[k]=(grid[k]||0)+1;
  }
 }
 var n=0;
 c2.strokeStyle="#ff4fd8";c2.lineWidth=4;
 for(var k in grid){
  if(grid[k]<12)continue;
  var p=k.split("_");
  c2.strokeRect((+p[0])*cell-24,(+p[1])*cell-24,48,48);
  n++;
 }
 tex.needsUpdate=true;
 return n;
}
document.getElementById("bFirML").onclick=function(){
 var n=firML(),o=document.getElementById("stats");
 if(o)o.innerHTML=n>=0?("ML-зон: "+n+" (розовые рамки) — RGB-эвристика тёплых зон; сверь с оранжевыми точками FIRMS"):(n==-1?"ML: открой FIRMS-вид":n==-2?"ML: снимок защищён CORS":"");
};
 document.getElementById("bList").onclick=function(){togglePanel("listCid");};
 document.getElementById("bDraw").onclick=function(){togglePanel("drawbar");};
 document.getElementById("bDebug").onclick=function(){togglePanel("debug");};
 document.getElementById("bInfo").onclick=function(){togglePanel("info");};
 document.getElementById("listX").onclick=function(){togglePanel(null);};
 document.getElementById("drawX").onclick=function(){togglePanel(null);};
 document.getElementById("debugX").onclick=function(){togglePanel(null);};
 document.getElementById("infoX").onclick=function(){togglePanel(null);};
 document.getElementById("bSortNum").onclick=function(){resortMode="num";refreshList();};
 document.getElementById("bSortBurn").onclick=function(){resortMode="burn";refreshList();};
 document.getElementById("bApply").onclick=function(){resortTiles();build();};
 document.getElementById("qCid").oninput=refreshList;
 document.getElementById("p1").onclick=function(){setPaint(paint.mode=="p1"?null:"p1");};
 document.getElementById("p2").onclick=function(){setPaint(paint.mode=="p2"?null:"p2");};
 document.getElementById("p3").onclick=function(){setPaint(paint.mode=="p3"?null:"p3");};
 document.getElementById("pE").onclick=function(){setPaint(paint.mode=="pE"?null:"pE");};
 document.getElementById("pClear").onclick=function(){
 paint.hist[paintCid()]=[];
 var c=paint.canvases[paintCid()];c.getContext("2d").clearRect(0,0,512,512);
 paint.tex.needsUpdate=true;};
document.getElementById("pUndo").onclick=undoStroke;
 document.getElementById("pTest").onclick=runTest;
 document.getElementById("pCsv").onclick=runCsv;
 ["th1","th2","th3"].forEach(function(id){
  document.getElementById(id).onclick=function(){
   setPaint({th1:"p1",th2:"p2",th3:"p3"}[id]);};});
 document.getElementById("inSave").onclick=function(){
  var v={firms:document.getElementById("inFirms").value.trim(),
   eeProj:document.getElementById("inEeProj").value.trim(),
   eeKey:document.getElementById("inEeKey").value.trim()};
  try{localStorage.setItem("cosmo_keys2",JSON.stringify(v));
   document.getElementById("pingout").textContent="ключи сохранены (localStorage этого браузера)";}
  catch(e){document.getElementById("pingout").textContent="не удалось сохранить: "+e;}};
 document.getElementById("inReset").onclick=function(){fillKeys();
  document.getElementById("pingout").textContent="показаны значения по умолчанию";};
 document.getElementById("inPing").onclick=function(){
  var k=Object.assign({},DEF_KEYS,loadKeys());
  k.firms=document.getElementById("inFirms").value.trim()||k.firms;
  var out=document.getElementById("pingout");
  var api="https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY="+k.firms;
  var q=encodeURIComponent(api);
  out.textContent="пингуем FIRMS...";
  var done=function(t){
   try{
    var j=JSON.parse(t);
    var txt="FIRMS MAP_KEY: OK";
    if(j.transaction_limit!=null)txt+=" · транзакций "+(j.current_transactions!=null?j.current_transactions:"?")+"/"+j.transaction_limit+" за "+(j.transaction_interval||"интервал");
    else txt+=" · "+JSON.stringify(j);
    out.textContent=txt;
   }catch(e){
    out.innerHTML='ответ не JSON (проверьте NASA FIRMS MAP_KEY в панели Debug): '+String(t).slice(0,120);
   }
  };
  if(!tryText([api,"https://api.allorigins.win/raw?url="+q,"https://api.codetabs.com/v1/proxy?quest="+q],0,done)){
   out.innerHTML='CORS блокирует прямые запросы — проверь вручную: <a href="'+api+'" target="_blank">'+api+'</a>';
  }};
 addEventListener("keydown",function(e){
  var ae=document.activeElement;
  if(ae&&(ae.tagName=="INPUT"||ae.tagName=="TEXTAREA"))return;
  keys[e.code]=true;
  var k=e.key.toLowerCase();
  if(k=="escape"){
   if(view.tile)closeTile();
   setPaint(null);
   togglePanel(null);
  }
 if((e.ctrlKey||e.metaKey)&&(k=="z"||k=="я")){e.preventDefault();undoStroke();return;}
  if(paint.mode)return;
  if(k==",")cycle(-1);
  if(k==".")cycle(1);
  if(k=="r")home();
  if(k=="t")topView();
  if(k=="1"){layerMode="true";build();syncSeg();}
  if(k=="2"){layerMode="false";build();syncSeg();}
  if(k=="3"){layerMode="dnbr";build();syncSeg();}
 });
 addEventListener("keyup",function(e){keys[e.code]=false;});
 renderer.domElement.addEventListener("mousedown",function(e){
 if(!paint.mode||e.button!==0)return;
 paint.stroke=null;paint.drawing=true;paintDot(e,paint.mode=="pE");});
 renderer.domElement.addEventListener("mousemove",function(e){
  if(paint.drawing){paintDot(e,paint.mode=="pE");return;}
  if(allMode&&!view.tile&&!paint.drawing){
   var t=pickTile(e.clientX,e.clientY);
   var tip=document.getElementById("tip");
   if(hovered&&(!t||t!==hovered)){hovered.mesh.material.color.set(0xffffff);hovered.mesh.material.emissive.set(0x000000);hovered=null;}
   if(t&&t!==hovered){hovered=t;t.mesh.material.color.set(0xcfe6ff);t.mesh.material.emissive.set(0x1a4a80);}
   renderer.domElement.style.cursor=t?"pointer":"";
   if(t){tip.textContent=t.cid;tip.style.display="block";}
   else tip.style.display="none";
  }
 });
 addEventListener("mouseup",function(){
 if(paint.drawing&&paint.stroke&&paint.stroke.dots.length)
  (paint.hist[paintCid()]=paint.hist[paintCid()]||[]).push(paint.stroke);
 paint.stroke=null;paint.drawing=false;});
 renderer.domElement.addEventListener("dblclick",function(e){
  if(paint.mode||!allMode||view.tile)return;
  var t=pickTile(e.clientX,e.clientY);
  if(t){view.tile=t;home();build();tabs();syncSeg();}
 });
 addEventListener("resize",function(){camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
 pingHelper();
build();tabs();syncSeg();
 (function loop(){requestAnimationFrame(loop);
  if(controls.enabled&&!paint.mode){
   var sp=camera.position.distanceTo(controls.target)*0.02+0.2;
   var dir=new THREE.Vector3();camera.getWorldDirection(dir);
   var fwd=new THREE.Vector3(dir.x,dir.y,0);
   if(fwd.lengthSq()<1e-6)fwd.set(0,1,0);
   fwd.normalize();
   var right=new THREE.Vector3(fwd.y,-fwd.x,0);
   var mv=new THREE.Vector3();
   if(keys.KeyW)mv.add(fwd);if(keys.KeyS)mv.sub(fwd);
   if(keys.KeyD)mv.add(right);if(keys.KeyA)mv.sub(right);
   if(keys.KeyE)mv.add(new THREE.Vector3(0,0,1));if(keys.KeyQ)mv.sub(new THREE.Vector3(0,0,1));
   if(mv.lengthSq()>0){mv.normalize().multiplyScalar(sp);camera.position.add(mv);controls.target.add(mv);}
  }
  controls.update();renderer.render(scene,camera);
 })();
}
init();
</script></body></html>
"""
if __name__ == "__main__":
    raise SystemExit(main())
