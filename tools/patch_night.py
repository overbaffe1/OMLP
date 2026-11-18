"""Разовая правка: ночные пороги VIIRS в af.py.

Запускается один раз, результат закоммичен в ``fires/af.py``; скрипт оставлен как след правки.
"""
import pathlib
import re

p = pathlib.Path(__file__).resolve().parents[1] / "fires" / "af.py"
s = p.read_text(encoding="utf-8")

old = """    sat_t4: float = 367.0            # насыщение канала I4
    absolute_t4: float = 360.0       # абсолютный порог «очень яркого» очага
    candidate_t4: float = 320.0      # минимум для контекстного теста"""
new = """    sat_t4: float = 367.0            # насыщение канала I4 (день)
    absolute_t4: float = 360.0       # абсолютный порог «очень яркого» очага (день)
    candidate_t4: float = 320.0      # минимум для контекстного теста (день)
    #: Ночные пороги. Ночью солнечный нагрев выключен: фон в I4 падает примерно до 283 K,
    #: поэтому «дневная» планка 320 K отсекала бы почти все ночные очаги.
    sat_t4_night: float = 340.0
    absolute_t4_night: float = 305.0
    candidate_t4_night: float = 293.0"""
assert old in s, "params block not found"
s = s.replace(old, new, 1)

pat = re.compile(
    r"    # 2\. абсолютные тесты\n"
    r"    saturated = i4 >= p\.sat_t4\n"
    r"    absolute = i4 >= p\.absolute_t4\n"
    r"    candidate = screen & \(\(i4 >= p\.candidate_t4\) \| absolute \| saturated\)\n"
    r"\n"
    r"    # 3\. контекстный тест\n"
    r"    mean_bg, std_bg, counts = _background_stats\(i4, candidate, screen, p\.bg_window\)\n"
    r"    mean_bg = np\.where\(counts > 0, mean_bg, np\.nanmean\(i4\[screen\]\) if screen\.any\(\) else i4\)\n"
    r"    delta = i4 - i5\n"
    r"    night = sun_zenith is not None and float\(np\.median\(np\.asarray\(sun_zenith\)\)\) >= p\.night_sun_zenith\n"
    r"    delta_threshold = p\.delta_night if night else p\.delta_day"
)
rep = """    # 2. абсолютные тесты; пороги зависят от времени суток
    night = bool(sun_zenith is not None and float(np.nanmedian(np.asarray(sun_zenith))) >= p.night_sun_zenith)
    sat_threshold = p.sat_t4_night if night else p.sat_t4
    absolute_threshold = p.absolute_t4_night if night else p.absolute_t4
    candidate_threshold = p.candidate_t4_night if night else p.candidate_t4
    saturated = i4 >= sat_threshold
    absolute = i4 >= absolute_threshold
    candidate = screen & ((i4 >= candidate_threshold) | absolute | saturated)

    # 3. контекстный тест
    mean_bg, std_bg, counts = _background_stats(i4, candidate, screen, p.bg_window)
    mean_bg = np.where(counts > 0, mean_bg, np.nanmean(i4[screen]) if screen.any() else i4)
    delta = i4 - i5
    delta_threshold = p.delta_night if night else p.delta_day"""
s, n = pat.subn(rep, s, count=1)
assert n == 1, "absolute test block not found"

s, n1 = re.subn(r"& \(delta < delta_threshold \+ 3\.0\) & \(i4 < p\.absolute_t4\)",
                "& (delta < delta_threshold + 3.0) & (i4 < absolute_threshold)", s)
s, n2 = re.subn(r"\| \(i4 >= p\.sat_t4\)", "| (i4 >= sat_threshold)", s)
assert n1 == 1 and n2 == 1, (n1, n2)

s, n3 = re.subn(r'            "background_mean_t4": float\(np\.nanmean\(mean_bg\)\),',
                '            "background_mean_t4": float(np.nanmean(mean_bg)),\n'
                '            "night": bool(night),\n'
                '            "thresholds": {"sat": sat_threshold, "absolute": absolute_threshold, "candidate": candidate_threshold},',
                s)
assert n3 == 1, n3
p.write_text(s, encoding="utf-8")
print("af.py patched")
