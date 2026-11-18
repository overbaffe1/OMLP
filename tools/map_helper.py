#!/usr/bin/env python3
"""Локальный хелпер карты: страница POST-ит разметку — файлы пишутся прямо в папку.

    py -3 tools/map_helper.py          # порт 8765
    py -3 tools/map_helper.py 8777     # другой порт

Страница карты отправляет POST /overlay {"overlay": {"BS_te_000002": [[u, v, deg], ...]}}:
  1) пишет paint_overlay.json
  2) запускает tools/apply_paint.py (база: submission_probe_35.csv, если есть)
     --out submission_map.csv --score --score-dir <данные>/train
  3) возвращает {ok, score, iou_burn, miou_sev, csv, base, log}

GET /ping — проверка живости. Окно можно сверить; остановка — Ctrl+C.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parents[1] if (ROOT.parents[1] / "train").exists() else ROOT.parent
BASE = ROOT / "submission_probe_35.csv"
if not BASE.exists():
    BASE = ROOT / "submission.csv"
OUT_JSON = ROOT / "paint_overlay.json"
OUT_CSV = ROOT / "submission_map.csv"

LAST: dict = {"ok": None, "text": "хелпер запущен, разметки ещё не было", "ts": 0}


def run_overlay(doc: dict) -> dict:
    overlay = doc.get("overlay") if isinstance(doc, dict) else None
    if not overlay:
        return {"ok": False, "text": "пустой оверлей — нарисуй точки кистью"}
    OUT_JSON.write_text(json.dumps({"overlay": overlay}, ensure_ascii=False), encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "tools" / "apply_paint.py"),
           "--overlay", str(OUT_JSON), "--submission", str(BASE),
           "--out", str(OUT_CSV), "--score", "--score-dir", str(DATA / "train")]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, encoding="utf-8")
        out = ((p.stdout or "") + (p.stderr or "")).strip()
    except Exception as e:
        LAST.update({"ok": False, "text": f"ошибка запуска apply_paint: {e}", "ts": 0})
        return {"ok": False, "text": f"ошибка запуска apply_paint: {e}"}
    m = re.search(r"score=([0-9.]+)", out)
    iou = re.search(r"iou_burn=([0-9.]+)", out)
    miou = re.search(r"miou_sev=([0-9.]+)", out)
    LAST.update({"ok": p.returncode == 0, "text": out[-600:], "ts": 0})
    return {"ok": p.returncode == 0,
            "score": m.group(1) if m else None,
            "iou_burn": iou.group(1) if iou else None,
            "miou_sev": miou.group(1) if miou else None,
            "csv": OUT_CSV.name, "json": OUT_JSON.name, "base": BASE.name,
            "log": out[-600:]}


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/ping", "/status"):
            self._json({"ping": True, **LAST})
        elif self.path in ("/catalog", "/service/catalog.json"):
            f = Path(__file__).resolve().parent.parent / "service" / "catalog.json"
            if not f.exists():
                self._json({"error": "catalog not found"}, 404)
                return
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        if self.path != "/overlay":
            self._json({"ok": False, "text": "только POST /overlay"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            doc = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            self._json({"ok": False, "text": f"битый JSON: {e}"}, 400)
            return
        self._json(run_overlay(doc))

    def log_message(self, *a):
        pass


def main(argv=None):
    argv = argv if argv is not None else sys.argv
    port = int(argv[1]) if len(argv) > 1 else 8765
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"хелпер карты: http://127.0.0.1:{port}  (ROOT={ROOT})")
    print(f"база: {BASE.name} -> результат: {OUT_CSV.name} + {OUT_JSON.name}")
    print("окно можно сверить; остановка — Ctrl+C")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("остановлено")


if __name__ == "__main__":
    main(sys.argv)
