from __future__ import annotations

import atexit
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.serving import make_server

from config import CFG
from vision import VisionEngine

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder="templates", static_folder="static")
vision = VisionEngine()
_server = None
_shutdown_lock = threading.Lock()


@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(ok=True, source="browser-camera", version="2.1.1")


@app.route("/api/config", methods=["GET", "POST"])
def config_api():
    if request.method == "GET":
        return jsonify(CFG.as_dict())
    patch = request.get_json(silent=True) or {}
    return jsonify(CFG.update(**patch))


@app.post("/api/frame")
def frame_api():
    # One in-flight request on the client means no stale frame queue can build up.
    payload = request.get_data(cache=False, as_text=False)
    if len(payload) > 1_500_000:
        return jsonify(error="frame too large"), 413
    return jsonify(vision.process_jpeg(payload))


@app.get("/api/state")
def state_api():
    return jsonify(vision.state())


@app.post("/api/action")
def action_api():
    body = request.get_json(silent=True) or {}
    action = str(body.get("action", "")).strip().lower()
    if action not in {"fire", "charge", "reset_charge"}:
        return jsonify(error="unknown action"), 400
    source = str(body.get("source", "manual")).strip().lower() or "manual"
    return jsonify(vision.action(action, source=source))


@app.post("/api/reset")
def reset_api():
    return jsonify(vision.reset_runtime())


@app.post("/api/score")
def score_api():
    body = request.get_json(silent=True) or {}
    try:
        score = int(body.get("score", 0))
    except (TypeError, ValueError):
        return jsonify(error="invalid score"), 400
    if score > CFG.best_score:
        CFG.best_score = score
        CFG.save()
    return jsonify(best_score=CFG.best_score)


@app.post("/shutdown")
def shutdown_api():
    global _server
    with _shutdown_lock:
        server = _server
        if server is None:
            return jsonify(ok=False, error="server unavailable"), 503
        threading.Thread(target=_delayed_shutdown, args=(server,), daemon=True).start()
    return jsonify(ok=True)


def _delayed_shutdown(server) -> None:
    time.sleep(0.15)
    server.shutdown()


@atexit.register
def _cleanup() -> None:
    try:
        vision.close()
    except Exception:
        pass


def main() -> None:
    global _server
    print("SmileGun v2.1.1: http://127.0.0.1:5000", flush=True)
    print("Камеру и микрофон открывает браузер; Python анализирует только кадры.", flush=True)
    _server = make_server("127.0.0.1", 5000, app, threaded=True)
    try:
        _server.serve_forever()
    finally:
        _server = None


if __name__ == "__main__":
    main()
