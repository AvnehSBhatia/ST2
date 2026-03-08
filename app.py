from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from simulation_backend import get_job, start_simulation, stream_job


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")


@app.get("/api/health")
def health() -> Response:
    return jsonify({"ok": True})


@app.post("/api/simulate")
def simulate() -> Response:
    payload = request.get_json(silent=True) or {}
    narrative = str(payload.get("narrative", "")).strip()
    if not narrative:
        return jsonify({"error": "narrative is required"}), 400
    seed = payload.get("seed")
    if seed is not None:
        seed = int(seed)
    job = start_simulation(narrative=narrative, seed=seed)
    return jsonify({"job_id": job.job_id})


@app.get("/api/jobs/<job_id>")
def get_job_state(job_id: str) -> Response:
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job.snapshot())


@app.get("/api/jobs/<job_id>/events")
def job_events(job_id: str) -> Response:
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404

    def event_stream():
        for snapshot in stream_job(job):
            yield f"data: {json.dumps(snapshot, ensure_ascii=True)}\n\n"

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
def index() -> Response:
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/<path:asset_path>")
def assets(asset_path: str) -> Response:
    full_path = BASE_DIR / asset_path
    if not full_path.exists() or not full_path.is_file():
        abort(404)
    return send_from_directory(BASE_DIR, asset_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True, threaded=True)
