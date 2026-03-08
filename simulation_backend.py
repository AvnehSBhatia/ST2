from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI

from answer_bank import ANSWER_BANK_PATH, load_answer_bank
from embedding_engine import EmbeddingEngine
from kmean_graph import get_clustering_output
from network import kmeans_auto_k, normalize_vectors, pick_recipients
from preprocessor import PREPROCESSOR_CHECKPOINT, load_preprocessor
from run_full_pipeline import HYBRID_DATASET_PATH, NUM_AGENTS, get_100_agent_descriptions
from test_pipeline import (
    NEGATIVE_SENTIMENTS,
    NEUTRAL_TIE_THRESHOLD,
    POSITIVE_SENTIMENTS,
    resolve_batch_topk_results,
)
from train import get_device
from train_answer_predictor import CHECKPOINT_PATH, load_checkpoint, predict_answer_embeddings_batch


FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"
FEATHERLESS_MODEL = "deepseek-ai/DeepSeek-V3.2"
DEFAULT_FEATHERLESS_API_KEY = "rc_bdd22b5defe34bf473fb57147a3bff37fe6a5aaef9e34f193b5e0e6cd43d493b"
SIMULATION_BATCH_SIZE = 64
TOP_K = 5
SHARE_CHANCE_DENOMINATOR = 10
PROGRESS_UPDATE_INTERVAL = 100
GRAPH_UPDATE_INTERVAL = 1024
DEFAULT_DATASET_PATH = Path(HYBRID_DATASET_PATH)


def sentiment_to_like_value(sentiment_label: str) -> float:
    mapping = {
        "strong_like": 1.0,
        "like": 0.6,
        "neutral": 0.0,
        "dislike": -0.6,
        "strong_dislike": -1.0,
    }
    return mapping.get(sentiment_label, 0.0)


def sentiment_to_action(sentiment_label: str, should_share: bool) -> str:
    if sentiment_label in POSITIVE_SENTIMENTS:
        return "LIKE_SHARE" if should_share else "LIKE"
    if sentiment_label in NEGATIVE_SENTIMENTS:
        return "DISLIKE_SHARE" if should_share else "DISLIKE"
    return "NOTHING"


def display_action_from_action(action_name: str) -> str:
    if action_name.startswith("LIKE"):
        return "liked"
    if action_name.startswith("DISLIKE"):
        return "disliked"
    return "neutral"


def short_text(text: str, limit: int = 150) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "..."


def make_chart_series(history: list[dict[str, float]]) -> dict[str, Any]:
    if not history:
        history = [{"progress": 0.0, "liked": 0.0, "neutral": 100.0, "disliked": 0.0}]
    labels = [f"{round(point['progress'] * 100):d}%" for point in history]
    return {
        "labels": labels,
        "series": [
            {"key": "liked", "label": "Liked", "color": "#10b981", "values": [point["liked"] for point in history]},
            {"key": "neutral", "label": "Neutral", "color": "#9ca3af", "values": [point["neutral"] for point in history]},
            {"key": "disliked", "label": "Disliked", "color": "#ef4444", "values": [point["disliked"] for point in history]},
        ],
    }


def compute_stats(agents: list[dict[str, Any]], processed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    relevant = [agent for agent in agents[:processed] if agent.get("sentiment_label")]
    counts = Counter(agent["action"] for agent in relevant)
    total = max(processed, 1)
    stats = {
        "impressions": processed,
        "likes": counts.get("LIKE", 0) + counts.get("LIKE_SHARE", 0),
        "dislikes": counts.get("DISLIKE", 0) + counts.get("DISLIKE_SHARE", 0),
        "shares": counts.get("LIKE_SHARE", 0) + counts.get("DISLIKE_SHARE", 0),
        "comments": 0,
        "nothing": counts.get("NOTHING", 0),
    }
    reaction_bar = {
        "liked": round(100 * stats["likes"] / total, 1),
        "disliked": round(100 * stats["dislikes"] / total, 1),
        "shared": round(100 * stats["shares"] / total, 1),
        "comment": 0.0,
        "none": round(100 * stats["nothing"] / total, 1),
    }
    return stats, reaction_bar


def build_history_point(agents: list[dict[str, Any]], processed: int) -> dict[str, float]:
    relevant = [agent for agent in agents[:processed] if agent.get("sentiment_label")]
    total = max(len(relevant), 1)
    liked = sum(1 for agent in relevant if agent["sentiment_label"] in POSITIVE_SENTIMENTS)
    disliked = sum(1 for agent in relevant if agent["sentiment_label"] in NEGATIVE_SENTIMENTS)
    neutral = total - liked - disliked
    return {
        "progress": processed / max(len(agents), 1),
        "liked": round(100 * liked / total, 1),
        "neutral": round(100 * neutral / total, 1),
        "disliked": round(100 * disliked / total, 1),
    }


def build_analysis_snapshot(
    agents: list[dict[str, Any]],
    stats: dict[str, Any],
    processed: int,
    total: int,
    stage_text: str,
) -> dict[str, Any]:
    processed_agents = [agent for agent in agents[:processed] if agent.get("response_to_media")]
    responses = [short_text(agent["response_to_media"], limit=70) for agent in processed_agents if agent["response_to_media"]]
    positives = [text for agent, text in zip(processed_agents, responses) if agent["sentiment_label"] in POSITIVE_SENTIMENTS]
    negatives = [text for agent, text in zip(processed_agents, responses) if agent["sentiment_label"] in NEGATIVE_SENTIMENTS]
    neutrals = [text for agent, text in zip(processed_agents, responses) if agent["sentiment_label"] == "neutral"]
    mixed = positives[:2] + negatives[:2] + neutrals[:2]
    while len(mixed) < 4:
        mixed.append("Models are still evaluating the narrative.")

    globe_labels = mixed[:4]
    scroll_messages = [
        f"{stage_text} {processed}/{total} agents profiled.",
        f"Likes {stats['likes']} | Dislikes {stats['dislikes']} | Shares {stats['shares']}.",
        short_text(responses[-1], limit=80) if responses else "Waiting for the first model responses...",
    ]
    paragraphs = [
        f"The simulation has processed {processed} of {total} agents so far, with {stats['likes']} positive and {stats['dislikes']} negative reactions.",
        "As new batches complete, the belief-shift chart and network graph update from actual model outputs rather than placeholder data.",
        short_text("Representative response: " + (responses[-1] if responses else "No representative response yet."), limit=220),
    ]
    return {
        "globe_status": stage_text,
        "globe_labels": globe_labels,
        "scroll_messages": scroll_messages,
        "paragraphs": paragraphs,
    }


def generate_final_analysis(narrative: str, state: dict[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get("FEATHERLESS_API_KEY", DEFAULT_FEATHERLESS_API_KEY)
    if not api_key:
        return state["analysis"]

    stats = state["stats"]
    agents = state["agents"]
    samples = [
        {
            "uid": agent["uid"],
            "sentiment": agent["sentiment_label"],
            "action": agent["action"],
            "response": short_text(agent["response_to_media"], 180),
        }
        for agent in agents
        if agent.get("response_to_media")
    ][:12]

    client = OpenAI(base_url=FEATHERLESS_BASE_URL, api_key=api_key)
    prompt = {
        "narrative": narrative,
        "stats": stats,
        "reaction_bar": state["reaction_bar"],
        "sample_responses": samples,
        "request": {
            "return_format": "json",
            "keys": {
                "paragraphs": "three short paragraphs under 220 chars each",
                "globe_labels": "four short lines under 48 chars each",
                "scroll_messages": "three short status lines under 90 chars each",
                "globe_status": "one short status line under 60 chars",
            },
        },
    }
    response = client.chat.completions.create(
        model=FEATHERLESS_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that summarizes social-simulation output. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
        ],
    )
    content = response.model_dump()["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return {
        "globe_status": short_text(parsed.get("globe_status", state["analysis"]["globe_status"]), limit=60),
        "globe_labels": [short_text(text, limit=48) for text in parsed.get("globe_labels", state["analysis"]["globe_labels"])[:4]],
        "scroll_messages": [short_text(text, limit=90) for text in parsed.get("scroll_messages", state["analysis"]["scroll_messages"])[:3]],
        "paragraphs": [short_text(text, limit=220) for text in parsed.get("paragraphs", state["analysis"]["paragraphs"])[:3]],
    }


@dataclass
class ModelResources:
    device: torch.device
    engine: EmbeddingEngine
    preprocessor: Any
    predictor: Any
    bank: dict[str, Any]
    bank_norm: torch.Tensor


_MODEL_RESOURCES: ModelResources | None = None
_MODEL_LOCK = threading.Lock()


def get_model_resources() -> ModelResources:
    global _MODEL_RESOURCES
    with _MODEL_LOCK:
        if _MODEL_RESOURCES is not None:
            return _MODEL_RESOURCES

        device = get_device()
        engine = EmbeddingEngine()
        preprocessor = load_preprocessor(PREPROCESSOR_CHECKPOINT, device=device)
        preprocessor.eval()
        predictor, _checkpoint = load_checkpoint(CHECKPOINT_PATH, device=device)
        predictor.eval()
        bank = load_answer_bank(ANSWER_BANK_PATH, device=device)
        bank_norm = F.normalize(bank["answer_embeddings"].to(device), dim=-1)
        _MODEL_RESOURCES = ModelResources(
            device=device,
            engine=engine,
            preprocessor=preprocessor,
            predictor=predictor,
            bank=bank,
            bank_norm=bank_norm,
        )
        return _MODEL_RESOURCES


@dataclass
class SimulationJob:
    job_id: str
    narrative: str
    state: dict[str, Any]
    done: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    subscribers: list[Queue] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.state))

    def publish(self) -> None:
        snapshot = self.snapshot()
        for queue in list(self.subscribers):
            queue.put(snapshot)

    def update(self, **updates: Any) -> None:
        with self.lock:
            self.state.update(updates)
        self.publish()

    def subscribe(self) -> Queue:
        queue: Queue = Queue()
        with self.lock:
            self.subscribers.append(queue)
        queue.put(self.snapshot())
        return queue

    def unsubscribe(self, queue: Queue) -> None:
        with self.lock:
            if queue in self.subscribers:
                self.subscribers.remove(queue)


JOBS: dict[str, SimulationJob] = {}
JOBS_LOCK = threading.Lock()


def build_initial_state(narrative: str) -> dict[str, Any]:
    return {
        "job_id": "",
        "narrative": narrative,
        "status": "queued",
        "progress": {"stage": "queued", "processed": 0, "total": NUM_AGENTS, "percent": 0},
        "stats": {"impressions": 0, "likes": 0, "dislikes": 0, "shares": 0, "comments": 0, "nothing": 0},
        "reaction_bar": {"liked": 0, "disliked": 0, "shared": 0, "comment": 0, "none": 100},
        "chart": make_chart_series([]),
        "graph": {"nodes": [], "links": []},
        "shares": [],
        "agents": [],
        "analysis": {
            "globe_status": "Queueing simulation...",
            "globe_labels": [
                "Awaiting first model cluster...",
                "Awaiting first model cluster...",
                "Awaiting first model cluster...",
                "Awaiting first model cluster...",
            ],
            "scroll_messages": [
                "Simulation queued.",
                "Waiting for model resources...",
                "Live chart will fill as batches complete.",
            ],
            "paragraphs": [
                "The dashboard will switch from placeholder values to live model output as soon as the first batch completes.",
                "Belief shift, likes, dislikes, shares, and the network graph all update from the backend stream.",
                "Final commentary is generated from the run using the Featherless-hosted model.",
            ],
        },
        "error": None,
    }


def get_job(job_id: str) -> SimulationJob | None:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def start_simulation(narrative: str, seed: int | None = None) -> SimulationJob:
    job_id = uuid.uuid4().hex
    state = build_initial_state(narrative)
    state["job_id"] = job_id
    job = SimulationJob(job_id=job_id, narrative=narrative, state=state)
    with JOBS_LOCK:
        JOBS[job_id] = job
    thread = threading.Thread(target=_run_simulation, args=(job, seed), daemon=True)
    thread.start()
    return job


def stream_job(job: SimulationJob):
    queue = job.subscribe()
    try:
        while True:
            try:
                snapshot = queue.get(timeout=15)
                yield snapshot
                if snapshot.get("status") in {"completed", "error"}:
                    break
            except Empty:
                yield job.snapshot()
                if job.done:
                    break
    finally:
        job.unsubscribe(queue)


def _run_simulation(job: SimulationJob, seed: int | None) -> None:
    rng = random.Random(seed if seed is not None else int(time.time() * 1000))
    try:
        job.update(
            status="loading",
            progress={"stage": "loading_models", "processed": 0, "total": NUM_AGENTS, "percent": 2},
        )

        resources = get_model_resources()
        descriptions = get_100_agent_descriptions(DEFAULT_DATASET_PATH, seed=seed)
        uids = list(range(len(descriptions)))

        stage_text = "Encoding agent personas..."
        analysis = build_analysis_snapshot([], job.state["stats"], 0, len(descriptions), stage_text)
        job.update(
            status="running",
            progress={"stage": "encoding_personas", "processed": 0, "total": len(descriptions), "percent": 8},
            analysis=analysis,
        )

        with torch.inference_mode():
            desc_embeddings = resources.engine.model.encode(
                descriptions,
                batch_size=min(SIMULATION_BATCH_SIZE, len(descriptions)),
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            desc_embeddings = desc_embeddings.float().to(resources.device)
            persona_tensor = resources.preprocessor(desc_embeddings)
            personality_vectors = persona_tensor.detach().cpu().numpy().astype(np.float64)
        normalized_personality_vectors = normalize_vectors(personality_vectors)

        labels, _, _ = kmeans_auto_k(personality_vectors)
        coords, _ = get_clustering_output(
            uids,
            personality_vectors,
            labels=labels,
            shares=[],
            pca_random_state=42,
            cluster_spacing_strength=1.1,
        )

        agents: list[dict[str, Any]] = []
        for uid in uids:
            agents.append(
                {
                    "uid": uid,
                    "action": "NOTHING",
                    "display_action": "neutral",
                    "like_value": 0.0,
                    "similarity_score": 0.0,
                    "description": descriptions[uid],
                    "response_to_media": "",
                    "cluster": int(labels[uid]),
                    "sentiment_label": "",
                    "shared": False,
                }
            )

        graph_nodes = [
            {
                "id": uid,
                "x": float(coords[uid, 0]),
                "y": float(coords[uid, 1]),
                "cluster": int(labels[uid]),
                "action": "NOTHING",
                "display_action": "neutral",
                "like_value": 0.0,
                "similarity_score": 0.0,
                "shared": False,
                "description": descriptions[uid],
                "description_short": short_text(descriptions[uid]),
                "response": "Analyzing narrative fit...",
                "response_short": "Analyzing narrative fit...",
            }
            for uid in uids
        ]
        job.update(
            graph={"nodes": graph_nodes, "links": []},
            agents=agents,
            analysis=build_analysis_snapshot(agents, job.state["stats"], 0, len(descriptions), "Clusters mapped. Running answer model..."),
            progress={"stage": "clusters_ready", "processed": 0, "total": len(descriptions), "percent": 18},
        )

        question = f"What is your opinion on this content: {job.narrative}"
        question_embedding = torch.tensor(
            resources.engine.encode([question]),
            dtype=torch.float32,
            device=resources.device,
        )
        persona_tensor = torch.tensor(personality_vectors, dtype=torch.float32, device=resources.device)

        history: list[dict[str, float]] = [{"progress": 0.0, "liked": 0.0, "neutral": 100.0, "disliked": 0.0}]
        shares: list[list[Any]] = []
        share_lookup: set[int] = set()

        for start in range(0, len(uids), SIMULATION_BATCH_SIZE):
            end = min(start + SIMULATION_BATCH_SIZE, len(uids))
            batch_slice = slice(start, end)
            with torch.inference_mode():
                predicted = predict_answer_embeddings_batch(
                    persona_tensor[batch_slice],
                    question_embedding,
                    resources.predictor,
                )
                pred_norm = F.normalize(predicted, dim=-1)
                similarities = pred_norm @ resources.bank_norm.T

            resolved = resolve_batch_topk_results(
                similarities,
                resources.bank,
                top_k=TOP_K,
                threshold=NEUTRAL_TIE_THRESHOLD,
            )

            for offset, (answer_text, sentiment_label, score, _counts) in enumerate(resolved):
                uid = start + offset
                like_value = sentiment_to_like_value(sentiment_label)
                should_share = sentiment_label != "neutral" and rng.randint(1, SHARE_CHANCE_DENOMINATOR) == 1
                action_name = sentiment_to_action(sentiment_label, should_share)
                similarity_score = float(score)
                agents[uid].update(
                    {
                        "action": action_name,
                        "display_action": display_action_from_action(action_name),
                        "like_value": like_value,
                        "similarity_score": similarity_score,
                        "response_to_media": answer_text,
                        "sentiment_label": sentiment_label,
                        "shared": should_share,
                    }
                )
                graph_nodes[uid].update(
                    {
                        "action": action_name,
                        "display_action": display_action_from_action(action_name),
                        "like_value": like_value,
                        "similarity_score": similarity_score,
                        "shared": should_share,
                        "response": answer_text,
                        "response_short": short_text(answer_text),
                    }
                )

                if should_share and uid not in share_lookup:
                    already_shared_to_me = {int(sharer) for sharer, recipients in shares if uid in recipients}
                    recipients = pick_recipients(
                        uid,
                        uids,
                        personality_vectors,
                        labels,
                        rng=rng,
                        exclude_uids=already_shared_to_me,
                        unit_vectors=normalized_personality_vectors,
                    )
                    if recipients:
                        shares.append([uid, recipients])
                        share_lookup.add(uid)

            processed = end
            is_final_batch = processed == len(uids)
            should_publish_progress = is_final_batch or processed % PROGRESS_UPDATE_INTERVAL == 0
            if not should_publish_progress:
                continue

            stats, reaction_bar = compute_stats(agents, processed)
            history.append(build_history_point(agents, processed))
            percent = min(92, 18 + round(70 * processed / len(uids)))
            stage_text = f"Running model batch {processed}/{len(uids)}"
            analysis = build_analysis_snapshot(agents, stats, processed, len(uids), stage_text)
            update_payload: dict[str, Any] = {
                "status": "running",
                "progress": {"stage": "predicting_responses", "processed": processed, "total": len(uids), "percent": percent},
                "stats": stats,
                "reaction_bar": reaction_bar,
                "chart": make_chart_series(history),
                "analysis": analysis,
            }
            should_update_graph = is_final_batch or processed % GRAPH_UPDATE_INTERVAL == 0
            if should_update_graph:
                graph_links = [
                    {"source": int(source), "target": int(target)}
                    for source, recipients in shares
                    for target in recipients
                ]
                update_payload["graph"] = {"nodes": graph_nodes, "links": graph_links}
                update_payload["shares"] = shares
                update_payload["agents"] = agents

            job.update(
                **update_payload,
            )

        final_state = job.snapshot()
        try:
            final_analysis = generate_final_analysis(job.narrative, final_state)
        except Exception:
            final_analysis = final_state["analysis"]

        final_stats, final_reaction_bar = compute_stats(agents, len(agents))
        job.update(
            status="completed",
            progress={"stage": "completed", "processed": len(agents), "total": len(agents), "percent": 100},
            stats=final_stats,
            reaction_bar=final_reaction_bar,
            chart=make_chart_series(history),
            analysis=final_analysis,
        )
        job.done = True
    except Exception as exc:
        job.update(
            status="error",
            error=str(exc),
            progress={"stage": "error", "processed": 0, "total": NUM_AGENTS, "percent": 100},
        )
        job.done = True
