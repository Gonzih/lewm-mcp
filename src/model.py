#!/usr/bin/env python3
"""
LeWorldModel Python subprocess — ViT encoder for visual surprise detection.
Communicates via stdin/stdout JSON protocol (newline-delimited).
Stays alive between tool calls so the model loads once and stays warm.
"""

import sys
import json
import base64
import io
import math
import os
import traceback
from typing import Optional, Any

# Lazy imports so we can report errors cleanly
_torch = None
_np = None
_PIL_Image = None
_model = None
_processor = None
_model_info: dict[str, Any] = {
    "loaded": False,
    "checkpoint": None,
    "param_count": 0,
    "device": "cpu",
    "embed_dim": 192,
}
_previous_embedding: Optional[list[float]] = None
_baseline_surprise: Optional[float] = None  # rolling mean for normalization


def _import_deps():
    global _torch, _np, _PIL_Image
    if _torch is None:
        import torch
        _torch = torch
    if _np is None:
        import numpy as np
        _np = np
    if _PIL_Image is None:
        from PIL import Image
        _PIL_Image = Image


def _get_device() -> str:
    _import_deps()
    if _torch.backends.mps.is_available():
        return "mps"
    if _torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(checkpoint: Optional[str] = None) -> dict:
    global _model, _processor, _model_info
    _import_deps()

    from transformers import ViTModel, ViTImageProcessor, ViTConfig

    device = _get_device()

    if checkpoint and os.path.exists(checkpoint):
        # Load from local checkpoint
        config = ViTConfig.from_pretrained(checkpoint)
        model = ViTModel.from_pretrained(checkpoint)
        ckpt_label = checkpoint
    else:
        # Use tiny ViT config — fast, low memory, works on CPU
        config = ViTConfig(
            hidden_size=192,
            num_hidden_layers=3,
            num_attention_heads=3,
            intermediate_size=768,
            patch_size=16,
            image_size=224,
        )
        model = ViTModel(config)
        ckpt_label = "tiny-vit-random"

    model = model.to(device)
    model.eval()

    processor = ViTImageProcessor(
        size={"height": 224, "width": 224},
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_resize=True,
        do_normalize=True,
        do_rescale=True,
    )

    param_count = sum(p.numel() for p in model.parameters())

    _model = model
    _processor = processor
    _model_info = {
        "loaded": True,
        "checkpoint": ckpt_label,
        "param_count": param_count,
        "device": device,
        "embed_dim": config.hidden_size,
    }
    return _model_info


def _ensure_model():
    if not _model_info["loaded"]:
        _load_model()


def _decode_image(source: str):
    """Accept file path or base64-encoded image data."""
    _import_deps()
    if os.path.exists(source):
        img = _PIL_Image.open(source).convert("RGB")
    else:
        # Try base64
        try:
            # Strip data URL prefix if present
            if "," in source:
                source = source.split(",", 1)[1]
            data = base64.b64decode(source)
            img = _PIL_Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Cannot decode image: {e}")
    return img


def _encode_image(img) -> list[float]:
    """Run ViT encoder on a PIL image, return CLS embedding as list."""
    _ensure_model()
    _import_deps()
    inputs = _processor(images=img, return_tensors="pt")
    inputs = {k: v.to(_model_info["device"]) for k, v in inputs.items()}
    with _torch.no_grad():
        outputs = _model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
    return cls_embedding.cpu().tolist()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    _import_deps()
    va = _np.array(a, dtype=_np.float32)
    vb = _np.array(b, dtype=_np.float32)
    denom = (_np.linalg.norm(va) * _np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(_np.dot(va, vb) / denom)


def _mse(a: list[float], b: list[float]) -> float:
    _import_deps()
    va = _np.array(a, dtype=_np.float32)
    vb = _np.array(b, dtype=_np.float32)
    return float(_np.mean((va - vb) ** 2))


def _surprise_score(a: list[float], b: list[float]) -> float:
    """Surprise = 1 - cosine_similarity (0 = identical, 2 = opposite)."""
    return 1.0 - _cosine_similarity(a, b)


def _update_baseline(score: float):
    global _baseline_surprise
    if _baseline_surprise is None:
        _baseline_surprise = score
    else:
        # Exponential moving average
        _baseline_surprise = 0.9 * _baseline_surprise + 0.1 * score


def _normalized_surprise(score: float) -> float:
    if _baseline_surprise is None or _baseline_surprise == 0:
        return 1.0
    return score / _baseline_surprise


# ──────────────────────── Tool handlers ────────────────────────

def handle_load_model(params: dict) -> dict:
    checkpoint = params.get("checkpoint")
    info = _load_model(checkpoint)
    return {"status": "ok", "model_info": info}


def handle_get_model_status(_params: dict) -> dict:
    return {"model_info": _model_info}


def handle_analyze_screenshot(params: dict) -> dict:
    global _previous_embedding
    source = params["source"]
    previous_source = params.get("previous_source")

    img = _decode_image(source)
    embedding = _encode_image(img)

    result: dict[str, Any] = {
        "embedding": embedding,
        "embed_dim": len(embedding),
        "surprise_score": None,
        "normalized_surprise": None,
        "anomaly": False,
        "anomaly_threshold": params.get("anomaly_threshold", 2.0),
    }

    compare_to = None
    if previous_source:
        prev_img = _decode_image(previous_source)
        compare_to = _encode_image(prev_img)
    elif _previous_embedding is not None:
        compare_to = _previous_embedding

    if compare_to is not None:
        score = _surprise_score(compare_to, embedding)
        norm = _normalized_surprise(score)
        _update_baseline(score)
        result["surprise_score"] = score
        result["normalized_surprise"] = norm
        result["cosine_similarity"] = _cosine_similarity(compare_to, embedding)
        result["mse"] = _mse(compare_to, embedding)
        result["anomaly"] = norm > params.get("anomaly_threshold", 2.0)

    _previous_embedding = embedding
    return result


def handle_compare_states(params: dict) -> dict:
    source_expected = params["expected"]
    source_actual = params["actual"]

    img_expected = _decode_image(source_expected)
    img_actual = _decode_image(source_actual)

    emb_expected = _encode_image(img_expected)
    emb_actual = _encode_image(img_actual)

    sim = _cosine_similarity(emb_expected, emb_actual)
    mse_val = _mse(emb_expected, emb_actual)
    surprise = _surprise_score(emb_expected, emb_actual)
    threshold = params.get("anomaly_threshold", 0.1)

    return {
        "cosine_similarity": sim,
        "mse": mse_val,
        "surprise_score": surprise,
        "anomaly": surprise > threshold,
        "anomaly_threshold": threshold,
        "match": sim > (1.0 - threshold),
    }


def handle_analyze_video(params: dict) -> dict:
    _import_deps()
    video_path = params["video_path"]
    top_n = params.get("top_n", 5)
    frame_sample_rate = params.get("frame_sample_rate", 1)  # every N seconds

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python required for video analysis: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(fps * frame_sample_rate))

    timestamps = []
    embeddings = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            ts = frame_idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = _PIL_Image.fromarray(rgb)
            emb = _encode_image(img)
            timestamps.append(ts)
            embeddings.append(emb)
        frame_idx += 1

    cap.release()

    if len(embeddings) < 2:
        return {
            "timestamps": timestamps,
            "surprise_scores": [],
            "normalized_scores": [],
            "anomaly_windows": [],
            "top_anomalies": [],
        }

    # Compute frame-to-frame surprise
    scores = []
    for i in range(1, len(embeddings)):
        s = _surprise_score(embeddings[i - 1], embeddings[i])
        scores.append(s)
        _update_baseline(s)

    # Normalize
    _import_deps()
    scores_arr = _np.array(scores)
    mean_s = float(scores_arr.mean())
    std_s = float(scores_arr.std()) if len(scores_arr) > 1 else 1.0
    if std_s == 0:
        std_s = 1.0
    norm_scores = ((scores_arr - mean_s) / std_s).tolist()

    # Anomaly windows: consecutive frames >2σ
    sigma_threshold = params.get("sigma_threshold", 2.0)
    anomaly_flags = [ns > sigma_threshold for ns in norm_scores]
    anomaly_windows = []
    in_window = False
    win_start = 0
    for i, flag in enumerate(anomaly_flags):
        if flag and not in_window:
            in_window = True
            win_start = i
        elif not flag and in_window:
            in_window = False
            anomaly_windows.append({
                "start_ts": timestamps[win_start],
                "end_ts": timestamps[i],
                "max_zscore": max(norm_scores[win_start:i]),
            })
    if in_window:
        anomaly_windows.append({
            "start_ts": timestamps[win_start],
            "end_ts": timestamps[-1],
            "max_zscore": max(norm_scores[win_start:]),
        })

    # Top N anomaly timestamps
    indexed = sorted(enumerate(norm_scores), key=lambda x: x[1], reverse=True)
    top_anomalies = []
    for rank, (idx, zscore) in enumerate(indexed[:top_n]):
        top_anomalies.append({
            "rank": rank + 1,
            "timestamp": timestamps[idx + 1],
            "zscore": zscore,
            "raw_surprise": scores[idx],
            "description": f"High visual change at t={timestamps[idx+1]:.2f}s (z={zscore:.2f}σ)",
        })

    return {
        "timestamps": timestamps[1:],  # aligned to score array
        "surprise_scores": scores,
        "normalized_scores": norm_scores,
        "anomaly_windows": anomaly_windows,
        "top_anomalies": top_anomalies,
        "stats": {"mean": mean_s, "std": std_s, "n_frames": len(embeddings)},
    }


def handle_run_surprise_detection(params: dict) -> dict:
    _import_deps()
    source_dir = params.get("directory")
    video_path = params.get("video_path")
    threshold_multiplier = params.get("threshold_multiplier", 2.0)

    if video_path:
        result = handle_analyze_video({
            "video_path": video_path,
            "sigma_threshold": threshold_multiplier,
        })
        return result

    if not source_dir or not os.path.exists(source_dir):
        raise ValueError(f"Directory not found: {source_dir}")

    # Collect image files sorted by name
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted([
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    if not files:
        raise ValueError(f"No image files found in {source_dir}")

    embeddings = []
    for f in files:
        img = _decode_image(f)
        emb = _encode_image(img)
        embeddings.append(emb)

    scores = []
    for i in range(1, len(embeddings)):
        s = _surprise_score(embeddings[i - 1], embeddings[i])
        scores.append(s)

    scores_arr = _np.array(scores)
    mean_s = float(scores_arr.mean())
    std_s = float(scores_arr.std()) if len(scores) > 1 else 1.0
    if std_s == 0:
        std_s = 1.0

    timeline = []
    for i, score in enumerate(scores):
        zscore = (score - mean_s) / std_s
        exceeds = zscore > threshold_multiplier
        timeline.append({
            "index": i + 1,
            "file": files[i + 1],
            "surprise_score": score,
            "zscore": zscore,
            "exceeds_threshold": exceeds,
        })

    exceeded = [t for t in timeline if t["exceeds_threshold"]]

    return {
        "timeline": timeline,
        "exceeded_threshold": exceeded,
        "threshold_multiplier": threshold_multiplier,
        "stats": {"mean": mean_s, "std": std_s, "n_frames": len(files)},
    }


# ──────────────────────── Main protocol loop ────────────────────────

HANDLERS = {
    "load_model": handle_load_model,
    "get_model_status": handle_get_model_status,
    "analyze_screenshot": handle_analyze_screenshot,
    "compare_states": handle_compare_states,
    "analyze_video": handle_analyze_video,
    "run_surprise_detection": handle_run_surprise_detection,
}


def main():
    # Eagerly load model at startup
    try:
        _load_model()
    except Exception as e:
        # Non-fatal — report on get_model_status
        sys.stderr.write(f"[lewm-mcp] Model load warning: {e}\n")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            tool = msg.get("tool")
            params = msg.get("params", {})
            req_id = msg.get("id")

            if tool not in HANDLERS:
                response = {"id": req_id, "error": f"Unknown tool: {tool}"}
            else:
                result = HANDLERS[tool](params)
                response = {"id": req_id, "result": result}
        except Exception as e:
            response = {"id": req_id if "req_id" in dir() else None, "error": str(e), "traceback": traceback.format_exc()}

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
