# lewm-mcp

**MCP server for LeWorldModel** — visual anomaly detection for Claude Code agents and other MCP clients.

Uses a JEPA-style ViT encoder to compute surprise scores between frames, enabling agents to detect unexpected UI changes, video anomalies, and state mismatches.

## Quick start

```bash
npx lewm-mcp
```

Or install globally:

```bash
npm install -g lewm-mcp
lewm-mcp
```

## Python requirements

The model runs in a Python subprocess. Install dependencies:

```bash
pip install torch transformers Pillow numpy
# For video analysis:
pip install opencv-python
```

> **Remote inference:** To run on a GPU/MPS server (e.g. `100.105.97.18`), start `lewm-mcp` there and connect via MCP over SSH or Tailscale.

## Tools

### `load_model`
Load the ViT encoder into memory. Call once before other tools.

```json
{ "checkpoint": "/path/to/checkpoint" }
```

Returns model info (param count, device, embed_dim, status).

### `get_model_status`
Check if the model is loaded, which checkpoint, param count, device (mps/cuda/cpu).

### `analyze_screenshot`
Encode a screenshot and compute surprise vs a previous frame.

```json
{
  "source": "/path/to/screenshot.png",
  "previous_source": "/path/to/previous.png",
  "anomaly_threshold": 2.0
}
```

`source` accepts a file path **or** base64-encoded image data.

Returns: `embedding`, `surprise_score`, `normalized_surprise`, `cosine_similarity`, `mse`, `anomaly`.

### `compare_states`
Compare expected vs actual screen states in embedding space.

```json
{
  "expected": "/path/to/expected.png",
  "actual": "/path/to/actual.png",
  "anomaly_threshold": 0.1
}
```

Returns: `cosine_similarity`, `mse`, `surprise_score`, `match`, `anomaly`.

### `analyze_video`
Extract frames from a video, run through ViT, return surprise timeline.

```json
{
  "video_path": "/path/to/recording.mp4",
  "frame_sample_rate": 1,
  "sigma_threshold": 2.0,
  "top_n": 5
}
```

Returns: `timestamps`, `surprise_scores`, `normalized_scores`, `anomaly_windows`, `top_anomalies`.

### `run_surprise_detection`
Full pipeline on a directory of screenshots or a video file.

```json
{
  "directory": "/path/to/screenshots/",
  "threshold_multiplier": 2.0
}
```

Returns: `timeline`, `exceeded_threshold`, `stats`.

## Architecture

```
Claude Code agent
       │
       │ MCP (stdio)
       ▼
  lewm-mcp (TypeScript)
       │
       │ stdin/stdout JSON protocol
       ▼
  model.py (Python subprocess)
       │
       ▼
  transformers ViTModel (tiny: hidden=192, layers=3, patch=16)
  runs on: mps → cuda → cpu
```

The Python process stays alive between tool calls — the model loads once and stays warm.

## Configure in Claude Code

Add to `~/.claude/claude_desktop_config.json` (or equivalent MCP config):

```json
{
  "mcpServers": {
    "lewm-mcp": {
      "command": "npx",
      "args": ["lewm-mcp"]
    }
  }
}
```

## Model details

Default model: tiny ViT initialized with random weights.
- `hidden_size`: 192
- `num_hidden_layers`: 3
- `num_attention_heads`: 3
- `patch_size`: 16
- `image_size`: 224

Pass a `checkpoint` path to `load_model` to use a fine-tuned or pretrained checkpoint (must be a `transformers` ViTModel checkpoint).

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LEWM_PYTHON` | `python3` | Python executable to use for model subprocess |

## License

MIT
