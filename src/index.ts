#!/usr/bin/env node
/**
 * lewm-mcp — MCP server for LeWorldModel visual anomaly detection.
 * Exposes JEPA-based ViT encoder capabilities to Claude Code agents.
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { callPython, shutdownPython } from "./python-bridge.js";

const server = new Server(
  { name: "lewm-mcp", version: "0.1.0" },
  { capabilities: { tools: {} } }
);

// ──────────────────────── Tool definitions ────────────────────────

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "load_model",
      description:
        "Load the ViT world-model encoder into memory. Call this before other tools for faster inference. Defaults to a tiny pretrained ViT (hidden_size=192, 3 layers, patch_size=16).",
      inputSchema: {
        type: "object",
        properties: {
          checkpoint: {
            type: "string",
            description:
              "Optional path to a local model checkpoint directory. Omit to use the default tiny ViT.",
          },
        },
      },
    },
    {
      name: "get_model_status",
      description:
        "Check whether the model is loaded, which checkpoint is active, parameter count, and which device (mps/cuda/cpu) is in use.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "analyze_screenshot",
      description:
        "Encode a screenshot through the ViT encoder and optionally compute a surprise score vs a previous frame. Returns embedding vector, cosine similarity, MSE, and anomaly flag.",
      inputSchema: {
        type: "object",
        required: ["source"],
        properties: {
          source: {
            type: "string",
            description:
              "File path to an image (png/jpg/webp) OR base64-encoded image data (with or without data URL prefix).",
          },
          previous_source: {
            type: "string",
            description:
              "Optional: file path or base64 of a previous screenshot to compare against. If omitted, uses the last screenshot passed to this tool.",
          },
          anomaly_threshold: {
            type: "number",
            description:
              "Normalized surprise multiplier above which to flag as anomaly (default 2.0 = 2× baseline).",
          },
        },
      },
    },
    {
      name: "compare_states",
      description:
        "Compare two screenshots in embedding space. Useful for 'does this screen match what I expected?' Returns cosine similarity, MSE, surprise score, and match/anomaly flags.",
      inputSchema: {
        type: "object",
        required: ["expected", "actual"],
        properties: {
          expected: {
            type: "string",
            description: "File path or base64 of the expected/reference state.",
          },
          actual: {
            type: "string",
            description: "File path or base64 of the actual/current state.",
          },
          anomaly_threshold: {
            type: "number",
            description:
              "Surprise score above which to flag as anomaly (default 0.1). Lower = stricter.",
          },
        },
      },
    },
    {
      name: "analyze_video",
      description:
        "Extract frames from a video file, run them through the ViT encoder, and compute frame-to-frame surprise scores. Returns timestamp array, surprise scores, z-score normalized scores, anomaly windows (>2σ spikes), and top N anomaly timestamps.",
      inputSchema: {
        type: "object",
        required: ["video_path"],
        properties: {
          video_path: {
            type: "string",
            description: "Path to an mp4 or webm video file.",
          },
          frame_sample_rate: {
            type: "number",
            description:
              "Sample one frame every N seconds (default 1). Increase for long videos.",
          },
          sigma_threshold: {
            type: "number",
            description:
              "Z-score threshold to flag as anomaly window (default 2.0).",
          },
          top_n: {
            type: "number",
            description: "Number of top anomaly timestamps to return (default 5).",
          },
        },
      },
    },
    {
      name: "run_surprise_detection",
      description:
        "Run full surprise detection pipeline on a directory of screenshots or a video file. Returns annotated timeline, list of frames exceeding threshold, and summary stats.",
      inputSchema: {
        type: "object",
        properties: {
          directory: {
            type: "string",
            description:
              "Path to a directory of image files (sorted alphabetically). Provide this or video_path.",
          },
          video_path: {
            type: "string",
            description: "Path to a video file. Provide this or directory.",
          },
          threshold_multiplier: {
            type: "number",
            description:
              "Z-score multiplier for anomaly detection (default 2.0 = 2σ above mean).",
          },
        },
      },
    },
  ],
}));

// ──────────────────────── Tool dispatch ────────────────────────

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const params = (args ?? {}) as Record<string, unknown>;

  const pythonTools = new Set([
    "load_model",
    "get_model_status",
    "analyze_screenshot",
    "compare_states",
    "analyze_video",
    "run_surprise_detection",
  ]);

  if (!pythonTools.has(name)) {
    return {
      content: [{ type: "text", text: `Unknown tool: ${name}` }],
      isError: true,
    };
  }

  try {
    const result = await callPython(name, params);
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return {
      content: [{ type: "text", text: `Error: ${message}` }],
      isError: true,
    };
  }
});

// ──────────────────────── Start server ────────────────────────

process.on("SIGINT", () => {
  shutdownPython();
  process.exit(0);
});
process.on("SIGTERM", () => {
  shutdownPython();
  process.exit(0);
});

const transport = new StdioServerTransport();
await server.connect(transport);
process.stderr.write("[lewm-mcp] Server started\n");
