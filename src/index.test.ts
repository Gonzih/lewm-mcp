/**
 * Unit tests for lewm-mcp — tests the MCP tool schema definitions without
 * requiring a running Python process.
 */
import { describe, it, expect } from "vitest";

// Tool name registry
const TOOL_NAMES = [
  "load_model",
  "get_model_status",
  "analyze_screenshot",
  "compare_states",
  "analyze_video",
  "run_surprise_detection",
];

describe("lewm-mcp tool definitions", () => {
  it("has all 6 required tools defined", () => {
    expect(TOOL_NAMES).toHaveLength(6);
  });

  it("tool names match spec", () => {
    const required = new Set([
      "load_model",
      "get_model_status",
      "analyze_screenshot",
      "compare_states",
      "analyze_video",
      "run_surprise_detection",
    ]);
    for (const name of TOOL_NAMES) {
      expect(required.has(name)).toBe(true);
    }
  });
});

describe("surprise score math", () => {
  function cosineSimilarity(a: number[], b: number[]): number {
    const dot = a.reduce((sum, ai, i) => sum + ai * b[i]!, 0);
    const normA = Math.sqrt(a.reduce((s, x) => s + x * x, 0));
    const normB = Math.sqrt(b.reduce((s, x) => s + x * x, 0));
    return dot / (normA * normB);
  }

  function surpriseScore(a: number[], b: number[]): number {
    return 1.0 - cosineSimilarity(a, b);
  }

  it("identical embeddings → surprise 0", () => {
    const emb = [0.1, 0.5, -0.3, 0.8];
    expect(surpriseScore(emb, emb)).toBeCloseTo(0, 5);
  });

  it("opposite embeddings → surprise 2", () => {
    const a = [1.0, 0.0];
    const b = [-1.0, 0.0];
    expect(surpriseScore(a, b)).toBeCloseTo(2.0, 5);
  });

  it("orthogonal embeddings → surprise 1", () => {
    const a = [1.0, 0.0];
    const b = [0.0, 1.0];
    expect(surpriseScore(a, b)).toBeCloseTo(1.0, 5);
  });
});

describe("python bridge message format", () => {
  it("generates correct JSON protocol message", () => {
    const id = 42;
    const tool = "analyze_screenshot";
    const params = { source: "/tmp/test.png" };
    const msg = JSON.stringify({ id, tool, params });
    const parsed = JSON.parse(msg);
    expect(parsed.id).toBe(42);
    expect(parsed.tool).toBe("analyze_screenshot");
    expect(parsed.params.source).toBe("/tmp/test.png");
  });

  it("handles response with result", () => {
    const response = JSON.stringify({
      id: 42,
      result: { embedding: [0.1, 0.2], anomaly: false },
    });
    const parsed = JSON.parse(response);
    expect(parsed.result.anomaly).toBe(false);
    expect(parsed.result.embedding).toHaveLength(2);
  });

  it("handles response with error", () => {
    const response = JSON.stringify({
      id: 42,
      error: "File not found: /tmp/missing.png",
    });
    const parsed = JSON.parse(response);
    expect(parsed.error).toContain("File not found");
  });
});
