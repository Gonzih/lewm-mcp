/**
 * Python subprocess bridge — spawns model.py and maintains a persistent
 * JSON-RPC-style protocol over stdin/stdout.
 */

import { spawn, ChildProcess } from "child_process";
import { createInterface } from "readline";
import { fileURLToPath } from "url";
import { existsSync } from "fs";
import path from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
}

let proc: ChildProcess | null = null;
let nextId = 1;
const pending = new Map<number, PendingRequest>();

function resolveModelPath(): string {
  const candidates = [
    path.join(__dirname, "model.py"),
    path.join(__dirname, "..", "src", "model.py"),
    path.join(__dirname, "..", "model.py"),
  ];
  for (const c of candidates) {
    if (existsSync(c)) return c;
  }
  return candidates[0];
}

function ensureProcess(): ChildProcess {
  if (proc && !proc.killed) return proc;

  const modelPath = resolveModelPath();
  const python = process.env.LEWM_PYTHON ?? "python3";

  proc = spawn(python, [modelPath], {
    stdio: ["pipe", "pipe", "pipe"],
  });

  proc.stderr?.on("data", (data: Buffer) => {
    process.stderr.write(`[lewm-python] ${data.toString()}`);
  });

  const rl = createInterface({ input: proc.stdout! });
  rl.on("line", (line: string) => {
    if (!line.trim()) return;
    try {
      const msg = JSON.parse(line) as { id: number; result?: unknown; error?: string };
      const req = pending.get(msg.id);
      if (req) {
        pending.delete(msg.id);
        if (msg.error) {
          req.reject(new Error(msg.error));
        } else {
          req.resolve(msg.result);
        }
      }
    } catch {
      process.stderr.write(`[lewm-mcp] Failed to parse Python response: ${line}\n`);
    }
  });

  proc.on("exit", (code) => {
    process.stderr.write(`[lewm-mcp] Python process exited (code=${code})\n`);
    proc = null;
    for (const [, req] of pending) {
      req.reject(new Error(`Python process exited with code ${code}`));
    }
    pending.clear();
  });

  return proc;
}

export async function callPython(
  tool: string,
  params: Record<string, unknown>
): Promise<unknown> {
  const p = ensureProcess();
  const id = nextId++;

  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    const msg = JSON.stringify({ id, tool, params }) + "\n";
    p.stdin!.write(msg, (err) => {
      if (err) {
        pending.delete(id);
        reject(err);
      }
    });
    // Timeout after 5 minutes (large video analysis)
    setTimeout(() => {
      if (pending.has(id)) {
        pending.delete(id);
        reject(new Error(`Python call timeout for tool: ${tool}`));
      }
    }, 300_000);
  });
}

export function shutdownPython(): void {
  if (proc && !proc.killed) {
    proc.stdin?.end();
    proc.kill("SIGTERM");
    proc = null;
  }
}
