import type { CellProfile, InferResponse, Reference } from "./types";

/** Empty in local dev (Vite proxies `/api`). Set for a static host such as Vercel. */
const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} failed (${res.status})`);
  return res.json() as Promise<T>;
}

export async function wakeHealth(): Promise<void> {
  await fetch(apiUrl("/api/health")).catch(() => undefined);
}

export async function fetchReference(): Promise<Reference> {
  return getJson<Reference>(apiUrl("/api/reference"));
}

export async function runInfer(cell: CellProfile, topK = 8): Promise<InferResponse> {
  await wakeHealth();
  const res = await fetch(apiUrl("/api/infer"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ genes: cell.genes, values: cell.values, top_k: topK }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = (body as { detail?: unknown }).detail;
    const msg =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join("; ")
          : (body as { message?: string }).message || `infer failed (${res.status})`;
    throw new Error(msg);
  }
  return body as InferResponse;
}
