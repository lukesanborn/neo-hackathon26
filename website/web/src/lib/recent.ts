import type { CellProfile, RecentUpload } from "../types";

const RECENT_KEY = "blood-jepa-recent-v1";
export const RECENT_MAX = 6;

export function loadRecent(): RecentUpload[] {
  try {
    const raw = JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
    return Array.isArray(raw) ? raw : [];
  } catch {
    return [];
  }
}

export function saveRecent(items: RecentUpload[]): void {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(items.slice(0, RECENT_MAX)));
  } catch {
    // localStorage full or blocked
  }
}

export function relativeTime(ts: number): string {
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function rememberUpload(
  label: string,
  profile: CellProfile,
  topHit?: string | null,
): RecentUpload {
  const items = loadRecent().filter((it) => it.label !== label);
  const entry: RecentUpload = {
    id: `u_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    label,
    ts: Date.now(),
    nGenes: profile.genes.length,
    genes: profile.genes,
    values: profile.values,
    topHit: topHit || null,
  };
  items.unshift(entry);
  saveRecent(items);
  return entry;
}

export function updateRecentHit(activeId: string | null, topHit: string | null): RecentUpload[] {
  if (!activeId || !topHit) return loadRecent();
  const items = loadRecent();
  const i = items.findIndex((it) => it.id === activeId);
  if (i < 0) return items;
  items[i] = { ...items[i], topHit };
  saveRecent(items);
  return items;
}
