export type CellProfile = {
  genes: string[];
  values: number[];
};

export type Reference = {
  source: string;
  genes: string[];
  values: number[];
};

export type GeneDelta = {
  gene: string;
  delta: number;
  query?: number;
  ref?: number;
};

export type Match = {
  gene: string;
  cosine: number;
  weight: number;
  deltas: GeneDelta[];
};

export type HeatmapColumn = {
  gene: string;
  cosine: number;
};

export type HeatmapPayload = {
  genes: string[];
  columns: HeatmapColumn[];
  values: number[][];
  unit: string;
  query: number[] | null;
};

export type InferResponse = {
  matches: Match[];
  abnormal: GeneDelta[];
  heatmap?: HeatmapPayload;
  n_tokens?: number;
  disclaimer?: string;
  ckpt?: Record<string, string>;
};

export type RecentUpload = {
  id: string;
  label: string;
  ts: number;
  nGenes: number;
  genes: string[];
  values: number[];
  topHit: string | null;
};

export type GeneInfo = {
  accession: string;
  proteinName: string;
  aliases: string[];
  function: string;
  locations: string[];
  diseases: string[];
  length?: number;
};

export type StageMode = "idle" | "loading" | "results";
