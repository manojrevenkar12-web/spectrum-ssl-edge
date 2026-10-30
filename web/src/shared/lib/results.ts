// The shape of reports/results.json that the explorer reads (written by `sse report`).
export interface Summary {
  mean: number;
  min: number;
  max: number;
}
export interface SnrBin {
  lo: number;
  hi: number;
  n: number;
  accuracy: number;
}
export interface Calib {
  ece: number;
  nll: number;
  brier: number;
  accuracy: number;
  aurc: number;
  risk_at_80: number;
}
export interface RelBin {
  lo: number;
  hi: number;
  n: number;
  confidence: number | null;
  accuracy: number | null;
}
export interface OpenRow {
  near: { auroc: number; fpr95: number };
  far: { auroc: number; fpr95: number };
  [cls: string]: { auroc: number; fpr95?: number };
}
export interface SceneBox {
  cls: string;
  channel: number;
  start: number;
  end: number;
  novel: boolean;
  snr_db?: number;
  confidence?: number;
}
export interface SceneExample {
  rows: number;
  cols: number;
  channels: number;
  pixels: number[];
  truth: SceneBox[];
  detections: SceneBox[];
}
export interface SceneTotals {
  emitters: number;
  detections: number;
  found: number;
  false_detections: number;
  known_found: number;
  known_correct: number;
  known_flagged_novel: number;
  novel_found: number;
  novel_flagged: number;
}
export interface Edge {
  weight_bytes: number;
  fold_max_abs_diff: number;
  accuracy: Record<string, { float: number; int8: number; agreement: number }>;
  rust: {
    available: boolean;
    windows?: number;
    mismatched_windows?: number;
    bit_exact?: boolean;
    macs?: number;
    latency?: { median_us: number; p99_us: number };
  };
  onnx: {
    max_abs_diff_vs_pytorch: number;
    int8_qdq_accuracy: Record<string, number>;
    latency_us: { float: number; int8_qdq: number };
  };
  scenes?: { novelty_threshold: number; domains: Record<string, SceneTotals>; example: SceneExample | null };
  cpu: string;
}
export interface Results {
  profile: string;
  generated_at: string;
  classes: { known: string[]; novel: string[]; novel_groups: Record<string, string[]> };
  domains: string[];
  budgets: number[];
  seeds: number[];
  model: { parameters: number; macs_per_window: number; length: number };
  deployed: { name: string; method: string; rule: string };
  label_efficiency: Record<string, Record<string, Record<string, Summary>>>;
  accuracy_by_snr: Record<string, Record<string, SnrBin[]>>;
  calibration: Record<string, Record<string, Calib>>;
  reliability: Record<string, Record<string, RelBin[]>>;
  openset: Record<string, Record<string, Record<string, OpenRow>>>;
  edge?: Edge;
  heldout_first_run?: { recorded_at: string; domain_sha256: string; accuracy_full_labels: Record<string, number> };
}

export function isResults(v: unknown): v is Results {
  const r = v as Results;
  return !!r && typeof r === 'object' && Array.isArray(r.domains) && !!r.label_efficiency && !!r.deployed;
}
