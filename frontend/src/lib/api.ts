// Typed client for the EvalForge backend. The browser only ever talks to the backend's
// public API; provider keys stay server-side and are never part of these payloads.

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type Provider = "openai" | "anthropic" | "gemini" | "sample";

export interface ModelInfo {
  provider: Provider;
  model_id: string;
  available: boolean;
  reason: string | null;
  sample: boolean;
}

export interface ModelsOut {
  models: ModelInfo[];
  judge: { provider: Provider | null; model_id: string | null; available: boolean; reason: string | null };
}

export interface TestCaseIn {
  id: string;
  input: string;
  context?: string | null;
  expected_answer: string;
  required_facts?: string[];
  metadata?: Record<string, string | number | boolean | null>;
}

export interface DatasetValidation {
  valid: boolean;
  errors: string[];
  case_count: number | null;
  dataset_hash: string | null;
}

export interface MetricSummary {
  mean: number | null;
  n: number;
}

export interface CostSummary {
  total: number | null;
  n_priced: number;
  complete: boolean;
}

export interface RunAggregate {
  total_cases: number;
  responses_ok: number;
  candidate_errors: number;
  judge_errors: number;
  correctness: MetricSummary;
  groundedness: MetricSummary;
  required_facts: MetricSummary;
  latency_ms_mean: number | null;
  latency_ms_median: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  candidate_cost: CostSummary;
  judge_cost: CostSummary;
}

export type RunStatus = "pending" | "running" | "completed" | "partial" | "failed";
export type ExperimentStatus = "draft" | "running" | "completed" | "partial" | "failed";

export interface RunOut {
  id: string;
  provider: Provider;
  model_id: string;
  status: RunStatus;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  judge_provider: string | null;
  judge_model_id: string | null;
  eval_config: Record<string, unknown>;
  eval_config_hash: string;
  pricing_version: string | null;
  sample: boolean;
  aggregate: RunAggregate | null;
}

export interface ExperimentOut {
  id: string;
  name: string;
  system_prompt: string;
  created_at: string;
  baseline_experiment_id: string | null;
  regression_threshold: number;
  status: ExperimentStatus;
  dataset_hash: string;
  prompt_hash: string;
  case_count: number;
  runs: RunOut[];
}

export type RegressionIndicator = "regression" | "ok" | "no_verdict" | null;

export interface ExperimentSummary {
  id: string;
  name: string;
  created_at: string;
  status: ExperimentStatus;
  case_count: number;
  run_count: number;
  models: string[];
  baseline_experiment_id: string | null;
  dataset_hash: string;
  regression_threshold: number;
  sample: boolean;
  regression: RegressionIndicator;
}

export interface CaseResult {
  run_id: string;
  provider: Provider;
  model_id: string;
  sample: boolean;
  run_status: RunStatus;
  response: string | null;
  correctness_score: number | null;
  groundedness_score: number | null;
  required_facts_score: number | null;
  missing_facts: string[];
  latency_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  estimated_cost: number | null;
  judge_provider: string | null;
  judge_model_id: string | null;
  judge_reason: string | null;
  judge_input_tokens: number | null;
  judge_output_tokens: number | null;
  judge_cost: number | null;
  error: string | null;
  judge_error: string | null;
  flags: string[];
}

export interface CaseOut {
  case_id: string;
  input: string;
  context: string | null;
  expected_answer: string;
  required_facts: string[];
  metadata: Record<string, string | number | boolean | null>;
  results: CaseResult[];
}

export interface ResultsOut {
  experiment_id: string;
  total_cases: number;
  low_score_threshold: number;
  cases: CaseOut[];
}

export type MetricVerdict = "regression" | "ok" | "insufficient_data" | "not_applicable";
export type ModelVerdict = "regression" | "ok" | "insufficient_data" | "not_comparable";

export interface MetricComparison {
  metric: "correctness" | "groundedness";
  applicable_cases: number;
  paired_cases: number;
  baseline_mean: number | null;
  candidate_mean: number | null;
  delta_pp: number | null;
  verdict: MetricVerdict;
  note: string | null;
}

export interface ModelComparison {
  provider: Provider;
  model_id: string;
  verdict: ModelVerdict;
  reasons: string[];
  baseline_run_id: string | null;
  candidate_run_id: string | null;
  metrics: MetricComparison[];
  informational: { name: string; baseline: number | null; candidate: number | null; delta: number | null; paired_cases: number }[];
  largest_drops: { case_id: string; metric: string; baseline: number; candidate: number; delta_pp: number }[];
}

export interface ComparisonOut {
  experiment_id: string;
  baseline_experiment_id: string | null;
  threshold_pp: number;
  comparable: boolean;
  reasons: string[];
  prompt_changed: boolean | null;
  has_regression: boolean | null;
  models: ModelComparison[];
  note: string;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public details: string[],
  ) {
    super(details.join("; ") || `Request failed with status ${status}`);
  }
}

function detailMessages(body: unknown): string[] {
  const detail = (body as { detail?: unknown })?.detail;
  if (typeof detail === "string") return [detail];
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      if (typeof item === "string") return item;
      const { loc, msg } = item as { loc?: unknown[]; msg?: string };
      const where = Array.isArray(loc) ? loc.filter((part) => part !== "body").join(".") : "";
      return where ? `${where}: ${msg}` : String(msg ?? JSON.stringify(item));
    });
  }
  return [];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(0, [`Cannot reach the backend at ${API_URL}. Is it running?`]);
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(response.status, detailMessages(body));
  return body as T;
}

export const api = {
  models: () => request<ModelsOut>("/models"),
  experiments: () => request<ExperimentSummary[]>("/experiments"),
  experiment: (id: string) => request<ExperimentOut>(`/experiments/${encodeURIComponent(id)}`),
  results: (id: string, params: { case_id?: string } = {}) => {
    const query = params.case_id ? `?case_id=${encodeURIComponent(params.case_id)}` : "";
    return request<ResultsOut>(`/experiments/${encodeURIComponent(id)}/results${query}`);
  },
  compare: (id: string) => request<ComparisonOut>(`/experiments/${encodeURIComponent(id)}/compare`),
  sampleDataset: () => request<{ name?: string; cases: TestCaseIn[] }>("/datasets/sample"),
  validateDataset: (dataset: unknown) =>
    request<DatasetValidation>("/datasets/validate", { method: "POST", body: JSON.stringify(dataset) }),
  createExperiment: (body: {
    name: string;
    system_prompt: string;
    cases: TestCaseIn[];
    baseline_experiment_id?: string | null;
    regression_threshold?: number;
  }) => request<ExperimentOut>("/experiments", { method: "POST", body: JSON.stringify(body) }),
  run: (id: string, models: { provider: Provider; model_id: string }[]) =>
    request<{ experiment_id: string; status: string; runs: RunOut[] }>(`/experiments/${encodeURIComponent(id)}/run`, {
      method: "POST",
      body: JSON.stringify({ models }),
    }),
};
