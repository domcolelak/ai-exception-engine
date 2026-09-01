/**
 * Typed client for the AI Exception Engine API.
 *
 * All calls run on the Next.js server, so the tenant API key never reaches the
 * browser.
 */

const BASE_URL = process.env.API_BASE_URL ?? "http://localhost:8000";
const API_KEY = process.env.API_KEY ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${BASE_URL}/v1${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
      ...(init.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new ApiError(`${path} failed: ${(await response.text()).slice(0, 300)}`, response.status);
  }
  return (await response.json()) as T;
}

/** Returns `null` instead of throwing when the backend is unreachable. */
export async function tryRequest<T>(path: string): Promise<T | null> {
  try {
    return await request<T>(path);
  } catch {
    return null;
  }
}

// --- types ---------------------------------------------------------------

export interface Reason {
  feature: string;
  observed: unknown;
  expected_range: [number, number] | null;
  deviation_score: number;
  explanation: string;
  scope: string;
  detector: string;
}

export interface ExceptionCase {
  id: string;
  schema_id: string;
  entity_type: string;
  entity_id: string;
  event_external_id: string;
  fingerprint: string;
  title: string;
  detected_at: string;
  anomaly_score: number;
  severity: "critical" | "high" | "medium" | "low";
  confidence: number;
  baseline_scope: string;
  detector_versions: Record<string, string>;
  reasons: Reason[];
  status: string;
  assignee: string | null;
  resolution_note: string;
  narrative: Record<string, unknown>;
}

export interface DetectorOutput {
  detector: string;
  version: string;
  raw_score: number;
  score: number;
  applicable: boolean;
  reasons: Reason[];
  evidence: Record<string, unknown>;
}

export interface ExceptionDetail extends ExceptionCase {
  evidence: {
    weights?: Record<string, number>;
    weighted_mean?: number;
    strongest_detector?: string;
    detectors?: DetectorOutput[];
  };
  event_payload: Record<string, unknown>;
  recomputed_score: number;
  feedback: { label: string; note: string; author: string; created_at: string }[];
}

export interface EventSchema {
  id: string;
  name: string;
  entity_type: string;
  entity_key: string;
  event_key: string;
  timestamp_field: string;
  numeric_features: string[];
  categorical_features: string[];
  context_dimensions: string[];
  rate_features: string[];
  detector_weights: Record<string, number>;
  min_score: number;
  created_at: string;
  event_count: number;
  open_exceptions: number;
}

export interface Policy {
  id: string;
  schema_id: string | null;
  name: string;
  match: Record<string, unknown>;
  detectors: string[];
  max_score: number;
  reason: string;
  active: boolean;
  created_by: string;
  created_at: string;
}

export interface Overview {
  schema_count: number;
  event_count: number;
  open_exceptions: number;
  critical_exceptions: number;
  resolved_exceptions: number;
  labelled_exceptions: number;
  detector_precision: {
    detector: string;
    total: number;
    confirmed: number;
    rejected: number;
    precision: number;
  }[];
  top_exceptions: ExceptionCase[];
}

export interface SimilarMatch {
  exception_id: string;
  score: number;
  components: Record<string, number>;
  status: string;
  feedback_label: string | null;
  resolution_note: string;
  title: string;
}

export interface SimilarResponse {
  exception_id: string;
  similar: SimilarMatch[];
  note: string;
}

export interface RetrainingProposal {
  schema_id: string;
  labelled_count: number;
  detector_quality: {
    detector: string;
    total: number;
    confirmed: number;
    rejected: number;
    precision: number;
  }[];
  suggested_weights: Record<string, number>;
  suggested_min_score: number | null;
  proposed_policies: {
    name: string;
    match: Record<string, string>;
    detectors: string[];
    max_score: number;
    reason: string;
    supporting_exceptions: string[];
  }[];
  notes: string[];
  applied: boolean;
}

// --- endpoints -----------------------------------------------------------

export const api = {
  overview: () => tryRequest<Overview>("/overview"),
  schemas: () => tryRequest<EventSchema[]>("/schemas"),
  exceptions: (params: Record<string, string> = {}) => {
    const query = new URLSearchParams(params).toString();
    return tryRequest<ExceptionCase[]>(`/exceptions${query ? `?${query}` : ""}`);
  },
  exception: (id: string) => tryRequest<ExceptionDetail>(`/exceptions/${id}`),
  similar: (id: string) => tryRequest<SimilarResponse>(`/exceptions/${id}/similar`),
  policies: () => tryRequest<Policy[]>("/policies"),
  retrainingProposal: (schemaId: string) =>
    tryRequest<RetrainingProposal>(`/retraining/proposal?schema_id=${schemaId}`),
};
