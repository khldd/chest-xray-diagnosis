// Thin client for the FastAPI backend. In dev and in Docker Compose, /api is proxied to it.
const BASE = "/api";

export interface TopItem {
  disease: string;
  probability: number;
}

export interface Prediction {
  probabilities: Record<string, number>;
  top: TopItem[];
  latency_ms: number;
  model?: string;
  disease?: string;
  overlay_png_base64?: string;
}

export interface ModelEntry {
  model_id: string;
  model_name: string;
  display_name: string;
  default: boolean;
  calibrated: boolean;
  val_mean_auc?: number | null;
  test_mean_auc?: number | null;
  n_parameters?: number | null;
  temperature?: number;
  ensemble_of?: string[] | null;
}

export interface Comparison {
  disease: string | null;
  results: Prediction[];
}

export interface ModelMetadata {
  model_name: string;
  task: string;
  classes: string[];
  input_size: number;
  val_mean_auc: number;
  test_mean_auc: number | null;
  onnx_sha256: string | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new Error("Cannot reach the API. Is the backend running?");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body: keep the generic message */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

function upload(file: Blob, filename: string): RequestInit {
  const form = new FormData();
  form.append("file", file, filename);
  return { method: "POST", body: form };
}

function query(params: Record<string, string | null | undefined>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v) q.set(k, v);
  const s = q.toString();
  return s ? `?${s}` : "";
}

export const predict = (file: Blob, model?: string | null, filename = "xray.png") =>
  request<Prediction>(`/predict${query({ model })}`, upload(file, filename));

export const visualize = (
  file: Blob,
  disease: string,
  model?: string | null,
  filename = "xray.png",
) => request<Prediction>(`/predict/visualize${query({ disease, model })}`, upload(file, filename));

export const compare = (file: Blob, disease?: string | null, filename = "xray.png") =>
  request<Comparison>(`/predict/compare${query({ disease })}`, upload(file, filename));

export const getModels = () => request<ModelEntry[]>("/models");

export const getMetadata = () => request<ModelMetadata>("/model/metadata");

export const getHealth = () =>
  request<{ status: string; model_loaded: boolean; model_file_present: boolean }>("/health");
