import { useEffect, useRef, useState } from "react";
import {
  compare,
  getModels,
  predict,
  visualize,
  type Comparison,
  type ModelEntry,
  type Prediction,
} from "./api";
import CompareView from "./components/CompareView";
import ExampleGallery from "./components/ExampleGallery";
import GradcamPanel from "./components/GradcamPanel";
import ModelInfo from "./components/ModelInfo";
import ModelSelector from "./components/ModelSelector";
import ProbabilityBars from "./components/ProbabilityBars";
import UploadPanel from "./components/UploadPanel";

export default function App() {
  const [file, setFile] = useState<Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [selected, setSelected] = useState<string | null>(null);
  const [overlay, setOverlay] = useState<string | null>(null);
  const [overlayLoading, setOverlayLoading] = useState(false);
  const [overlayError, setOverlayError] = useState<string | null>(null);
  const [roundTripMs, setRoundTripMs] = useState<number | null>(null);
  const [overlayMs, setOverlayMs] = useState<number | null>(null);

  const [models, setModels] = useState<ModelEntry[]>([]);
  const [model, setModel] = useState<string | null>(null);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const compareRun = useRef(0);

  useEffect(() => {
    getModels()
      .then((list) => {
        setModels(list);
        setModel(list.find((m) => m.default)?.model_id ?? list[0]?.model_id ?? null);
      })
      .catch(() => setModels([]));
  }, []);

  // Monotonic counters: a response is applied only if no newer request has started since.
  const predictRun = useRef(0);
  const overlayRun = useRef(0);

  // Release the object URL when the image changes or the component unmounts.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const loadOverlay = async (blob: Blob, disease: string, modelId = model) => {
    const id = ++overlayRun.current;
    setSelected(disease);
    setOverlay(null);
    setOverlayMs(null);
    setOverlayError(null);
    setOverlayLoading(true);
    try {
      const r = await visualize(blob, disease, modelId);
      if (id === overlayRun.current) {
        setOverlay(r.overlay_png_base64 ?? null);
        setOverlayMs(r.latency_ms);
      }
    } catch (e) {
      if (id === overlayRun.current) {
        setOverlayError(e instanceof Error ? e.message : "Heatmap failed");
      }
    } finally {
      if (id === overlayRun.current) setOverlayLoading(false);
    }
  };

  const runCompare = async (blob: Blob, disease: string | null) => {
    const id = ++compareRun.current;
    setComparison(null);
    setCompareError(null);
    setCompareLoading(true);
    try {
      const r = await compare(blob, disease);
      if (id === compareRun.current) setComparison(r);
    } catch (e) {
      if (id === compareRun.current) {
        setCompareError(e instanceof Error ? e.message : "Comparison failed");
      }
    } finally {
      if (id === compareRun.current) setCompareLoading(false);
    }
  };

  const analyze = async (blob: Blob, modelId = model) => {
    const id = ++predictRun.current;
    overlayRun.current++; // cancel any heatmap still in flight for the previous image
    compareRun.current++;
    setComparison(null);
    setCompareLoading(false);
    if (blob !== file) {
      setFile(blob);
      setPreviewUrl(URL.createObjectURL(blob));
    }
    setResult(null);
    setSelected(null);
    setOverlay(null);
    setOverlayMs(null);
    setOverlayError(null);
    setOverlayLoading(false);
    setError(null);
    setLoading(true);
    try {
      const t0 = performance.now();
      const r = await predict(blob, modelId);
      if (id !== predictRun.current) return;
      setRoundTripMs(Math.round(performance.now() - t0));
      setResult(r);
      void loadOverlay(blob, r.top[0].disease, modelId);
    } catch (e) {
      if (id === predictRun.current) {
        setError(e instanceof Error ? e.message : "Prediction failed");
      }
    } finally {
      if (id === predictRun.current) setLoading(false);
    }
  };

  return (
    <main className="container">
      <header>
        <h1>Chest X-ray Diagnosis</h1>
        <p className="disclaimer">
          Research and educational demo. Not a diagnostic tool and not medical advice.
        </p>
        <ModelInfo model={models.find((m) => m.model_id === model) ?? null} />
        <ModelSelector
          models={models}
          value={model}
          disabled={loading}
          onChange={(id) => {
            setModel(id);
            if (file) void analyze(file, id);
          }}
        />
      </header>

      <ExampleGallery disabled={loading} onPick={analyze} />

      <div className="grid">
        <UploadPanel previewUrl={previewUrl} disabled={loading} onFile={analyze} />

        <section className="panel">
          <h2>Predictions</h2>
          {error && <p className="error">{error}</p>}
          {loading && <p className="muted">Analyzing…</p>}
          {!loading && !error && !result && (
            <p className="muted">Upload an X-ray to see the 14 disease probabilities.</p>
          )}
          {result && (
            <>
              <p className="muted small">Click a disease to see where the model looked.</p>
              <ProbabilityBars
                probabilities={result.probabilities}
                selected={selected}
                onSelect={(d) => file && loadOverlay(file, d)}
              />
              <p className="muted small latency">
                Inference {result.latency_ms} ms on the server
                {roundTripMs !== null && <> · {roundTripMs} ms round trip</>}
              </p>
              {models.length > 1 && file && (
                <button
                  type="button"
                  className="btn"
                  disabled={compareLoading}
                  onClick={() => void runCompare(file, selected)}
                >
                  Compare all models{selected && ` on ${selected.replace("_", " ")}`}
                </button>
              )}
            </>
          )}
        </section>
      </div>

      {result && previewUrl && selected && (
        <GradcamPanel
          originalUrl={previewUrl}
          disease={selected}
          overlayBase64={overlay}
          loading={overlayLoading}
          error={overlayError}
          latencyMs={overlayMs}
        />
      )}

      {(comparison || compareLoading || compareError) && (
        <CompareView
          comparison={comparison}
          models={models}
          loading={compareLoading}
          error={compareError}
        />
      )}
    </main>
  );
}
