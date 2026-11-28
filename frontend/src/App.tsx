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
import Findings from "./components/Findings";
import { AlertIcon, LungsIcon } from "./components/Icons";
import ModelInfo from "./components/ModelInfo";
import ModelSelector from "./components/ModelSelector";
import Viewer from "./components/Viewer";

export default function App() {
  const [file, setFile] = useState<Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [sourceName, setSourceName] = useState<string | null>(null);
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

  const analyze = async (blob: Blob, modelId = model, name?: string) => {
    const id = ++predictRun.current;
    overlayRun.current++; // cancel any heatmap still in flight for the previous image
    compareRun.current++;
    setComparison(null);
    setCompareLoading(false);
    if (blob !== file) {
      setFile(blob);
      setPreviewUrl(URL.createObjectURL(blob));
      setSourceName(name ?? "Uploaded image");
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

  const current = models.find((m) => m.model_id === model) ?? null;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            <LungsIcon />
          </span>
          <div>
            <span className="brand-name">Chest X-ray Diagnosis</span>
            <span className="brand-sub">14 thoracic findings · NIH ChestX-ray14</span>
          </div>
        </div>
        <ModelSelector
          models={models}
          value={model}
          disabled={loading}
          onChange={(id) => {
            setModel(id);
            if (file) void analyze(file, id);
          }}
        />
        <span
          className="disclaimer"
          title="Research and educational demo. Not a diagnostic tool and not medical advice."
        >
          <AlertIcon /> Research demo · not for diagnosis
        </span>
      </header>

      <ModelInfo model={current} />

      <main className="workspace">
        <aside className="rail">
          <ExampleGallery
            disabled={loading}
            active={sourceName}
            onPick={(blob, name) => void analyze(blob, model, name)}
          />
        </aside>

        <Viewer
          previewUrl={previewUrl}
          sourceName={sourceName}
          disease={selected}
          overlayBase64={overlay}
          overlayLoading={overlayLoading}
          overlayError={overlayError}
          overlayMs={overlayMs}
          analyzing={loading}
          disabled={loading}
          onFile={(blob, name) => void analyze(blob, model, name)}
        />

        <Findings
          result={result}
          loading={loading}
          error={error}
          selected={selected}
          calibrated={current?.calibrated ?? false}
          roundTripMs={roundTripMs}
          canCompare={models.length > 1 && !!file}
          comparing={compareLoading}
          onSelect={(d) => file && void loadOverlay(file, d)}
          onCompare={() => file && void runCompare(file, selected)}
        />
      </main>

      {(comparison || compareLoading || compareError) && (
        <CompareView
          comparison={comparison}
          models={models}
          loading={compareLoading}
          error={compareError}
        />
      )}

      <footer className="foot">
        Research and educational demo built on NIH ChestX-ray14. Labels were extracted from
        radiology reports by text mining; scores are not a diagnosis and the models were not
        validated outside this dataset.
      </footer>
    </div>
  );
}
