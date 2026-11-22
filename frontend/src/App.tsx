import { useEffect, useRef, useState } from "react";
import { predict, visualize, type Prediction } from "./api";
import GradcamPanel from "./components/GradcamPanel";
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

  // Monotonic counters: a response is applied only if no newer request has started since.
  const predictRun = useRef(0);
  const overlayRun = useRef(0);

  // Release the object URL when the image changes or the component unmounts.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const loadOverlay = async (blob: Blob, disease: string) => {
    const id = ++overlayRun.current;
    setSelected(disease);
    setOverlay(null);
    setOverlayError(null);
    setOverlayLoading(true);
    try {
      const r = await visualize(blob, disease);
      if (id === overlayRun.current) setOverlay(r.overlay_png_base64 ?? null);
    } catch (e) {
      if (id === overlayRun.current) {
        setOverlayError(e instanceof Error ? e.message : "Heatmap failed");
      }
    } finally {
      if (id === overlayRun.current) setOverlayLoading(false);
    }
  };

  const analyze = async (blob: Blob) => {
    const id = ++predictRun.current;
    overlayRun.current++; // cancel any heatmap still in flight for the previous image
    setFile(blob);
    setPreviewUrl(URL.createObjectURL(blob));
    setResult(null);
    setSelected(null);
    setOverlay(null);
    setOverlayError(null);
    setOverlayLoading(false);
    setError(null);
    setLoading(true);
    try {
      const r = await predict(blob);
      if (id !== predictRun.current) return;
      setResult(r);
      void loadOverlay(blob, r.top[0].disease);
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
      </header>

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
        />
      )}
    </main>
  );
}
