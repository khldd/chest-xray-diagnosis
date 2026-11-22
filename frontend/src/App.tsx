import { useEffect, useState } from "react";
import { predict, type Prediction } from "./api";
import ProbabilityBars from "./components/ProbabilityBars";
import UploadPanel from "./components/UploadPanel";

export default function App() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Release the object URL when the image changes or the component unmounts.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const analyze = async (blob: Blob) => {
    setPreviewUrl(URL.createObjectURL(blob));
    setResult(null);
    setError(null);
    setLoading(true);
    try {
      setResult(await predict(blob));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
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
          {result && <ProbabilityBars probabilities={result.probabilities} />}
        </section>
      </div>
    </main>
  );
}
