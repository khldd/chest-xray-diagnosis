import type { Comparison, ModelEntry } from "../api";

interface Props {
  comparison: Comparison | null;
  models: ModelEntry[];
  loading: boolean;
  error: string | null;
}

/** Same X-ray through every model: score for the selected finding, top 3, heatmap. */
export default function CompareView({ comparison, models, loading, error }: Props) {
  const name = (id?: string) =>
    id === "ensemble" ? "Ensemble" : (models.find((m) => m.model_id === id)?.display_name ?? id);
  const disease = comparison?.disease ?? null;
  return (
    <section className="panel">
      <h2>Compare models{disease && <>: {disease.replace("_", " ")}</>}</h2>
      <p className="muted small">
        The same image through every model. Agreement between architectures is more convincing
        than one model alone; disagreement is a reason for caution.
      </p>
      {loading && <p className="muted">Running every model and its heatmap…</p>}
      {error && <p className="error">{error}</p>}
      {comparison && !loading && (
        <div className="compare-grid">
          {comparison.results.map((r) => (
            <figure key={r.model} className="compare-card">
              {r.overlay_png_base64 && (
                <img src={`data:image/png;base64,${r.overlay_png_base64}`} alt={`${r.model}`} />
              )}
              <figcaption>
                <strong>{name(r.model)}</strong>
                {disease && (
                  <span className="score">
                    {disease.replace("_", " ")}: {(r.probabilities[disease] * 100).toFixed(1)}%
                  </span>
                )}
                <span className="muted small">
                  Top: {r.top.slice(0, 3).map((t) => t.disease.replace("_", " ")).join(", ")}
                </span>
                <span className="muted small">{r.latency_ms} ms</span>
              </figcaption>
            </figure>
          ))}
        </div>
      )}
    </section>
  );
}
