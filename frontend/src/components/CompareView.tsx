import type { Comparison, ModelEntry } from "../api";

interface Props {
  comparison: Comparison | null;
  models: ModelEntry[];
  loading: boolean;
  error: string | null;
}

const pretty = (d: string) => d.replace("_", " ");

/** Same X-ray through every model: heatmap, score for the chosen finding, top 3, and agreement. */
export default function CompareView({ comparison, models, loading, error }: Props) {
  const name = (id?: string) =>
    id === "ensemble" ? "Ensemble" : (models.find((m) => m.model_id === id)?.display_name ?? id);
  const auc = (id?: string) => models.find((m) => m.model_id === id)?.test_mean_auc ?? null;
  const disease = comparison?.disease ?? null;

  const singles = comparison?.results.filter((r) => r.model !== "ensemble") ?? [];
  const scores = disease ? singles.map((r) => r.probabilities[disease]) : [];
  const lo = scores.length ? Math.min(...scores) : 0;
  const hi = scores.length ? Math.max(...scores) : 0;
  const agreeTop = disease ? singles.filter((r) => r.top[0]?.disease === disease).length : 0;

  return (
    <section className="compare card">
      <div className="card-head">
        <div>
          <span className="eyebrow">Model comparison</span>
          <h2 className="compare-title">
            {disease ? pretty(disease) : "Top findings"} across {comparison?.results.length ?? "all"}{" "}
            models
          </h2>
        </div>
        {disease && singles.length > 0 && !loading && (
          <div className="agreement">
            <div>
              <span className="agree-value">
                {agreeTop}/{singles.length}
              </span>
              <span className="agree-label">rank it first</span>
            </div>
            <div>
              <span className="agree-value">
                {(lo * 100).toFixed(0)}–{(hi * 100).toFixed(0)}%
              </span>
              <span className="agree-label">spread of scores</span>
            </div>
          </div>
        )}
      </div>
      <p className="muted small-inline">
        Same image, every architecture. Agreement is more convincing than one model alone;
        disagreement is a reason for caution.
      </p>

      {loading && (
        <div className="compare-grid">
          {Array.from({ length: 5 }, (_, i) => (
            <div key={i} className="mcard skeleton-card" />
          ))}
        </div>
      )}
      {error && <p className="error">{error}</p>}

      {comparison && !loading && (
        <div className="compare-grid">
          {comparison.results.map((r) => {
            const p = disease ? r.probabilities[disease] : r.top[0].probability;
            const isEns = r.model === "ensemble";
            return (
              <figure key={r.model} className={`mcard${isEns ? " ensemble" : ""}`}>
                <div className="mcard-img">
                  {r.overlay_png_base64 ? (
                    <img src={`data:image/png;base64,${r.overlay_png_base64}`} alt={`${name(r.model)} heatmap`} />
                  ) : (
                    <div className="mcard-noimg muted">No heatmap</div>
                  )}
                  {isEns && <span className="badge">Ensemble</span>}
                </div>
                <figcaption>
                  <div className="mcard-head">
                    <strong>{name(r.model)}</strong>
                    {auc(r.model) != null && (
                      <span className="muted tiny">AUC {auc(r.model)!.toFixed(3)}</span>
                    )}
                  </div>
                  <div className="mcard-score">
                    <span className="mcard-value">{(p * 100).toFixed(1)}%</span>
                    <span className="muted tiny">{disease ? pretty(disease) : pretty(r.top[0].disease)}</span>
                  </div>
                  <div className="mini-track">
                    <span style={{ width: `${(p * 100).toFixed(1)}%` }} />
                  </div>
                  <div className="mcard-top">
                    {r.top.slice(0, 3).map((t) => (
                      <span key={t.disease} className={`tchip${t.disease === disease ? " hit" : ""}`}>
                        {pretty(t.disease)}
                      </span>
                    ))}
                  </div>
                  <span className="muted tiny">{Math.round(r.latency_ms)} ms</span>
                </figcaption>
              </figure>
            );
          })}
        </div>
      )}
    </section>
  );
}
