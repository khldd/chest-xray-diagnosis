import type { Prediction } from "../api";
import { ClockIcon, GridIcon } from "./Icons";
import ProbabilityBars from "./ProbabilityBars";

interface Props {
  result: Prediction | null;
  loading: boolean;
  error: string | null;
  selected: string | null;
  calibrated: boolean;
  roundTripMs: number | null;
  canCompare: boolean;
  comparing: boolean;
  onSelect: (disease: string) => void;
  onCompare: () => void;
}

const pretty = (d: string) => d.replace("_", " ");

export default function Findings({
  result,
  loading,
  error,
  selected,
  calibrated,
  roundTripMs,
  canCompare,
  comparing,
  onSelect,
  onCompare,
}: Props) {
  const focus = result && (selected ?? result.top[0].disease);
  const focusP = result && focus ? result.probabilities[focus] : null;
  const above = result ? Object.values(result.probabilities).filter((p) => p >= 0.5).length : 0;

  return (
    <section className="findings card">
      <div className="card-head">
        <span className="eyebrow">Findings</span>
        {result && (
          <span className="muted small-inline">
            {above === 0 ? "none above 50%" : `${above} above 50%`}
          </span>
        )}
      </div>

      {error && <p className="error">{error}</p>}

      {!result && !error && (
        <div className="skeleton" aria-busy={loading}>
          <div className="sk-hero" />
          {Array.from({ length: 8 }, (_, i) => (
            <div key={i} className="sk-row" style={{ opacity: 1 - i * 0.1 }} />
          ))}
          <p className="muted center">
            {loading ? "Running the model…" : "Load an X-ray to see all 14 findings."}
          </p>
        </div>
      )}

      {result && focus && focusP !== null && (
        <>
          <div className="hero">
            <span className="hero-label">{focus === result.top[0].disease ? "Top finding" : "Selected finding"}</span>
            <div className="hero-row">
              <span className="hero-name">{pretty(focus)}</span>
              <span className="hero-value">
                {(focusP * 100).toFixed(1)}
                <small>%</small>
              </span>
            </div>
            <div className="hero-track">
              <span style={{ width: `${(focusP * 100).toFixed(1)}%` }} />
            </div>
            <span className="hero-note">
              {calibrated
                ? "Calibrated probability that this finding is labelled in the report."
                : "Ranking score, not a probability."}
            </span>
          </div>

          <p className="muted small-inline list-hint">Click a finding to see where the model looked.</p>
          <ProbabilityBars
            probabilities={result.probabilities}
            selected={focus}
            onSelect={onSelect}
          />

          <div className="findings-foot">
            <span className="muted small-inline">
              <ClockIcon /> {Math.round(result.latency_ms)} ms on the server
              {roundTripMs !== null && <> · {roundTripMs} ms round trip</>}
            </span>
            {canCompare && (
              <button type="button" className="btn primary wide" disabled={comparing} onClick={onCompare}>
                <GridIcon /> {comparing ? "Comparing…" : `Compare all models on ${pretty(focus)}`}
              </button>
            )}
          </div>
        </>
      )}
    </section>
  );
}
