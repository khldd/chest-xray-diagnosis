interface Props {
  probabilities: Record<string, number>;
  selected?: string | null;
  onSelect?: (disease: string) => void;
}

/** All 14 findings, highest first. Click a row to show its heatmap. */
export default function ProbabilityBars({ probabilities, selected, onSelect }: Props) {
  const rows = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <ol className="bars">
      {rows.map(([disease, p], i) => (
        <li key={disease}>
          <button
            type="button"
            className={`bar-row${selected === disease ? " selected" : ""}${p >= 0.5 ? " high" : ""}`}
            onClick={() => onSelect?.(disease)}
            disabled={!onSelect}
            aria-pressed={selected === disease}
          >
            <span className="bar-rank">{i + 1}</span>
            <span className="bar-label">{disease.replace("_", " ")}</span>
            <span className="bar-track">
              <span className="bar-fill" style={{ width: `${Math.max(p * 100, 0.8).toFixed(1)}%` }} />
            </span>
            <span className="bar-value">{(p * 100).toFixed(1)}%</span>
          </button>
        </li>
      ))}
    </ol>
  );
}
