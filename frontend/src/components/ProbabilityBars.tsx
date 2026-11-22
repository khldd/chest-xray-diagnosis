interface Props {
  probabilities: Record<string, number>;
  selected?: string | null;
  onSelect?: (disease: string) => void;
}

export default function ProbabilityBars({ probabilities, selected, onSelect }: Props) {
  const rows = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <ul className="bars">
      {rows.map(([disease, p]) => (
        <li key={disease}>
          <button
            type="button"
            className={`bar-row${selected === disease ? " selected" : ""}`}
            onClick={() => onSelect?.(disease)}
            disabled={!onSelect}
          >
            <span className="bar-label">{disease.replace("_", " ")}</span>
            <span className="bar-track">
              <span className="bar-fill" style={{ width: `${(p * 100).toFixed(1)}%` }} />
            </span>
            <span className="bar-value">{(p * 100).toFixed(1)}%</span>
          </button>
        </li>
      ))}
    </ul>
  );
}
