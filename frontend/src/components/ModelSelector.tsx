import type { ModelEntry } from "../api";

interface Props {
  models: ModelEntry[];
  value: string | null;
  disabled?: boolean;
  onChange: (id: string) => void;
}

const auc = (v?: number | null) => (v === null || v === undefined ? "–" : v.toFixed(3));

/** One pill per served model (plus the ensemble), with its test AUC. */
export default function ModelSelector({ models, value, disabled, onChange }: Props) {
  if (models.length < 2) return null;
  return (
    <div className="models">
      <span className="muted small">Model</span>
      {models.map((m) => (
        <button
          key={m.model_id}
          type="button"
          className={`pill${value === m.model_id ? " active" : ""}`}
          disabled={disabled}
          onClick={() => onChange(m.model_id)}
          title={m.display_name}
        >
          {m.model_id === "ensemble" ? "Ensemble" : m.display_name}
          {m.test_mean_auc != null && <small> AUC {auc(m.test_mean_auc)}</small>}
        </button>
      ))}
    </div>
  );
}
