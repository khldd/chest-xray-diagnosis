import type { ModelEntry } from "../api";

interface Props {
  models: ModelEntry[];
  value: string | null;
  disabled?: boolean;
  onChange: (id: string) => void;
}

const auc = (v?: number | null) => (v === null || v === undefined ? null : v.toFixed(3));

/** Segmented control: one segment per served model plus the ensemble, with its test AUC. */
export default function ModelSelector({ models, value, disabled, onChange }: Props) {
  if (models.length < 2) return null;
  return (
    <div className="segmented" role="radiogroup" aria-label="Model">
      {models.map((m) => {
        const active = value === m.model_id;
        const score = auc(m.test_mean_auc);
        return (
          <button
            key={m.model_id}
            type="button"
            role="radio"
            aria-checked={active}
            className={`segment${active ? " active" : ""}`}
            disabled={disabled}
            onClick={() => onChange(m.model_id)}
          >
            <span className="segment-name">
              {m.model_id === "ensemble" ? "Ensemble" : m.display_name}
            </span>
            <span className="segment-sub">{score ? `AUC ${score}` : "mean of 4"}</span>
          </button>
        );
      })}
    </div>
  );
}
