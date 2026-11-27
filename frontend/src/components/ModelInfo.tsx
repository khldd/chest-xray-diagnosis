import { useEffect, useState } from "react";
import { getMetadata, type ModelEntry, type ModelMetadata } from "../api";

/** Facts about the model in use: the selected registry entry, or the API's default model. */
export default function ModelInfo({ model }: { model: ModelEntry | null }) {
  const [meta, setMeta] = useState<ModelMetadata | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    getMetadata()
      .then(setMeta)
      .catch(() => setOffline(true));
  }, []);

  if (offline) return <p className="error small">API unreachable: start the backend.</p>;
  if (!meta) return <p className="muted small">Connecting to the API…</p>;

  const fmt = (v?: number | null) => (v === null || v === undefined ? "–" : v.toFixed(3));
  const name = model?.display_name ?? meta.model_name;
  const val = model ? model.val_mean_auc : meta.val_mean_auc;
  const test = model ? model.test_mean_auc : meta.test_mean_auc;
  const calibrated = model?.calibrated ?? false;
  return (
    <ul className="meta">
      <li>
        <span>Model</span> {name} · {meta.classes.length} classes · {meta.input_size}px
        {model?.n_parameters ? ` · ${(model.n_parameters / 1e6).toFixed(1)}M params` : ""}
      </li>
      {model?.model_id !== "ensemble" && (
        <li>
          <span>Mean AUC</span> val {fmt(val)} · test {fmt(test)}
        </li>
      )}
      <li>
        <span>Scores</span>{" "}
        {calibrated
          ? "calibrated on validation (temperature scaling): read them as probabilities"
          : "not calibrated: a ranking score, not a probability"}
      </li>
    </ul>
  );
}
