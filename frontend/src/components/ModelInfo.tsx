import { useEffect, useState } from "react";
import { getMetadata, type ModelEntry, type ModelMetadata } from "../api";

/** One-line model card under the top bar: size, AUCs, calibration, plus API status. */
export default function ModelInfo({ model }: { model: ModelEntry | null }) {
  const [meta, setMeta] = useState<ModelMetadata | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    getMetadata()
      .then(setMeta)
      .catch(() => setOffline(true));
  }, []);

  if (offline)
    return (
      <div className="modelcard">
        <span className="status off">API offline</span>
        <span className="muted">Start the backend, then reload.</span>
      </div>
    );
  if (!meta)
    return (
      <div className="modelcard">
        <span className="status wait">Connecting…</span>
      </div>
    );

  const fmt = (v?: number | null) => (v === null || v === undefined ? "–" : v.toFixed(3));
  const isEnsemble = model?.model_id === "ensemble";
  const name = isEnsemble ? "Ensemble" : (model?.display_name ?? meta.model_name);
  const val = model ? model.val_mean_auc : meta.val_mean_auc;
  const test = model ? model.test_mean_auc : meta.test_mean_auc;
  const calibrated = model?.calibrated ?? false;

  return (
    <div className="modelcard">
      <span className="status on">API online</span>
      <span className="mc-name">{name}</span>
      {isEnsemble ? (
        <span className="mc-item">average of {model?.ensemble_of?.length ?? 4} models</span>
      ) : (
        <>
          {model?.n_parameters ? (
            <span className="mc-item">{(model.n_parameters / 1e6).toFixed(1)}M params</span>
          ) : null}
          <span className="mc-item">
            val AUC <b>{fmt(val)}</b>
          </span>
          <span className="mc-item">
            test AUC <b>{fmt(test)}</b>
          </span>
        </>
      )}
      <span className="mc-item">
        {meta.classes.length} findings · {meta.input_size}px
      </span>
      <span
        className={`tag${calibrated ? " ok" : ""}`}
        title={
          calibrated
            ? "Temperature scaling fitted on the validation split: scores read as probabilities."
            : "Not calibrated: scores rank cases but are not probabilities."
        }
      >
        {calibrated ? "Calibrated" : "Uncalibrated scores"}
      </span>
    </div>
  );
}
