import { useEffect, useState } from "react";
import { getMetadata, type ModelMetadata } from "../api";

export default function ModelInfo() {
  const [meta, setMeta] = useState<ModelMetadata | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    getMetadata()
      .then(setMeta)
      .catch(() => setOffline(true));
  }, []);

  if (offline) return <p className="error small">API unreachable: start the backend.</p>;
  if (!meta) return <p className="muted small">Connecting to the API…</p>;

  const fmt = (v: number | null) => (v === null ? "not evaluated yet" : v.toFixed(3));
  return (
    <ul className="meta">
      <li>
        <span>Model</span> {meta.model_name} · {meta.classes.length} classes · {meta.input_size}px
      </li>
      <li>
        <span>Mean AUC</span> val {fmt(meta.val_mean_auc)} · test {fmt(meta.test_mean_auc)}
      </li>
      {meta.onnx_sha256 && (
        <li>
          <span>ONNX</span> sha256 {meta.onnx_sha256.slice(0, 12)}
        </li>
      )}
    </ul>
  );
}
