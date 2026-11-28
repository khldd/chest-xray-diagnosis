import { useEffect, useState } from "react";

interface Example {
  file: string;
  label: string;
}

interface Props {
  disabled: boolean;
  active: string | null;
  onPick: (blob: Blob, name: string) => void;
}

// Images live in public/examples/ and are listed in public/examples/manifest.json:
//   [{ "file": "00000001_000.png", "label": "Cardiomegaly" }, ...]
export default function ExampleGallery({ disabled, active, onPick }: Props) {
  const [examples, setExamples] = useState<Example[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/examples/manifest.json")
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => setExamples(Array.isArray(data) ? data : []))
      .catch(() => setExamples([])); // no manifest (or not JSON): hide the list
  }, []);

  if (examples.length === 0) return null;

  const pick = async (ex: Example) => {
    setError(null);
    try {
      const res = await fetch(`/examples/${encodeURIComponent(ex.file)}`);
      if (!res.ok) throw new Error();
      onPick(await res.blob(), ex.file);
    } catch {
      setError(`Could not load ${ex.file}`);
    }
  };

  return (
    <nav className="studies" aria-label="Example studies">
      <div className="rail-head">
        <span className="eyebrow">Test-set studies</span>
        <span className="count">{examples.length}</span>
      </div>
      <ul>
        {examples.map((ex) => (
          <li key={ex.file}>
            <button
              type="button"
              className={`study${active === ex.file ? " active" : ""}`}
              disabled={disabled}
              onClick={() => pick(ex)}
            >
              <img src={`/examples/${encodeURIComponent(ex.file)}`} alt="" loading="lazy" />
              <span className="study-text">
                <span className="study-label">{ex.label.replace("_", " ")}</span>
                <span className="study-file">{ex.file.replace(".png", "")}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
      {error && <p className="error small">{error}</p>}
    </nav>
  );
}
