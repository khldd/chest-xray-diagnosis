import { useEffect, useState } from "react";

interface Example {
  file: string;
  label: string;
}

interface Props {
  disabled: boolean;
  onPick: (blob: Blob) => void;
}

// Images live in public/examples/ and are listed in public/examples/manifest.json:
//   [{ "file": "00000001_000.png", "label": "Cardiomegaly" }, ...]
export default function ExampleGallery({ disabled, onPick }: Props) {
  const [examples, setExamples] = useState<Example[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/examples/manifest.json")
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => setExamples(Array.isArray(data) ? data : []))
      .catch(() => setExamples([])); // no manifest (or not JSON): hide the gallery
  }, []);

  if (examples.length === 0) return null;

  const pick = async (ex: Example) => {
    setError(null);
    try {
      const res = await fetch(`/examples/${encodeURIComponent(ex.file)}`);
      if (!res.ok) throw new Error();
      onPick(await res.blob());
    } catch {
      setError(`Could not load ${ex.file}`);
    }
  };

  return (
    <section className="panel gallery">
      <h2>Try an example</h2>
      <div className="thumbs">
        {examples.map((ex) => (
          <button
            key={ex.file}
            type="button"
            className="thumb"
            disabled={disabled}
            onClick={() => pick(ex)}
            title={ex.label}
          >
            <img src={`/examples/${encodeURIComponent(ex.file)}`} alt={ex.label} loading="lazy" />
            <span>{ex.label}</span>
          </button>
        ))}
      </div>
      {error && <p className="error small">{error}</p>}
    </section>
  );
}
