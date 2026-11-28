import { useRef, useState, type DragEvent } from "react";
import { ClockIcon, LayersIcon, LungsIcon, UploadIcon } from "./Icons";

type Mode = "split" | "original" | "heatmap";

interface Props {
  previewUrl: string | null;
  sourceName: string | null;
  disease: string | null;
  overlayBase64: string | null;
  overlayLoading: boolean;
  overlayError: string | null;
  overlayMs: number | null;
  analyzing: boolean;
  disabled: boolean;
  onFile: (file: Blob, name?: string) => void;
}

const pretty = (d: string) => d.replace("_", " ");

/**
 * The image stage. The API's overlay is the whole X-ray resized to a square, so the original is
 * drawn into the same square: the two layers line up pixel for pixel and a split slider can reveal
 * the heatmap over the original.
 */
export default function Viewer({
  previewUrl,
  sourceName,
  disease,
  overlayBase64,
  overlayLoading,
  overlayError,
  overlayMs,
  analyzing,
  disabled,
  onFile,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [mode, setMode] = useState<Mode>("split");
  const [split, setSplit] = useState(50);

  const handle = (files: FileList | null) => {
    const file = files?.[0];
    if (file) onFile(file, file.name);
  };
  const choose = () => !disabled && inputRef.current?.click();

  const dropProps = {
    onDragOver: (e: DragEvent) => {
      e.preventDefault();
      setDragging(true);
    },
    onDragLeave: () => setDragging(false),
    onDrop: (e: DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (!disabled) handle(e.dataTransfer.files);
    },
  };

  const overlaySrc = overlayBase64 ? `data:image/png;base64,${overlayBase64}` : null;
  const showOverlay = overlaySrc && !overlayLoading && mode !== "original";
  const clip = mode === "split" ? `inset(0 0 0 ${split}%)` : "none";

  return (
    <section className={`viewer card${dragging ? " dragging" : ""}`} {...dropProps}>
      <div className="viewer-head">
        <div className="viewer-title">
          <span className="eyebrow">Study</span>
          <span className="viewer-name">{sourceName ?? "No image loaded"}</span>
        </div>
        {previewUrl && (
          <div className="viewer-tools">
            <div className="mini-seg" role="radiogroup" aria-label="View">
              {(["original", "split", "heatmap"] as Mode[]).map((m) => (
                <button
                  key={m}
                  type="button"
                  role="radio"
                  aria-checked={mode === m}
                  className={mode === m ? "active" : ""}
                  onClick={() => setMode(m)}
                >
                  {m === "original" ? "Original" : m === "split" ? "Split" : "Heatmap"}
                </button>
              ))}
            </div>
            <button type="button" className="btn ghost" onClick={choose} disabled={disabled}>
              <UploadIcon size={16} /> Replace
            </button>
          </div>
        )}
      </div>

      {!previewUrl ? (
        <div
          className="empty-stage"
          onClick={choose}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && choose()}
        >
          <div className="empty-icon">
            <LungsIcon size={40} />
          </div>
          <p className="empty-title">Drop a frontal chest X-ray</p>
          <p className="empty-sub">PNG or JPEG · or pick a study from the list</p>
          <span className="btn primary">
            <UploadIcon size={16} /> Choose a file
          </span>
        </div>
      ) : (
        <div className="stage">
          <img className="layer" src={previewUrl} alt="Chest X-ray" />
          {showOverlay && (
            <img
              className="layer"
              src={overlaySrc}
              alt={disease ? `Grad-CAM heatmap for ${pretty(disease)}` : "Grad-CAM heatmap"}
              style={{ clipPath: clip }}
            />
          )}
          {showOverlay && mode === "split" && (
            <>
              <div className="split-line" style={{ left: `${split}%` }}>
                <span className="split-knob" />
              </div>
              <input
                className="split-input"
                type="range"
                min={0}
                max={100}
                value={split}
                onChange={(e) => setSplit(Number(e.target.value))}
                aria-label="Split position between original and heatmap"
              />
              <span className="corner left">Original</span>
              <span className="corner right">Heatmap</span>
            </>
          )}
          {(analyzing || overlayLoading) && (
            <div className="scan">
              <span className="scan-bar" />
              <span className="scan-label">
                {analyzing ? "Reading the image…" : "Computing heatmap…"}
              </span>
            </div>
          )}
          {overlayError && !overlayLoading && mode !== "original" && (
            <div className="stage-error">{overlayError}</div>
          )}
        </div>
      )}

      {previewUrl && (
        <div className="viewer-foot">
          <span className="chip">
            <LayersIcon size={14} />
            Grad-CAM{disease ? ` · ${pretty(disease)}` : ""}
          </span>
          {overlayMs !== null && !overlayLoading && (
            <span className="muted small-inline">
              <ClockIcon /> heatmap in {Math.round(overlayMs)} ms
            </span>
          )}
          <span className="muted small-inline hint">
            Red = regions that pushed the score up. Where the model looked, not a diagnosis.
          </span>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          handle(e.target.files);
          e.target.value = ""; // allow re-selecting the same file
        }}
      />
    </section>
  );
}
