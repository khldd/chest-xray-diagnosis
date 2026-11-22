import { useRef, useState } from "react";
import WebcamCapture from "./WebcamCapture";

interface Props {
  previewUrl: string | null;
  disabled: boolean;
  onFile: (file: Blob) => void;
}

export default function UploadPanel({ previewUrl, disabled, onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handle = (files: FileList | null) => {
    const file = files?.[0];
    if (file) onFile(file);
  };

  return (
    <section className="panel">
      <h2>X-ray</h2>
      <div
        className={`dropzone${dragging ? " dragging" : ""}`}
        onClick={() => !disabled && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) handle(e.dataTransfer.files);
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && !disabled && inputRef.current?.click()}
      >
        {previewUrl ? (
          <img src={previewUrl} alt="Uploaded chest X-ray" className="preview" />
        ) : (
          <p>Drop a chest X-ray here, or click to choose a file</p>
        )}
      </div>
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
      <WebcamCapture disabled={disabled} onCapture={onFile} />
    </section>
  );
}
