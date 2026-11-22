import { useEffect, useRef, useState } from "react";

interface Props {
  disabled: boolean;
  onCapture: (blob: Blob) => void;
}

// Meant for photographing a printed film or a screen showing an X-ray.
// getUserMedia only works on https:// or localhost.
export default function WebcamCapture({ disabled, onCapture }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stop = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setActive(false);
  };

  useEffect(() => stop, []); // release the camera on unmount

  const start = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      setActive(true);
      // The <video> mounts on the next render, so attach the stream after it exists.
      requestAnimationFrame(() => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      });
    } catch {
      setError("Camera unavailable. Check browser permissions or that a camera is connected.");
    }
  };

  const capture = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")!.drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) {
        stop();
        onCapture(blob);
      }
    }, "image/png");
  };

  return (
    <div className="webcam">
      {!active && (
        <button type="button" className="btn" disabled={disabled} onClick={start}>
          Use webcam
        </button>
      )}
      {active && (
        <>
          <video ref={videoRef} autoPlay playsInline muted className="preview" />
          <div className="row">
            <button type="button" className="btn primary" disabled={disabled} onClick={capture}>
              Capture and analyze
            </button>
            <button type="button" className="btn" onClick={stop}>
              Cancel
            </button>
          </div>
        </>
      )}
      {error && <p className="error small">{error}</p>}
    </div>
  );
}
