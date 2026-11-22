interface Props {
  originalUrl: string;
  disease: string;
  overlayBase64: string | null;
  loading: boolean;
  error: string | null;
}

export default function GradcamPanel({ originalUrl, disease, overlayBase64, loading, error }: Props) {
  return (
    <section className="panel">
      <h2>Grad-CAM: {disease.replace("_", " ")}</h2>
      <p className="muted small">
        Highlights the image regions that pushed the model toward this finding. The first heatmap
        after the server starts can take about 10 seconds.
      </p>
      <div className="compare">
        <figure>
          <img src={originalUrl} alt="Original X-ray" />
          <figcaption>Original</figcaption>
        </figure>
        <figure>
          {overlayBase64 && !loading && (
            <img src={`data:image/png;base64,${overlayBase64}`} alt={`Grad-CAM for ${disease}`} />
          )}
          {loading && <div className="placeholder muted">Computing heatmap…</div>}
          {error && !loading && <div className="placeholder error">{error}</div>}
          <figcaption>Grad-CAM overlay</figcaption>
        </figure>
      </div>
    </section>
  );
}
