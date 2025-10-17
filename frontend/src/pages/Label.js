import React, { useState, useEffect, useRef } from "react";

const API_BASE =
  process.env.REACT_APP_API_BASE ||
  process.env.NEXT_PUBLIC_API_BASE ||
  "http://localhost:8000";

const OCR_PATH = "/ocr";

export default function Label() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const imgRef = useRef(null);

  function handleFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
    setResult(null); // reset any prior result
  }

  useEffect(() => {
    if (!preview) return;
    const i = new Image();
    i.onload = () => setImgSize({ w: i.width, h: i.height });
    i.src = preview;
  }, [preview]);

  const onSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);

      const url = new URL(OCR_PATH, API_BASE);
      url.search = new URLSearchParams({
        return_image: "true",
      }).toString();

      const res = await fetch(url, { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `Status ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
      setImgSize({ w: data.width, h: data.height });
    } catch (e) {
      console.error(e);
      setResult({ error: "OCR failed: " + e.message });
    } finally {
      setLoading(false);
    }
  };

  const words = result?.words || [];
  const first = result?.first_word || null;

  const scalePoly = (poly) => {
    if (!imgRef.current || !imgSize.w || !imgSize.h) return poly;
    const dispW = imgRef.current.clientWidth;
    const dispH = imgRef.current.clientHeight;
    const sx = dispW / imgSize.w,
      sy = dispH / imgSize.h;
    return poly.map(([x, y]) => [Math.round(x * sx), Math.round(y * sy)]);
  };

  // SHOW DETAILS ONLY AFTER GENERATION COMPLETES
  const hasOutput = !!(result && !loading);

  return (
    <div className="container">
      <h2 className="pt-5 pb-2 text-center">Label Automation</h2>
      <h5 className="pb-3 text-center text-muted">
        Using Convolution Neural Networks
      </h5>

      <div className="card w-50 mx-auto p-4 rounded-4 shadow-lg">
        <label
          htmlFor="file-upload"
          className="btn btn-outline-secondary w-100 mb-4 rounded-3 shadow-sm cursor-pointer"
        >
          {file ? `File Selected: ${file.name}` : "Choose Box Image"}
        </label>
        <p className="text-muted text-center">
          In production the ability to choose a picture will not be necessary.
        </p>
        <p className="text-muted text-center">
          Images of labels will be pulled live from the camera.
        </p>

        <input
          id="file-upload"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="d-none"
        />

        <button
          onClick={onSubmit}
          disabled={!file || loading}
          className="btn btn-primary w-100 shadow-sm"
        >
          {loading ? "Generating..." : "Generate Label"}
        </button>

        {result?.error && (
          <div className="alert alert-danger mt-3">{result.error}</div>
        )}

        {/* IMAGE AREA */}
        <div className="mt-4">
          {/* Before generation: show preview (if selected) */}
          {!hasOutput && preview && (
            <div className="text-center">
              <img
                ref={imgRef}
                src={preview}
                alt="preview"
                style={{ maxWidth: "100%", borderRadius: 12 }}
              />
            </div>
          )}

          {/* After generation: show annotated image if provided */}
          {hasOutput && result?.annotated_image_b64 && (
            <div className="meta text-center">
              <h5 className="mt-2 mb-2">Server Annotated</h5>
              <img
                ref={imgRef}
                src={`data:image/jpeg;base64,${result.annotated_image_b64}`}
                alt="annotated"
                style={{ maxWidth: "100%", borderRadius: 12 }}
              />
            </div>
          )}
        </div>

        {/* DETAILS PANEL — HIDDEN UNTIL WE HAVE OUTPUT */}
        {hasOutput && (
          <div className="row">
            <div className="side">
              <h4 className="mt-4 mb-2 text-center">Box ID</h4>
              {first ? (
                <div className="text-center">
                  <b>Text:</b> {first.text} | <b>Conf:</b>{" "}
                  {first.prob.toFixed(2)}
                </div>
              ) : (
                <div className="text-center text-muted">No text found.</div>
              )}

              <h4 className="mt-3 mb-2 text-center">Label Details</h4>
              {words.length > 0 ? (
                <ul className="list">
                  {words.map((w, i) => (
                    <li key={i}>
                      <div>
                        <b>Text:</b> {w.text || <i>(empty)</i>}
                      </div>
                      <div>
                        <b>Conf:</b> {w.prob.toFixed(2)}
                      </div>
                      <div style={{ fontSize: 12, color: "#64748b" }}>
                        [
                        {w.poly
                          .map((p) => p.map((v) => Math.round(v)).join(","))
                          .join(" | ")}
                        ]
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="text-center text-muted">No words found.</div>
              )}

              {result?.label_png_b64 && (
                <div style={{ marginTop: 12 }} className="text-center">
                  <a
                    download="label.png"
                    href={`data:image/png;base64,${result.label_png_b64}`}
                  >
                    Download label PNG
                  </a>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}