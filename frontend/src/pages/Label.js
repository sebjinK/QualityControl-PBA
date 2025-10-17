import React, { useState, useEffect, useRef } from "react";

const API_BASE =
  process.env.REACT_APP_API_BASE ||
  process.env.NEXT_PUBLIC_API_BASE ||
  "http://localhost:8000";

const OCR_PATH = "/ocr";
const COMPARE_PATH = "/compare";

export default function Label() {
  // ----- OCR (single image) -----
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
    setResult(null);
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
      url.search = new URLSearchParams({ return_image: "true" }).toString();

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
  const hasOutput = !!(result && !loading);

  const scalePoly = (poly) => {
    if (!imgRef.current || !imgSize.w || !imgSize.h) return poly;
    const dispW = imgRef.current.clientWidth;
    const dispH = imgRef.current.clientHeight;
    const sx = dispW / imgSize.w,
      sy = dispH / imgSize.h;
    return poly.map(([x, y]) => [Math.round(x * sx), Math.round(y * sy)]);
  };

  // ----- Compare (two images) -----
  const [fileA, setFileA] = useState(null);
  const [fileB, setFileB] = useState(null);
  const [cmp, setCmp] = useState(null);
  const [cmpLoading, setCmpLoading] = useState(false);
  const [textWeight, setTextWeight] = useState(0.5); // optional knob

  const onCompare = async () => {
    if (!fileA || !fileB) return;
    setCmpLoading(true);
    setCmp(null);
    try {
      const fd = new FormData();
      fd.append("file1", fileA);
      fd.append("file2", fileB);

      const url = new URL(COMPARE_PATH, API_BASE);
      url.search = new URLSearchParams({
        text_weight: String(textWeight), // optional tunable 0..1
      }).toString();

      const res = await fetch(url, { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `Status ${res.status}`);
      }
      const data = await res.json();
      setCmp(data);
    } catch (e) {
      console.error(e);
      setCmp({ same: false, reason: "Compare failed: " + e.message, scores: {} });
    } finally {
      setCmpLoading(false);
    }
  };

  return (
    <div className="container">
      <h2 className="pt-5 pb-2 text-center">Label Automation</h2>
      <h5 className="pb-3 text-center text-muted">Using Convolution Neural Networks</h5>

      {/* -------- OCR CARD -------- */}
      <div className="card w-50 mx-auto p-4 rounded-4 shadow-lg mb-5">
        <h4 className="mb-3">Read Label (OCR)</h4>

        <label
          htmlFor="file-upload"
          className="btn btn-outline-secondary w-100 mb-3 rounded-3 shadow-sm cursor-pointer"
        >
          {file ? `File Selected: ${file.name}` : "Choose Box Image"}
        </label>
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
          {loading ? "Reading…" : "Read Text"}
        </button>

        {result?.error && <div className="alert alert-danger mt-3">{result.error}</div>}

        {/* IMAGE AREA */}
        <div className="mt-3">
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
          {hasOutput && result?.annotated_image_b64 && (
            <div className="meta text-center">
              <h6 className="mt-2 mb-2">Server Annotated</h6>
              <img
                ref={imgRef}
                src={`data:image/jpeg;base64,${result.annotated_image_b64}`}
                alt="annotated"
                style={{ maxWidth: "100%", borderRadius: 12 }}
              />
            </div>
          )}
        </div>

        {/* DETAILS */}
        {hasOutput && (
          <div className="mt-3">
            <h5 className="text-center">Box ID</h5>
            {first ? (
              <div className="text-center">
                <b>Text:</b> {first.text} &nbsp;|&nbsp; <b>Conf:</b> {first.prob.toFixed(2)}
              </div>
            ) : (
              <div className="text-center text-muted">No text found.</div>
            )}

            <h6 className="mt-3 mb-2">All Words</h6>
            {words.length > 0 ? (
              <ul className="list">
                {words.map((w, i) => (
                  <li key={i}>
                    <div><b>Text:</b> {w.text || <i>(empty)</i>}</div>
                    <div><b>Conf:</b> {w.prob.toFixed(2)}</div>
                    <div style={{ fontSize: 12, color: "#64748b" }}>
                      [{w.poly.map((p) => p.map((v) => Math.round(v)).join(",")).join(" | ")}]
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="text-muted">No words found.</div>
            )}

            {result?.label_png_b64 && (
              <div style={{ marginTop: 12 }} className="text-center">
                <a download="label.png" href={`data:image/png;base64,${result.label_png_b64}`}>
                  Download label PNG
                </a>
              </div>
            )}
          </div>
        )}
      </div>

      {/* -------- COMPARE CARD -------- */}
      <div className="card w-50 mx-auto p-4 rounded-4 shadow-lg">
        <h4 className="mb-3">Compare Two Labels</h4>

        <div className="mb-2">
          <label className="form-label">Image A</label>
          <input type="file" accept="image/*" className="form-control"
            onChange={(e) => setFileA(e.target.files?.[0] || null)} />
        </div>

        <div className="mb-3">
          <label className="form-label">Image B</label>
          <input type="file" accept="image/*" className="form-control"
            onChange={(e) => setFileB(e.target.files?.[0] || null)} />
        </div>

        <div className="mb-3">
          <label className="form-label">Text Weight (0–1): {textWeight}</label>
          <input
            type="range" min="0" max="1" step="0.05" value={textWeight}
            onChange={(e) => setTextWeight(parseFloat(e.target.value))}
            className="form-range"
          />
          <small className="text-muted">
            Higher weight relies more on OCR text similarity; lower weight relies more on image similarity.
          </small>
        </div>

        <button
          className="btn btn-secondary w-100"
          disabled={!fileA || !fileB || cmpLoading}
          onClick={onCompare}
        >
          {cmpLoading ? "Comparing…" : "Compare"}
        </button>

        {cmp && (
          <div className="card mt-3 p-3">
            <h5 className="text-center pt-1 pb-3">Same? {cmp.same ? "Yes" : "No"}</h5>
            <div className="mb-4">{cmp.reason}</div>
            {cmp.scores && (
              <pre style={{ background:"#f8fafc", padding:12, borderRadius:8}}>
                {JSON.stringify(cmp.scores, null, 2)}
              </pre>
            )}
            <div className="row pt-3">
              <div className="col">
                <h6>Text A</h6>
                <p>{cmp.text1 || <i>(none)</i>}</p>
              </div>
              <div className="col">
                <h6>Text B</h6>
                <p>{cmp.text2 || <i>(none)</i>}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
