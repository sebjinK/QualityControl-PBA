import React, { useState, useEffect, useRef } from "react";
// Removed: import "./App.css"; as CSS is typically imported in the root component or main index file,
// or you'll need to create a Caliber.css file if the styling isn't global.

// Define API endpoint here. Assuming the backend is running locally on port 8000 (standard FastAPI setup).

// works in CRA, Vite, Next.js, and falls back to localhost
// CRA/Next-friendly, with fallback
const API_BASE =
  process.env.REACT_APP_API_BASE ||
  process.env.NEXT_PUBLIC_API_BASE ||
  "http://localhost:8000";


const OCR_PATH = "/ocr";

export default function Label() {
  const [file, setFile] = useState(null);
  const [imgURL, setImgURL] = useState(null);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const imgRef = useRef(null);

  const onPick = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setImgURL(URL.createObjectURL(f));
    setResult(null);
  };

  useEffect(() => {
    if (!imgURL) return;
    const i = new Image();
    i.onload = () => setImgSize({ w: i.width, h: i.height });
    i.src = imgURL;
  }, [imgURL]);

  const onSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);

      const url = new URL(OCR_PATH, API_BASE);
      url.search = new URLSearchParams({
        return_image: "true",   // server can also return annotated source
        // backend now always returns all words + first_word (top-left)
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
    const sx = dispW / imgSize.w, sy = dispH / imgSize.h;
    return poly.map(([x, y]) => [Math.round(x * sx), Math.round(y * sy)]);
  };

  return (
    <div className="container">
      <h1>OCR Label Reader</h1>
      <div className="card">
        <input type="file" accept="image/*" onChange={onPick} />
        <button disabled={!file || loading} onClick={onSubmit}>
          {loading ? "Reading..." : "Read Text"}
        </button>

        {result?.error && <div className="alert alert-danger mt-3">{result.error}</div>}

        <div className="row">
          

          <div className="side">
            <h3>Top-left</h3>
            {first ? (
              <>
                <div><b>Text:</b> {first.text}</div>
                <div><b>Conf:</b> {first.prob.toFixed(2)}</div>
              </>
            ) : <div>No text found.</div>}

            <h3 style={{ marginTop: 16 }}>All words</h3>
            <ul className="list">
              {words.map((w, i) => (
                <li key={i}>
                  <div><b>Text:</b> {w.text || <i>(empty)</i>}</div>
                  <div><b>Conf:</b> {w.prob.toFixed(2)}</div>
                  <div style={{ fontSize: 12, color: "#64748b" }}>
                    [{w.poly.map(p => p.map(v => Math.round(v)).join(",")).join(" | ")}]
                  </div>
                </li>
              ))}
            </ul>

            {/* Download synthesized label PNG built from ALL words */}
            {result?.label_png_b64 && (
              <div style={{ marginTop: 12 }}>
                <a download="label.png" href={`data:image/png;base64,${result.label_png_b64}`}>
                  Download label PNG
                </a>
              </div>
            )}
          </div>
        </div>

        {/* optional server-annotated image (only first word in our backend) */}
        {result?.annotated_image_b64 && (
          <div className="meta">
            <h3>Server Annotated</h3>
            <img
              src={`data:image/jpeg;base64,${result.annotated_image_b64}`}
              alt="annotated"
              style={{ maxWidth: "100%" }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
