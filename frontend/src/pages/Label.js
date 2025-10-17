import React, { useState, useEffect, useRef } from "react";
// Removed: import "./App.css"; as CSS is typically imported in the root component or main index file,
// or you'll need to create a Caliber.css file if the styling isn't global.

// Define API endpoint here. Assuming the backend is running locally on port 8000 (standard FastAPI setup).
const API = "http://localhost:8000/ocr"; // <-- FIXED: Defined the missing API constant

function Label() { // Renamed from App to Label
  const [file, setFile] = useState(null);
  const [imgURL, setImgURL] = useState(null);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const imgRef = useRef(null); // <-- FIX: Changed to useRef(null), ensuring it's available.

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
      const res = await fetch(API + "?return_image=true", { method: "POST", body: fd });

      if (!res.ok) {
        // --- CRITICAL FIX: Read the detailed JSON error message from the server ---
        const errorBody = await res.json().catch(() => ({
          // Fallback if server response is not valid JSON (e.g., Uvicorn crash page)
          detail: `Server returned status ${res.status} (${res.statusText}). Response body was unreadable or empty.`
        }));

        // Throw a new Error using the 'detail' field provided by FastAPI's HTTPException
        throw new Error(errorBody.detail || `Server Error: Status ${res.status} received.`);
      }

      const data = await res.json();
      setResult(data);
      setImgSize({ w: data.width, h: data.height }); // scale polygons correctly

    } catch (e) {
      // This catch block now receives the detailed error message thrown above.
      console.error("OCR failed:", e);
      // Display the detailed error message in the UI
      setResult({ error: "OCR failed: " + e.message });

    } finally {
      setLoading(false);
    }
  };

  const words = result?.words || [];

  const scalePoly = (poly) => {
    if (!imgRef.current || !imgSize.w || !imgSize.h) return poly;
    const dispW = imgRef.current.clientWidth;
    const dispH = imgRef.current.clientHeight;
    const sx = dispW / imgSize.w;
    const sy = dispH / imgSize.h;
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

        {/* Display error message if one exists */}
        {result?.error && <div className="alert alert-danger mt-3">{result.error}</div>}

        <div className="row">
          <div className="preview">
            {imgURL && <img ref={imgRef} src={imgURL} alt="preview" />}
            {imgURL && words.map((w, i) => {
              const poly = scalePoly(w.poly);
              const points = poly.map(([x, y]) => `${x},${y}`).join(" ");
              return (
                <svg key={i} className="poly" viewBox={`0 0 ${imgRef.current?.clientWidth || 0} ${imgRef.current?.clientHeight || 0}`}>
                  <polygon points={points} />
                  {/* label near first point */}
                  <text x={poly[0][0] + 4} y={Math.max(12, poly[0][1] - 6)}>{w.text}</text>
                </svg>
              );
            })}
          </div>

          <div className="side">
            <h3>Words</h3>
            <ul className="list">
              {words.map((w, i) => (
                <li key={i}>
                  <div><b>Text:</b> {w.text || <i>(empty)</i>}</div>
                  <div><b>Conf:</b> {w.prob.toFixed(2)}</div>
                  <div><b>Poly:</b> [{w.poly.map(p => p.map(v => Math.round(v)).join(",")).join(" | ")}]</div>
                </li>
              ))}
            </ul>
          </div>
        </div>

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

export default Label;
