import React, { useState } from "react";
// Removed: import "./App.css"; as CSS is typically imported in the root component or main index file,
// or you'll need to create a Caliber.css file if the styling isn't global.

function Caliber() { // Renamed from App to Caliber
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  function handleFileChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
    setResult(null);
  }

  async function handleUpload(e) {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      // NOTE: Using hardcoded endpoint. Consider using process.env.REACT_APP_BACKURL like in Landing.js
      const resp = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) throw new Error(await resp.text());
      const json = await resp.json();
      setResult(json);
    } catch (err) {
      alert("Upload failed: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1>Square Size Predictor</h1>

      <form onSubmit={handleUpload} className="card">
        <input type="file" accept="image/*" onChange={handleFileChange} />
        {preview && (
          <div className="preview-container">
            <img src={preview} alt="preview" className="preview-img" />
            {result && (
              <div className="overlay">
                <span className="label">
                  {result.predicted_side_cm.toFixed(2)} cm
                </span>
              </div>
            )}
          </div>
        )}
        <button type="submit" disabled={!file || loading}>
          {loading ? "Predicting..." : "Upload & Predict"}
        </button>
      </form>

      {result && (
        <div className="result card">
          <h2>Prediction Result</h2>
          <p>
            <strong>Predicted Side:</strong> {result.predicted_side_cm.toFixed(2)} cm
          </p>
          <p>
            <strong>Confidence:</strong> {result.confidence}
          </p>
        </div>
      )}
    </div>
  );
}

export default Caliber;