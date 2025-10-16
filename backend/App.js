import React, { useState } from "react";
import "./App.css";

function App() {
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

export default App;
