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
      <h2 className="pt-5 pb-2 text-center">Tile Caliber Analysis</h2>
      <h5 className="pb-3 text-center text-muted">Using Convolusion Neural Networks</h5>

      <form onSubmit={handleUpload} className="card w-50 mx-auto p-4 rounded-4 shadow-lg"> {/* ADDED rounded-4 and shadow-lg */}

        {/* 1. Styled Label/Button */}
        <label htmlFor="file-upload" className="btn btn-outline-secondary w-100 mb-4 rounded-3 shadow-sm cursor-pointer">
          {/* Display file name or default text */}
          {file ? `File Selected: ${file.name}` : 'Choose Tile Image for Calibre Check'}
        </label>

        {/* 2. Hidden Native Input */}
        <input
          id="file-upload" // Linked to label via htmlFor
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="d-none" // Hides the ugly default input
        />

        {preview && (
          <div className="preview-container">
            <img src={preview} alt="preview" className="preview-img shadow-lg d-block mx-auto mb-4" />
            {result && (
              <div className="overlay">
                <span className="label">
                </span>
              </div>
            )}
          </div>
        )}

        <p className="text-muted text-center">In production the ability to choose a picture will not be necessary.</p>
        <p className="text-muted text-center">Images of tiles will be pulled live from the Qualitron.</p>

        {/* Use mt-4 for separation from the preview, keeping it inside the padded card */}
        <button type="submit" disabled={!file || loading} className="btn btn-primary w-100 shadow-sm">
          {loading ? "Predicting..." : "Upload & Predict"}
        </button>
      </form>

      {result && (
        <div className="card w-50 mx-auto p-4 rounded-4 shadow-lg">
          <h3 className="text-center mb-3">Prediction Result</h3>
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