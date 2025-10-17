from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import uvicorn
import base64
from typing import List, Optional
import cv2
from paddleocr import PaddleOCR
import traceback

# --- Config ---
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("square_regressor.pt")

# --- Initialization and Configuration (Combined) ---
app = FastAPI(title="Tile Quality Control API (CNN + OCR)")
print(f"DEBUG: Initializing FastAPI application. Device is set to {DEVICE}.")

# Load OCR model once at startup (downloads models on first run)
# NOTE: Using use_textline_orientation=True replaces the deprecated use_angle_cls=True.
ocr = PaddleOCR(use_textline_orientation=True, lang='en')
print("DEBUG: PaddleOCR model initialized.")

# Allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model definition for CNN Calibre Prediction ---
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 1)
    def forward(self, x):
        f = self.net(x).flatten(1)
        return self.head(f)

# --- Load trained weights ---
model = TinyCNN().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print(f"✅ Loaded TinyCNN from {MODEL_PATH}")

# --- Schemas for CNN Prediction ---
class PredictionResponse(BaseModel):
    predicted_side_cm: float
    confidence: float = 1.0

# --- Health Check/Session Endpoint (To resolve 404 errors from frontend) ---
@app.get("/api/dashboard")
async def check_dashboard_session():
    # This route exists purely to satisfy the frontend's session check on mount.
    return {"status": "active"}

# --- Endpoint 1: CNN Calibre Prediction ---
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    print(f"DEBUG: POST /predict called for file '{file.filename}'.")
    try:
        # Load image
        print("DEBUG: Task: Reading image bytes for CNN.")
        img_bytes = await file.read()
        pil = Image.open(BytesIO(img_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

        # Convert to tensor
        print("DEBUG: Task: Converting PIL image to PyTorch tensor.")
        x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(DEVICE)

        # Predict
        print("DEBUG: Task: Running CNN prediction.")
        with torch.no_grad():
            y_pred = model(x).cpu().numpy().squeeze().item()

        print(f"DEBUG: CNN Prediction complete. Result: {round(y_pred, 3)} cm.")
        return PredictionResponse(predicted_side_cm=round(y_pred, 3))
    except Exception as e:
        print(f"CRITICAL CNN ERROR in /predict: {type(e).__name__}: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during CNN prediction: {type(e).__name__}: {e}")

# --- Schemas for OCR Label Reading ---
class Word(BaseModel):
    poly: List[List[float]]       # 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    text: str
    prob: float

class OCRResponse(BaseModel):
    width: int
    height: int
    words: List[Word]
    annotated_image_b64: Optional[str] = None

def draw_poly(img, poly, text):
    """Draws bounding box and text label on the image."""
    pts = np.array(poly, dtype=np.int32)
    cv2.polylines(img, [pts], True, (0,255,0), 2)
    x = int(min(p[0] for p in poly)); y = int(min(p[1] for p in poly))
    label = text[:32]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, max(0,y-th-6)), (x+tw+8, y), (0,255,0), -1)
    cv2.putText(img, label, (x+4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

# --- Endpoint 2: OCR Label Reading ---
@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...), return_image: bool = Query(False)):
    print(f"DEBUG: POST /ocr called for file '{file.filename}'. return_image={return_image}.")
    try:
        # Read image bytes
        print("DEBUG: Task: Reading and decoding image bytes for OCR (using OpenCV).")
        data = np.frombuffer(await file.read(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        # Check for invalid image early
        if bgr is None:
            print("ERROR: Invalid image file detected.")
            raise ValueError("Invalid image: The file could not be decoded by OpenCV. Ensure it is a standard image format.")

        H, W = bgr.shape[:2]
        
        # Run OCR on the image
        print("DEBUG: Task: Starting PaddleOCR processing.")
        out = ocr.ocr(bgr)  
        print("DEBUG: Task: PaddleOCR finished.")

        words: List[Word] = []
        # Process OCR results
        print("DEBUG: Task: Processing and unpacking OCR results.")
        
        # CRITICAL FIX: Use the proven nested unpacking pattern (poly, (text, prob))
        if out and out[0] and isinstance(out[0], list):
            for line in out[0]:
                try:
                    # Use the proven nested unpacking pattern
                    poly, (text, prob) = line 
                    words.append(Word(poly=poly, text=text, prob=float(prob)))
                except Exception as e:
                    # This catches lines that might be metadata or wrongly structured, preventing a crash.
                    print(f"WARNING: Failed to unpack OCR line (Skipped due to structure issue). Line: {line}. Error: {e}")
        else:
            print(f"WARNING: PaddleOCR returned an unexpected top-level structure: {out}")


        print(f"DEBUG: Found {len(words)} words.")
        b64 = None
        if return_image:
            print("DEBUG: Task: Annotating image and encoding to Base64.")
            # Create an annotated image copy
            anno = bgr.copy()
            for w in words:
                draw_poly(anno, w.poly, w.text)
            
            # Encode annotated image to JPG bytes
            ok, buf = cv2.imencode(".jpg", anno)
            if ok:
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                print("DEBUG: Annotated image successfully Base64 encoded.")
            else:
                print("ERROR: Failed to encode annotated image to JPG.")

        print("DEBUG: OCR processing complete. Returning response.")
        return OCRResponse(width=W, height=H, words=words, annotated_image_b64=b64)

    except ValueError as e:
        # Catch known input errors (like invalid image format)
        print(f"ERROR: Caught ValueError: {e}")
        raise HTTPException(status_code=400, detail=f"Bad Request: {e}")
    except Exception as e:
        # Catch all unexpected server errors and return detailed traceback
        error_detail = f"Internal Server Error during OCR processing: {type(e).__name__}: {e}. Traceback: {traceback.format_exc()}"
        print(f"CRITICAL OCR ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)


# --- Run ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
