from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
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
import cv2
from paddleocr import PaddleOCR
import traceback
from PIL import Image, ImageDraw, ImageFont
import statistics
from pydantic import BaseModel, Field

# --- Config ---
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "~/QualityControl-PBA/backend/square_regressor.pt"

# --- Initialization and Configuration (Combined) ---
app = FastAPI(title="Tile Quality Control API (CNN + OCR)")
print(f"DEBUG: Initializing FastAPI application. Device is set to {DEVICE}.")

# Load OCR model once at startup (downloads models on first run)
# NOTE: Using use_textline_orientation=True replaces the deprecated use_angle_cls=True.
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


# ---------- setup ----------
# app.py




ocr = PaddleOCR(use_angle_cls=True, lang='en')

class Word(BaseModel):
    poly: List[List[float]]
    text: str
    prob: float

class Line(BaseModel):
    box: List[int]   # [x1, y1, x2, y2] of the merged line
    text: str
    prob: float      # avg prob of words in the line




class OCRResponse(BaseModel):
    width: int
    height: int
    words: List[Word]
    first_word: Optional[Word] = None
    lines: List[Line] = Field(default_factory=list)    # ✅
    first_line: Optional[Line] = None
    annotated_image_b64: Optional[str] = None
    label_png_b64: Optional[str] = None




def poly_bbox(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

def poly_height(poly):
    _, y1, _, y2 = poly_bbox(poly)
    return y2 - y1

def words_to_line_groups(words: List[Word]) -> List[Line]:
    if not words:
        return []

    # estimate a dynamic row threshold from median word height
    med_h = statistics.median(max(8, poly_height(w.poly)) for w in words)
    row_thresh = max(12, 0.7 * med_h)

    # compute centers for sorting & grouping
    items = []
    for w in words:
        xs = [p[0] for p in w.poly]; ys = [p[1] for p in w.poly]
        cx = sum(xs) / 4.0; cy = sum(ys) / 4.0
        items.append((cy, cx, w))

    # sort by y then x
    items.sort(key=lambda t: (t[0], t[1]))

    # group into rows by y-distance
    rows: List[List[Word]] = []
    for cy, cx, w in items:
        if not rows:
            rows.append([w])
        else:
            # compare to last row center
            last_row = rows[-1]
            last_row_cy = sum(sum(p[1] for p in ww.poly) / 4.0 for ww in last_row) / len(last_row)
            if abs(cy - last_row_cy) <= row_thresh:
                rows[-1].append(w)
            else:
                rows.append([w])

    # inside each row, sort by x and merge
    lines: List[Line] = []
    for row in rows:
        row.sort(key=lambda w: min(p[0] for p in w.poly))  # left→right
        text = " ".join((w.text or "").strip() for w in row if (w.text or "").strip())
        if not text:
            continue
        # merged bbox
        x1 = min(min(p[0] for p in w.poly) for w in row)
        y1 = min(min(p[1] for p in w.poly) for w in row)
        x2 = max(max(p[0] for p in w.poly) for w in row)
        y2 = max(max(p[1] for p in w.poly) for w in row)
        avgp = float(sum(w.prob for w in row) / len(row))
        lines.append(Line(box=[int(x1), int(y1), int(x2), int(y2)], text=text, prob=avgp))

    # top-to-bottom order
    lines.sort(key=lambda L: (L.box[1], L.box[0]))
    return lines

def draw_line_box(img, box, text):
    x1,y1,x2,y2 = box
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    label = text[:50]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 10, y1), (0,255,0), -1)
    cv2.putText(img, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

def draw_poly(img, poly, text):
    pts = np.array(poly, dtype=np.int32)
    cv2.polylines(img, [pts], True, (0,255,0), 2)
    x = int(min(p[0] for p in poly)); y = int(min(p[1] for p in poly))
    label = text[:32]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, max(0, y - th - 6)), (x + tw + 8, y), (0,255,0), -1)
    cv2.putText(img, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def words_to_lines(words: List[Word], row_thresh: float = 18.0) -> List[str]:
    """
    Convert word polygons to reading-order lines by grouping on vertical centers.
    Simple heuristic that works well for labels.
    """
    items = []
    for w in words:
        xs = [p[0] for p in w.poly]; ys = [p[1] for p in w.poly]
        cx = float(sum(xs) / 4.0); cy = float(sum(ys) / 4.0)
        items.append((cy, cx, w.text))

    # sort by y then x
    items.sort(key=lambda t: (t[0], t[1]))

    # cluster rows by y distance
    rows: List[List[tuple]] = []
    for cy, cx, text in items:
        if not rows or abs(cy - rows[-1][0][0]) > row_thresh:
            rows.append([(cy, cx, text)])
        else:
            rows[-1].append((cy, cx, text))

    # within each row, sort by x, then join
    lines = []
    for row in rows:
        row_sorted = sorted(row, key=lambda t: t[1])
        lines.append(" ".join(t[2] for t in row_sorted))
    return lines

def render_label_png(words: List[Word], width: int = 1100, height: int = 650) -> str:
    """
    Make a clean white PNG with all text laid out as lines.
    Returns base64 PNG string.
    """
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)

    # Choose fonts (fallback to default if not found)
    try:
        big = ImageFont.truetype("arial.ttf", 36)
        med = ImageFont.truetype("arial.ttf", 28)
        small = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        big = ImageFont.load_default()
        med = ImageFont.load_default()
        small = ImageFont.load_default()

    lines = words_to_lines(words)

    y = 24
    for i, line in enumerate(lines):
        font = big if i == 0 else (med if i < 3 else small)
        d.text((24, y), line, fill=(0, 0, 0), font=font)
        # advance y using text bbox for consistent spacing
        bbox = d.textbbox((24, y), line, font=font)
        line_h = bbox[3] - bbox[1]
        y += line_h + 14
        if y > height - 40:
            break  # avoid overflow

    # encode to base64
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(
    file: UploadFile = File(...),
    return_image: bool = Query(False)
):
    try:
        data = np.frombuffer(await file.read(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Invalid image: could not be decoded.")
        H, W = bgr.shape[:2]

        out = ocr.ocr(bgr, cls=True)

        # parse paddle output
        words: List[Word] = []
        if out and isinstance(out[0], list):
            for line in out[0]:
                try:
                    poly, (text, prob) = line
                    words.append(Word(poly=poly, text=text, prob=float(prob)))
                except Exception:
                    continue

        # sort by top-left (y then x)
        # sort words by top-left for consistency
        def top_left_key(w: Word):
            xs=[p[0] for p in w.poly]; ys=[p[1] for p in w.poly]
            return (min(ys), min(xs))
        words.sort(key=top_left_key)   # ✅


        first_word = words[0] if words else None

        # NEW: merge to lines
        lines = words_to_line_groups(words)
        first_line = lines[0] if lines else None

        # Optional annotated image: draw WHOLE top line box instead of single word
        b64_anno = None
        if return_image and first_line:
            anno = bgr.copy()
            draw_line_box(anno, first_line.box, first_line.text)
            ok, buf = cv2.imencode(".jpg", anno)
            if ok:
                b64_anno = base64.b64encode(buf.tobytes()).decode("utf-8")

        # keep your label_png_b64 from all words (unchanged)
        label_png_b64 = render_label_png(words) if words else None

        return OCRResponse(
            width=W, height=H,
            words=words,
            first_word=first_word,
            lines=lines,                # NEW
            first_line=first_line,      # NEW
            annotated_image_b64=b64_anno,
            label_png_b64=label_png_b64
        )


    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Bad Request: {e}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal Error: {type(e).__name__}: {e}. Traceback: {traceback.format_exc()}"
        )


# --- Run ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
