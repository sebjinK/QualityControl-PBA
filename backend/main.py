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
import os
from pydantic import BaseModel
from difflib import SequenceMatcher

# --- Config ---
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.expanduser("~\\Downloads\\QualitiyControl\\QualityControl-PBA\\backend\\square_regressor.pt")

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

# --- models ---


class CompareResponse(BaseModel):
    same: bool
    reason: str
    scores: dict
    text1: str
    text2: str

# --- helpers ---
def read_bgr(upload: UploadFile):
    data = np.frombuffer(upload.file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Invalid image: {upload.filename}")
    return img

def phash64(gray):
    """Perceptual hash (64-bit) via DCT; returns a 64-length boolean array."""
    g = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    g = np.float32(g)
    dct = cv2.dct(g)      # 32x32
    dct_low = dct[:8, :8] # keep top-left 8x8
    med = np.median(dct_low[1:, 1:])  # ignore DC term
    bits = (dct_low > med).flatten()
    return bits

def hamming(a, b):
    return int(np.count_nonzero(a ^ b))

def ssim_gray(a, b):
    """Simple SSIM for grayscale images of the same size."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

def ocr_text_from_bgr(bgr):
    out = ocr.ocr(bgr, cls=True)
    chunks = []
    if out and isinstance(out[0], list):
        for line in out[0]:
            try:
                _, (text, prob) = line
                t = (text or "").strip()
                if t:
                    chunks.append(t)
            except Exception:
                continue
    return " ".join(chunks)

# --- endpoint ---
@app.post("/compare", response_model=CompareResponse)
async def compare(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    text_weight: float = Query(0.5, ge=0.0, le=1.0, description="weight for OCR text similarity in final decision")
):
    try:
        # read both
        bgr1 = read_bgr(file1)
        bgr2 = read_bgr(file2)

        # resize to common size for SSIM / hist (keep aspect roughly)
        target = (512, 512)
        g1 = cv2.cvtColor(cv2.resize(bgr1, target, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(bgr2, target, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

        # pHash
        h1 = phash64(g1)
        h2 = phash64(g2)
        phash_dist = hamming(h1, h2)      # 0..64 (lower = more similar)

        # SSIM
        ssim_val = ssim_gray(g1, g2)      # -1..1 (1 = identical; typically 0..1)

        # Color histogram similarity (correlation)
        hsv1 = cv2.cvtColor(cv2.resize(bgr1, target), cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(cv2.resize(bgr2, target), cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([hsv1], [0,1], None, [32,32], [0,180, 0,256])
        hist2 = cv2.calcHist([hsv2], [0,1], None, [32,32], [0,180, 0,256])
        cv2.normalize(hist1, hist1); cv2.normalize(hist2, hist2)
        hist_corr = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))  # -1..1

        # OCR text similarity
        txt1 = ocr_text_from_bgr(bgr1)
        txt2 = ocr_text_from_bgr(bgr2)
        text_sim = float(SequenceMatcher(None, txt1, txt2).ratio())  # 0..1

        # Decision heuristic (tweak to your data)
        # Consider "same" if both image structure and text match well.
        # pHash <= 10 and SSIM >= 0.85 usually means visually the same.
        img_ok = (phash_dist <= 10 and ssim_val >= 0.85) or (ssim_val >= 0.95)
        # blend image vs text evidence
        blended = (1 - text_weight) * max(0.0, (1 - phash_dist/64.0)*0.6 + (ssim_val)*0.4) + text_weight * text_sim
        same = bool(img_ok or blended >= 0.85)

        reason = (
            f"pHashDist={phash_dist:.0f} (<=10 good), "
            f"SSIM={ssim_val:.3f}, HistCorr={hist_corr:.3f}, "
            f"TextSim={text_sim:.3f}, Score={blended:.3f}"
        )

        return CompareResponse(
            same=same,
            reason=reason,
            scores={
                "phash_distance": phash_dist,
                "ssim": ssim_val,
                "hist_correlation": hist_corr,
                "text_similarity": text_sim,
                "blended_score": blended,
            },
            text1=txt1,
            text2=txt2,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Compare failed: {type(e).__name__}: {e}")



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
