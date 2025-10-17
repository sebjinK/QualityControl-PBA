# backend/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import uvicorn

# --- Config ---
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "~/QualityControl-PBA/backend/square_regressor.pt"

app = FastAPI()

# Allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model definition (copied from your code) ---
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

# --- Input/output schema ---
class PredictionResponse(BaseModel):
    predicted_side_cm: float
    confidence: float = 1.0

# --- Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Load image
    img_bytes = await file.read()
    pil = Image.open(BytesIO(img_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    # Convert to tensor
    x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        y_pred = model(x).cpu().numpy().squeeze().item()

    return PredictionResponse(predicted_side_cm=round(y_pred, 3))

# --- Run ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
