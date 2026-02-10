import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from ultralytics import YOLO
from PIL import Image


# =========================
# Paths (your structure)
# =========================
BASE_DIR = Path(__file__).resolve().parent          # .../python
MODEL_PATH = BASE_DIR / "model" / "best.pt"         # .../python/model/best.pt
TEMP_DIR = BASE_DIR / "temp_upload"                 # .../python/temp_upload
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Load model ONCE
# =========================
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

app = FastAPI(title="Seaweed Classifier API")

# =========================
# CORS (allow frontend calls)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev only; restrict later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Helpers
# =========================
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

async def _save_upload(file: UploadFile) -> Path:
    ext = Path(file.filename).suffix.lower()

    if file.content_type and file.content_type not in ALLOWED_MIME:
        raise ValueError(f"Unsupported file type: {file.content_type}")

    if ext not in ALLOWED_EXT:
        ext = ".jpg"

    out_path = TEMP_DIR / f"{uuid.uuid4().hex}{ext}"
    content = await file.read()

    with open(out_path, "wb") as f:
        f.write(content)

    return out_path

def _predict_image(img_path: Path) -> Dict[str, Any]:
    with Image.open(img_path).convert("RGB") as im:
        results = model.predict(im, verbose=False)

    r0 = results[0]
    if r0.probs is None:
        return {"label": "Unknown", "confidence": 0.0, "top5": []}

    probs = r0.probs
    names = model.names

    def _name(cid: int) -> str:
        return names[cid] if isinstance(names, (list, tuple)) else names.get(cid, str(cid))

    top1_id = int(probs.top1)
    top1_conf = float(probs.top1conf)

    top5_ids = probs.top5
    top5_confs = probs.top5conf

    top5 = []
    for i in range(min(5, len(top5_ids))):
        cid = int(top5_ids[i])
        cconf = float(top5_confs[i])
        top5.append({"label": _name(cid), "confidence": cconf})

    return {"label": _name(top1_id), "confidence": top1_conf, "top5": top5}

# =========================
# Routes
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>Seaweed Classifier API is running ✅</h2>
    <ul>
      <li><a href="/api/health">/api/health</a></li>
      <li><a href="/docs">/docs</a> (Swagger UI)</li>
      <li><a href="/redoc">/redoc</a></li>
    </ul>
    """

# Render sometimes pings with HEAD /
@app.head("/")
def home_head():
    return HTMLResponse("")

@app.get("/api/health")
def health():
    return {"status": "ok", "model": str(MODEL_PATH)}

@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    saved = None
    try:
        saved = await _save_upload(image)
        pred = _predict_image(saved)
        return JSONResponse(pred)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if saved and saved.exists():
            try:
                saved.unlink()
            except Exception:
                pass

@app.post("/api/predict_top5")
async def predict_top5(image: UploadFile = File(...)):
    saved = None
    try:
        saved = await _save_upload(image)
        pred = _predict_image(saved)
        return JSONResponse({"top5": pred.get("top5", [])})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if saved and saved.exists():
            try:
                saved.unlink()
            except Exception:
                pass


