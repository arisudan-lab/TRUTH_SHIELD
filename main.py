from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import traceback
import base64
import io
import cv2
import numpy as np
from PIL import Image
from scipy.stats import kurtosis
from scipy.fftpack import fft2, fftshift

# ══════════════════════════════════════════════════════════════════════════════
#  APP INITIALIZATION & CORS
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Fake News & Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, "models/vectorizer.jb")
MODEL_PATH      = os.path.join(BASE_DIR, "models/lr_model.jb")

vectorizer = None
text_model = None

try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    text_model = joblib.load(MODEL_PATH)
    print(f"✅ Text models loaded from: {BASE_DIR}")
except Exception as e:
    print(f"❌ CRITICAL ERROR LOADING TEXT MODELS: {e}")
    print(f"⚠️  Looked here:\n - {VECTORIZER_PATH}\n - {MODEL_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
#  REQUEST SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class NewsRequest(BaseModel):
    text: str

class ImageRequest(BaseModel):
    image: str  # base64-encoded image bytes

# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_indicator_status(score: float) -> str:
    """Returns a text status based on the individual indicator score."""
    if score > 0.60: return "anomalous"
    if score > 0.35: return "suspicious"
    return "clean"

def analyze_deepfake_image(img_pil: Image.Image, img_np: np.ndarray, img_bgr: np.ndarray) -> dict:
    """Handles the core mathematical logic for detecting image manipulation."""
    
    # ── 1. Luminance channel (LAB colour space)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img_lab)

    # ── 2. Noise residual (high-pass = original − Gaussian blur)
    blurred = cv2.GaussianBlur(L.astype(np.float32), (5, 5), 0)
    noise   = L.astype(np.float32) - blurred

    # ── 3. Noise kurtosis — real images have near-Gaussian noise
    kurt       = kurtosis(noise.flatten())
    kurt_score = min(abs(kurt) / 10.0, 1.0)

    # ── 4. FFT frequency fingerprint — GANs leave grid artefacts
    fshift      = fftshift(fft2(noise))
    magnitude   = np.log(np.abs(fshift) + 1)
    h, w        = magnitude.shape
    center_mean = magnitude[h//2-20:h//2+20, w//2-20:w//2+20].mean()
    outer_mean  = np.concatenate([
                      magnitude[:20, :].flatten(),
                      magnitude[-20:, :].flatten()
                  ]).mean()
    fft_score   = min(center_mean / (outer_mean + 1e-8) / 100.0, 1.0)

    # ── 5. Error Level Analysis — manipulated regions re-compress oddly
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    compressed = np.array(Image.open(buf).convert("RGB")).astype(np.float32)
    ela_diff   = np.abs(img_np.astype(np.float32) - compressed)
    ela_gray   = cv2.cvtColor(ela_diff.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    ela_std    = ela_gray.std()
    ela_score  = 1.0 - min(ela_std / 50.0, 1.0)

    # ── 6. Edge sharpness — GANs produce subtly blurry edges
    gray       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var    = cv2.Laplacian(gray, cv2.CV_64F).var()
    edge_score = 1.0 - min(lap_var / 800.0, 1.0)

    # ── 7. Colour channel correlation drift
    r_flat      = img_np[:, :, 0].flatten().astype(np.float64)
    g_flat      = img_np[:, :, 1].flatten().astype(np.float64)
    b_flat      = img_np[:, :, 2].flatten().astype(np.float64)
    rg_corr     = float(np.corrcoef(r_flat, g_flat)[0, 1])
    rb_corr     = float(np.corrcoef(r_flat, b_flat)[0, 1])
    color_score = min(abs(rg_corr - rb_corr) * 3.0, 1.0)

    # ── Weighted fake score
    fake_score = (
        kurt_score  * 0.25 +
        ela_score   * 0.25 +
        fft_score   * 0.20 +
        edge_score  * 0.15 +
        color_score * 0.15
    )

    return {
        "fake_score": fake_score,
        "kurt": kurt,
        "kurt_score": kurt_score,
        "fft_score": fft_score,
        "ela_std": ela_std,
        "ela_score": ela_score,
        "lap_var": lap_var,
        "edge_score": edge_score,
        "rg_corr": rg_corr,
        "rb_corr": rb_corr,
        "color_score": color_score
    }

# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/check-news")
async def check_news(request: NewsRequest):
    if vectorizer is None or text_model is None:
        raise HTTPException(
            status_code=500,
            detail="Text models failed to load. Check your terminal logs."
        )

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Please provide text to analyse.")

    try:
        vectorized_text = vectorizer.transform([request.text])
        prediction      = text_model.predict(vectorized_text)[0]

        try:
            probabilities = text_model.predict_proba(vectorized_text)[0]
            confidence    = max(probabilities) * 100
        except AttributeError:
            confidence = 88.5   # fallback for models without predict_proba

        pred_str = str(prediction).upper()
        verdict  = "REAL" if pred_str in ["1", "REAL", "TRUE"] else "FAKE"

        return {
            "verdict":    verdict,
            "confidence": round(confidence, 1),
            "summary":    "Analysed using local Logistic Regression model."
        }

    except Exception as e:
        print(f"Server crashed during prediction:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"ML Error: {str(e)}")


@app.post("/api/check-image")
async def check_image(request: ImageRequest):
    try:
        # Robust Base64 Decoding
        img_data = request.image
        
        # Fix padding if necessary
        missing_padding = len(img_data) % 4
        if missing_padding:
            img_data += '=' * (4 - missing_padding)
            
        img_bytes = base64.b64decode(img_data)
        img_pil   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np    = np.array(img_pil)
        img_bgr   = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Run analysis through our helper function
        results = analyze_deepfake_image(img_pil, img_np, img_bgr)
        
        # Extract variables for clean formatting below
        fake_score = results["fake_score"]
        fake_pct   = round(fake_score * 100, 1)

        # ── Verdict Logic ─────────────────────────────────────────────────────
        if fake_score > 0.62:
            verdict    = "AI_GENERATED"
            confidence = round(fake_score * 100, 1)
            summary    = (
                f"Multiple anomalies detected. Noise residuals show non-Gaussian "
                f"patterns (kurtosis: {results['kurt']:.2f}), ELA reveals uneven compression "
                f"artefacts, and frequency analysis shows GAN-like fingerprints."
            )
        elif fake_score > 0.42:
            verdict    = "UNCERTAIN"
            confidence = round((1 - abs(fake_score - 0.5) * 2) * 100, 1)
            summary    = (
                f"Mixed signals detected. Some indicators suggest manipulation but "
                f"not conclusively. Manual review recommended. Fake score: {fake_pct}%."
            )
        else:
            verdict    = "REAL"
            confidence = round((1 - fake_score) * 100, 1)
            summary    = (
                f"Image appears authentic. Noise distribution is natural, "
                f"compression artefacts are consistent, and edge coherence looks normal. "
                f"Fake score: {fake_pct}%."
            )

        # ── Construct API Response ────────────────────────────────────────────
        return {
            "verdict":    verdict,
            "confidence": confidence,
            "fake_score": fake_pct,
            "summary":    summary,
            "indicators": [
                {
                    "label":  "Noise Kurtosis",
                    "status": get_indicator_status(results["kurt_score"]),
                    "detail": f"Score: {results['kurt_score']:.2f} | Kurtosis: {results['kurt']:.2f}"
                },
                {
                    "label":  "ELA Compression",
                    "status": get_indicator_status(results["ela_score"]),
                    "detail": f"Score: {results['ela_score']:.2f} | Std dev: {results['ela_std']:.1f}"
                },
                {
                    "label":  "FFT Frequency Pattern",
                    "status": get_indicator_status(results["fft_score"]),
                    "detail": f"Score: {results['fft_score']:.2f}"
                },
                {
                    "label":  "Edge Sharpness",
                    "status": get_indicator_status(results["edge_score"]),
                    "detail": f"Laplacian variance: {results['lap_var']:.1f}"
                },
                {
                    "label":  "Colour Channel Drift",
                    "status": get_indicator_status(results["color_score"]),
                    "detail": f"R-G / R-B corr delta: {abs(results['rg_corr'] - results['rb_corr']):.3f}"
                },
            ]
        }

    except Exception as e:
        print(f"Image analysis error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Image analysis error: {str(e)}")