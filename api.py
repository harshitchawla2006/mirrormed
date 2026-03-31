from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from inference import mc_predict
import asyncio
import httpx
import logging
import time
from datetime import datetime
from collections import defaultdict

# ── logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── in-memory metrics store ───────────────────────────────────
metrics = {
    "total_predictions" : 0,
    "class_distribution": defaultdict(int),
    "avg_confidence"    : 0.0,
    "avg_uncertainty"   : 0.0,
    "total_confidence"  : 0.0,
    "total_uncertainty" : 0.0,
    "startup_time"      : datetime.utcnow().isoformat(),
    "errors"            : 0,
}

app = FastAPI(
    title       = "MirrorMed Dermatology API",
    description = "AI-powered skin lesion classification using EfficientNet-B3 with MC Dropout uncertainty estimation.",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── root ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
    <head><title>MirrorMed API</title></head>
    <body style="font-family:sans-serif;padding:2rem;background:#f0f4f8;max-width:600px;margin:auto">
        <h1 style="color:#1a365d">🔬 MirrorMed Dermatology API</h1>
        <p style="color:#4a5568">AI-powered skin lesion classification · EfficientNet-B3 · MC Dropout</p>
        <hr style="border-color:#e2e8f0">
        <p><strong style="color:green">● Status: Running</strong></p>
        <p>
            <a href="/health" style="color:#2b6cb0">Health Check</a> ·
            <a href="/metrics" style="color:#2b6cb0">Metrics</a> ·
            <a href="/docs" style="color:#2b6cb0">API Docs</a> ·
            <a href="/redoc" style="color:#2b6cb0">ReDoc</a>
        </p>
        <hr style="border-color:#e2e8f0">
        <h3 style="color:#1a365d">Endpoints</h3>
        <ul style="color:#4a5568;line-height:2">
            <li><code>GET /health</code> — health check</li>
            <li><code>GET /metrics</code> — prediction statistics</li>
            <li><code>POST /predict</code> — classify skin lesion image</li>
            <li><code>POST /predict/batch</code> — classify multiple images</li>
        </ul>
        <hr style="border-color:#e2e8f0">
        <p style="color:#718096;font-size:0.85rem">Model: EfficientNet-B3 · Dataset: HAM10000 (10,015 images) · Classes: 7</p>
    </body>
    </html>
    """

# ── health ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"      : "ok",
        "version"     : "1.0.0",
        "model"       : "efficientnet_b3",
        "dataset"     : "HAM10000",
        "num_classes" : 7,
        "timestamp"   : datetime.utcnow().isoformat()
    }

# ── metrics ───────────────────────────────────────────────────
@app.get("/metrics")
def get_metrics():
    return {
        "total_predictions" : metrics["total_predictions"],
        "class_distribution": dict(metrics["class_distribution"]),
        "avg_confidence"    : round(metrics["avg_confidence"], 4),
        "avg_uncertainty"   : round(metrics["avg_uncertainty"], 6),
        "errors"            : metrics["errors"],
        "uptime_since"      : metrics["startup_time"],
        "timestamp"         : datetime.utcnow().isoformat()
    }

# ── predict ───────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # input validation
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        metrics["errors"] += 1
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")

    image_bytes = await file.read()

    # validate file size (max 10MB)
    if len(image_bytes) > 10 * 1024 * 1024:
        metrics["errors"] += 1
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 10MB")

    start_time = time.time()

    try:
        result = mc_predict(image_bytes)
    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    elapsed = round(time.time() - start_time, 3)

    # update metrics
    metrics["total_predictions"]            += 1
    metrics["class_distribution"][result["class"]] += 1
    metrics["total_confidence"]             += result["confidence"]
    metrics["total_uncertainty"]            += result["uncertainty"]
    metrics["avg_confidence"]                = metrics["total_confidence"] / metrics["total_predictions"]
    metrics["avg_uncertainty"]               = metrics["total_uncertainty"] / metrics["total_predictions"]

    # log prediction
    logger.info(
        f"PREDICT | class={result['class']} | "
        f"confidence={result['confidence']} | "
        f"uncertainty={result['uncertainty']} | "
        f"time={elapsed}s | file={file.filename}"
    )

    return {
        **result,
        "inference_time_s" : elapsed,
        "model_version"    : "1.0.0",
        "mc_passes"        : 10
    }

# ── batch predict ─────────────────────────────────────────────
@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 images per batch")

    results = []
    for file in files:
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            results.append({"filename": file.filename, "error": "Invalid file type"})
            continue
        try:
            image_bytes = await file.read()
            result      = mc_predict(image_bytes)
            results.append({"filename": file.filename, **result})

            metrics["total_predictions"]                   += 1
            metrics["class_distribution"][result["class"]] += 1

        except Exception as e:
            metrics["errors"] += 1
            results.append({"filename": file.filename, "error": str(e)})

    logger.info(f"BATCH PREDICT | {len(files)} images processed")
    return {"results": results, "total": len(results)}

# ── keep alive ────────────────────────────────────────────────
@app.on_event("startup")
async def start_keep_alive():
    logger.info("MirrorMed API started successfully")
    asyncio.create_task(keep_alive())

async def keep_alive():
    await asyncio.sleep(60)
    while True:
        try:
            async with httpx.AsyncClient() as client:
                await client.get("https://lordaizen55-mirrormed-api.hf.space/health", timeout=10)
                logger.info("Keep-alive ping sent")
        except Exception as e:
            logger.warning(f"Keep-alive failed: {e}")
        await asyncio.sleep(600)