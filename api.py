from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import mc_predict
import asyncio
import httpx

app = FastAPI(title="MirrorMed Dermatology API")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

@app.get("/")
def root():
    return {"message": "MirrorMed API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported"
        )
    image_bytes = await file.read()
    result      = mc_predict(image_bytes)
    return result

# ── keep alive — pings itself every 10 minutes ──────────────
@app.on_event("startup")
async def start_keep_alive():
    asyncio.create_task(keep_alive())

async def keep_alive():
    await asyncio.sleep(60)
    while True:
        try:
            async with httpx.AsyncClient() as client:
                await client.get("https://mirrormed-backend.onrender.com/health", timeout=10)
                print("Keep-alive ping sent")
        except Exception as e:
            print(f"Keep-alive failed: {e}")
        await asyncio.sleep(600)  # ping every 10 minutes