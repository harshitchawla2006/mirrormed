from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <body style="font-family:sans-serif;padding:2rem;background:#f0f4f8">
            <h1>🔬 MirrorMed API</h1>
            <p>Status: <strong style="color:green">Running ✓</strong></p>
            <p><a href="/health">Health Check</a> · <a href="/docs">API Docs</a></p>
        </body>
    </html>
    """

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

@app.on_event("startup")
async def start_keep_alive():
    asyncio.create_task(keep_alive())

async def keep_alive():
    await asyncio.sleep(60)
    while True:
        try:
            async with httpx.AsyncClient() as client:
                await client.get("https://lordaizen55-mirrormed-api.hf.space/health", timeout=10)
                print("Keep-alive ping sent")
        except Exception as e:
            print(f"Keep-alive failed: {e}")
        await asyncio.sleep(600)