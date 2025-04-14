import os
import logging
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import uvicorn
from io import BytesIO
from functools import lru_cache

from app import LeffaPredictor  # This imports your existing class from app.py

# ------------------------- Setup -------------------------
app = FastAPI(title="Leffa Inference API")

# CORS middleware (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local testing, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------- Lazy Predictor ---------------------
@lru_cache()
def get_predictor():
    if not os.path.exists("./ckpts"):
        raise FileNotFoundError("Checkpoints directory not mounted. Please mount the volume containing ckpts")
    logger.info("Initializing LeffaPredictor...")
    return LeffaPredictor()

# --------------------- Utility ---------------------
def load_image_from_upload(upload_file: UploadFile):
    return Image.open(BytesIO(upload_file.file.read()))

def save_temp_image(image: Image.Image, path: str):
    image.save(path)

# -------------------- API Endpoints --------------------
@app.post("/predict/virtual_tryon")
async def predict_virtual_tryon(
    src_image: UploadFile = File(...),
    ref_image: UploadFile = File(...),
    vt_model_type: str = Form("viton_hd"),
    vt_garment_type: str = Form("upper_body"),
    step: int = Form(50),
    scale: float = Form(2.5),
    seed: int = Form(42),
    vt_repaint: bool = Form(False),
    ref_acceleration: bool = Form(False),
    preprocess_garment: bool = Form(False),
):
    try:
        logger.info("Receiving images...")
        src_img = load_image_from_upload(src_image)
        ref_img = load_image_from_upload(ref_image)

        # Save to temporary disk paths
        os.makedirs("temp", exist_ok=True)
        src_path = f"temp/{src_image.filename}"
        ref_path = f"temp/{ref_image.filename}"
        save_temp_image(src_img, src_path)
        save_temp_image(ref_img, ref_path)

        logger.info("Calling predictor...")
        predictor = get_predictor()

        result, mask, dense = predictor.leffa_predict_vt(
            src_path,
            ref_path,
            ref_acceleration,
            step,
            scale,
            seed,
            vt_model_type,
            vt_garment_type,
            vt_repaint,
            preprocess_garment,
        )

        logger.info("Inference complete.")

        # Save the result to disk (optional)
        result_path = "temp/result.jpg"
        Image.fromarray(result).save(result_path)

        return JSONResponse({
            "status": "success",
            "result_path": result_path,
        })

    except Exception as e:
        logger.exception("Error during inference")
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------- Server Start --------------------
if __name__ == "__main__":
    uvicorn.run("leffa_api:app", host="0.0.0.0", port=8000, reload=True)
