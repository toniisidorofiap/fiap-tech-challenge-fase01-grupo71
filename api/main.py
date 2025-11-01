from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from typing import List
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
import aiofiles

app = FastAPI()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

app = FastAPI()

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    model_path = Path(__file__).parent / "model" / "MobileNetV2.h5"
    ml_models["MobileNetV2"] = tf.keras.models.load_model(str(model_path))
    yield
    # Clean up the ML models and release the resources
    ml_models.clear


@app.post("/analyze-images")
async def analyze_images(files: List[UploadFile] = File(...)):
    results = []
    
    for file in files:
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": "Invalid file type. Only PNG, JPG and JPEG are allowed"
            })
            continue
            
        # Save the file
        file_path = UPLOAD_DIR / file.filename
        try:
            content = await file.read()
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)

            # Load image, preprocess and predict with the model
            image = Image.open(file_path).convert('RGB')
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)  # batch dimension

            model = ml_models["MobileNetV2"]
            prediction = model.predict(image_array)
            probability = float(prediction[0][0])
            has_disease = probability >= 0.91

            results.append({
                "filename": file.filename,
                "status": "success",
                "file_path": str(file_path),
                "disease_detected": has_disease,
                "probability": probability
            })
            
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
            
    return {"results": results}
