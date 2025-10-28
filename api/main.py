from fastapi import FastAPI, UploadFile, File
from typing import List
from pathlib import Path
import aiofiles
import asyncio

app = FastAPI()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

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
                
            # Placeholder for calling the disease detection model.
            # Replace the following asynchronous sleep and dummy logic with a real model call.
            await asyncio.sleep(2)
            # Simulated response (dummy logic for demonstration)
            has_disease = len(file.filename) % 2 == 0  # Dummy logic for demonstration
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "file_path": str(file_path),
                "disease_detected": has_disease
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
            
    return {"results": results}
