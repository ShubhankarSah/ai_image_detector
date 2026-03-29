import os
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager

app = FastAPI(title="V2 AI Image Forensics API - Diagnostic Mode")

@app.get("/api/health")
def handler(request):
    return {
        "statusCode": 200,
        "body": "Working!"
    }
def health_check():
    return {
        "status": "ok",
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "files_in_api": os.listdir(os.path.join(os.getcwd(), "api")) if os.path.exists("api") else "None",
        "files_in_root": os.listdir(os.getcwd())
    }

@app.get("/api/test-imports")
def test_imports():
    results = {}
    
    # Core Libraries
    try:
        import numpy
        results["numpy"] = numpy.__version__
    except Exception as e:
        results["numpy"] = f"FAILED: {str(e)}"
        
    try:
        import PIL
        results["PIL"] = "Loaded"
    except Exception as e:
        results["PIL"] = f"FAILED: {str(e)}"

    # Problematic CV2
    try:
        import cv2
        results["cv2"] = cv2.__version__
    except Exception as e:
        results["cv2"] = f"FAILED: {str(e)}"

    # Local Imports
    try:
        from .src.extractors.texture import calculate_texture_anomaly
        results["local_texture"] = "Loaded"
    except Exception as e:
        results["local_texture"] = f"FAILED: {str(e)}"

    return results

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    return {
        "verdict": "Diagnostic Mode Active",
        "message": "AI engines are temporarily disabled for stability testing.",
        "filename": file.filename
    }
