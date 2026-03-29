import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

from src.extractors.texture import calculate_texture_anomaly
from src.extractors.frequency import calculate_frequency_anomalies
from src.extractors.reconstruction import calculate_reconstruction_similarity
from src.extractors.metadata import calculate_metadata_authenticity
from src.extractors.spatial import calculate_spatial_anomalies
from src.llm.orchestrator import ForensicsOrchestrator
from dotenv import load_dotenv

# Load env variables from the parent directory if they exist
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    global orchestrator
    orchestrator = ForensicsOrchestrator()
    yield
    # Teardown
    pass

app = FastAPI(title="V2 AI Image Forensics API", lifespan=lifespan)

@app.post("/api/v2/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyzes an image using CV extraction heuristics and Gemini 2.0 Flash orchestration.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Run CV Feature Extraction
        print("Running Spatial Analysis...")
        spatial_data = calculate_spatial_anomalies(file_path)
        
        print("Running Texture Analysis (SRM)...")
        texture_data = calculate_texture_anomaly(file_path)
        
        print("Running Frequency Analysis (FFT)...")
        frequency_data = calculate_frequency_anomalies(file_path)
        
        print("Running Reconstruction Analysis (SSIM)...")
        reconstruction_data = calculate_reconstruction_similarity(file_path)
        
        print("Running Metadata Analysis (EXIF)...")
        metadata_data = calculate_metadata_authenticity(file_path)
        
        # Aggregate raw scores
        cv_scores = {
            **spatial_data,
            **texture_data,
            **frequency_data,
            **reconstruction_data,
            **metadata_data
        }
        
        # 2. LLM Orchestration & Structuring
        print("Sending signals to LLM Orchestrator...")
        final_report = orchestrator.generate_forensic_report(file_path, cv_scores)
        
        return final_report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    # Try multiple paths to find index.html in Vercel's environment
    possible_paths = [
        os.path.join(os.getcwd(), "index.html"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "index.html"),
        "/var/task/index.html"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read()
    raise HTTPException(status_code=404, detail=f"index.html not found. Checked: {possible_paths}")

@app.get("/health")
def health_check():
    return {"status":"ok", "version":"2.0", "system":"AI Image Forensics Segmenter"}
