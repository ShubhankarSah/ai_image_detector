import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.extractors.texture import calculate_texture_anomaly
from src.extractors.frequency import calculate_frequency_anomalies
from src.extractors.reconstruction import calculate_reconstruction_similarity
from src.extractors.metadata import calculate_metadata_authenticity
from src.extractors.spatial import calculate_spatial_anomalies
from src.llm.orchestrator import ForensicsOrchestrator
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    orchestrator = ForensicsOrchestrator()
    yield

app = FastAPI(title="V2 AI Image Forensics API", lifespan=lifespan)

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        cv_scores = {
            **calculate_spatial_anomalies(file_path),
            **calculate_texture_anomaly(file_path),
            **calculate_frequency_anomalies(file_path),
            **calculate_reconstruction_similarity(file_path),
            **calculate_metadata_authenticity(file_path)
        }
        
        return orchestrator.generate_forensic_report(file_path, cv_scores)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/api/health")
def health_check():
    return {"status":"ok", "system":"AI Image Forensics Segmenter"}
