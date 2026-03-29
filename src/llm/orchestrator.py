import os
import json
from google import genai
from google.genai import types

class ForensicsOrchestrator:
    def __init__(self):
        # The user's system likely uses google-genai 0.3.0 based on new docs
        # We will use the standard initialized client if API key is in environment
        # or configure it manually.
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
             print("WARNING: GEMINI_API_KEY not found in environment")
        self.client = genai.Client(api_key=api_key)
        self.model_id = 'models/gemini-2.5-flash'

    def generate_forensic_report(self, image_path: str, cv_scores: dict) -> dict:
        """
        Combines the raw CV extraction scores with Gemini's visual analysis 
        to produce the final JSON verdict based strictly on user guidelines.
        """
        prompt = f"""
You are an AI image forensic analyst.

Your job is to evaluate whether an image is AI-generated or real based on multiple forensic signals provided below, as well as your own visual analysis of the attached image.

You must:
- Analyze each factor independently
- Assign a score (0–100) for each factor
- Provide reasoning for each score
- Combine all signals into a final probability
- Output a structured JSON verdict

Weightage:
- CRITICAL Importance (70% weight): Your own Multimodal Visual Inspection (look for AI hallmarks like perfect symmetry, synthetic skin textures, dream-like backgrounds, or structural absurdities in the image itself).
- Medium Importance (30% weight): The provided CV mathematical anomaly scores. (Only rely on them if they show extremely high anomalies).

RULES:
- You are an expert AI multimodal forensic model. You MUST closely inspect the image visually.
- Do NOT blindly trust the mathematical CV scores if your visual inspection strongly contradicts them (e.g., if you see a classic AI-generated face, override low CV anomaly scores and declare it AI).
- Do NOT output 'Uncertain' unless the physical metrics are perfectly split 50/50. If the majority of anomaly scores (especially Recon/Frequency) are < 40, output 'Real'. If the majority are > 60, output 'AI Generated'.
- Final output must reflect decisive confidence if signals align.

Image Analysis Data (Computed from raw CV Extractor tools):
(CRITICAL NOTE: All mathematical scores below are ANOMALY SCORES. 0 = Definitely Real, 100 = Definitely AI. Do not invert the logic!)

Spatial Analysis:
- anomalies_detected: {cv_scores.get('anomalies_detected')}
- anomaly_score: {cv_scores.get('anomaly_score')}

Lighting Analysis:
- lighting_consistency: {cv_scores.get('lighting_consistency')}

Texture Analysis:
- noise_pattern_score: {cv_scores.get('noise_pattern_score')}
- texture_artifacts: {cv_scores.get('texture_artifacts')}

Text & Detail Analysis:
- text_accuracy: 50 (Please analyze the image visually for text rendering errors)

Metadata Analysis:
- metadata_present: {cv_scores.get('metadata_present')}
- camera_signature_score: {cv_scores.get('camera_signature_score')}

Frequency Analysis:
- spectral_anomaly_score: {cv_scores.get('spectral_anomaly_score')}

Reconstruction Analysis:
- reconstruction_similarity_score: {cv_scores.get('reconstruction_similarity_score')}

Contextual Analysis:
- logical_consistency_score: 50 (Please analyze the image visually for impossible geometry)

Scoring Guidelines:
- Spatial inconsistencies > 70 -> strong AI signal
- Frequency anomalies > 65 -> strong AI signal
- Reconstruction similarity anomaly > 75 -> strong AI signal
- Missing metadata -> weak AI signal (not decisive)

Weightage:
- Frequency + Reconstruction = HIGH importance
- Spatial + Texture = MEDIUM
- Metadata + Text = LOW

Final Probability Calculation:
- Combine weighted scores
- Normalize to 0–100

Verdict Mapping:
- 0–20 -> Real
- 20–40 -> Likely Real
- 40–60 -> Uncertain
- 60–80 -> Likely AI
- 80–100 -> AI Generated

Generate output strictly in the following JSON format:
{{
  "verdict": "Real / Likely Real / Uncertain / Likely AI / AI Generated",
  "confidence": <float 0-100>,
  "forgery_probability": <float 0-100 based on mapping>,
  "strength_of_evidence": "Weak / Moderate / Strong",
  "factor_scores": {{
    "spatial_inconsistency": <float 0-100>,
    "lighting_mismatch": <float 0-100>,
    "texture_anomaly": <float 0-100>,
    "text_errors": <float 0-100>,
    "metadata_authenticity": <float 0-100>,
    "frequency_artifacts": <float 0-100>,
    "reconstruction_behavior": <float 0-100>,
    "contextual_logic": <float 0-100>
  }},
  "reasoning": {{
    "spatial": "string",
    "lighting": "string",
    "texture": "string",
    "text": "string",
    "metadata": "string",
    "frequency": "string",
    "reconstruction": "string",
    "context": "string"
  }},
  "final_explanation": "Explain in 2-3 lines why this image is likely real or AI generated."
}}

Respond ONLY with valid JSON. Do not use markdown wrappers.
        """
        
        from PIL import Image
        import time
        
        try:
            img = Image.open(image_path)
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Using the new google-genai SDK format
                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=[prompt, img],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        )
                    )
                    break # Success
                except Exception as e:
                    err_str = str(e)
                    if attempt < max_retries - 1 and ("400" not in err_str):
                        print(f"API Error ({err_str}). Retrying in 10s... (Attempt {attempt+1})")
                        time.sleep(10)
                    else:
                        raise e
            
            # Parse the strict JSON response
            text = response.text.strip()
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            
            return json.loads(text.strip())
            
        except Exception as e:
            print(f"Orchestration Error: {e}")
            return {"error": str(e), "message": "Failed to generate forensic report."}
