import cv2
import numpy as np

def calculate_spatial_anomalies(image_path: str) -> dict:
    """
    Basic mathematical edge/contrast inconsistency check.
    We rely heavily on the LLM for deep spatial reasoning (anatomy/lighting),
    but this provides a baseline CV metric for unnatural edge gradients.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"anomalies_detected": False, "anomaly_score": 50, "lighting_consistency": 50}
            
        # Calculate Laplacian variance (focus score / edge strength)
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        
        anomalies_detected = False
        anomaly_score = 50
        lighting_consistency = 50
        
        # Extremely high or low variance can be suspicious. 
        # But this is a very weak signal on its own.
        if laplacian_var > 5000:
            anomaly_score = 70
            anomalies_detected = True
        elif laplacian_var < 50:
            anomaly_score = 65
            anomalies_detected = True
        else:
            anomaly_score = 20
            lighting_consistency = 80 # Just a placeholder metric based on overall spread
            
        return {
            "anomalies_detected": anomalies_detected,
            "anomaly_score": round(anomaly_score, 1),
            "lighting_consistency": round(lighting_consistency, 1)
        }
            
    except Exception as e:
        print(f"Spatial extraction error: {e}")
        return {"anomalies_detected": False, "anomaly_score": 50, "lighting_consistency": 50}
