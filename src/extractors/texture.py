import numpy as np
import cv2
from scipy.signal import convolve2d

def calculate_texture_anomaly(image_path: str) -> dict:
    """
    Applies a basic Spatial Rich Model (SRM) filter to extract the noise residual.
    High variance or sharp discontinuities in the noise residual suggest AI manipulation.
    """
    try:
        # Load grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"texture_artifacts": 50, "noise_pattern_score": 50}

        img = img.astype(np.float32)

        # Basic SRM filter (High-pass) to extract noise residual
        srm_filter = np.array([
            [-1,  2, -1],
            [ 2, -4,  2],
            [-1,  2, -1]
        ]) / 4.0

        noise_residual = convolve2d(img, srm_filter, mode='same', boundary='symm')
        
        # Calculate local variance of the noise
        local_var = cv2.GaussianBlur(noise_residual**2, (5, 5), 0) - cv2.GaussianBlur(noise_residual, (5, 5), 0)**2
        
        # Anomaly score based on the spread and inconsistency of the noise variance
        mean_var = np.mean(local_var)
        std_var = np.std(local_var)
        
        # Natural photos have a lot of noise variety. We need to be less sensitive.
        # Scale back the penalty significantly for real noise variance.
        anomaly_score = min(max((std_var / (mean_var + 1e-6)) * 0.5, 0), 100)
        
        noise_pattern_score = min(max(np.max(local_var) / (mean_var + 1e-6) * 0.1, 0), 100)
        
        return {
            "texture_artifacts": round(anomaly_score, 1),
            "noise_pattern_score": round(noise_pattern_score, 1)
        }
    except Exception as e:
        print(f"Texture extraction error: {e}")
        return {"texture_artifacts": 50, "noise_pattern_score": 50}
