import numpy as np
import cv2

def calculate_frequency_anomalies(image_path: str) -> dict:
    """
    Analyzes the 2D FFT magnitude spectrum to find unnatural periodic high-frequency peaks.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"spectral_anomaly_score": 50}
            
        # Compute 2D FFT
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        H, W = magnitude_spectrum.shape
        cy, cx = H // 2, W // 2
        
        # Ignore low frequencies (center)
        y, x = np.ogrid[-cy:H-cy, -cx:W-cx]
        mask = x*x + y*y > (min(H, W) * 0.15)**2
        
        high_freq_vals = magnitude_spectrum[mask]
        
        if len(high_freq_vals) == 0:
            return {"spectral_anomaly_score": 0}
            
        max_val = np.max(high_freq_vals)
        mean_val = np.mean(high_freq_vals)
        std_val = np.std(high_freq_vals)
        
        # Number of std deviations the maximum peak is away from the mean high-freq energy
        peakiness = (max_val - mean_val) / (std_val + 1e-8)
        
        # Scale peakiness to 0-100 (Peakiness > 5 is very suspicious)
        score = min(max((peakiness - 2.5) * 20, 0), 100)
        
        return {
            "spectral_anomaly_score": round(score, 1)
        }
    except Exception as e:
        print(f"Frequency extraction error: {e}")
        return {"spectral_anomaly_score": 50}
