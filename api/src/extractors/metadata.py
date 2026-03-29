from PIL import Image
from PIL.ExifTags import TAGS

def calculate_metadata_authenticity(image_path: str) -> dict:
    """
    Extracts EXIF metadata.
    A complete lack of metadata is slightly suspicious (weak signal).
    Presence of known generative software tags is a definitive signal.
    """
    result = {
        "metadata_present": False,
        "camera_signature_score": 50  # Default neutral
    }
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        if exif_data:
            result["metadata_present"] = True
            
            suspect_keywords = ["midjourney", "dall-e", "stable diffusion", "ai", "generated", "adobe photoshop"]
            found_suspect = False
            
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if isinstance(value, str):
                    val_lower = value.lower()
                    if any(kw in val_lower for kw in suspect_keywords):
                        found_suspect = True
                        break
            
            if found_suspect:
                # Strong signal if explicitly stating generating software
                result["camera_signature_score"] = 95
            else:
                # Good sign if metadata exists and has no suspect keywords
                result["camera_signature_score"] = 10
        else:
            # Metadata missing: Weak signal (not decisive)
            # 15 means low anomaly (typical internet image, likely real/neutral)
            result["camera_signature_score"] = 15
            
    except Exception as e:
        print(f"Metadata extraction error: {e}")
        
    return result
