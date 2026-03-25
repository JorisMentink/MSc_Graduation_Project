import numpy as np
from PIL import Image

def normalize_mri_to_uint8(volume: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    vol = volume.astype(np.float32)
    lo, hi = np.percentile(vol, (p_low, p_high))
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-8)
    vol = (vol * 255.0).astype(np.uint8)
    return vol


def resize_grayscale_stack_to_rgb(volume_uint8: np.ndarray, image_size: int) -> np.ndarray:
    d, h, w = volume_uint8.shape
    out = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d):
        img_pil = Image.fromarray(volume_uint8[i])
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        arr = np.asarray(img_resized, dtype=np.float32) / 255.0
        out[i] = arr.transpose(2, 0, 1)
    return out