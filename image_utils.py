import numpy as np
import os
import cv2
import tifffile
from PIL import Image

def to_8bit(img):
    """Convert image to 8-bit uint8."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.copy()
    imin, imax = float(img.min()), float(img.max())
    if imax > imin:
        return ((255.0 * (img - imin) / (imax - imin)).clip(0, 255)).astype(np.uint8)
    return np.zeros(img.shape, dtype=np.uint8)

def apply_clahe(img8, clipLimit=2.0, tileGridSize=(8, 8)):
    """Apply CLAHE for contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=float(clipLimit), tileGridSize=tileGridSize)
    return clahe.apply(img8)

def load_image(path):
    """Load image from various formats."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(('.npy', '.npz')):
        data = np.load(path)
        return _normalize_to_u8(data)
    if path.lower().endswith(('.tif', '.tiff')):
        return _normalize_to_u8(tifffile.imread(path))
    # PIL fallback
    pil = Image.open(path)
    if pil.mode != 'L':
        pil = pil.convert('L')
    return np.array(pil, dtype=np.uint8)

def _normalize_to_u8(img):
    """Internal normalization."""
    img = np.asarray(img)
    if img.ndim > 2:
        img = img[..., 0] if img.shape[-1] > 1 else img
    return to_8bit(img)