"""Dermoscopy-specific classical CV preprocessing: hair removal + color
normalization. Optional (config.use_dermoscopy_preprocessing), applied to a
PIL image before the standard torchvision transform pipeline.

Hair removal: DullRazor-style — blackhat morphological filter to isolate
dark hair-like structures against skin, threshold, inpaint over them.
Color normalization: gray-world white balance — scales each RGB channel so
its mean matches the overall gray-level mean, correcting for the varied
lighting/color casts across dermoscopy captures.
"""
import cv2
import numpy as np
from PIL import Image


def remove_hair(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(bgr, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)


def gray_world_normalize(bgr: np.ndarray) -> np.ndarray:
    bgr = bgr.astype(np.float32)
    channel_means = bgr.reshape(-1, 3).mean(axis=0)
    overall_mean = channel_means.mean()
    scale = overall_mean / np.maximum(channel_means, 1e-6)
    normalized = bgr * scale
    return np.clip(normalized, 0, 255).astype(np.uint8)


def apply_dermoscopy_preprocessing(image: Image.Image) -> Image.Image:
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    bgr = remove_hair(bgr)
    bgr = gray_world_normalize(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
