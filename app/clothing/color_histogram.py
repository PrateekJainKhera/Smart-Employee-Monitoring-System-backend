"""
Color histogram extractor for clothing-based person re-identification.

Extracts a normalized BGR color histogram from the body region of a
person crop (bottom 60% of bounding box — skips the face/head area).

Used only when recognition_mode != 'face_only'.
"""
import cv2
import numpy as np


def extract_clothing_histogram(person_crop: np.ndarray) -> np.ndarray | None:
    """
    Extract a normalized color histogram from the clothing region of a person crop.

    Args:
        person_crop: BGR image of the full person bounding box (from YOLO)

    Returns:
        1D numpy array (histogram vector) or None if crop is too small
    """
    if person_crop is None or person_crop.size == 0:
        return None

    h, w = person_crop.shape[:2]
    if h < 40 or w < 20:
        return None

    # Take bottom 60% of crop — skips head/face, focuses on clothing
    clothing_region = person_crop[int(h * 0.4):, :]

    # Compute histogram for each BGR channel — 32 bins each
    hist_b = cv2.calcHist([clothing_region], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([clothing_region], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([clothing_region], [2], None, [32], [0, 256])

    # Normalize each channel
    cv2.normalize(hist_b, hist_b)
    cv2.normalize(hist_g, hist_g)
    cv2.normalize(hist_r, hist_r)

    # Concatenate into single 96-dim vector
    hist = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
    return hist


def histogram_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compute similarity between two histograms using Bhattacharyya distance.
    Returns value in [0, 1] where 1 = identical, 0 = completely different.
    """
    if hist1 is None or hist2 is None:
        return 0.0
    # Bhattacharyya: 0 = identical, 1 = completely different — we invert it
    dist = cv2.compareHist(
        hist1.astype(np.float32),
        hist2.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA
    )
    return float(1.0 - dist)
