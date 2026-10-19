"""Shared thread-geometry extraction used by both training and the app."""
import cv2
import numpy as np
from scipy.signal import find_peaks

MIN_COLUMNS = 80      # minimum image columns the thread must span
PEAK_DISTANCE = 80    # min px between crests/roots (tune to your image size)
PEAK_PROMINENCE = 8   # min peak prominence in px


def extract_thread_features(img):
    """Return a dict of thread geometry from a BGR image, or None on failure."""
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.createCLAHE(2.0, (8, 8)).apply(gray)

    edges = cv2.Canny(img_eq, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=len).reshape(-1, 2)
    x_pts, y_pts = contour[:, 0], contour[:, 1]

    x = np.sort(np.unique(x_pts))
    if len(x) < MIN_COLUMNS:
        return None

    y_top = np.array([y_pts[x_pts == xi].min() for xi in x])
    y_bottom = np.array([y_pts[x_pts == xi].max() for xi in x])

    profile = (y_top + y_bottom) / 2
    profile = profile - profile.mean()

    crests, _ = find_peaks(profile, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    roots, _ = find_peaks(-profile, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    if len(crests) < 3 or len(roots) < 3:
        return None

    pitch_px = float(np.median(np.diff(x[crests])))
    depth_px = float(np.median(y_bottom[roots]) - np.median(y_top[crests]))

    return {
        "features": [pitch_px, depth_px, 0.0, 0.0],
        "x": x, "y_top": y_top, "y_bottom": y_bottom,
        "crests": crests, "roots": roots,
        "pitch_px": pitch_px, "depth_px": depth_px,
    }
