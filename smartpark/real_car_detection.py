"""
real_car_detection.py

A more robust replacement for inference.py's find_car_bbox_in_spot(), aimed
at real photos (shadows, texture, varied lighting) rather than clean
synthetic asphalt. Real cars reliably produce dense edges (window frames,
panel lines, wheel wells, reflections) whereas asphalt -- even shadowed
asphalt -- stays comparatively edge-sparse. Edge density is far more
lighting-invariant than raw pixel color, which is why the original
color-distance approach broke on real photos.

Auto-thresholded Canny (via the image's own median intensity) means no
per-photo manual tuning is needed for exposure differences between lots.
"""
import cv2
import numpy as np


def _auto_canny(gray, sigma=0.33):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def find_car_bbox_real(img, spot_bbox, pad_ratio=0.12, min_area_ratio=0.08):
    """Edge-density based car localization, for real photos.

    pad_ratio is intentionally small (real PKLot spaces are marked tightly
    around each stall already, unlike our synthetic generator's looser
    layout) to avoid bleeding into neighboring cars.
    """
    x0, y0, x1, y1 = spot_bbox
    w, h = x1 - x0, y1 - y0
    pad_x, pad_y = max(int(w * pad_ratio), 2), max(int(h * pad_ratio), 2)
    H, W = img.shape[:2]
    rx0, ry0 = max(x0 - pad_x, 0), max(y0 - pad_y, 0)
    rx1, ry1 = min(x1 + pad_x, W), min(y1 + pad_y, H)
    region = img[ry0:ry1, rx0:rx1]
    if region.size == 0:
        return None

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = _auto_canny(gray)

    # thicken + connect nearby edges into solid blobs representing the car body
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    region_area = region.shape[0] * region.shape[1]
    # merge all contours above the noise threshold into one bounding box,
    # rather than trusting a single "biggest" blob (edge maps often
    # fragment one car into several disconnected pieces)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 0.01 * region_area]
    if not boxes:
        return None

    xs0 = [bx for bx, by, bw, bh in boxes]
    ys0 = [by for bx, by, bw, bh in boxes]
    xs1 = [bx + bw for bx, by, bw, bh in boxes]
    ys1 = [by + bh for bx, by, bw, bh in boxes]
    cx0, cy0, cx1, cy1 = min(xs0), min(ys0), max(xs1), max(ys1)

    merged_area = (cx1 - cx0) * (cy1 - cy0)
    if merged_area < min_area_ratio * region_area:
        return None  # too small / sparse edges -> probably not actually a car

    return [rx0 + cx0, ry0 + cy0, rx0 + cx1, ry0 + cy1]
