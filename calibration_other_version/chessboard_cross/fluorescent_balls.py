"""Fluorescent pink ball candidates from BGR images; only OpenCV and NumPy.

No camera intrinsics, PnP, fixed marker count, or physical ball numbering.
Centres are in original image pixels, even when detection uses an ROI.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DetectionConfig:
    hue_min: int = 135
    hue_max: int = 168
    saturation_min: int = 105
    value_min: int = 50
    min_radius_px: float = 3.0
    min_area_px: float = 20.0
    max_area_px: float = 2500.0

    def validate(self):
        if not (0 <= self.hue_min <= 179 and 0 <= self.hue_max <= 179
                and 0 <= self.saturation_min <= 255 and 0 <= self.value_min <= 255
                and np.isfinite([self.min_radius_px, self.min_area_px, self.max_area_px]).all()
                and self.min_radius_px > 0 and 0 < self.min_area_px < self.max_area_px):
            raise ValueError("Invalid HSV/blob settings.")


def read_image(path):
    path = Path(path)
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot read image: {}".format(path))
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError("Cannot encode image: {}".format(path))
    encoded.tofile(str(path))


def detect_balls(image, config=None, roi=None):
    """Segment pink, then split touching blobs with distance-transform peaks.

    Return all candidate centres, not an arbitrary eight largest contours.
    Physical marker identities and geometry must be validated separately.
    """
    config = config or DetectionConfig()
    config.validate()
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected an 8-bit BGR image.")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low = (config.hue_min, config.saturation_min, config.value_min)
    high = (config.hue_max, 255, 255)
    if config.hue_min <= config.hue_max:
        mask = cv2.inRange(hsv, low, high)
    else:
        mask = cv2.inRange(hsv, low, (179, 255, 255)) | cv2.inRange(
            hsv, (0, config.saturation_min, config.value_min), high)
    if roi is not None:
        x, y, w, h = roi
        if min(x, y) < 0 or min(w, h) <= 0 or x+w > image.shape[1] or y+h > image.shape[0]:
            raise ValueError("ROI must be x y width height inside the original image.")
        region = np.zeros_like(mask)
        region[y:y+h, x:x+w] = mask[y:y+h, x:x+w]
        mask = region
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if not config.min_area_px <= area <= config.max_area_px:
            continue
        component = np.uint8(labels[y:y+h, x:x+w] == label)
        # Pad: otherwise a foreground component touching the crop boundary
        # incorrectly has infinite/overestimated distance to the background.
        distance = cv2.distanceTransform(np.pad(component, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        peaks = ((distance >= cv2.dilate(distance, np.ones((5, 5), np.uint8))-1e-6)
                 & (distance >= config.min_radius_px)).astype(np.uint8)
        n, _, _, centers = cv2.connectedComponentsWithStats(peaks)
        seeds = sorted(centers[1:n], key=lambda p: distance[int(round(p[1])), int(round(p[0]))],
                       reverse=True)
        selected = []
        for seed in seeds:
            radius = float(distance[int(round(seed[1])), int(round(seed[0]))])
            if all(np.linalg.norm(seed-p) >= max(6.0, 0.9*(radius+r)) for p, r in selected):
                selected.append((seed, radius))
        if not selected:
            continue
        pixels = np.column_stack(np.where(component)[::-1]).astype(float)
        seed_points = np.array([p for p, _ in selected])
        assignment = np.argmin(np.linalg.norm(pixels[:, None] - seed_points, axis=2), axis=1)
        for i, (seed, radius) in enumerate(selected):
            part = pixels[assignment == i]
            if len(part) < config.min_area_px:
                continue
            extent = np.ptp(part, axis=0) + 1
            if min(extent)/max(extent) < 0.4:
                continue
            center = part.mean(axis=0) + [x, y]
            candidates.append({"center_px": center.tolist(), "area_px": int(len(part)),
                               "radius_px": radius, "split_component": len(selected) > 1})
    candidates.sort(key=lambda c: (c["center_px"][1], c["center_px"][0]))
    return candidates, mask


