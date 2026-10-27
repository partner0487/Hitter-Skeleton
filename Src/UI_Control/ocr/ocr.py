from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any
import numpy as np
import cv2
from paddleocr import PaddleOCR

# Skip network source checking to reduce startup latency.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract numeric values from ROIs using PaddleOCR and return JSON",
    )
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument(
        "--roi-file",
        type=Path,
        default=Path("roi_hitter.json"),
        help="ROI JSON file with named entries (default: roi_hitter.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output JSON path",
    )
    parser.add_argument(
        "--det-model",
        default="PP-OCRv5_mobile_det",
        help="PaddleOCR detection model (default: PP-OCRv5_mobile_det)",
    )
    parser.add_argument(
        "--rec-model",
        default="PP-OCRv5_mobile_rec",
        help="PaddleOCR recognition model (default: PP-OCRv5_mobile_rec)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to run OCR on (default: cpu)",
    )
    parser.add_argument(
        "--use-tensorrt",
        action="store_true",
        default=False,
        help="Use TensorRT optimization (GPU only)",
    )
    parser.add_argument(
        "--enable-hpi",
        action="store_true",
        default=False,
        help="Enable HPIPE for faster OCR (GPU only)",
    )
    
    return parser.parse_args()


def load_rois(roi_file: Path) -> dict[str, tuple[int, int, int, int]]:
    if not roi_file.exists():
        raise FileNotFoundError(f"ROI file does not exist: {roi_file}")

    with roi_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("ROI file format is invalid (expected JSON object)")

    rois: dict[str, tuple[int, int, int, int]] = {}
    for name, item in data.items():
        if not isinstance(item, dict):
            raise ValueError(f"ROI entry for {name} is invalid")
        try:
            x = int(item["x"])
            y = int(item["y"])
            w = int(item["w"])
            h = int(item["h"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"ROI entry for {name} must contain integer x,y,w,h") from exc

        if min(x, y, w, h) < 0 or w <= 0 or h <= 0:
            raise ValueError(f"ROI entry for {name} has invalid coordinates")

        rois[name] = (x, y, w, h)

    if not rois:
        raise ValueError(f"No ROI entries found in: {roi_file}")

    return rois


def _to_number(text: str) -> float | None:
    t = text.replace(":", ".")
    t = re.sub(r"[^0-9.+-]", "", t)
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _format_number(value: float, decimals: int) -> str:
    if decimals == 0:
        return f"{value:.0f}"
    return f"{value:.{decimals}f}"


def normalize_field_value(text: str, field_name: str) -> str:
    value = _to_number(text)
    if value is None:
        return ""

    name = field_name.lower()

    if "exit" in name:
        while value > 100 and value >= 10:
            value /= 10.0
        value = max(60.0, min(100.0, value))
        return _format_number(value, 1)

    if "angle" in name:
        while value > 90 and value >= 10:
            value /= 10.0
        value = max(0.0, min(90.0, value))
        return _format_number(value, 1)

    if "distance" in name:
        while value > 500 and value >= 10:
            value /= 10.0
        value = max(0.0, min(500.0, value))
        return _format_number(value, 0)

    return text.strip()


def score_candidate(token: str, conf: float, field_name: str) -> tuple[int, int, int, int, int]:
    name = field_name.lower()
    value = _to_number(token)
    if value is None:
        return (0, 0, 0, 0, 0)

    in_range = 0
    prefers_decimal = 0
    digits = sum(ch.isdigit() for ch in token)

    if "exit" in name:
        in_range = 1 if 60 <= value <= 100 else 0
        prefers_decimal = 1 if "." in token else 0
    elif "angle" in name:
        in_range = 1 if 0 <= value <= 90 else 0
        prefers_decimal = 1 if "." in token else 0
    elif "distance" in name:
        in_range = 1 if 0 <= value <= 500 else 0
        # Distance usually 3 digits in this workflow.
        prefers_decimal = 1 if token.isdigit() and len(token) >= 3 else 0

    conf_i = int(conf * 1000)
    return (in_range, prefers_decimal, digits, conf_i, len(token))


def extract_best_number_from_roi(
    roi_img,
    field_name: str,
    ocr: PaddleOCR,
) -> str:
    res_list = ocr.predict(roi_img)
    if not res_list:
        return ""

    res = res_list[0]
    texts = res.get("rec_texts", []) or []
    scores = res.get("rec_scores", []) or []

    best_token = ""
    best_score = (0, 0, 0, 0, 0)

    for i, raw_text in enumerate(texts):
        conf = float(scores[i]) if i < len(scores) else 0.0
        for token in re.findall(r"[-+]?\d+(?:[\.:]\d+)?", str(raw_text)):
            s = score_candidate(token, conf, field_name)
            if s > best_score:
                best_score = s
                best_token = token

    return normalize_field_value(best_token, field_name)


def extract_from_ndarray(
    image: np.ndarray,
    rois: dict[str, tuple[int, int, int, int]],
    ocr: PaddleOCR,
) -> dict[str, str]:
    """
    Extract values from an image (numpy array) using provided ROIs and OCR model.
    """
    img_h, img_w = image.shape[:2]
    out: dict[str, str] = {}

    for field_name, (x, y, w, h) in rois.items():
        if x + w > img_w or y + h > img_h:
            raise ValueError(
                f"ROI out of bounds for {field_name}: image={img_w}x{img_h}, roi={x},{y},{w},{h}"
            )

        roi_img = image[y : y + h, x : x + w]
        out[field_name] = extract_best_number_from_roi(roi_img, field_name, ocr)

    return out


def extract_from_image_and_rois(
    image_path: Path,
    roi_file: Path,
    det_model: str = "PP-OCRv5_mobile_det",
    rec_model: str = "PP-OCRv5_mobile_rec",
    use_tensorrt: bool = False,
    enable_hpi: bool = False,
    device: str = "cpu",
    ocr: PaddleOCR | None = None,
) -> dict[str, str]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    rois = load_rois(roi_file)

    if ocr is None:
        ocr = PaddleOCR(
            text_detection_model_name=det_model,
            text_recognition_model_name=rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_tensorrt=use_tensorrt,
            enable_hpi=enable_hpi,
            device=device,
        )

    return extract_from_ndarray(image, rois, ocr)


def main() -> None:
    args = parse_args()

    result = extract_from_image_and_rois(
        image_path=args.image,
        roi_file=args.roi_file,
        det_model=args.det_model,
        rec_model=args.rec_model,
        device=args.device,
        use_tensorrt=args.use_tensorrt,
        enable_hpi=args.enable_hpi,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
