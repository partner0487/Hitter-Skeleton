# PaddleOCR Baseball Metrics Extraction

Extract baseball player metrics (Exit Velocity, Launch Angle, Distance) from Rapsodo/SoftToss frame images using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).

## Installation

### Basic Setup
```bash
pip install paddleocr opencv-python
```

Or follow the [official PaddleOCR installation guide](https://www.paddleocr.ai/latest/en/version3.x/installation.html).

---

## Quick Start

### Basic Usage (CPU)
```bash
# Extract metrics from a single image
python ocr.py path/to/image.jpg --roi-file roi_hitter.json
```

### Output
Returns JSON with extracted values:
```json
{
  "ExitVelocity": "70.3",
  "LaunchAngle": "27.6",
  "Distance": "227"
}
```

---

## Device Selection

### CPU (Default - No Setup Required)
```bash
python ocr.py image.jpg
```
- No additional dependencies
- Suitable for slower throughput, lower hardware cost
- ~2-5 seconds per image

### GPU Acceleration
```bash
python ocr.py image.jpg --device gpu```


---

## Inference Acceleration

### 1. **Model Selection** (Fastest)

Use lightweight mobile models (default & fastest):
```bash
python ocr.py image.jpg \
  --det-model PP-OCRv5_mobile_det \
  --rec-model PP-OCRv5_mobile_rec
```
- Speed: **~2-3 seconds** (CPU) / **~0.5-1s** (GPU)
- Best for real-time sequential inference

Use high-accuracy server models (slower, more accurate):
```bash
python ocr.py image.jpg \
  --det-model PP-OCRv5_det \
  --rec-model PP-OCRv5_rec
```

### 2. **TensorRT Acceleration** (GPU Only - Fastest)

Convert to TensorRT and use:
```bash
python ocr.py image.jpg \
  --use-gpu \
  --use-tensorrt
```
- Speed: **~0.2-0.5s** per image
- Requires: NVIDIA TensorRT library, CUDA 11.0+
- Not portable across different GPU architectures

### 3. **Batch Processing** (Recommended for Throughput)

Create a batch script:
```python
from pathlib import Path
from ocr import extract_from_image_and_rois
import json

roi_file = Path("roi_hitter.json")
images = sorted(Path(".").glob("*.jpg"))

for img_path in images:
    result = extract_from_image_and_rois(img_path, roi_file)
    print(f"{img_path.name}: {result}")
```
- Reuses loaded models across images
- ~0.3-0.8s per image after first image load

---

## Advanced Configuration

### Custom Output Path
```bash
python ocr.py image.jpg --output results.json
```

### Full Example with All Options
```bash
python ocr.py frame.jpg \
  --roi-file roi_hitter.json \
  --output output.json \
  --det-model PP-OCRv5_mobile_det \
  --rec-model PP-OCRv5_mobile_rec
```

### Environment Options
```bash
# Skip model source checking (faster startup)
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Limit multi-threading (more deterministic)
export OMP_NUM_THREADS=1

# Then run
python ocr.py image.jpg
```

---

## Performance Comparison

| Config | Speed | Accuracy | Notes |
|--------|-------|----------|-------|
| CPU + Mobile Models | 2-3s | √√√ | Default, no setup |
| GPU + Mobile Models | 0.5-1s | √√√ | Fastest practical |
| GPU + TensorRT | 0.2-0.5s | √√√ | Fastest overall, requires TensorRT |
| Server Models | 5-8s | √√√√ | Highest accuracy, slowest |

---

## Input Files

### ROI Configuration (`roi_hitter.json`)
```json
{
  "ExitVelocity": {"x": 826, "y": 239, "w": 95, "h": 58},
  "LaunchAngle": {"x": 825, "y": 385, "w": 84, "h": 58},
  "Distance": {"x": 827, "y": 532, "w": 81, "h": 54}
}
```
- Define bounding boxes (ROI) for each metric
- Coordinates: `x, y` (top-left), `w, h` (width, height)

---

## Troubleshooting

### Memory Issues (GPU Out of Memory)
```bash
# Use mobile models instead
python ocr.py image.jpg --det-model PP-OCRv5_mobile_det
```

### Slow First Run
First run downloads models (~25 MB). Subsequent runs are cached.

### CPU Floating Point Errors
```bash
export OPENBLAS_NUM_THREADS=1
python ocr.py image.jpg
```

---

## References

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [Installation Guide](https://www.paddleocr.ai/latest/en/version3.x/installation.html)
- [Model Zoo](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/models_list_en.md)