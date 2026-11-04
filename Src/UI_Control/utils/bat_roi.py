"""在打者雙手附近偵測球棒（ROI 偵測）。

整張 1920×1080 縮到 YOLO 輸入大小時，揮棒最快的模糊球棒很小、信心低，常偵測不到或誤抓手臂。
改成以雙手腕中點為中心、打者身高 × ROI_SCALE 為邊長切出區塊再跑 YOLO：
- 球棒在輸入中變大；區塊內門檻降到 CONF（區塊外本來就看不到，不會多出背景誤判）
- 握把在手上：手腕中點到遮罩輪廓的距離需 ≤ 身高 × GRIP_SCALE
  （手會遮住握把，遮罩常離手腕 60 px 以上；收棒時更遠）
- 多個候選取信心最高者
依 WeiShu 2026-10-29 等 34 支片段：著地到擊球的有效遮罩 561 → 640 幀（共 755），
側拋 H04 模糊的第 40、41 幀（原本沒偵測到／誤抓手臂）都抓到球棒。
"""
from typing import Optional, Tuple

import cv2
import numpy as np

BAT_CLASS = 34        # COCO baseball bat
CONF = 0.05
ROI_SCALE = 1.2       # 區塊邊長 = 打者身高（person bbox 高）× ROI_SCALE，約可涵蓋兩倍棒長
GRIP_SCALE = 0.22     # 遮罩離雙手腕中點的容許距離 = 身高 × GRIP_SCALE
IMGSZ = 640
MIN_WRIST_CONF = 0.3


def hands_and_height(keypoints, scores, bbox) -> Optional[Tuple[np.ndarray, float]]:
    """(雙手腕中點, 身高 px)；Halpe26 的 9 左腕、10 右腕，只有一手可信時用那一手；都不可信回傳 None"""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).ravel()
    bbox = np.asarray(bbox, dtype=np.float32).ravel()
    pts = [keypoints[i, :2] for i in (9, 10) if keypoints[i, 0] > 0 and scores[i] >= MIN_WRIST_CONF]
    height = float(bbox[3] - bbox[1]) if len(bbox) >= 4 else 0.0
    if not pts or height <= 0:
        return None
    return np.mean(pts, axis=0), height


def roi_box(center, height, shape):
    """以 center 為中心、邊長 height × ROI_SCALE 的正方形，裁在畫面內 (x0, y0, x1, y1)"""
    h, w = shape[:2]
    side = int(round(height * ROI_SCALE))
    x0 = int(np.clip(center[0] - side / 2, 0, max(w - side, 0)))
    y0 = int(np.clip(center[1] - side / 2, 0, max(h - side, 0)))
    return x0, y0, min(x0 + side, w), min(y0 + side, h)


def detect_bat_near_hands(yolo_model, image, wrist, height):
    """回傳 (bbox [x1, y1, x2, y2], 輪廓 (n, 2), 信心)，座標為整張畫面；沒找到回傳 None"""
    x0, y0, x1, y1 = roi_box(wrist, height, image.shape)
    result = yolo_model.predict(image[y0:y1, x0:x1], classes=[BAT_CLASS], conf=CONF, imgsz=IMGSZ,
                                retina_masks=True, verbose=False)[0]
    if result.masks is None:
        return None
    grip_tol = height * GRIP_SCALE
    best = None
    for box, poly, conf in zip(result.boxes.xyxy.cpu().numpy(), result.masks.xy, result.boxes.conf.cpu().numpy()):
        if len(poly) < 3:
            continue
        poly = (poly + np.array([x0, y0], dtype=np.float32)).astype(np.float32)
        # pointPolygonTest：在輪廓內為正、外為負；握把在手上，手腕中點應在遮罩內或很近
        dist = -cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(wrist[0]), float(wrist[1])), True)
        if dist > grip_tol:
            continue
        if best is None or conf > best[2]:
            best = ((box + np.array([x0, y0, x0, y0])).tolist(), poly, float(conf))
    return best
