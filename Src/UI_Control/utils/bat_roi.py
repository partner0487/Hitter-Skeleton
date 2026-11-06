"""在打者雙手附近偵測球棒（ROI 偵測）。

整張 1920×1080 縮到 YOLO 輸入大小時，揮棒最快的模糊球棒很小、信心低，常偵測不到或誤抓手臂。
改成以雙手腕中點為中心、打者身高 × ROI_SCALE 為邊長切出區塊再跑 YOLO：
- 球棒在輸入中變大；區塊內門檻降到 CONF（區塊外本來就看不到，不會多出背景誤判）
- 握把在手上：較近那隻手腕到遮罩輪廓的距離需 ≤ 身高 × GRIP_SCALE
  （手會遮住握把，遮罩常離手腕 60 px 以上；收棒時更遠）。量到較近的那隻手而非兩手中點：
  收棒單手放開時中點落在胸前、離球棒很遠（TSG 從打者後方拍，手腕常被身體擋住也是如此）
- 多個候選取握把離手腕最近者（打者手上那支一定最近，即使揮動中模糊、信心較低）
- 離手稍遠（> 身高 × FAR_GRIP_SCALE）的候選，長度需 ≥ 身高 × MIN_FAR_LENGTH：
  後方其他球員的球棒離相機遠、畫面上短（TSG 為身高的 0.06～0.13），打者手上的球棒中位數 0.30
依 WeiShu 2026-10-29 等 34 支片段：著地到擊球的有效遮罩 561 → 640 幀（共 755），
側拋 H04 模糊的第 40、41 幀（原本沒偵測到／誤抓手臂）都抓到球棒。
"""
from typing import Optional, Tuple

import cv2
import numpy as np

BAT_CLASS = 34        # COCO baseball bat
CONF = 0.05
ROI_SCALE = 1.2       # 區塊邊長 = 打者身高（person bbox 高）× ROI_SCALE，約可涵蓋兩倍棒長
GRIP_SCALE = 0.22     # 遮罩離較近那隻手腕的容許距離 = 身高 × GRIP_SCALE
# WeiShu、ChengXuan 的真球棒：手腕距離 / 身高 中位數 0.09、95% 0.16；棒長 / 身高 1% 0.11、5% 0.16
FAR_GRIP_SCALE = 0.15
MIN_FAR_LENGTH = 0.12
IMGSZ = 640
MIN_WRIST_CONF = 0.3


def hands_and_height(keypoints, scores, bbox) -> Optional[Tuple[np.ndarray, float]]:
    """(可信的手腕座標 (k, 2), 身高 px)；Halpe26 的 9 左腕、10 右腕，只有一手可信時只有那一手；都不可信回傳 None"""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).ravel()
    bbox = np.asarray(bbox, dtype=np.float32).ravel()
    pts = [keypoints[i, :2] for i in (9, 10) if keypoints[i, 0] > 0 and scores[i] >= MIN_WRIST_CONF]
    height = float(bbox[3] - bbox[1]) if len(bbox) >= 4 else 0.0
    if not pts or height <= 0:
        return None
    return np.asarray(pts, dtype=np.float32), height


def roi_box(center, height, shape):
    """以 center 為中心、邊長 height × ROI_SCALE 的正方形，裁在畫面內 (x0, y0, x1, y1)"""
    h, w = shape[:2]
    side = int(round(height * ROI_SCALE))
    x0 = int(np.clip(center[0] - side / 2, 0, max(w - side, 0)))
    y0 = int(np.clip(center[1] - side / 2, 0, max(h - side, 0)))
    return x0, y0, min(x0 + side, w), min(y0 + side, h)


def detect_bat_near_hands(yolo_model, image, wrists, height):
    """在雙手腕中點附近偵測，回傳 (bbox [x1, y1, x2, y2], 輪廓 (n, 2), 信心)，座標為整張畫面；沒找到回傳 None"""
    return detect_bats_near_hands(yolo_model, [(image, wrists, height)])[0]


def detect_bats_near_hands(yolo_model, jobs):
    """多張畫面一起送 YOLO（雙相機同一幀），jobs 為 [(image, wrists, height), ...]，回傳各自的 detect_bat_near_hands 結果"""
    boxes = [roi_box(np.asarray(wrists, dtype=np.float32).reshape(-1, 2).mean(axis=0), height, image.shape)
             for image, wrists, height in jobs]
    crops = [image[y0:y1, x0:x1] for (image, _, _), (x0, y0, x1, y1) in zip(jobs, boxes)]
    if not crops:
        return []
    results = yolo_model.predict(crops, classes=[BAT_CLASS], conf=CONF, imgsz=IMGSZ, retina_masks=True, verbose=False)
    return [_pick_bat(result, box[:2], wrists, height) for result, box, (_, wrists, height) in zip(results, boxes, jobs)]


def _pick_bat(result, origin, wrists, height):
    """區塊內的候選中挑打者手上那支（握把離手腕最近），座標從區塊換回整張畫面"""
    x0, y0 = origin
    if result.masks is None:
        return None
    best, best_dist = None, None
    for box, poly, conf in zip(result.boxes.xyxy.cpu().numpy(), result.masks.xy, result.boxes.conf.cpu().numpy()):
        if len(poly) < 3:
            continue
        poly = (poly + np.array([x0, y0], dtype=np.float32)).astype(np.float32)
        dist = grip_distance(poly, wrists, height)
        if dist is None:
            continue
        if best is None or (max(dist, 0.0), -conf) < (max(best_dist, 0.0), -best[2]):
            best, best_dist = ((box + np.array([x0, y0, x0, y0])).tolist(), poly, float(conf)), dist
    return best


def grip_distance(poly, wrists, height) -> Optional[float]:
    """球棒在打者手上時回傳較近那隻手腕到遮罩的距離（px，在遮罩內為負），不在手上回傳 None：
    距離 > 身高 × GRIP_SCALE，或離手稍遠（> 身高 × FAR_GRIP_SCALE）又太短（遠處其他球員的球棒）"""
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(poly) < 3 or height <= 0:
        return None
    # pointPolygonTest：在輪廓內為正、外為負；握把在手上，至少一隻手腕應在遮罩內或很近
    contour = poly.reshape(-1, 1, 2)
    dist = min(-cv2.pointPolygonTest(contour, (float(x), float(y)), True)
               for x, y in np.asarray(wrists, dtype=np.float32).reshape(-1, 2))
    if dist > height * GRIP_SCALE:
        return None
    if dist > height * FAR_GRIP_SCALE:
        length = float(np.linalg.norm(poly[:, None] - poly[None], axis=-1).max())
        if length < height * MIN_FAR_LENGTH:
            return None
    return float(dist)
