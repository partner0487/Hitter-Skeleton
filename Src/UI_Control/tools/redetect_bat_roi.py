"""在打者雙手附近重新偵測球棒（ROI 重偵測），結果另存，不覆蓋原本的 bat_side.json。

原本的偵測（skeleton/detect_skeleton_dual.py）是整張 1920×1080 縮到 YOLO 輸入大小、門檻 0.15；
揮棒最快時球棒模糊，常常偵測不到或誤抓手臂。這裡改成：
- 以雙手腕中點為中心、打者身高 × ROI_SCALE 為邊長切出區塊，只在區塊內跑 YOLO（球棒在輸入中變大）
- 區塊內門檻降到 CONF（區塊外的東西本來就看不到，不會多出背景誤判）
- 候選遮罩必須碰到雙手附近（握把在手上）：手腕中點到遮罩輪廓的距離 ≤ 身高 × GRIP_SCALE
- 多個候選時取信心最高者
- 模型與原本相同（Db/pretrain/yolo11m-seg.pt），可用 --model 換成其他權重比較

輸出（每個資料夾）：
  bat_side_roi.json   與 bat_side.json 相同格式（bbox、mask、frame_number），另加 conf、candidates；
                      沒偵測到的幀不輸出（不沿用上一幀）
用法（Pitcher 環境）：
  python Src/UI_Control/tools/redetect_bat_roi.py "Db/WeiShu/2026-10-29/側拋/WeiShu_H04" [更多資料夾…]
"""
import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = ROOT / "Db" / "pretrain" / "yolo11m-seg.pt"
BAT_CLASS = 34        # COCO baseball bat
CONF = 0.05
ROI_SCALE = 1.2       # 區塊邊長 = 打者身高（person bbox 高）× ROI_SCALE，約可涵蓋兩倍棒長
GRIP_SCALE = 0.22     # 遮罩離雙手腕中點的容許距離 = 身高 × GRIP_SCALE（手會遮住握把，遮罩常離手腕 60 px 以上；收棒時更遠）
IMGSZ = 640
MIN_WRIST_CONF = 0.3


def read_records(path):
    return pd.read_json(path, orient="records") if path.exists() else pd.DataFrame()


def batter_frames(person_df):
    """{frame: (雙手腕中點, 身高 px)}；每幀取 bbox 最高的人（打者）"""
    out = {}
    for frame, rows in person_df.groupby("frame_number"):
        best = None
        for _, row in rows.iterrows():
            bbox = np.asarray(row["bbox"], dtype=float).ravel()
            height = bbox[3] - bbox[1]
            if best is None or height > best[1]:
                best = (row, height)
        row, height = best
        kpts = np.asarray(row["keypoints"], dtype=float)
        pts = [kpts[i, :2] for i in (9, 10) if kpts[i, 0] > 0 and kpts[i, 2] >= MIN_WRIST_CONF]
        if pts and height > 0:
            out[int(frame)] = (np.mean(pts, axis=0), float(height))
    return out


def roi_box(center, height, shape):
    """以 center 為中心、邊長 height × ROI_SCALE 的正方形，裁在畫面內 (x0, y0, x1, y1)"""
    h, w = shape[:2]
    side = int(round(height * ROI_SCALE))
    x0 = int(np.clip(center[0] - side / 2, 0, max(w - side, 0)))
    y0 = int(np.clip(center[1] - side / 2, 0, max(h - side, 0)))
    return x0, y0, min(x0 + side, w), min(y0 + side, h)


def detect_frame(model, image, wrist, height):
    x0, y0, x1, y1 = roi_box(wrist, height, image.shape)
    result = model.predict(image[y0:y1, x0:x1], classes=[BAT_CLASS], conf=CONF, imgsz=IMGSZ,
                           retina_masks=True, verbose=False)[0]
    if result.masks is None:
        return None, 0
    grip_tol = height * GRIP_SCALE
    best = None
    for box, poly, conf in zip(result.boxes.xyxy.cpu().numpy(), result.masks.xy, result.boxes.conf.cpu().numpy()):
        if len(poly) < 3:
            continue
        poly = poly + np.array([x0, y0], dtype=np.float32)
        # pointPolygonTest：在輪廓內為正、外為負；握把在手上，手腕中點應在遮罩內或很近
        dist = -cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(wrist[0]), float(wrist[1])), True)
        if dist > grip_tol:
            continue
        if best is None or conf > best["conf"]:
            best = {"bbox": (box + np.array([x0, y0, x0, y0])).tolist(), "poly": poly, "conf": float(conf)}
    return best, len(result.boxes)


def process(folder, model):
    person = read_records(folder / "person_side.json")
    videos = glob.glob(str(folder / "CS_*.mp4"))
    if person.empty or not videos:
        print("   缺 person_side.json 或 CS_*.mp4，略過")
        return
    batter = batter_frames(person)
    cap = cv2.VideoCapture(videos[0])
    rows, frame = [], 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if frame in batter:
            wrist, height = batter[frame]
            best, n = detect_frame(model, image, wrist, height)
            if best is not None:
                # 與原本格式相同：mask = [序號, [輪廓]]
                rows.append({"bbox": best["bbox"], "mask": [0, [best["poly"].round(1).tolist()]],
                             "frame_number": frame, "conf": round(best["conf"], 4), "candidates": n})
        frame += 1
    cap.release()
    with open(folder / "bat_side_roi.json", "w", encoding="utf-8") as f:
        json.dump(rows, f)
    print(f"   {frame} 幀，有打者手腕 {len(batter)} 幀，偵測到球棒 {len(rows)} 幀 → bat_side_roi.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args()
    from ultralytics import YOLO
    model = YOLO(args.model)
    for name in args.folders:
        folder = Path(name).resolve()
        print(f"\n== {folder}")
        process(folder, model)


if __name__ == "__main__":
    main()
