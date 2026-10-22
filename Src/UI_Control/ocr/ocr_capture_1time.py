import cv2
import json
import time
import os
from pathlib import Path
from paddleocr import PaddleOCR
from .ocr import extract_from_ndarray, load_rois
import numpy as np

# Global cache for OCR engine to avoid slow re-initialization
_cached_ocr = None

def get_ocr_engine(device="cpu"):
    global _cached_ocr
    if _cached_ocr is None:
        print(f"Initializing OCR engine on {device}...")
        _cached_ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )
    return _cached_ocr

def run_ocr_and_save(video_save_path: str, camera_index: int = 0):
    """
    開啟相機讀取一幀，執行 OCR 辨識，並將結果存入指定的 video_save_path。
    存檔內容：data.json, sprayChart.jpg
    """
    # 1. 確保路徑存在
    save_dir = Path(video_save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. 讀取 ROI 設定
    # 假設 roi_hitter.json 與此腳本在同一目錄或 Src/UI_Control 根目錄
    roi_file = Path(__file__).parent / "roi_hitter.json"
    if not roi_file.exists():
        roi_file = Path("roi_hitter.json")
    
    try:
        rois = load_rois(roi_file)
    except Exception as e:
        print(f"Error loading ROI file: {e}")
        return False

    # 3. 初始化/取得 OCR 引擎
    ocr = get_ocr_engine()

    # 4. 開啟相機讀取一幀
    print(f"Opening camera {camera_index} for single capture...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return False

    # 設定解析度
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # 讀取幾幀以確保影像穩定 (部分擷取卡第一幀會黑屏或閃爍)
    frame = None
    for _ in range(5):
        ret, frame = cap.read()
    
    cap.release()

    if frame is None:
        print("Error: Failed to grab frame.")
        return False

    # 5. 執行 OCR
    start_time = time.time()
    try:
        # 過濾掉噴散圖，只對數據部分進行 OCR
        ocr_rois = {k: v for k, v in rois.items() if k != "SprayChart"}
        results = extract_from_ndarray(frame, ocr_rois, ocr)

        # 6. 存檔數據 (data.json)
        json_path = save_dir / "data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 7. 存檔噴散圖 (sprayChart.jpg)
        if "SprayChart" in rois:
            x, y, w, h = rois["SprayChart"]
            # 檢查範圍是否合法
            img_h, img_w = frame.shape[:2]
            if x + w <= img_w and y + h <= img_h:
                spray_img = frame[y:y+h, x:x+w]
                img_path = save_dir / "sprayChart.jpg"
                cv2.imwrite(str(img_path), spray_img)
            else:
                print(f"Warning: SprayChart ROI {x,y,w,h} is out of frame bounds {img_w, img_h}")

        print(f"Successfully processed OCR and saved to: {save_dir}")
        print(f"Time taken: {time.time() - start_time:.2f}s")
        print(f"Results: {json.dumps(results, indent=2)}")
        return True

    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return False

def main():
    # 測試範例
    test_path = "./Db/TestOCR"
    run_ocr_and_save(test_path, camera_index=0)

if __name__ == "__main__":
    main()
