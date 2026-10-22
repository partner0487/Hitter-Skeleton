import cv2
import json
import time
from pathlib import Path
from paddleocr import PaddleOCR
from ocr import extract_from_ndarray, load_rois

def main():
    # --- Configuration ---
    CAMERA_INDEX = 0  # Try 0, 1, 2... for your capture card
    ROI_FILE = Path("roi_hitter.json")
    DEVICE = "cpu"    # Change to "gpu" if you have CUDA installed
    # ---------------------

    print(f"Initializing OCR engine on {DEVICE}...")
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=DEVICE,
    )

    rois = load_rois(ROI_FILE)
    
    print(f"Opening camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}.")
        print("Tip: If you have multiple cameras, try changing CAMERA_INDEX in the script.")
        return

    # Recommended settings for many capture cards
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("\nControls:")
    print("  's' - Capture and Run OCR")
    print("  'q' - Quit")
    
    last_results = {}
    ocr_in_progress = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        display_frame = frame.copy()

        # Draw ROI boxes for reference
        for name, (x, y, w, h) in rois.items():
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, name, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw last results on screen
        y0, dy = 30, 30
        for i, (key, value) in enumerate(last_results.items()):
            text = f"{key}: {value}"
            cv2.putText(display_frame, text, (20, y0 + i*dy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if ocr_in_progress:
            cv2.putText(display_frame, "Processing OCR...", (20, display_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 254), 2)

        cv2.imshow("iPad Capture OCR (Press 's' to Capture)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        # Periodic check (once every 1 second)
        if not hasattr(main, "last_check_time"):
            main.last_check_time = 0
            main.last_stable_data = None
            main.last_stable_spray_img = None

        current_time = time.time()
        if current_time - main.last_check_time >= 1.0 and not ocr_in_progress:
            main.last_check_time = current_time
            
            # Filter out SprayChart for OCR extraction
            ocr_rois = {k: v for k, v in rois.items() if k != "SprayChart"}
            current_results = extract_from_ndarray(frame, ocr_rois, ocr)
            last_results = current_results
            
            # Check if current_results has valid numbers (at least ExitVelocity)
            is_valid = bool(current_results.get("ExitVelocity", "").strip())
            
            if main.last_stable_data is None:
                main.last_stable_data = current_results
            else:
                if current_results != main.last_stable_data:
                    # Data changed! That means last_stable_data is now officially the "previous" hit.
                    # We should save last_stable_data BEFORE updating it.
                    
                    if bool(main.last_stable_data.get("ExitVelocity", "").strip()):
                        print(f"\n[Auto-Trigger] Data changed. Saving PREVIOUS data...")
                        
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        save_dir = Path("Db") / timestamp
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        json_path = save_dir / "data.json"
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(main.last_stable_data, f, indent=2, ensure_ascii=False)
                            
                        if main.last_stable_spray_img is not None:
                            img_path = save_dir / "spray_chart.jpg"
                            cv2.imwrite(str(img_path), main.last_stable_spray_img)
                            
                        print(f"Saved data and image to: {save_dir}")
                        print(json.dumps(main.last_stable_data, indent=2))
                    
                    # Update to the new data (could be empty or a new valid number)
                    main.last_stable_data = current_results
            
            # Continuously update the spray chart image so we always have the latest frame of the stable data
            if is_valid and "SprayChart" in rois:
                x, y, w, h = rois["SprayChart"]
                main.last_stable_spray_img = frame[y:y+h, x:x+w]

        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"\nManual capture: Running OCR...")
            ocr_in_progress = True
            
            # Show "Processing" immediately
            cv2.putText(display_frame, "Processing OCR...", (20, display_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 254), 2)
            cv2.imshow("iPad Capture OCR (Press 's' to Capture)", display_frame)
            cv2.waitKey(1)

            start_time = time.time()
            try:
                # Create timestamp directory inside Db
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_dir = Path("Db") / timestamp
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Filter out SprayChart for OCR extraction
                ocr_rois = {k: v for k, v in rois.items() if k != "SprayChart"}
                results = extract_from_ndarray(frame, ocr_rois, ocr)
                last_results = results
                
                # Save results to JSON
                json_path = save_dir / "data.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                # Save SprayChart image if it exists in ROIs
                if "SprayChart" in rois:
                    x, y, w, h = rois["SprayChart"]
                    spray_img = frame[y:y+h, x:x+w]
                    img_path = save_dir / "spray_chart.jpg"
                    cv2.imwrite(str(img_path), spray_img)

                print(f"Saved data and image to: {save_dir}")
                print(f"OCR Results (took {time.time() - start_time:.2f}s):")
                print(json.dumps(results, indent=2))
            except Exception as e:
                print(f"Error during OCR: {e}")
            
            ocr_in_progress = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
