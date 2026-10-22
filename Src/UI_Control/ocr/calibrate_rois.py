import cv2
import json
from pathlib import Path

def main():
    CAMERA_INDEX = 0
    ROI_FILE = Path("roi_hitter.json")

    print(f"Opening camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}.")
        return

    # Set same resolution as capture script
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Read a few frames to let the camera adjust
    for _ in range(10):
        ret, frame = cap.read()
        
    # Load existing ROIs to allow skipping
    existing_rois = {}
    if ROI_FILE.exists():
        try:
            with open(ROI_FILE, "r", encoding="utf-8") as f:
                existing_rois = json.load(f)
        except Exception:
            pass

    rois = {}
    fields = ["ExitVelocity", "LaunchAngle", "Distance", "SprayChart"]

    print("\n" + "="*50)
    print("ROI CALIBRATION MODE")
    print("="*50)
    print("For each field, please draw a rectangle around the value.")
    print("Drag with your mouse to draw, then press ENTER or SPACE to confirm.")
    print("Press 'c' to cancel the current drawing and try again.")
    print("If you want to KEEP the existing ROI for a field, just press ENTER without drawing.")
    print("="*50 + "\n")

    for field in fields:
        print(f"Please select the region for: {field} (Press ENTER to skip/keep existing)")
        roi = cv2.selectROI(f"Select {field}", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(f"Select {field}")
        
        if roi == (0, 0, 0, 0):
            if field in existing_rois:
                print(f"Keeping existing ROI for {field}.")
                rois[field] = existing_rois[field]
            else:
                print(f"Warning: Empty ROI selected for {field} and no existing ROI found. Skipping.")
        else:
            rois[field] = {
                "x": int(roi[0]),
                "y": int(roi[1]),
                "w": int(roi[2]),
                "h": int(roi[3])
            }
            print(f"Saved {field}: {rois[field]}")

    with open(ROI_FILE, "w", encoding="utf-8") as f:
        json.dump(rois, f, indent=2)

    print(f"\nSuccessfully updated {ROI_FILE}!")
    
    # Draw them on the frame to show the final result
    display_frame = frame.copy()
    for name, roi_data in rois.items():
        x, y, w, h = roi_data["x"], roi_data["y"], roi_data["w"], roi_data["h"]
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_frame, name, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
    cv2.imshow("Final ROIs (Press any key to close)", display_frame)
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
