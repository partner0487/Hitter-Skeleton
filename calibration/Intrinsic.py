import os
import glob
import json
import argparse
import logging
import re
from datetime import datetime
import cv2
import numpy as np

from utils import get_starting_index, setup_logging


def has_images(target_dir: str) -> bool:
    """檢查指定的 result_dual_intrinsic 資料夾內是否有相機圖片"""
    front_img_dir = os.path.join(target_dir, "front_camera", "chessboard_images")
    side_img_dir = os.path.join(target_dir, "side_camera", "chessboard_images")
    
    # 支援舊版 left/right 與新版 front/side 目錄搜尋
    old_left = os.path.join(target_dir, "left_camera", "chessboard_images")
    old_right = os.path.join(target_dir, "right_camera", "chessboard_images")

    check_dirs = [front_img_dir, side_img_dir, old_left, old_right]
    
    for d in check_dirs:
        if os.path.exists(d):
            imgs = glob.glob(os.path.join(d, "*.*"))
            imgs = [p for p in imgs if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(imgs) > 0:
                return True
    return False


def get_latest_result_dir(specified_date: str) -> str:
    """如果指定日期的資料夾不存在或裡面沒有圖片，自動尋找最近日期且含有圖片的資料夾"""
    target_dir = f"./result_dual_intrinsic_{specified_date}"
    
    # 若指定的資料夾存在且裡面確實有圖片，直接使用
    if os.path.exists(target_dir) and has_images(target_dir):
        return target_dir

    pattern = re.compile(r"^result_dual_intrinsic_(\d{8})$")
    matching_dirs = []

    for entry in os.listdir("."):
        if os.path.isdir(entry):
            match = pattern.match(entry)
            if match:  # 修正此處：去掉多餘的冒號
                matching_dirs.append((match.group(1), entry))

    if matching_dirs:
        # 依據 YYYYMMDD 日期降冪排序（從最新到最舊）
        matching_dirs.sort(key=lambda x: x[0], reverse=True)
        
        # 優先尋找「有圖片」的最新資料夾
        for date_str, dir_name in matching_dirs:
            if has_images(dir_name):
                print(f"⚠️ 資料夾 '{target_dir}' 無有效圖片，自動切換至最近有資料的歷史資料夾: '{dir_name}'")
                return dir_name
        
        # 如果都沒有圖片，退回使用最新建立的資料夾
        latest_dir = matching_dirs[0][1]
        print(f"⚠️ 自動選取最近日期的資料夾: '{latest_dir}'")
        return latest_dir

    print(f"ℹ️ 未找到任何既存的 result_dual_intrinsic 資料夾，將建立新資料夾: '{target_dir}'")
    return target_dir


def setup_dual_directories(base_dir: str):
    """建立正面與側面相機各自儲存圖片與參數的目錄結構"""
    dirs = {
        "front_img": os.path.join(base_dir, "front_camera", "chessboard_images"),
        "front_calib": os.path.join(base_dir, "front_camera", "calibration"),
        "side_img": os.path.join(base_dir, "side_camera", "chessboard_images"),
        "side_calib": os.path.join(base_dir, "side_camera", "calibration"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def save_single_camera_json(camera_matrix: np.ndarray, dist_coeffs: np.ndarray, output_path: str, rms_error: float):
    """標準化儲存單目標定參數至 JSON 檔"""
    data = {
        "rms_reprojection_error": float(rms_error),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "camera_matrix_shape": list(camera_matrix.shape),
        "dist_coeffs_shape": list(dist_coeffs.shape)
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"💾 參數已成功儲存至: {output_path}")

def calibrate_camera_points(objpoints, imgpoints, image_size):
    """
    標準 8mm 側面廣角鏡頭標定優化：
    - 設定光學中心初值 (畫面中心)
    - 固定切向畸變 p1=p2=0 與高階畸變 k3=0
    """
    w, h = image_size
    init_K = np.array([
        [1000.0, 0.0, w / 2.0],
        [0.0, 1000.0, h / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    init_D = np.zeros((5, 1), dtype=np.float64)

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS +  # 提供光心與焦距初值引導
        cv2.CALIB_FIX_ASPECT_RATIO +     # 強制 fx == fy (正方形像素)
        cv2.CALIB_ZERO_TANGENT_DIST +    # 忽略切向畸變 p1, p2
        cv2.CALIB_FIX_K3                 # 固定 k3 = 0，避免高階畸變在邊緣發散
    )
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, init_K, init_D, flags=flags
    )
    return ret, camera_matrix, dist_coeffs


def calibrate_camera_points_telephoto(objpoints, imgpoints, image_size):
    """
    專門為 35mm 望遠大距離校正設計的穩定優化函式：
    - 強制鎖定光心在畫面中心附近的合理範圍 (CALIB_FIX_PRINCIPAL_POINT)
    - 鎖定高階畸變 k2=0, k3=0 (長焦鏡頭幾乎無高階桶形/枕形畸變，防止發散至 -26.02)
    - 固定切向畸變 p1=p2=0
    """
    w, h = image_size
    # 提供 35mm 在當前解析度下的合理光學初值 (~5800) 與圖像正中心
    init_K = np.array([
        [5800.0, 0.0, w / 2.0],
        [0.0, 5800.0, h / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    init_D = np.zeros((5, 1), dtype=np.float64)

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS +    # 使用給定初值引導收斂
        cv2.CALIB_FIX_ASPECT_RATIO +       # 強制 fx == fy (維持正方形像素)
        cv2.CALIB_FIX_PRINCIPAL_POINT +    # 🎯 鎖定光心 (cx, cy) 在影像中心，防止偏至 1320
        cv2.CALIB_ZERO_TANGENT_DIST +      # 切向畸變 p1=p2=0
        cv2.CALIB_FIX_K2 +                 # 🎯 鎖定 k2 = 0，防止像前次一樣發散至 -26.02
        cv2.CALIB_FIX_K3                   # 固定 k3 = 0
    )

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, init_K, init_D, flags=flags
    )
    return ret, camera_matrix, dist_coeffs


def process_camera_folder(img_dir: str, pattern_size: tuple, square_size_mm: float, cam_name: str):
    """從資料夾中批次讀取圖片進行角點檢測與亞像素精準化"""
    print(f"\n📂 開始讀取 {cam_name} 資料夾影像: {img_dir}")
    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_paths:
        print(f"⚠️ {cam_name} 未找到任何圖片！")
        return None, None, None

    # 世界座標定義 (Z=0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size_mm
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []
    imgpoints = []
    image_size = None
    valid_count = 0

    for idx, path in enumerate(image_paths, 1):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if found:
            sub_c = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
            objpoints.append(objp)
            imgpoints.append(sub_c)
            valid_count += 1
            print(f"[{idx}/{len(image_paths)}] ✅ 成功偵測角點: {os.path.basename(path)}")
        else:
            print(f"[{idx}/{len(image_paths)}] ❌ 未偵測到棋盤格: {os.path.basename(path)}")

    print(f"📊 {cam_name} 有效角點樣本: {valid_count} / {len(image_paths)}")
    return objpoints, imgpoints, image_size


def run_folder_intrinsic_calculation(dirs: dict, pattern_front: tuple, square_front: float,
                                     pattern_side: tuple, square_side: float):
    """離線計算模式：直接讀取現有圖片目錄計算內參"""
    print("\n================ 開始離線資料夾內參計算 ================")

    # 1. 處理正面相機 (正面 35mm 望遠)
    obj_f, img_f, size_f = process_camera_folder(dirs["front_img"], pattern_front, square_front, "正面相機 (Front 35mm)")
    if obj_f and len(obj_f) >= 5:
        print(f"⏳ 正在計算正面相機內參 (有效樣本: {len(obj_f)})...")
        rms_f, K_f, D_f = calibrate_camera_points_telephoto(obj_f, img_f, size_f)
        print(f"✅ 正面相機校正完成 | RMS 重投影誤差: {rms_f:.4f} pixels")
        print(f"   fx: {K_f[0,0]:.2f}, fy: {K_f[1,1]:.2f}, cx: {K_f[0,2]:.2f}, cy: {K_f[1,2]:.2f}")
        print(f"   畸變係數 D: {D_f.flatten()}")
        save_single_camera_json(
            K_f, D_f, 
            os.path.join(dirs["front_calib"], "single_camera_parameters.json"),
            rms_f
        )
    else:
        print("❌ 正面相機有效樣本不足 5 組，跳過計算。")

    # 2. 處理側面相機 (側面 8mm 廣角)
    obj_s, img_s, size_s = process_camera_folder(dirs["side_img"], pattern_side, square_side, "側面相機 (Side 8mm)")
    if obj_s and len(obj_s) >= 5:
        print(f"⏳ 正在計算側面相機內參 (有效樣本: {len(obj_s)})...")
        rms_s, K_s, D_s = calibrate_camera_points(obj_s, img_s, size_s)
        print(f"✅ 側面相機校正完成 | RMS 重投影誤差: {rms_s:.4f} pixels")
        print(f"   fx: {K_s[0,0]:.2f}, fy: {K_s[1,1]:.2f}, cx: {K_s[0,2]:.2f}, cy: {K_s[1,2]:.2f}")
        print(f"   畸變係數 D: {D_s.flatten()}")
        save_single_camera_json(
            K_s, D_s, 
            os.path.join(dirs["side_calib"], "single_camera_parameters.json"),
            rms_s
        )
    else:
        print("❌ 側面相機有效樣本不足 5 組，跳過計算。")

    print("========================================================\n")


def run_realtime_capture(dirs: dict, pattern_front: tuple, square_front: float,
                         pattern_side: tuple, square_side: float):
    """即時連線雙相機擷取與即時標定"""
    from camera_objects import FlirCameraSystem

    config_front = "./camera_config/GH3_camera_config.yaml"
    serial_front = "25083472"  # 正面相機 (35mm)
    config_side = "./camera_config/GH3_camera_config_2.yaml"
    serial_side = "21091470"   # 側面相機 (8mm)

    print("🎥 正在初始化雙相機硬體串流...")
    cam_front = FlirCameraSystem(config_front, serial_front)
    cam_side = FlirCameraSystem(config_side, serial_side)

    objp_f = np.zeros((pattern_front[0] * pattern_front[1], 3), np.float32)
    objp_f[:, :2] = np.mgrid[0:pattern_front[0], 0:pattern_front[1]].T.reshape(-1, 2) * square_front

    objp_s = np.zeros((pattern_side[0] * pattern_side[1], 3), np.float32)
    objp_s[:, :2] = np.mgrid[0:pattern_side[0], 0:pattern_side[1]].T.reshape(-1, 2) * square_side

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints_f, imgpoints_f = [], []
    objpoints_s, imgpoints_s = [], []
    image_size_f, image_size_s = None, None

    # 正面與側面共用同一個編號：同一次按下空白鍵拍的照片檔名相同，PnP.py 才能用檔名正確配對
    saved_count = max(get_starting_index(dirs["front_img"]),
                      get_starting_index(dirs["side_img"]))

    window_name = "Dual Camera Real-Time Intrinsic Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[即時操作說明]")
    print(" - 空白鍵 (SPACE) : 採樣 (只要任一相機有綠燈角點，該相機就採樣並存檔)")
    print(" - 'c' 鍵         : 雙相機分別執行內參計算並匯出 JSON")
    print(" - 'r' 鍵         : 清空記憶體中已採樣的角點資料")
    print(" - 'q' 鍵         : 退出程式\n")

    try:
        while True:
            ret_frame_f, gray_f = cam_front.get_grayscale_image()
            ret_frame_s, gray_s = cam_side.get_grayscale_image()

            if not ret_frame_f or not ret_frame_s:
                logging.error("影像擷取失敗！")
                break

            if image_size_f is None:
                image_size_f = (gray_f.shape[1], gray_f.shape[0])
                image_size_s = (gray_s.shape[1], gray_s.shape[0])

            vis_f = cv2.cvtColor(gray_f, cv2.COLOR_GRAY2BGR)
            vis_s = cv2.cvtColor(gray_s, cv2.COLOR_GRAY2BGR)

            # 0.5x 降採樣加速檢測
            small_f = cv2.resize(gray_f, (0, 0), fx=0.5, fy=0.5)
            small_s = cv2.resize(gray_s, (0, 0), fx=0.5, fy=0.5)
            fast_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

            found_f, corners_f = cv2.findChessboardCorners(small_f, pattern_front, flags=fast_flags)
            found_s, corners_s = cv2.findChessboardCorners(small_s, pattern_side, flags=fast_flags)

            if found_f:
                cv2.drawChessboardCorners(vis_f, pattern_front, corners_f * 2.0, found_f)
            if found_s:
                cv2.drawChessboardCorners(vis_s, pattern_side, corners_s * 2.0, found_s)

            color_f = (0, 255, 0) if found_f else (0, 0, 255)
            color_s = (0, 255, 0) if found_s else (0, 0, 255)

            cv2.putText(vis_f, f"Front Samples: {len(objpoints_f)}", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis_f, "DETECTED" if found_f else "SEARCHING...", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_f, 2)

            cv2.putText(vis_s, f"Side Samples: {len(objpoints_s)}", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis_s, "DETECTED" if found_s else "SEARCHING...", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_s, 2)

            preview_canvas = np.hstack([vis_f, vis_s])
            cv2.imshow(window_name, preview_canvas)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == 32:  # 空白鍵採樣
                sampled = []
                if found_f:
                    _, full_c_f = cv2.findChessboardCorners(gray_f, pattern_front, None)
                    if full_c_f is not None:
                        sub_c_f = cv2.cornerSubPix(gray_f, full_c_f, (11, 11), (-1, -1), subpix_criteria)
                        objpoints_f.append(objp_f)
                        imgpoints_f.append(sub_c_f)
                        save_p = os.path.join(dirs["front_img"], f"chessboard_image{saved_count}.png")
                        cv2.imwrite(save_p, gray_f)
                        sampled.append("正面")

                if found_s:
                    _, full_c_s = cv2.findChessboardCorners(gray_s, pattern_side, None)
                    if full_c_s is not None:
                        sub_c_s = cv2.cornerSubPix(gray_s, full_c_s, (11, 11), (-1, -1), subpix_criteria)
                        objpoints_s.append(objp_s)
                        imgpoints_s.append(sub_c_s)
                        save_p = os.path.join(dirs["side_img"], f"chessboard_image{saved_count}.png")
                        cv2.imwrite(save_p, gray_s)
                        sampled.append("側面")

                if sampled:
                    # 只有一台偵測到時仍可用於內參，但另一邊沒有同名檔案，不會被 PnP.py 當成外參配對
                    tag = "雙機配對" if len(sampled) == 2 else f"僅{sampled[0]}，不會用於外參"
                    print(f"📸 第 {saved_count} 組採樣成功（{tag}）-> "
                          f"正面相機: {len(objpoints_f)} 張, 側面相機: {len(objpoints_s)} 張")
                    saved_count += 1
                else:
                    print("⚠️ 正面與側面皆未捕捉到完整棋盤格，無法採樣。")


            elif key == ord('r'):
                objpoints_f.clear()
                imgpoints_f.clear()
                objpoints_s.clear()
                imgpoints_s.clear()
                print("🔄 已重設正面與側面相機的所有角點樣本！")

            elif key == ord('c'):
                run_folder_intrinsic_calculation(dirs, pattern_front, square_front, pattern_side, square_side)

    finally:
        cam_front.release()
        cam_side.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="雙相機單目內參校正工具 (支援即時串流與離線直接計算)")
    parser.add_argument("--calculate", action="store_true", help="指定此旗標時不開相機，直接讀取現有圖片資料夾計算內參")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y%m%d"), help="指定日期資料夾 (預設為當日 YYYYMMDD)")
    args = parser.parse_args()

    # 尋找資料夾：若當天無圖片，會自動搜尋並改用歷史最新有圖片的資料夾 (例如 20261008)
    base_dir = get_latest_result_dir(args.date)
    dirs = setup_dual_directories(base_dir)
    setup_logging(base_dir)

    # 標定板規格配置 (內角點數量 與 格子邊長 mm)
    pattern_size_front = (8, 7)
    square_size_front_mm = 100.0

    pattern_size_side = (8, 7)
    square_size_side_mm = 100.0

    if args.calculate:
        # 模式一：直接讀取已存在資料夾內的圖片做計算
        run_folder_intrinsic_calculation(dirs, pattern_size_front, square_size_front_mm, pattern_size_side, square_size_side_mm)
    else:
        # 模式二：即時串流擷取與互動計算
        run_realtime_capture(dirs, pattern_size_front, square_size_front_mm, pattern_size_side, square_size_side_mm)


if __name__ == "__main__":
    main()