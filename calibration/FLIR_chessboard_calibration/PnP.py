import os
import glob
import json
import cv2
import numpy as np
from datetime import datetime


CHECKERBOARD = (8   , 7)
SQUARE_SIZE = 100.0  # mm

MAX_STEREO_RMS = 3.0  # px，雙目標定輸出門檻

def load_camera_params(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到相機內參：{json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    K = np.array(
        data.get("camera_matrix", data.get("K1", data.get("K2"))),
        dtype=np.float64
    )

    D_raw = data.get(
        "dist_coeffs",
        data.get("dist_coeff", data.get("D1", data.get("D2")))
    )

    if D_raw is None:
        D = np.zeros((5, 1), dtype=np.float64)
    else:
        D = np.asarray(D_raw, dtype=np.float64).reshape(-1, 1)

    if K.shape != (3, 3):
        raise ValueError(f"相機內參矩陣形狀錯誤：{K.shape}")

    return K, D


def detect_corners(image, criteria):
    """先用一般模式偵測，失敗再使用較穩健的 SB 模式。"""
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    found, corners = cv2.findChessboardCorners(
        image, CHECKERBOARD, flags
    )

    if not found and hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            image, CHECKERBOARD
        )

    if not found:
        return False, None

    corners = cv2.cornerSubPix(
        image,
        corners.astype(np.float32),
        (11, 11),
        (-1, -1),
        criteria
    )

    return True, corners


def reprojection_error(objpoints, imgpoints, K, D, rvec, tvec):
    """計算單一相機的平均角點重投影誤差，單位為 pixel。"""
    projected, _ = cv2.projectPoints(
        objpoints, rvec, tvec, K, D
    )

    return float(
        cv2.norm(imgpoints, projected, cv2.NORM_L2)
        / np.sqrt(len(objpoints))
    )

def generate_corner_order_candidates(corners, rows=7, cols=8):
    """
    corners: OpenCV 角點，shape=(rows * cols, 1, 2)
    rows=7、cols=8：對應 9×8 黑白格子的內角點
    回傳保留原始長寬的 4 種方向候選。
    """
    grid = np.asarray(corners).reshape(rows, cols, 2)

    return [
        grid.reshape(-1, 1, 2).copy(),
        np.fliplr(grid).reshape(-1, 1, 2).copy(),
        np.flipud(grid).reshape(-1, 1, 2).copy(),
        np.rot90(grid, 2).reshape(-1, 1, 2).copy(),
    ]



def run_stereo_calibration():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 搜尋符合前綴的所有資料夾
    pattern = os.path.join(base_dir, "result_dual_intrinsic_*")
    candidate_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

    if not candidate_dirs:
        raise FileNotFoundError("找不到符合 'result_dual_intrinsic_*' 規則的資料夾")

    # 依資料夾最後修改時間排序，取得最新的資料夾
    intrinsic_dir = max(candidate_dirs, key=os.path.getmtime)
    print(f"📁 使用最新的內參資料夾：{os.path.basename(intrinsic_dir)}")

    # 取得最新資料夾對應的日期標籤（可用於輸出檔名）
    dir_suffix = os.path.basename(intrinsic_dir).replace("result_dual_intrinsic_", "")

    front_json_path = os.path.join(
        intrinsic_dir, "front_camera", "calibration",
        "single_camera_parameters.json"
    )
    side_json_path = os.path.join(
        intrinsic_dir, "side_camera", "calibration",
        "single_camera_parameters.json"
    )

    front_img_dir = os.path.join(
        intrinsic_dir, "front_camera", "chessboard_images"
    )
    side_img_dir = os.path.join(
        intrinsic_dir, "side_camera", "chessboard_images"
    )

    K1, D1 = load_camera_params(front_json_path)
    K2, D2 = load_camera_params(side_json_path)

    print("Front K:\n", K1)
    print("Front D:", D1.ravel())
    print("Side K:\n", K2)
    print("Side D:", D2.ravel())

    # 棋盤格 3D 座標，單位 mm
    objp = np.zeros(
        (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
        dtype=np.float32
    )
    objp[:, :2] = (
        np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]]
        .T.reshape(-1, 2) * SQUARE_SIZE
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5
    )

    front_images = glob.glob(os.path.join(front_img_dir, "*.png"))
    side_images = glob.glob(os.path.join(side_img_dir, "*.png"))

    # 用檔名建立配對，避免字串排序造成 1、10、11、2 的混淆
    side_by_name = {
        os.path.basename(p): p for p in side_images
    }
    valid_pairs = [
        (p, side_by_name[os.path.basename(p)])
        for p in front_images
        if os.path.basename(p) in side_by_name
    ]
    valid_pairs.sort(
        key=lambda pair: int(
            os.path.splitext(os.path.basename(pair[0]))[0]
            .replace("chessboard_image", "")
        )
        if os.path.splitext(os.path.basename(pair[0]))[0]
        .replace("chessboard_image", "").isdigit()
        else os.path.basename(pair[0])
    )

    print(f"\n配對影像數量：{len(valid_pairs)}")

    objpoints = []
    imgpoints_f = []
    imgpoints_s = []
    pair_names = []
    image_size_f = None
    image_size_s = None

    for path_f, path_s in valid_pairs:
        name = os.path.basename(path_f)
        img_f = cv2.imread(path_f, cv2.IMREAD_GRAYSCALE)
        img_s = cv2.imread(path_s, cv2.IMREAD_GRAYSCALE)

        if img_f is None or img_s is None:
            print(f"❌ 影像讀取失敗：{name}")
            continue

        size_f = (img_f.shape[1], img_f.shape[0])
        size_s = (img_s.shape[1], img_s.shape[0])

        if image_size_f is None:
            image_size_f = size_f
            image_size_s = size_s

        if size_f != image_size_f or size_s != image_size_s:
            print(
                f"❌ 影像尺寸不一致：{name}，"
                f"Front={size_f}，Side={size_s}"
            )
            continue

        found_f, corners_f = detect_corners(img_f, criteria)
        found_s, corners_s = detect_corners(img_s, criteria)

        if not found_f or not found_s:
            print(
                f"❌ 跳過 {name}："
                f"Front={'OK' if found_f else 'FAIL'}，"
                f"Side={'OK' if found_s else 'FAIL'}"
            )
            continue

        # 印出每組影像四個角點，檢查兩台相機的角點排列
        corners_f_grid = corners_f.reshape(7, 7, 2)
        corners_s_grid = corners_s.reshape(7, 7, 2)

        print(f"\n[{os.path.basename(path_f)}]")
        print("Front 四角：")
        print(corners_f_grid[0, 0], corners_f_grid[0, -1])
        print(corners_f_grid[-1, 0], corners_f_grid[-1, -1])

        print("Side 四角：")
        print(corners_s_grid[0, 0], corners_s_grid[0, -1])
        print(corners_s_grid[-1, 0], corners_s_grid[-1, -1])

        objpoints.append(objp.copy())
        imgpoints_f.append(corners_f)
        imgpoints_s.append(corners_s)
        pair_names.append(name)

    if len(objpoints) < 10:
        raise RuntimeError(
            f"有效配對僅 {len(objpoints)} 組，資料不足，停止標定。"
        )

    print(f"\n有效雙目配對：{len(objpoints)} 組")
    print("Front 影像尺寸：", image_size_f)
    print("Side 影像尺寸：", image_size_s)

    # 兩台相機的影像尺寸可以不同；OpenCV 的 imageSize
    # 在 stereoCalibrate 中代表同一組標定影像尺寸參數。
    # 為避免不同尺寸造成誤用，這份流程要求兩台影像尺寸相同。
    if image_size_f != image_size_s:
        raise RuntimeError(
            "Front 與 Side 影像解析度不同。"
            "請先確認內參各自對應正確的原始影像尺寸，"
            "不要直接拿其中一台的尺寸標定。"
        )

    # 使用已知內參，只估計兩台相機之間的相對姿態
    flags = cv2.CALIB_FIX_INTRINSIC

    rms, K1_out, D1_out, K2_out, D2_out, R_raw, T_raw, E, F = (
        cv2.stereoCalibrate(
            objpoints,
            imgpoints_f,
            imgpoints_s,
            K1.copy(), D1.copy(),
            K2.copy(), D2.copy(),
            image_size_f,
            criteria=criteria,
            flags=flags
        )
    )

    # 檢查 Front -> Side 外參的基本數值
    print("\n========== 外參方向檢查 ==========")
    print("Stereo RMS:", rms)
    print("R_raw:\n", R_raw)
    print("T_raw:\n", T_raw.reshape(3))

    # 檢查旋轉矩陣是否有效
    print("det(R_raw):", np.linalg.det(R_raw))
    print(
        "R_raw 正交性誤差:",
        np.linalg.norm(R_raw.T @ R_raw - np.eye(3))
    )

    # 外參反向轉換：Side -> Front
    R_reverse = R_raw.T
    T_reverse = -R_raw.T @ T_raw

    print("R_reverse:\n", R_reverse)
    print("T_reverse:\n", T_reverse.reshape(3))
    print("==================================\n")

    # 先用 Front -> Side 的外參，計算每組影像的雙目重投影誤差
    per_pair_errors = []

    for i, name in enumerate(pair_names):
        ok_f, rvec_f, tvec_f = cv2.solvePnP(
            objpoints[i], imgpoints_f[i], K1_out, D1_out
        )
        ok_s, rvec_s, tvec_s = cv2.solvePnP(
            objpoints[i], imgpoints_s[i], K2_out, D2_out
        )

        if not ok_f or not ok_s:
            continue

        err_f = reprojection_error(
            objpoints[i], imgpoints_f[i], K1_out, D1_out,
            rvec_f, tvec_f
        )
        err_s = reprojection_error(
            objpoints[i], imgpoints_s[i], K2_out, D2_out,
            rvec_s, tvec_s
        )

        # 將 Front 相機座標中的棋盤格點轉換到 Side 相機座標，
        # 再投影到 Side 影像。這裡的 R_raw、T_raw 是
        # stereoCalibrate 回傳的 Front -> Side 外參。
        R_board_f, _ = cv2.Rodrigues(rvec_f)
        board_in_side = (
            R_raw @ (R_board_f @ objpoints[i].T + tvec_f)
            + T_raw
        ).T

        projected_s, _ = cv2.projectPoints(
            board_in_side,
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            K2_out,
            D2_out
        )

        # 將兩個點集統一轉為 float32 且保證形狀一致 (N, 2)
        pts_actual = np.array(imgpoints_s[i], dtype=np.float32).reshape(-1, 2)
        pts_proj = np.array(projected_s, dtype=np.float32).reshape(-1, 2)

        # 使用實際 Side 角點與經相對外參投影的點比較
        err_stereo = float(
            cv2.norm(
                pts_actual, pts_proj, cv2.NORM_L2
            ) / np.sqrt(len(objpoints[i]))
        )
        if name in {
            "chessboard_image1.png",
            "chessboard_image15.png",
            "chessboard_image29.png",
        }:
            print(
                f"[誤差檢查] {name}: "
                f"stereo={err_stereo:.2f}px, "
                f"front={err_f:.2f}px, "
                f"side={err_s:.2f}px"
            )

        per_pair_errors.append({
            "filename": name,
            "front_error_px": err_f,
            "side_error_px": err_s,
            "stereo_error_px": err_stereo
        })

        # 診斷每組影像的 Side 角點排列，不修改正式角點
        candidates = generate_corner_order_candidates(imgpoints_s[i])

        candidate_errors = []

        for candidate_idx, candidate in enumerate(candidates):
            candidate_pts = candidate.reshape(-1, 2)

            candidate_error = float(
                cv2.norm(
                    candidate_pts,
                    pts_proj,
                    cv2.NORM_L2
                ) / np.sqrt(len(objpoints[i]))
            )

            candidate_errors.append(
                (candidate_idx, candidate_error)
            )

        best_idx, best_error = min(
            candidate_errors,
            key=lambda item: item[1]
        )

        print(
            f"[角點排列診斷] {name}: "
            f"原始={err_stereo:.2f}px, "
            f"最佳 Candidate={best_idx}, "
            f"最佳誤差={best_error:.2f}px"
        )

    per_pair_errors.sort(
        key=lambda x: x["stereo_error_px"], reverse=True
    )
    
    print(f"\nStereo RMS：{rms:.4f} px")

    if rms > MAX_STEREO_RMS:
        print(
            f"❌ 雙目 RMS = {rms:.2f}px，誤差過大，"
            "這組外參僅供診斷，不會輸出。"
        )

    print("\n雙目誤差最大的 15 組：")

    for item in per_pair_errors[:15]:
        print(
            f"{item['filename']}: "
            f"stereo={item['stereo_error_px']:.2f}px, "
            f"front={item['front_error_px']:.2f}px, "
            f"side={item['side_error_px']:.2f}px"
        )
    
    if rms > MAX_STEREO_RMS:
        print("⛔ RMS 超標，停止輸出外參 JSON。")
        return


    # 反轉外參，讓 Side 為座標原點、轉換至 Front
    R_ui = R_raw.T
    T_ui = -R_raw.T @ T_raw

    output_dir = os.path.join(
        base_dir, f"result_stereo_{dir_suffix}"
    )
    os.makedirs(output_dir, exist_ok=True)

    output_json_path = os.path.join(
        output_dir, "stereo_camera_parameters_pnp.json"
    )

    stereo_params = {
        "K1": K1_out.tolist(),
        "D1": D1_out.flatten().tolist(),
        "K2": K2_out.tolist(),
        "D2": D2_out.flatten().tolist(),
        "R": R_ui.tolist(),
        "T": T_ui.tolist(),
        "rms_error": float(rms),
        "unit": "mm",
        "image_size_front": list(image_size_f),
        "image_size_side": list(image_size_s),
        "valid_pairs": pair_names,
        "per_pair_errors": per_pair_errors
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(stereo_params, f, indent=4, ensure_ascii=False)

    print("\n外參已輸出：", output_json_path)
    print("R Side -> Front:\n", R_ui)
    print("T Side -> Front (mm):\n", T_ui.ravel())


if __name__ == "__main__":
    run_stereo_calibration()
