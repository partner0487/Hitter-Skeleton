"""單台相機內參：角點偵測、calibrateCamera、品質判斷。

不含 Qt 與相機程式，內參視窗（intrinsic_window.py）、精靈（wizard.py）與
指令列離線重算（calibration.py --calculate）共用同一套計算與門檻。
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# 品質判斷門檻
RMS_GOOD = 0.5               # RMS 低於此值為良好
RMS_MAX = 1.0                # RMS 高於此值判定不合格
MIN_SAMPLES = 15             # 建議樣本數
MIN_CALIBRATE_SAMPLES = 10   # 少於此數不計算
MIN_TILT_DEG = 20.0          # 至少要有一組傾斜超過此角度
DUPLICATE_SHIFT_PX = 5.0     # 與上一組平均位移小於此值視為重複
PIXEL_SIZE_MM = 0.00586      # GS3-U3-23S6C 像素大小 5.86 µm，用來把鏡頭焦距換算成 fx
TELEPHOTO_MM = 25.0          # 焦距達此值視為長焦，鎖住光心與高階畸變
FX_TOLERANCE = 0.15          # fx 與焦距推算值相差超過 15% 就警告
AXIS_MIN_TILT_DEG = 10.0     # 傾斜超過此角度的樣本才拿來看「往哪個方向轉」
AXIS_CONCENTRATION = 0.8     # 轉軸方向集中度超過此值（1＝全部同一軸）就警告
REGION_NAMES = ["左上", "上", "右上", "左", "中", "右", "左下", "下", "右下"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class IntrinsicFit:
    rms: float
    K: np.ndarray
    dist: np.ndarray
    per_image_rms: np.ndarray
    tilts: np.ndarray            # 每組板子與光軸的夾角（度）
    tilt_axes: list = field(default_factory=list)  # 每組板子的傾斜轉軸方向（度，影像平面內）
    flags: int = 0


def object_points(pattern, square_mm):
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_mm
    return objp


def read_image(path):
    """讀取影像為 BGR；imdecode/fromfile 支援中文 Windows 路徑"""
    frame = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("無法解碼影像")
    return frame


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def find_corners(frame, pattern):
    """全解析度偵測棋盤格並做亞像素修正，偵測不到回傳 None"""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern, None)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)


def mean_shift(corners, previous):
    return float(np.mean(np.linalg.norm(corners.reshape(-1, 2) - previous.reshape(-1, 2), axis=1)))


def load_photo_samples(files, pattern, on_progress=None):
    """逐張偵測照片中的棋盤格，略過讀不到、尺寸不同、偵測失敗與重複的照片。

    on_progress(index, total, name) 回傳 False 代表取消，此時整個函式回傳 None。
    回傳 dict：points、paths、image_size、preview（最後一張有效照片）、skipped [(檔名, 原因)]
    """
    points, paths, skipped, size, preview = [], [], [], None, None
    for index, path in enumerate(files):
        if on_progress is not None and on_progress(index, len(files), path.name) is False:
            return None
        try:
            frame = read_image(path)
            current_size = (frame.shape[1], frame.shape[0])
            if size is not None and current_size != size:
                raise ValueError(f"影像尺寸 {current_size} 與本批 {size} 不同")
            corners = find_corners(frame, pattern)
            if corners is None:
                raise ValueError("未完整偵測到棋盤格")
            if points:
                shift = mean_shift(corners, points[-1])
                if shift < DUPLICATE_SHIFT_PX:
                    raise ValueError(f"與上一張有效照片幾乎相同（位移 {shift:.1f} px）")
            points.append(corners)
            paths.append(str(Path(path).resolve()))
            size, preview = current_size, frame
        except (OSError, ValueError, cv2.error) as exc:
            skipped.append((path.name, str(exc)))
    return {"points": points, "paths": paths, "image_size": size,
            "preview": preview, "skipped": skipped}


def coverage(imgpoints, image_size):
    """把畫面切成 3x3 九宮格，回傳 (已覆蓋區數, 未覆蓋的區名)"""
    if image_size is None or not imgpoints:
        return 0, list(REGION_NAMES)
    w, h = image_size
    pts = np.vstack([c.reshape(-1, 2) for c in imgpoints])
    col = np.clip((pts[:, 0] / w * 3).astype(int), 0, 2)
    row = np.clip((pts[:, 1] / h * 3).astype(int), 0, 2)
    hit = set((row * 3 + col).tolist())
    missing = [name for i, name in enumerate(REGION_NAMES) if i not in hit]
    return len(hit), missing


def is_telephoto(focal_mm):
    return bool(focal_mm) and focal_mm >= TELEPHOTO_MM


def calibration_guess(image_size, focal_mm=0.0, lock=False):
    """依鏡頭焦距決定 calibrateCamera 的初值與 flags，回傳 (K 初值或 None, flags)"""
    flags = 0
    init_K = None
    if focal_mm and focal_mm > 0:
        # 有焦距就用 焦距/像素大小 當 fx 初值、光心放畫面中心；
        # 切向畸變與 k3 在球場距離下估不準，固定為 0 避免邊緣發散
        w, h = image_size
        f_px = focal_mm / PIXEL_SIZE_MM
        init_K = np.array([[f_px, 0.0, w / 2.0],
                           [0.0, f_px, h / 2.0],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
                 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    if lock:
        # 長焦鏡頭視角窄，照片裡幾乎沒有光心與高階畸變的資訊，放著不鎖會發散
        # （2026-10-08 正面 50 mm 的 k2 = 33.5、舊版 35 mm 的 k3 = 274），只留 k1
        flags |= (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
                  | cv2.CALIB_ZERO_TANGENT_DIST)
    return init_K, flags


def tilt_and_axis(rvec):
    """板子的傾斜角（板子法向量與光軸夾角）與傾斜轉軸方向（影像平面內），單位度"""
    R, _ = cv2.Rodrigues(rvec)
    tilt = float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))
    # 板子是繞哪個軸轉離正對相機：轉軸 = 光軸 × 板子法向量（落在影像平面內）
    normal = R[:, 2] * np.sign(R[2, 2] or 1.0)
    return tilt, float(np.degrees(np.arctan2(normal[0], -normal[1])))


def estimate_pose(corners, pattern, square_mm, image_size, focal_mm):
    """拍攝當下還沒有內參，用鏡頭焦距推算的 K 估板子姿態；回傳 (傾斜角, 轉軸, 板子中心距離 mm)，沒填焦距回傳 None"""
    init_K, _ = calibration_guess(image_size, focal_mm)
    if init_K is None:
        return None
    objp = object_points(pattern, square_mm)
    ok, rvec, tvec = cv2.solvePnP(objp, corners, init_K, None)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    center = R @ objp.mean(axis=0) + tvec.ravel()
    return (*tilt_and_axis(rvec), float(np.linalg.norm(center)))


def estimate_tilt(corners, pattern, square_mm, image_size, focal_mm):
    """同 estimate_pose，只回傳 (傾斜角, 轉軸)"""
    pose = estimate_pose(corners, pattern, square_mm, image_size, focal_mm)
    return None if pose is None else pose[:2]


def calibrate(objpoints, imgpoints, image_size, focal_mm=0.0, lock=False):
    """執行校正並計算每組樣本的誤差、傾斜角與傾斜轉軸"""
    init_K, flags = calibration_guess(image_size, focal_mm, lock)
    init_dist = None if init_K is None else np.zeros((5, 1), dtype=np.float64)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, init_K, init_dist, flags=flags)
    per_image_rms, tilts, tilt_axes = [], [], []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        d = np.linalg.norm(imgpoints[i].reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        per_image_rms.append(float(np.sqrt(np.mean(d ** 2))))
        tilt, axis = tilt_and_axis(rvecs[i])
        tilts.append(tilt)
        tilt_axes.append(axis)
    return IntrinsicFit(rms=float(rms), K=K, dist=dist, per_image_rms=np.array(per_image_rms),
                        tilts=np.array(tilts), tilt_axes=tilt_axes, flags=int(flags))


def axis_concentration(tilt_axes, tilts):
    """傾斜轉軸方向的集中度（0～1）。轉軸沒有正負之分，所以角度先乘 2 再取平均向量長度；
    只朝同一方向轉（轉軸互相平行）時接近 1，這時焦距與距離分不開，fx 會算不準。"""
    axes = np.radians([a for a, t in zip(tilt_axes, tilts) if t >= AXIS_MIN_TILT_DEG])
    if len(axes) < 3:
        return None
    return float(np.hypot(np.mean(np.cos(2 * axes)), np.mean(np.sin(2 * axes))))


def sample_problems(imgpoints, image_size, tilts, tilt_axes):
    """拍攝時就能檢查的問題：畫面覆蓋、傾斜角度與方向、張數。精靈拍攝中即時顯示，校正後 assess 也用同一套"""
    problems = []
    covered, missing = coverage(imgpoints, image_size)
    if missing:
        problems.append(f"畫面覆蓋 {covered}/9 區，缺：{'、'.join(missing)}（邊角沒拍到，畸變會校不準）")
    concentration = axis_concentration(tilt_axes, tilts)
    if concentration is not None and concentration > AXIS_CONCENTRATION:
        problems.append(f"板子幾乎都朝同一個方向轉（集中度 {concentration:.2f}），fx 可能算不準："
                        "請讓板子上下、左右、斜向都傾斜（不要為了讓另一台相機看到而只朝一個方向轉）")
    if len(tilts) and max(tilts) < MIN_TILT_DEG:
        problems.append(f"最大傾斜僅 {max(tilts):.0f}°，請加入傾斜 {MIN_TILT_DEG:.0f}° 以上的樣本")
    if len(imgpoints) < MIN_SAMPLES:
        problems.append(f"樣本僅 {len(imgpoints)} 組，建議至少 {MIN_SAMPLES} 組")
    return problems


def assess(fit, imgpoints, image_size, focal_mm=0.0):
    """依門檻判斷品質，回傳 (結論, 問題清單, 異常樣本索引)"""
    problems = []
    rms, K, per_image_rms, tilts = fit.rms, fit.K, fit.per_image_rms, fit.tilts
    # RMS 小不代表 fx 對：長焦時 fx 與距離可以互相抵消，RMS 照樣很小，所以另外用鏡頭焦距檢查
    if focal_mm and focal_mm > 0:
        expected = focal_mm / PIXEL_SIZE_MM
        ratio = K[0, 0] / expected
        if abs(ratio - 1) > FX_TOLERANCE:
            problems.append(f"fx={K[0, 0]:.0f}，與 {focal_mm:g} mm 鏡頭推算的 {expected:.0f} 相差 "
                            f"{(ratio - 1) * 100:+.0f}%：請確認鏡頭焦距，並讓板子往上下、左右不同方向傾斜 30° 以上（只朝同一方向轉會讓 fx 算不準）")
    bad = [i for i, e in enumerate(per_image_rms) if e > RMS_MAX and e > 2 * rms]

    if rms > RMS_MAX:
        verdict = "不合格，建議重拍"
        problems.append(f"整體誤差 {rms:.3f} px 超過 {RMS_MAX} px")
    elif rms > RMS_GOOD:
        verdict = "可用，但建議補拍改善"
        problems.append(f"整體誤差 {rms:.3f} px 介於 {RMS_GOOD}～{RMS_MAX} px")
    else:
        verdict = "良好"

    if bad:
        problems.append(f"{len(bad)} 組樣本誤差異常（第 {', '.join(str(i + 1) for i in bad)} 組），"
                        "可能是模糊或晃動")
    problems += sample_problems(imgpoints, image_size, tilts, fit.tilt_axes)

    if verdict == "良好" and problems:
        verdict = "良好，但有改善空間"
    return verdict, problems, bad


def report_text(sn, fit, verdict, problems):
    text = (
        f"校正完成 SN={sn}｜判定：{verdict}\n"
        f"  重投影誤差 RMS: {fit.rms:.4f} px  (< {RMS_GOOD} 良好，> {RMS_MAX} 須重拍)\n"
        f"  單組誤差範圍: {fit.per_image_rms.min():.3f} ~ {fit.per_image_rms.max():.3f} px\n"
        f"  板子傾斜範圍: {fit.tilts.min():.0f}° ~ {fit.tilts.max():.0f}°\n"
        f"  fx={fit.K[0, 0]:.2f}  fy={fit.K[1, 1]:.2f}\n"
        f"  cx={fit.K[0, 2]:.2f}  cy={fit.K[1, 2]:.2f}\n"
        f"  畸變係數: {np.round(fit.dist.ravel(), 6)}"
    )
    if problems:
        text += "\n  待改善：\n" + "\n".join(f"   - {p}" for p in problems)
    return text


def make_entry(fit, verdict, image_size, pattern, square_mm, focal_mm, lock, source_images):
    """intrinsics.json／intrinsics_all.json 中一台相機的內參資料"""
    return {
        "intrinsic_matrix": fit.K.tolist(),
        "distortion_coefficients": fit.dist.tolist(),
        "fx": float(fit.K[0, 0]), "fy": float(fit.K[1, 1]),
        "cx": float(fit.K[0, 2]), "cy": float(fit.K[1, 2]),
        "image_size": list(image_size),
        "rms": float(fit.rms),
        "mean_reproj_err": float(np.mean(fit.per_image_rms)),
        "verdict": verdict,
        "pattern": [int(pattern[0]), int(pattern[1])],
        "square_size_mm": float(square_mm),
        "sample_count": len(fit.per_image_rms),
        "distortion_locked": bool(lock),
        "lens_focal_mm": focal_mm or None,
        "calibration_flags": int(fit.flags),
        "source_images": list(source_images),
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_npz(path, sn, fit, image_size, pattern, square_mm, objpoints, imgpoints, source_images):
    """單台完整校正資料（含角點），可離線重算或比較"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), mtx=fit.K, dist=fit.dist, image_size=np.array(image_size),
             rms=fit.rms, mean_reproj_err=float(np.mean(fit.per_image_rms)),
             per_image_rms=fit.per_image_rms, tilt_deg=fit.tilts,
             pattern=np.array(pattern), square_size=square_mm, serial_number=sn,
             sample_count=len(fit.per_image_rms),
             objpoints=np.asarray(objpoints), imgpoints=np.asarray(imgpoints),
             source_images=np.asarray(source_images, dtype=str))


def calibrate_with_outlier_removal(objpoints, imgpoints, image_size, focal_mm, lock, paths):
    """校正後若有異常樣本且剔除後仍夠多，剔除再算一次（與視窗按「是」相同）。

    回傳 (fit, verdict, problems, objpoints, imgpoints, paths, removed_names)
    """
    fit = calibrate(objpoints, imgpoints, image_size, focal_mm, lock)
    verdict, problems, bad = assess(fit, imgpoints, image_size, focal_mm)
    removed = []
    if bad and len(imgpoints) - len(bad) >= MIN_CALIBRATE_SAMPLES:
        removed = [Path(paths[i]).name for i in bad]
        keep = [i for i in range(len(imgpoints)) if i not in set(bad)]
        objpoints = [objpoints[i] for i in keep]
        imgpoints = [imgpoints[i] for i in keep]
        paths = [paths[i] for i in keep]
        fit = calibrate(objpoints, imgpoints, image_size, focal_mm, lock)
        verdict, problems, _ = assess(fit, imgpoints, image_size, focal_mm)
    return fit, verdict, problems, objpoints, imgpoints, paths, removed


def session_photos(session, role, sn):
    """一台相機在 session 中的內參照片：只有這台看到的 intrinsic/images_SN<序號>/，加上雙機配對的 cf/ 或 cs/"""
    session = Path(session)
    files = []
    for folder in (session / "intrinsic" / f"images_SN{sn}", session / ("cf" if role == "front" else "cs")):
        if folder.is_dir():
            files += list_images(folder)
    return files


def calibrate_session_camera(session, role, sn, pattern, square_mm, focal_mm, log=print):
    """用 session 中一台相機的照片算內參並存 npz，回傳 (intrinsics.json 的 entry, fit)；照片不足回傳 (None, None)"""
    session = Path(session)
    lock = is_telephoto(focal_mm)
    files = session_photos(session, role, sn)
    title = "正面" if role == "front" else "側面"
    log(f"== {title} SN{sn}｜{focal_mm or '未填'} mm{'（長焦鎖定）' if lock else ''}｜{len(files)} 張照片")
    loaded = load_photo_samples(files, pattern)
    points, paths, size = loaded["points"], loaded["paths"], loaded["image_size"]
    log(f"   有效 {len(points)} 張，略過 {len(loaded['skipped'])} 張"
        + (f"（例：{loaded['skipped'][0][0]} {loaded['skipped'][0][1]}）" if loaded["skipped"] else ""))
    if len(points) < MIN_CALIBRATE_SAMPLES:
        log(f"   ✘ 有效照片少於 {MIN_CALIBRATE_SAMPLES} 張，無法計算；請確認棋盤格規格是否正確")
        return None, None

    objpoints = [object_points(pattern, square_mm) for _ in points]
    fit, verdict, problems, objpoints, points, paths, removed = calibrate_with_outlier_removal(
        objpoints, points, size, focal_mm, lock, paths)
    if removed:
        log(f"   已剔除誤差異常的 {len(removed)} 張：{'、'.join(removed)}")
    log("   " + report_text(sn, fit, verdict, problems).replace("\n", "\n   "))

    photos = [Path(p).relative_to(session.resolve()).as_posix() for p in paths]
    save_npz(session / "intrinsic" / f"calib_SN{sn}.npz", sn, fit, size, pattern, square_mm,
             objpoints, points, photos)
    return make_entry(fit, verdict, size, pattern, square_mm, focal_mm, lock, photos), fit
