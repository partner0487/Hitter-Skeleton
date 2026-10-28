"""雙相機棋盤格校正入口。

    python calibration/calibration.py                 開啟校正精靈（內參 → 外參）
    python calibration/calibration.py --single        開啟單台相機內參視窗
    python calibration/calibration.py --calculate [SESSION] [--pattern 7x7]
                                                      用已拍好的照片離線重算，不開相機、不開視窗

--calculate 讀 Db/Calibration/sessions/<SESSION>/：
  內參：正面用 intrinsic/images_SN<正面序號>/，沒有就用 cf/；側面同理（images_SN<側面序號>/ 或 cs/）
  外參：cf/ 與 cs/ 的同名照片
相機序號、焦距、格子邊長讀該 session 的 session.json；棋盤格規格也是，--pattern 可覆寫。
結果寫回該 session；Stereo RMS ≤ 3 px 時與精靈一樣直接採用：
該日期起的錄影做 3D 分析時使用這次校正，若是日期最新的也更新 Db/Calibration/*.json。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_PATTERN = (8, 7)
DEFAULT_SQUARE_MM = 100.0
DEFAULT_FOCAL_MM = {"front": 35.0, "side": 8.0}


def parse_pattern(text):
    """'7x7'、'7*7'、'7,7' 皆可；回傳 (寬, 高) 內角點數"""
    for sep in ("x", "X", "*", ","):
        if sep in text:
            cols, rows = text.split(sep, 1)
            return int(cols), int(rows)
    raise argparse.ArgumentTypeError(f"棋盤格內角點格式應為 寬x高，例如 7x7：{text}")


def find_session(name):
    """名稱（sessions/ 底下）或路徑；未指定時取最新一個有 cf/ 與 cs/ 照片的 session"""
    import storage
    sessions = Path(storage.SESSIONS_DIR)
    if name:
        path = Path(name)
        for candidate in (path, sessions / name):
            if candidate.is_dir():
                return candidate.resolve()
        raise SystemExit(f"找不到 session：{name}（請給 {sessions} 底下的資料夾名稱或完整路徑）")
    found = sorted((p for p in sessions.iterdir()
                    if p.is_dir() and (p / "cf").is_dir() and (p / "cs").is_dir()),
                   key=lambda p: p.name) if sessions.is_dir() else []
    if not found:
        raise SystemExit(f"{sessions} 底下沒有含 cf/、cs/ 照片的 session")
    return found[-1].resolve()


def calculate_intrinsic(session, role, sn, pattern, square, focal, lock):
    """用一台相機的照片算內參，回傳 intrinsics.json 的 entry；失敗回傳 None"""
    import intrinsic
    folder = session / "intrinsic" / f"images_SN{sn}"
    if not (folder.is_dir() and intrinsic.list_images(folder)):
        folder = session / ("cf" if role == "front" else "cs")
    files = intrinsic.list_images(folder)
    title = "正面" if role == "front" else "側面"
    print(f"\n== {title} SN{sn}｜{focal or '未填'} mm{'（長焦鎖定）' if lock else ''}｜{folder.relative_to(session)}/ {len(files)} 張")
    loaded = intrinsic.load_photo_samples(files, pattern)
    points, paths, size = loaded["points"], loaded["paths"], loaded["image_size"]
    print(f"   有效 {len(points)} 張，略過 {len(loaded['skipped'])} 張"
          + (f"（例：{loaded['skipped'][0][0]} {loaded['skipped'][0][1]}）" if loaded["skipped"] else ""))
    if len(points) < intrinsic.MIN_CALIBRATE_SAMPLES:
        print(f"   ✘ 有效照片少於 {intrinsic.MIN_CALIBRATE_SAMPLES} 張，無法計算；請確認 --pattern 是否正確")
        return None

    objpoints = [intrinsic.object_points(pattern, square) for _ in points]
    fit, verdict, problems, objpoints, points, paths, removed = intrinsic.calibrate_with_outlier_removal(
        objpoints, points, size, focal, lock, paths)
    if removed:
        print(f"   已剔除誤差異常的 {len(removed)} 張：{'、'.join(removed)}")
    print("   " + intrinsic.report_text(sn, fit, verdict, problems).replace("\n", "\n   "))

    photos = [Path(p).relative_to(session).as_posix() for p in paths]
    intrinsic.save_npz(session / "intrinsic" / f"calib_SN{sn}.npz", sn, fit, size, pattern, square,
                       objpoints, points, photos)
    return intrinsic.make_entry(fit, verdict, size, pattern, square, focal, lock, photos)


def calculate(args):
    import intrinsic
    from stereo import MAX_STEREO_RMS, estimate_chessboard_stereo, publish_stereo, save_json
    import storage

    session = find_session(args.session)
    info_path = session / "session.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    pattern = args.pattern or tuple(info.get("pattern") or DEFAULT_PATTERN)
    square = info.get("square_size_mm") or DEFAULT_SQUARE_MM
    roles = (str(info.get("front") or "front"), str(info.get("side") or "side"))
    focal_info = info.get("lens_focal_mm") or {}
    focals = {role: focal_info.get(role, DEFAULT_FOCAL_MM[role]) for role in ("front", "side")}
    print(f"Session：{session}")
    print(f"棋盤格：內角點 {pattern[0]}x{pattern[1]}，每格 {square:g} mm｜正面 SN{roles[0]}、側面 SN{roles[1]}")

    entries = {}
    for role, sn in zip(("front", "side"), roles):
        focal = focals[role] or 0.0
        entry = calculate_intrinsic(session, role, sn, pattern, square, focal, intrinsic.is_telephoto(focal))
        if entry is None:
            return 1
        entries[sn] = entry
    save_json(session / "intrinsics.json", entries)

    print("\n== 外參（cf/、cs/ 同名照片）")
    try:
        report = estimate_chessboard_stereo(session, roles, entries, pattern, square)
    except (ValueError, KeyError) as exc:
        print(f"   ✘ 外參計算失敗：{exc}")
        return 1
    save_json(session / "stereo.json", report)
    fit = report["fit"]
    checks = report["held_out_board_checks"]
    print(f"   Stereo RMS {fit['stereo_rms_px']:.3f} px｜基線 {report['baseline_mm'] / 1000:.2f} m｜"
          f"訓練 {len(report['training_pairs'])}、驗證 {len(checks)}、剔除 {len(report['rejected_pairs'])} 組")
    print(f"   驗證組格子邊長誤差中位數 {sorted(c['square_rmse_mm'] for c in checks)[len(checks) // 2]:.2f} mm")
    if report["rejected_pairs"]:
        print(f"   姿態不一致而剔除：{'、'.join(report['rejected_pairs'])}")
    for warning in report["warnings"]:
        print(f"   REVIEW：{warning}")
    print(f"\n已寫入 {session / 'intrinsics.json'}、{session / 'stereo.json'}")

    if fit["stereo_rms_px"] > MAX_STEREO_RMS:
        print(f"✘ Stereo RMS 超過 {MAX_STEREO_RMS} px，不採用；3D 分析不會使用這次結果")
        return 1
    _, is_current = publish_stereo(storage.CALIB_DIR, session, roles, entries, report)
    day = session.name[:8]
    print(f"✔ 已採用：{day[:4]}-{day[4:6]}-{day[6:]} 起的錄影會使用這次校正（直到下一次較新的校正）")
    if is_current:
        print(f"✔ 也是日期最新的校正，已更新 {storage.STEREO_JSON}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="雙相機棋盤格校正：不加參數開啟精靈",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="範例：\n"
               "  python calibration/calibration.py --calculate\n"
               "  python calibration/calibration.py --calculate 20261019_legacy\n"
               "  python calibration/calibration.py --calculate 20261019_legacy --pattern 7x7")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--single", action="store_true", help="開啟單台相機內參視窗")
    mode.add_argument("--calculate", nargs="?", const="", metavar="SESSION", dest="session",
                      help="離線重算；SESSION 為 sessions/ 底下的資料夾名稱或路徑，省略則用最新的")
    parser.add_argument("--pattern", type=parse_pattern,
                        help="棋盤格內角點 寬x高，例如 7x7（預設讀 session.json，否則 8x7）")
    args = parser.parse_args(argv)

    if args.session is not None:
        return calculate(args)
    if args.pattern:
        parser.error("--pattern 只能搭配 --calculate 使用")
    if args.single:
        import intrinsic_window
        return intrinsic_window.main()
    import wizard
    return wizard.main()


if __name__ == "__main__":
    sys.exit(main())
