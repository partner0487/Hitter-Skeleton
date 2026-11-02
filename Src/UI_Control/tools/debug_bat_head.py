"""輸出 2D 擊球點判斷用的棒頭位置與速度，方便 debug。

與 2D 分析相同：讀錄影資料夾的 bat_side.json／person_side.json／ball_side.json，
用 utils/phase_detect2d.py 的 PhaseDetector 算棒頭軌跡（bat_head_track）與各階段。

每個資料夾輸出：
  debug_bat_head.csv   每幀棒頭位置、遮罩兩端、速度（px/幀，有 frame_times.csv 時另算 px/s）
  debug_bat_head.png   棒頭 x、y 與速度曲線，標出離地、著地、擊球（原始最高速幀與 + IMPACT_OFFSET 後）、偵測到球的幀
  debug_bat_head.mp4   （加 --video）側面影片畫上遮罩兩端（青）、手腕中點（綠十字）與棒頭
                       （紅圈；灰圈為不合理跳動、已排除）

用法（在任何目錄）：
  python Src/UI_Control/tools/debug_bat_head.py "Db/WeiShu/2026-10-29/T座/WeiShu_H01" "Db/WeiShu/2026-10-29/T座/WeiShu_H02"
  python Src/UI_Control/tools/debug_bat_head.py <資料夾> --left    # 左打（預設右打）
  python Src/UI_Control/tools/debug_bat_head.py <資料夾> --video   # 另存標註影片
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

UI_CONTROL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UI_CONTROL))

from utils.phase_detect2d import PhaseDetector  # noqa: E402

# 與 analysis2d_widget.detectPhases 相同的參數
DETECTOR_KWARGS = dict(th_lift=5.0, th_land=5.0, land_hold=3)
IMPACT_OFFSET = PhaseDetector.IMPACT_OFFSET  # detect_phases 對最高速幀加的偏移


def read_records(folder, name):
    """與 cv_control_dual.DualJsonLoader 相同的讀法"""
    path = folder / name
    return pd.read_json(path, orient="records") if path.exists() else pd.DataFrame()


def ball_frames(ball_df):
    if ball_df.empty or "frame_number" not in ball_df:
        return []
    return sorted(int(f) for f in ball_df["frame_number"])


def analyse(folder, batter_index):
    person = read_records(folder, "person_side.json")
    bat = read_records(folder, "bat_side.json")
    ball = read_records(folder, "ball_side.json")
    detector = PhaseDetector(person_df=person, batter_index=batter_index, ball_df=ball, bat_df=bat,
                             **DETECTOR_KWARGS)
    phases = detector.detect_phases()
    track = detector.bat_head_track()

    times_path = folder / "frame_times.csv"
    if not track.empty and times_path.exists():
        times = pd.read_csv(times_path).set_index("frame")["time_sec"]
        track["time_sec"] = track["frame"].map(times)
        dt = track["time_sec"].diff()
        gap = track["frame"].diff()
        # 與 px/幀 相同：只在間隔 ≤ 3 幀時計算
        step = track["speed_px_per_frame"] * gap
        track["speed_px_per_s"] = (step / dt).where(track["speed_px_per_frame"].notna())
    return phases, track, ball_frames(ball), len(bat)


def write_plot(path, folder, track, phases, balls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(track["frame"], track["head_x"], ".-", label="head x")
    axes[1].plot(track["frame"], track["head_y"], ".-", label="head y", color="tab:orange")
    rejected = track[track["rejected"]]
    axes[0].plot(rejected["frame"], rejected["head_x"], "rx", markersize=9, label="rejected (jump)")
    axes[1].plot(rejected["frame"], rejected["head_y"], "rx", markersize=9, label="rejected (jump)")
    axes[1].invert_yaxis()
    axes[2].plot(track["frame"], track["speed_px_per_frame"], ".-", label="speed (px/frame)", color="tab:green")
    raw_impact = phases.F_impact - IMPACT_OFFSET if phases.F_impact is not None else None
    marks = [("liftoff", phases.F_liftoff, "tab:blue"), ("landing", phases.F_landing, "tab:purple"),
             ("max speed", raw_impact, "tab:red"), (f"impact (+{IMPACT_OFFSET})", phases.F_impact, "black")]
    for ax in axes:
        for name, frame, color in marks:
            if frame is not None:
                ax.axvline(frame, color=color, linestyle="--", linewidth=1, label=name)
        for f in balls:
            ax.axvline(f, color="gold", alpha=0.25, linewidth=3)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].legend(loc="upper left", fontsize=8)
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].set_xlabel("frame（黃色直條：偵測到球的幀）")
    fig.suptitle(f"{folder.name}  bat head (2D side view)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_video(path, folder, track, phases):
    import cv2

    videos = sorted(glob.glob(str(folder / "CS_*.mp4")))
    if not videos:
        print("   找不到 CS_*.mp4，略過影片")
        return
    cap = cv2.VideoCapture(videos[0])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rows = track.set_index("frame")
    raw_impact = phases.F_impact - IMPACT_OFFSET if phases.F_impact is not None else None
    frame_num = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if frame_num in rows.index:
            r = rows.loc[frame_num]
            if r.get("source") == "mask":
                cv2.line(img, (int(r.end1_x), int(r.end1_y)), (int(r.end2_x), int(r.end2_y)), (255, 255, 0), 2)
                cv2.circle(img, (int(r.end1_x), int(r.end1_y)), 6, (255, 255, 0), -1)
                cv2.circle(img, (int(r.end2_x), int(r.end2_y)), 6, (255, 255, 0), -1)
            if pd.notna(r.get("wrist_x")):
                cv2.drawMarker(img, (int(r.wrist_x), int(r.wrist_y)), (0, 255, 0), cv2.MARKER_CROSS, 24, 3)
            cv2.circle(img, (int(r.head_x), int(r.head_y)), 10, (128, 128, 128) if r.rejected else (0, 0, 255), 3)
            speed = r.speed_px_per_frame
            text = f"head ({r.head_x:.0f},{r.head_y:.0f})  speed {speed:.1f} px/f" if pd.notna(speed) else \
                f"head ({r.head_x:.0f},{r.head_y:.0f})"
            if r.rejected:
                text += f"  REJECTED (jump {r.jump_px:.0f} px)"
            cv2.putText(img, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        label = f"frame {frame_num}"
        for name, frame in (("LIFTOFF", phases.F_liftoff), ("LANDING", phases.F_landing),
                            ("MAX SPEED", raw_impact), ("IMPACT", phases.F_impact)):
            if frame == frame_num:
                label += f"  <{name}>"
        cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        out.write(img)
        frame_num += 1
    cap.release()
    out.release()


def main():
    parser = argparse.ArgumentParser(description="輸出 2D 棒頭位置與速度（debug 擊球點）")
    parser.add_argument("folders", nargs="+", help="錄影資料夾（含 bat_side.json）")
    parser.add_argument("--left", action="store_true", help="左打（預設右打），影響離地／著地用哪一腳")
    parser.add_argument("--video", action="store_true", help="另存標註棒頭的側面影片")
    args = parser.parse_args()

    for name in args.folders:
        folder = Path(name).resolve()
        print(f"\n== {folder}")
        if not (folder / "bat_side.json").exists():
            print("   沒有 bat_side.json，略過")
            continue
        phases, track, balls, bat_rows = analyse(folder, 1 if args.left else 0)
        print(f"   bat_side.json {bat_rows} 筆，取得棒頭 {len(track)} 幀")
        print(f"   離地 {phases.F_liftoff}｜著地 {phases.F_landing}｜擊球 {phases.F_impact}"
              f"（最高速幀 + {IMPACT_OFFSET}）｜偵測到球的幀：{balls[:5]}{' …' if len(balls) > 5 else ''}")
        if track.empty:
            print("   沒有棒頭資料")
            continue
        rejected = track.loc[track["rejected"], "frame"].tolist()
        print(f"   沿用上一幀的遮罩（沒偵測到）：{track.loc[track['repeated'], 'frame'].tolist() or '無'}")
        print(f"   排除不合理跳動（> {PhaseDetector.JUMP_MAX_PX:g} px）：{rejected or '無'}")
        top = track.nlargest(5, "speed_px_per_frame")[["frame", "head_x", "head_y", "speed_px_per_frame"]]
        print("   速度前 5 名：")
        print(top.to_string(index=False, float_format=lambda v: f"{v:.1f}").replace("\n", "\n   ").join(["   ", ""]))
        track.to_csv(folder / "debug_bat_head.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
        write_plot(folder / "debug_bat_head.png", folder, track, phases, balls)
        print(f"   已輸出 debug_bat_head.csv、debug_bat_head.png")
        if args.video:
            write_video(folder / "debug_bat_head.mp4", folder, track, phases)
            print(f"   已輸出 debug_bat_head.mp4")


if __name__ == "__main__":
    main()
