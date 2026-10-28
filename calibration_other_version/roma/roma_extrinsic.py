"""RoMa 外參校正：不需校正道具，直接用正面（CF）與側面（CS）同時拍的錄影畫面求 F。

研究測試中（2026-10-13）。做法：每支錄影取幾幀，RoMa 稠密匹配兩台相機的畫面 → 每對用 MAGSAC 去除錯誤匹配
→ 全部合併再求一次 F。結果的方向慣例和 calibrate.py 的校正桿外參相同（x_front^T F x_side = 0），主程式可直接使用。

需在 Roma 環境執行（安裝方式見 docs/CALIBRATION.md 第 7 節），第一次執行會自動下載模型權重（需網路）：
    conda activate Roma
    python Src/calibration/roma/roma_extrinsic.py <錄影資料夾> [<錄影資料夾> ...] [選項]

錄影資料夾可以是單球（含 CF_*.mp4、CS_*.mp4），也可以是整天的資料夾（會往下找每一球）。
選項：
    --frames 0,30,60   每支錄影取哪幾幀（預設 0,30,60；超出影片長度的會略過）
    --pitcher-region   只用投手附近的匹配點（依 CS 骨架 JSON 的人物框，研究中、尚待驗證）
    --save             寫入 Db/Calibration/extrinsic/fundamental.json（之後錄影的校正快照會用它）
    --snapshot         另外替每一球寫入 calibration.json（已有的不覆蓋，加 --force 才覆蓋）
"""
import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # 專案根目錄：讓 Src.* 可匯入

import cv2
import numpy as np

from Src.calibration import calib_store

THRESHOLD_PX = 1.0         # MAGSAC 內點門檻（px）
SAMPLES_PER_PAIR = 5000    # 每對畫面取樣的匹配點數
UPSAMPLE = (1088, 1920)    # 接近原圖比例的解析度；預設 864x864 單對結果較不穩
REGION_GROW = 0.3          # 投手範圍：CS 骨架框聯集往外擴 30%


def find_pitches(folders):
    """回傳含 CF／CS 原始影片的每一球資料夾"""
    found = []
    for folder in folders:
        for cf in sorted(glob.glob(os.path.join(folder, '**', 'CF_*.mp4'), recursive=True)):
            if '_Sk' in os.path.basename(cf):
                continue
            cs = os.path.join(os.path.dirname(cf), 'CS' + os.path.basename(cf)[2:])
            if os.path.exists(cs):
                found.append((os.path.dirname(cf), cf, cs))
    return found


def read_frame(path, index):
    cap = cv2.VideoCapture(path)
    frame = None
    for _ in range(index + 1):          # 逐格讀，mp4 跳格不一定準
        ok, frame = cap.read()
        if not ok:
            frame = None
            break
    cap.release()
    return frame


def pitcher_box(folder):
    """CS 骨架 JSON 中人物框的聯集，往外擴 REGION_GROW；沒有 JSON 回傳 None"""
    import json
    files = [f for f in glob.glob(os.path.join(folder, 'CS_*.json')) if 'BRand' not in f and '_3D' not in f
             and 'keyframes' not in f]
    if not files:
        return None
    with open(files[0], encoding='utf-8') as f:
        boxes = np.array([d['bbox'][:4] for d in json.load(f) if d.get('bbox')])
    if not len(boxes):
        return None
    x0, y0 = boxes[:, :2].min(0)
    x1, y1 = boxes[:, 2:4].max(0)
    gx, gy = (x1 - x0) * REGION_GROW / 2, (y1 - y0) * REGION_GROW / 2
    return x0 - gx, y0 - gy, x1 + gx, y1 + gy


def magsac(a, b):
    F, mask = cv2.findFundamentalMat(a.astype(np.float64), b.astype(np.float64), cv2.USAC_MAGSAC,
                                     THRESHOLD_PX, 0.999999, 10000)
    if F is None:
        return None, np.zeros(len(a), bool)
    return F[:3], mask.ravel().astype(bool)


def epipolar_error(F, a, b):
    """對稱對極距離（px）；F 滿足 b^T F a = 0"""
    ah, bh = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
    l2, l1 = ah @ F.T, bh @ F
    num = np.abs(np.sum(bh * l2, axis=1))
    return 0.5 * (num / np.hypot(l2[:, 0], l2[:, 1]) + num / np.hypot(l1[:, 0], l1[:, 1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description='RoMa 外參校正（不需校正道具）')
    parser.add_argument('folders', nargs='+')
    parser.add_argument('--frames', default='0,30,60')
    parser.add_argument('--pitcher-region', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    pitches = find_pitches(args.folders)
    if not pitches:
        sys.exit('找不到同時有 CF_*.mp4 與 CS_*.mp4 的錄影資料夾')
    frames = [int(v) for v in args.frames.split(',')]
    print(f'共 {len(pitches)} 球，每球取第 {frames} 幀')

    import torch
    from PIL import Image
    from romatch import roma_outdoor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = roma_outdoor(device=device, upsample_res=UPSAMPLE)

    per_pitch = {}       # 每球的內點（CF 座標 a、CS 座標 b）
    for folder, cf, cs in pitches:
        box = pitcher_box(folder) if args.pitcher_region else None
        if args.pitcher_region and box is None:
            print(f'  {os.path.basename(folder)}：沒有 CS 骨架 JSON，改用整張畫面')
        pa, pb = [], []
        for index in frames:
            fa, fb = read_frame(cf, index), read_frame(cs, index)
            if fa is None or fb is None:
                continue
            ia = Image.fromarray(cv2.cvtColor(fa, cv2.COLOR_BGR2RGB))
            ib = Image.fromarray(cv2.cvtColor(fb, cv2.COLOR_BGR2RGB))
            warp, certainty = model.match(ia, ib, device=device)
            matches, _ = model.sample(warp, certainty, num=SAMPLES_PER_PAIR)
            a, b = model.to_pixel_coordinates(matches, fa.shape[0], fa.shape[1], fb.shape[0], fb.shape[1])
            a, b = a.cpu().numpy().astype(np.float64), b.cpu().numpy().astype(np.float64)
            if box is not None:
                keep = (b[:, 0] > box[0]) & (b[:, 0] < box[2]) & (b[:, 1] > box[1]) & (b[:, 1] < box[3])
                a, b = a[keep], b[keep]
            if len(a) < 50:
                continue
            _, mask = magsac(a, b)
            pa.append(a[mask])
            pb.append(b[mask])
        if pa:
            per_pitch[folder] = (np.vstack(pa), np.vstack(pb))
            print(f'  {os.path.basename(folder)}：{len(pa)} 幀、{len(per_pitch[folder][0])} 個內點')
    if len(per_pitch) < 1:
        sys.exit('沒有可用的匹配結果')

    all_a = np.vstack([v[0] for v in per_pitch.values()])
    all_b = np.vstack([v[1] for v in per_pitch.values()])
    F_cs_cf, mask = magsac(all_a, all_b)          # x_cs^T F x_cf = 0

    # 品質：每球輪流拿出來，用其他球算的 F 量它；某一球特別大，代表拍那球時相機可能動過
    if len(per_pitch) >= 3:
        print('\n留一球交叉驗證（對極誤差中位數，越小越好）：')
        medians = {}
        for folder, (a, b) in per_pitch.items():
            others = [v for k, v in per_pitch.items() if k != folder]
            F_o, _ = magsac(np.vstack([v[0] for v in others]), np.vstack([v[1] for v in others]))
            medians[folder] = float(np.median(epipolar_error(F_o, a, b)))
        overall = float(np.median(list(medians.values())))
        for folder, m in medians.items():
            flag = '  ← 明顯偏大，相機可能在這球前後被移動' if m > 3 * overall and m > 5 else ''
            print(f'  {os.path.basename(folder)}：{m:.2f} px{flag}')
        print(f'  整體中位數 {overall:.2f} px')

    F = F_cs_cf.T                                   # 轉成 x_front^T F x_side = 0（同 calibrate.py）
    F = F / F[2, 2]
    print(f'\nF（x_front^T F x_side = 0）：\n{np.array2string(F, precision=6)}')
    print(f'合併後內點 {int(mask.sum())}/{len(mask)}')

    if args.save or args.snapshot:
        roles = calib_store.load_roles()
        if roles is None:
            sys.exit(f'找不到 {calib_store.ROLES_JSON}，請先用 calib.py 第 4 區指定正面／側面相機')
        front_sn, side_sn = roles
        calib_store.save_fundamental(F, front_sn, side_sn, int(mask.sum()), len(mask), THRESHOLD_PX, extra={
            'method': 'RoMa（roma_extrinsic.py）',
            'sources': [os.path.abspath(k) for k in per_pitch],
            'frames': frames,
            'pitcher_region': bool(args.pitcher_region),
        })
        print(f'已寫入 {calib_store.FUNDAMENTAL_JSON}（正面 SN{front_sn}、側面 SN{side_sn}）')
    if args.snapshot:
        for folder in per_pitch:
            path = os.path.join(folder, calib_store.SNAPSHOT_NAME)
            if os.path.exists(path) and not args.force:
                print(f'  已有 {path}，略過（加 --force 覆蓋）')
                continue
            ok, message = calib_store.save_snapshot(folder)
            print(f'  {os.path.basename(folder)}：{message}')


if __name__ == '__main__':
    main()
