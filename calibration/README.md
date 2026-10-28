# 雙相機棋盤格校正（正面 35mm／側面 8mm）

用同一塊棋盤格完成兩台 FLIR 相機的內參與外參，結果給 `Src/UI_Control` 的 3D 分析使用。

## 執行

```bash
conda activate Pitcher
python calibration/calibration.py            # 校正精靈：內參 → 外參（可從任何目錄執行）
python calibration/calibration.py --single   # 只校正一台相機的內參（即時拍攝或讀取照片資料夾）
```

執行前請關閉 `main.py`、SpinView 等會占用相機的程式。

### 用已拍好的照片離線重算（不開相機、不開視窗）

```bash
python calibration/calibration.py --calculate                    # 最新的 session
python calibration/calibration.py --calculate 20261019_legacy    # 指定 session（名稱或路徑）
python calibration/calibration.py --calculate 20261019_legacy --pattern 7x7 --square 100
python calibration/calibration.py --calculate 20261019_legacy --publish      # 合格就設為目前使用中
python calibration/calibration.py --calculate 20261008_legacy --snapshot "Db/handsomeboy/2026-10-08/T座/handsomeboy_H01"   # 存成舊錄影的校正快照
```

- 棋盤格規格、相機序號、焦距預設讀該 session 的 `session.json`，沒有才用 8x7、100 mm、35／8 mm；參數可覆寫。
- 內參照片：`intrinsic/images_SN<序號>/`，沒有就用 `cf/`、`cs/`；外參用 `cf/`、`cs/` 的同名照片。
- 誤差異常的照片會自動剔除（與視窗按「是」相同），結果寫回該 session，不加 `--publish` 不會動到目前使用中的校正。

## 精靈流程

1. **設定**：選正面、側面相機序號；棋盤格**內角點**數（格子數 − 1，預設 8×7）與每格邊長（預設 100 mm）；兩台鏡頭焦距（預設 35／8 mm）。
2. **正面內參**、3. **側面內參**：一台一台拍，至少 15 組。畫面會即時提示九宮格覆蓋、傾斜角、轉軸方向是否足夠；焦距 ≥ 25 mm 自動固定光心與 k2、k3。
4. **雙機拍棋盤格**：需接 GPIO 同步線。兩台畫面都偵測到棋盤格時拍照鍵才會亮，照片以同名存進 `cf/`、`cs/`。建議 15 組以上，涵蓋打擊區不同位置、深度與傾斜方向。
5. **計算外參**：自動偵測角點、修正棋盤格旋轉造成的角點順序錯誤（`align_pairs`），固定內參求 R、t；每 5 組留 1 組驗證格子邊長。Stereo RMS > 3 px 不發布，回到第 4 步補拍。

已經拍好的照片可在第 4 步「匯入 cf / cs 照片資料夾」。勾選「沿用已有內參」可跳過第 2、3 步，只重做外參。

## 輸出：`Db/Calibration/`（不進 git）

```
Db/Calibration/
├─ camera_roles.json        目前的正面／側面相機序號
├─ intrinsics_all.json      目前各相機內參（以序號為鍵）
├─ stereo.json              目前的外參與兩台內參 ← 3D 分析讀這個
└─ sessions/
   ├─ <時間>/               精靈每次校正一個資料夾
   │  ├─ intrinsic/images_SN<序號>/   內參照片
   │  ├─ intrinsic/calib_SN<序號>.npz  內參完整資料
   │  ├─ cf/ cs/                       外參照片（同名＝同一時刻）
   │  ├─ corners/                      角點預覽（標出 0 號角點，可檢查方向）
   │  ├─ intrinsics.json  stereo.json  session.json
   │  └─ previous_calibration/         被取代的上一版，可手動還原
   ├─ <時間>_SN<序號>/      --single 單獨校正一台
   └─ <日期>_legacy/        2026-10 舊版 Intrinsic.py／PnP.py 的照片與結果（legacy/）
```

`stereo.json` 的外參為正面→側面：`X_cs = R_cs_from_cf @ X_cf + t_cs_from_cf_mm`（mm）。
3D 分析會轉成側面為世界座標、單位公尺。

## 檔案

| 檔案 | 用途 |
|---|---|
| `calibration.py` | 入口：開精靈、`--single`、`--calculate` |
| `wizard.py` | 校正精靈 UI |
| `intrinsic_window.py` | 單台內參 UI（精靈的內參步驟也是它） |
| `intrinsic.py` | 內參計算與品質判斷（不含 Qt，視窗與指令列共用） |
| `stereo.py` | 外參計算、驗證與發布（不含 Qt） |
| `storage.py` | `Db/Calibration` 的路徑與讀寫 |
| `chessboard_cross/` | 角點偵測、角點順序對齊、立體校正、驗證 |
| `camera_objects/`、`camera_config/`、`utils/` | FLIR 相機控制與設定 |
| `paths.py` | 路徑常數 |
