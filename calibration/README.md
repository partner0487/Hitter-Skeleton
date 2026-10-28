# 雙相機棋盤格校正

為 Hitter-Skeleton 的兩台 FLIR 相機（正面 35 mm、側面 8 mm）求出內參與外參，
讓 3D 分析能把兩個視角的 2D 關節點三角化成真實尺度（公尺）的 3D 座標。

這個資料夾可以獨立使用：不需要 ViTPose、YOLO 等模型，只要相機驅動與 OpenCV。

---

## 1. 環境

使用主程式的 `Pitcher` conda 環境即可。若要另外建立，需要：

| 套件 | 用途 |
|---|---|
| Python 3.8 | |
| `opencv-python`（4.x，含 `findChessboardCornersSB`） | 角點偵測、校正 |
| `numpy`、`scipy` | 計算 |
| `PyQt5` | 精靈與內參視窗（`--calculate` 不需要） |
| `PySpin`（Spinnaker Python SDK） | 連接 FLIR 相機（`--calculate` 不需要） |
| `PyYAML` | 讀相機設定檔 |

```bash
conda activate Pitcher
python calibration/calibration.py --help
```

## 2. 硬體與事前準備

| 項目 | 說明 |
|---|---|
| 正面相機 | 35 mm 鏡頭，設定檔 `camera_config/GH3_camera_config.yaml`，GPIO primary（Line2 輸出觸發） |
| 側面相機 | 8 mm 鏡頭，設定檔 `camera_config/GH3_camera_config_2.yaml`，GPIO secondary（Line3 接收觸發） |
| GPIO 同步線 | **拍外參時必須接**，否則側面收不到觸發，畫面會顯示「雙機取像失敗」。內參一台一台拍，不需要 |
| 棋盤格 | 預設 **9×8 格＝8×7 內角點**，每格 100 mm。板子要平、不反光 |

> **內角點**＝格子數 − 1。程式要填的是內角點數，填錯會完全偵測不到。
> 寬、高不相等的板子才能分辨方向；7×7 這種正方形板容易在兩台相機間轉錯 90°。

開始前：
1. 兩台相機架在**正式錄影的位置**，鏡頭焦距、光圈鎖好。校正後只要相機移動或轉焦距，就要重做外參。
2. 關閉 `main.py`、SpinView 等會占用相機的程式。

## 3. 快速開始

```bash
python calibration/calibration.py              # 校正精靈：正面內參 → 側面內參 → 雙機拍照 → 外參
python calibration/calibration.py --single     # 只校正一台相機的內參
python calibration/calibration.py --calculate  # 用已拍好的照片離線重算（見第 6 節）
```

可以從任何目錄執行。

## 4. 精靈流程

### 步驟 1：設定
- 正面、側面相機序號（自動掃描，也可手動輸入）。
- 棋盤格內角點寬、高與每格邊長。
- 兩台鏡頭焦距（預設 35 / 8 mm）。用來給 fx 初值並檢查結果；焦距 ≥ 25 mm 視為長焦，自動固定光心與 k2、k3，只估 k1。
- **沿用已有內參**：鏡頭沒換、只是相機移動時勾選，直接跳到步驟 4。
- **只做內參**：相機還沒架到球場時先做內參，外參之後再補。

### 步驟 2、3：正面內參、側面內參
一台一台拍，按**空白鍵**擷取，至少 15 組。每擷取一組會即時顯示九宮格覆蓋了幾區、還缺哪裡。

拍攝要點：
- **先調曝光**：畫面上方的「曝光時間」「增益」可以邊看邊調（方向鍵／滾輪即時套用，打字則按 Enter 套用）。調到棋盤格黑白分明、白格不過曝即可。優先加長曝光；增益會增加雜訊，盡量維持 0。曝光上限受幀率限制（180 fps 約 5500 μs），超過時會自動夾到上限。
  - 調好的數值會帶到下一台、下一步驟，並記在 `session.json` 的 `camera_settings`。
  - 按「存成預設」會寫回 `camera_config/*.yaml`，下次開相機就使用這組數值（只影響校正工具，Src 錄影用的是 `Src/UI_Control/camera_config/`）。
- **畫面四角都要拍到**：鏡頭畸變在邊角最明顯，只拍中間會校不準。
- **往不同方向傾斜 30° 以上**：上下、左右、斜向都要有。只朝同一方向轉的話，焦距和距離會分不開，fx 會算錯。
- **板子要停穩**再拍，模糊的照片會被判為異常。
- 跟上一組幾乎一樣的照片（平均位移 < 5 px）會自動略過。

按「計算內參並繼續」後會判斷品質，RMS > 1 px 不會進入下一步。

### 步驟 4：雙機拍棋盤格（外參）
按「連接雙相機」。兩台畫面都出現彩色角點、標題變成綠色 ✔ 時，拍照鍵才會亮；按空白鍵拍一組，兩台照片以同一個檔名存檔。

- 建議 **15 組以上**（至少 6 組）。
- 涵蓋**打擊區的左右、前後、高低**，並讓板子朝不同方向傾斜。
- 每台畫面下方各有一組曝光／增益，可以分別調整，用法同步驟 2、3。
- 滾輪可放大檢查角點，雙擊畫面恢復完整畫面。
- 已有照片可按「匯入 cf / cs 照片資料夾」（資料夾內需有 `cf/`、`cs/` 兩個子資料夾，同名配對）。

### 步驟 5：計算外參
按「計算外參並儲存」後自動完成：
1. 用 `findChessboardCornersSB` 偵測所有照片的角點。
2. **修正角點順序**：棋盤格在兩台相機中可能被編成相反方向（轉 180°），`align_pairs` 會用所有照片的姿態一致性挑出正確的排列，並剔除不一致的照片。
3. 固定兩台內參，用 `stereoCalibrate` 求正面→側面的 R、t。
4. 每 5 組留 1 組不參與計算，用來驗證三角化後的格子邊長是否接近 100 mm。

**Stereo RMS ≤ 3 px 才會採用**；超過會回到步驟 4，已拍的照片保留，可以補拍後再算。

## 5. 品質判斷標準

| 項目 | 標準 | 不符合時 |
|---|---|---|
| 內參 RMS | < 0.5 px 良好；0.5–1 px 可用；> 1 px 不合格 | 精靈不會進入下一步 |
| 單張誤差 | > 1 px 且 > 整體 RMS 兩倍視為異常 | 提示剔除後重算 |
| fx | 與「焦距 ÷ 像素大小 5.86 µm」相差 < 15% | 警告：通常是傾斜方向不夠多元 |
| 九宮格覆蓋 | 9 區都要有角點 | 警告缺哪幾區 |
| 傾斜角 | 至少一組 > 20° | 警告 |
| 轉軸集中度 | < 0.8（1 代表全部朝同一方向轉） | 警告 fx 可能不準 |
| 樣本數 | 內參建議 ≥ 15 組；外參建議 ≥ 15 組 | 警告 |
| Stereo RMS | ≤ 3 px | 不採用這次校正 |
| 驗證組格子邊長誤差 | 中位數 < 5 mm | 警告 |

## 6. 用已拍好的照片離線重算

不開相機、不開視窗，直接重算某次校正的內參與外參：

```bash
python calibration/calibration.py --calculate                                # 最新一次（有 cf/、cs/ 的 session）
python calibration/calibration.py --calculate 20261019                       # 指定日期（當天最新一次），也可給完整 session 名稱
python calibration/calibration.py --calculate 20261019 --pattern 7x7         # 覆寫棋盤格規格（也接受 7*7、7,7）
```

- 棋盤格規格、相機序號、焦距、格子邊長都讀該 session 的 `session.json`，沒有才用 8×7、100 mm、35 / 8 mm。
- 內參照片優先用 `intrinsic/images_SN<序號>/`，沒有就用 `cf/`（正面）、`cs/`（側面）；外參用 `cf/`、`cs/` 的同名照片。
- 誤差異常的照片會自動剔除後重算。
- 結果寫回該 session。**Stereo RMS ≤ 3 px 就和精靈一樣直接採用**：該日期起的錄影做 3D 分析時會使用這次校正（見第 8 節）；超過則不採用。

## 7. 輸出

所有結果都在 `Db/Calibration/`（不進 git）：

```
Db/Calibration/
├─ camera_roles.json        目前的正面／側面相機序號
├─ intrinsics_all.json      目前各相機的內參（以序號為鍵）
├─ stereo.json              日期最新的已採用校正（外參＋兩台內參）
└─ sessions/
   ├─ <YYYYMMDD_HHMMSS>/              精靈每次校正一個資料夾
   │  ├─ intrinsic/images_SN<序號>/   內參照片（灰階 PNG）
   │  ├─ intrinsic/calib_SN<序號>.npz  內參完整資料（角點、每張誤差、傾斜角）
   │  ├─ cf/  cs/                      外參照片，同名＝同一時刻
   │  ├─ corners/                      角點預覽，標出 0 號角點，可檢查兩台方向是否一致
   │  ├─ intrinsics.json               本次兩台內參
   │  ├─ stereo.json                   本次採用的外參完整報告（含 published_at）
   │  ├─ stereo_failed.json            Stereo RMS 不合格時的報告，供檢查；不影響已採用的 stereo.json
   │  ├─ session.json                  本次設定與進度
   │  └─ previous_calibration/         採用時被取代的上一版，可手動複製回去還原
   ├─ <YYYYMMDD_HHMMSS>_SN<序號>/     --single 單獨校正一台
   └─ <YYYYMMDD>_000000_legacy/       舊版 Intrinsic.py／PnP.py 的照片；舊輸出在 legacy/
```

`sessions/` 保留每一次校正的原始資料。精靈或 `--calculate` 算出 Stereo RMS ≤ 3 px 時「採用」該 session；
上層三個 json 永遠是**日期最新**的已採用校正，重算較舊日期的 session（例如 `_legacy`）不會蓋掉它們。

session 名稱固定為 `YYYYMMDD_HHMMSS[_後綴]`，名稱順序就是時間順序；舊資料用 `000000`，同一天若有新的校正會排在它之後。只有 `sessions/` 底下符合此格式的資料夾會被使用。

### `stereo.json` 主要欄位

| 欄位 | 說明 |
|---|---|
| `R_cs_from_cf`、`t_cs_from_cf_mm` | 正面→側面外參：`X_side = R @ X_front + t`，單位 mm |
| `baseline_mm` | 兩台相機的距離 |
| `intrinsics.cf`、`intrinsics.cs` | 正面、側面內參（`intrinsic_matrix`、`distortion_coefficients`、`image_size` …） |
| `fit.stereo_rms_px` | 雙目重投影誤差 |
| `held_out_board_checks` | 驗證組的格子邊長誤差 |
| `training_pairs`、`rejected_pairs` | 參與計算與被剔除的照片 |
| `front`、`side` | 相機序號 |
| `published_at` | 採用時間；沒有這個欄位（不合格）的 session 不會被 3D 分析使用 |
| `warnings` | 需要人工確認的項目 |

### 內參 entry（`intrinsics.json`、`intrinsics_all.json`）

`intrinsic_matrix`、`distortion_coefficients`、`fx`、`fy`、`cx`、`cy`、`image_size`、`rms`、`verdict`、`pattern`、`square_size_mm`、`lens_focal_mm`、`distortion_locked`、`calibration_flags`、`source_images`、`calibrated_at`。

## 8. 主程式如何使用校正結果

校正以**日期**為單位：相機有偏差、或 demo／測試前重新校正一次，之後的錄影都用這次的結果。

3D 分析載入錄影時，從路徑中的日期（`Db/<打者>/2026-10-08/...`，或 `20260421` 這種資料夾）挑**當天或之前最新的已採用校正**：

| 錄影日期 | 已採用的校正 | 使用 |
|---|---|---|
| 2026-10-30 | 20261008_000000_legacy、20261029_093000 | 20261029_093000 |
| 2026-10-19 | 同上 | 20261008_000000_legacy |
| 2026-10-05（早於所有校正）或路徑沒有日期 | | 最新的 `Db/Calibration/stereo.json` |

- 重新校正後，舊錄影仍使用當時的校正，不會被新參數影響。
- 3D 介面會在終端機印出這次用的是哪個校正檔。
- 都沒有校正時，才用程式內寫死的舊參數（沒有真實尺度）。
- 3D 介面以側面相機為世界座標，會自動把外參轉成「側面→正面」、單位換成公尺。

**2026-10 的舊錄影**要使用當時的校正，用對應日期的舊照片重算即可（合格就會採用）：

```bash
python calibration/calibration.py --calculate 20261008
```

## 9. 常見問題

| 狀況 | 可能原因與處理 |
|---|---|
| 掃描不到相機 | 其他程式占用相機（main.py、SpinView），或 USB 沒接好 |
| 「雙機取像失敗，請確認 GPIO 同步線」 | 拍外參時沒接同步線，或兩台的 primary／secondary 設定檔對調 |
| 一直偵測不到棋盤格 | 內角點數填錯（格子數 − 1）；板子沒有完整入鏡；反光或太暗 |
| 側面 fx 跟鏡頭推算值差很多 | 板子只朝同一方向轉、邊角沒拍到；照提示補拍 |
| 長焦畸變係數很大 | 確認焦距有填，長焦鎖定有勾 |
| Stereo RMS > 3 px | 內參不準、兩台不同步、配對錯誤；看 `corners/` 預覽確認兩台的 0 號角點方向，補拍更多不同位置的照片 |
| 想退回上一版校正 | 把該 session 的 `previous_calibration/` 裡的檔案複製回 `Db/Calibration/` |

## 10. 檔案

| 檔案 | 用途 |
|---|---|
| `calibration.py` | 入口：開精靈、`--single`、`--calculate` |
| `wizard.py` | 校正精靈 UI |
| `intrinsic_window.py` | 單台內參 UI（精靈的內參步驟也是它） |
| `intrinsic.py` | 內參計算與品質判斷（不含 Qt，視窗與指令列共用） |
| `stereo.py` | 外參計算、驗證與採用（不含 Qt） |
| `storage.py` | `Db/Calibration` 的路徑與讀寫 |
| `chessboard_cross/` | 角點偵測、角點順序對齊、立體校正、驗證（源自學長的 calibration_other_version） |
| `camera_objects/`、`camera_config/`、`utils/` | FLIR 相機控制與設定 |
| `paths.py` | 路徑常數 |
