# FLIR 相機棋盤格校正系統 (FLIR Chessboard Calibration)

本專案提供了一個基於 Python 和 OpenCV 的工具集，用於 FLIR 相機的棋盤格相機校正（Camera Calibration）。它包含兩個主要的腳本：一個支援連接相機即時拍攝與校正，另一個用於從本機資料夾載入圖片進行批次校正。

## 專案結構

- `flir_camera_system_opencv_app.py`：連接 FLIR 相機並顯示即時預覽、擷取棋盤格影像，並可直接進行校正的主程式。
- `loadFolderChessboard.py`：從指定的資料夾批次讀取棋盤格圖片，進行相機校正並輸出參數的腳本。
- `camera_config/`：存放相機相關的設定檔（例如 `.yaml`）。
- `camera_objects/`：FLIR 相機系統的相關物件與操作封裝。
- `opencv_objects/`：包含 OpenCV 棋盤格角點偵測與校正演算法的物件 (`ChessboardCalibrator`)。
- `utils/`：通用工具函數（日誌設定、資料夾建立等）。
- `result_YYYYMMDD/`：預設的輸出資料夾，用來存放拍攝的相片及產出的校正參數 JSON 檔。

## 系統需求

請確認已安裝以下相關套件與環境：
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- （若要使用即時拍攝功能）需要安裝 FLIR 的相關 SDK 及 Python 驅動（例如 Spinnaker / PySpin，相依於 `camera_objects` 的實作）。

## 核心功能與使用方式

本專案中的棋盤格規格預設如下，如有不同請於程式碼中自行修改：
- `pattern_size`: `(7, 7)` (棋盤格內角點的行與列數量)
- `square_size_mm`: `100` (單個棋盤格方塊的邊長，單位為毫米 mm)

### 1. 即時相機拍攝與校正 (`flir_camera_system_opencv_app.py`)

這支程式會啟動連接的 FLIR 相機，並顯示即時影像。您可以將棋盤格放置在鏡頭前，程式會自動嘗試尋找棋盤格的角點，並即時將角點繪製在顯示畫面上。

**設定與執行：**
1. 執行前請確認程式碼中的 `config_path`（相機設定檔路徑）與 `serial_number`（相機序號）是否正確。
2. 執行指令：
   ```bash
   python flir_camera_system_opencv_app.py
   ```
3. 程式開啟後，可以使用以下 **鍵盤快捷鍵** 來操作：
   - `s`：若畫面中成功偵測到角點，按下後會儲存該影像至 `result_YYYYMMDD/chessboard_images/`。
   - `f`：凍結 (Freeze) / 解除凍結 當下畫面，方便確認。
   - `c`：讀取已儲存的所有圖片，執行相機校正，並將結果儲存至 `./result_YYYYMMDD/single_camera_parameters.json`。
   - `q`：退出程式。

### 2. 資料夾圖片批次校正 (`loadFolderChessboard.py`)

如果您已經準備好了一組棋盤格圖片，且不需要連接相機即時拍攝，可以使用此腳本直接進行校正計算。

**設定與執行：**
1. 確保圖片放在正確的資料夾中。請修改程式內的 `base_dir` 以對應您的圖片目錄（預設為 `./result_20250903`，其下需有 `chessboard_images/` 資料夾存放圖片）。
2. 執行指令：
   ```bash
   python loadFolderChessboard.py
   ```
3. 程式會自動讀取目錄下的所有圖片，進行角點偵測與校正（校正至少需要 5 張有效的棋盤格圖片）。
4. 處理完成後，程式會在終端機印出 **RMS 重投影誤差**、**相機內參矩陣** 及 **畸變係數**。
5. 成功後，參數會被儲存至 `<base_dir>/calibration/single_camera_parameters.json` 檔案中。

## 校正結果輸出格式

校正計算完成後儲存的 `single_camera_parameters.json` 包含以下重要資訊：
- `camera_matrix`: 相機內參矩陣 (3x3)。
- `dist_coeffs`: 畸變係數陣列 (Distortion coefficients)。
- `camera_matrix_shape` / `dist_coeffs_shape`: 用來驗證資料維度。
