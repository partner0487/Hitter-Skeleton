# Hitter-Skeleton

棒球打擊動作分析：雙 FLIR 相機（正面、側面）同步錄影，ViTPose 偵測骨架，分析揮棒、動作階段與 3D 姿態。

## Usage
### Installation
1. Clone this repo.
2. Setup conda environment（在專案根目錄執行，Anaconda Prompt 或 PowerShell 皆可）:
    ```
    # 建立並啟用環境
    conda create -n hitter python=3.8 -y
    conda activate hitter

    # 解決 OMP 函式庫重複衝突 (libiomp5md.dll)，寫進環境設定，之後每次 activate 都會生效
    conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
    conda activate hitter

    # 安裝 PyQt5
    conda install -y -c defaults pyqt=5.15.10 qt-main=5.15.2

    # 安裝 Windows 專用的 Faiss GPU 版本（追蹤器 fast-reid 執行時會用到）
    conda install -y -c conda-forge faiss-gpu
    # 若 GPU 版後續有相容性問題，可退回: conda install -c conda-forge faiss-cpu

    # 先安裝 PyTorch 2.0.1 (CUDA 11.8)，避免 requirements 裡的套件先拉進 CPU 版 torch
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

    # 安裝 FLIR 工業相機驅動 (Spinnaker Python SDK)
    pip install "Src\spinnaker_python-4.0.0.116-cp38-cp38-win_amd64\spinnaker_python-4.0.0.116-cp38-cp38-win_amd64.whl"

    # cython_bbox 編譯時需要 Cython 與 numpy，要先裝
    pip install Cython==3.0.11 numpy==1.24.4

    # 安裝其餘 requirements
    pip install -r requirements.txt
    ```
3. setup openmim
    ```
    # 安裝 OpenMMLab 系列預編譯套件（避免本機 MSVC 編譯報錯）
    pip install -U openmim
    mim install "mmengine==0.10.5"
    mim install "mmcv==2.0.1"
    mim install "mmdet==3.1.0"

    # 本地 Editable 模式安裝專案模組（主程式直接 import mmpose，這步不能省略）
    pip install --no-build-isolation -e Src\mmpose_main
    pip install --no-build-isolation -e Src\mmpretrain_main
    pip install --no-build-isolation -e Src\mmyolo_main
    ```
4. 確認環境
    ```
    pip check
    python -c "import torch, mmcv, mmdet, mmpose, mmpretrain, mmyolo, faiss, PySpin, cv2; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
    ```
    `pip check` 應顯示 `No broken requirements found.`，第二行應印出 `CUDA True`。

    > 舊環境若裝過 `mmrotate`、`mmrazor`、`mmcls`，安裝 mmdet 3.1.0 時會出現
    > `mmrotate 1.0.0rc1 requires mmdet<3.1.0` 的警告。本專案沒有用到這三個套件，可以移除：
    > `pip uninstall -y mmrotate mmrazor mmcls`

### Data Preparation
ViTPose、YOLO 與 fast-reid 權重請從 https://drive.google.com/drive/folders/1D7Q5bTnTAfKkfLuppqUo4_8W4t0wrCmP?usp=sharing 下載，放到 `Db/pretrain/`。`Db/` 不進 git，整體結構如下：

    Hitter-Skeleton
    |-- Db
    |   |-- pretrain
    |   |   |-- epoch_210.pth
    |   |   |-- vitpose_Sk26.pth
    |   |   |-- yolo11m-seg.pt
    |   |   `-- yolov7_x_syncbn_fast_8x16b-300e_coco_20221124_215331-ef949a68.pth
    |   |-- Calibration                       # 雙相機校正（見下方「相機校正」）
    |   |   |-- stereo.json                   # 日期最新的內外參
    |   |   |-- intrinsics_all.json
    |   |   |-- camera_roles.json
    |   |   `-- sessions/<日期_時間>/         # 每次校正的照片與結果；3D 依錄影日期挑選
    |   `-- <打者>/<YYYY-MM-DD>/<T座|側拋|正投>/<打者>_H01/   # 每一球的錄影
    |       |-- CF_<打者>_H01.mp4             # 正面影片（60 fps）
    |       |-- CS_<打者>_H01.mp4             # 側面影片
    |       |-- frame_times.csv               # 每幀實際擷取時間（掉幀時用來算實際 fps）
    |       `-- person_*.json、bat_*.json …    # 骨架與球棒分析結果
    |-- calibration                           # 校正工具
    `-- Src
        |-- UI_Control                        # 主程式
        |-- mmpose_main
        |-- mmpretrain_main
        |-- mmyolo_main
        `-- tracker

### Demo
```
conda activate hitter
python Src\UI_Control\main.py
```
或直接執行 `start.bat`。

- **2D 相機**：兩台 FLIR 相機即時偵測骨架並錄影。相機 180 fps 擷取、每 3 幀取 1 存成 60 fps，觸發前後各保留一段（共 90 幀）。
- **2D 影片**：讀取錄好的雙視角影片做骨架、球棒與動作階段分析。
- **3D**：用校正結果把正面、側面的 2D 關節點三角化成 3D，計算關節速度、揮棒速度等（單位為公尺、m/s）。

### 相機校正
3D 分析需要兩台相機的內參與外參，校正工具在 [calibration/](calibration/)，可以獨立使用：

```
python calibration/calibration.py
```

環境、拍攝方式、品質標準、離線重算與輸出格式請見 **[calibration/README.md](calibration/README.md)**。
