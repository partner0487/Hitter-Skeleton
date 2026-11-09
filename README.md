# Hitter-Skeleton

棒球打擊動作分析：雙 FLIR 相機（正面、側面）同步錄影，YOLO26 偵測打者與球棒、ViTPose 偵測骨架，分析揮棒、動作階段與 3D 姿態。

## Usage
### Installation
1. Clone this repo.
2. Setup conda environment:
    ```
    # 建立並啟用環境
    conda create -n hitter python=3.8 -y
    conda activate hitter

    # 安裝 FLIR 工業相機驅動 (Spinnaker Python SDK)
    # 請替換為你本機的 wheel 檔實際路徑：
    pip install "Src\spinnaker_python-4.0.0.116-cp38-cp38-win_amd64\spinnaker_python-4.0.0.116-cp38-cp38-win_amd64.whl"

    # 安裝其餘 requirements
    pip install -r requirements.txt

    # 解決 OMP 函式庫重複衝突 (libiomp5md.dll)
    $env:KMP_DUPLICATE_LIB_OK="TRUE"

    # 安裝 PyQt5
    conda install -y -c defaults pyqt=5.15.10 qt-main=5.15.2

    # 安裝 PyTorch 2.0.1 (CUDA 11.8)
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

    # 安裝 Windows 專用的 Faiss GPU 或 CPU 版本
    conda install -c conda-forge faiss-gpu
    # 若 GPU 版後續有相容性問題，可退回: conda install -c conda-forge faiss-cpu

    # 安裝 Cython 及依賴庫
    pip install cython
    pip install cython_bbox
    ```
3. setup openmim
    ``` 
    #安裝 OpenMMLab 系列預編譯套件（避免本機 MSVC 編譯報錯）
    pip install -U openmim
    mim install "mmcv==2.0.1"
    mim install "mmdet==3.1.0"
    mim install "mmengine==0.10.5"

    #本地 Editable 模式安裝專案模組
    pip install --no-build-isolation -e Src\mmpose_main
    pip install --no-build-isolation -e Src\mmpretrain_main
    pip install --no-build-isolation -e Src\mmyolo_main
    ``` 

### Data Preparation
兩個模型權重都放到 `Db/pretrain/`：
- 骨架 ViTPose（`epoch_210.pth`，Halpe 26 個關節點）：從 https://drive.google.com/drive/folders/1D7Q5bTnTAfKkfLuppqUo4_8W4t0wrCmP?usp=sharing 下載
- 人與球棒偵測 YOLO26（`yolo26m-seg.pt`，需要 ultralytics 8.4 以上）：從 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt 下載

`Db/` 不進 git，整體結構如下：

    Hitter-Skeleton
    |-- Db
    |   |-- pretrain
    |   |   |-- epoch_210.pth                 # ViTPose（Halpe 26 點）
    |   |   `-- yolo26m-seg.pt                # YOLO26 分割（人、球棒、球）
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

- **錄影**：兩台 FLIR 相機即時偵測骨架並錄影（也可以只接側面相機）。相機 180 fps 擷取、每 3 幀取 1 存成 60 fps，觸發前後各保留一段（共 90 幀）。錄完先顯示 2D 結果，3D 在背景算完後換上。
- **分析**：讀取錄好的影片，用校正結果把正面、側面的 2D 關節點三角化成 3D，計算動作階段、關節速度、揮棒速度等（單位為公尺、m/s）；只有側面影片時做純側面 2D 分析。
- **比較**：兩支錄影並排比較，指標與分析頁相同。

### 相機校正
3D 分析需要兩台相機的內參與外參，校正工具在 [calibration/](calibration/)，可以獨立使用：

```
python calibration/calibration.py
```

環境、拍攝方式、品質標準、離線重算與輸出格式請見 **[calibration/README.md](calibration/README.md)**。
