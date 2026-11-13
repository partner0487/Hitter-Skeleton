# Hitter-Skeleton

棒球打擊動作分析：雙 FLIR 相機（正面、側面）同步錄影，YOLO26 偵測打者與球棒、ViTPose 偵測骨架，分析揮棒、動作階段與 3D 姿態。

## Usage
### Installation
適用 RTX 4090 與 RTX 5090：5090（Blackwell）需要 PyTorch 2.7 以上、CUDA 12.8，顯示卡驅動 572 版以上。以下都在專案根目錄、Anaconda Prompt 或 PowerShell 執行。

1. Clone this repo.
2. 建立環境並依序安裝（順序不能換：先裝 torch，requirements 才不會另外裝 CPU 版 torch）：
    ```
    # Python 3.10（torch 2.7 不支援 3.8）
    conda create -n hitter python=3.10 -y
    conda activate hitter

    # PyTorch 2.7.1 + CUDA 12.8（同時支援 4090 sm_89、5090 sm_120）
    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

    # mmcv 2.1.0：官方沒有 torch 2.7 + CUDA 12.8 的 Windows 版，用專案內自行編譯的 wheel（含 4090、5090 架構）
    # 這個 wheel 只能搭配 Python 3.10 + torch 2.7.1，換版本要重新編譯（見下方「重新編譯 mmcv」）
    pip install Src\mmcv_prebuilt\mmcv-2.1.0-cp310-cp310-win_amd64.whl

    # FLIR 工業相機驅動（Spinnaker Python SDK，Python 3.10 版）
    pip install Src\spinnaker_python-4.0.0.116-cp310-cp310-win_amd64\spinnaker_python-4.0.0.116-cp310-cp310-win_amd64.whl

    # PyQt5
    conda install -y -c defaults pyqt=5.15.10 qt-main=5.15.2

    # 其餘套件
    pip install -r requirements.txt
    pip install --no-build-isolation cython_bbox==0.1.5

    # mmpose、mmpretrain（本地 editable 安裝；mmyolo 程式沒用到，不用裝）
    pip install --no-build-isolation -e Src\mmpose_main
    pip install --no-build-isolation -e Src\mmpretrain_main
    ```
3. 確認安裝：
    ```
    python -c "import torch, mmcv.ops, mmpose, PySpin; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
    ```
    顯示卡名稱正確、清單裡有 `sm_89`（4090）或 `sm_120`（5090）即可。

#### 重新編譯 mmcv（只有換 Python 或 torch 版本時才需要）
需要 Visual Studio 2022（含「使用 C++ 的桌面開發」）。CUDA 12.8 編譯器用 conda 裝在環境內，不影響系統的 CUDA：
```
conda activate hitter
conda install -y -c "nvidia/label/cuda-12.8.1" cuda-nvcc cuda-cudart-dev cuda-cccl cuda-nvrtc-dev libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev cuda-profiler-api
pip install ninja wheel "setuptools<70"
pip download mmcv==2.1.0 --no-binary :all: --no-deps -d mmcv_src
```
解壓 `mmcv_src\mmcv-2.1.0.tar.gz` 後，在「x64 Native Tools Command Prompt for VS 2022」執行（約 20 分鐘）：
```
conda activate hitter
set CUDA_HOME=%CONDA_PREFIX%\Library
set MMCV_WITH_OPS=1
set FORCE_CUDA=1
set TORCH_CUDA_ARCH_LIST=8.9;12.0
set DISTUTILS_USE_SDK=1
cd mmcv-2.1.0
pip wheel . --no-build-isolation --no-deps -w dist
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
