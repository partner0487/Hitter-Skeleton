# Pitch-Skeleton-User-Interface

## Usage
### Installation
1. Clone this repo.
2. Setup conda environment:
    ```
    # 建立並啟用環境
    conda create -n Pitcher python=3.8 -y
    conda activate Pitcher

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
To obtain the vitpose、yolo and fast-reid wights, it can be downloaded from the https://drive.google.com/drive/folders/1D7Q5bTnTAfKkfLuppqUo4_8W4t0wrCmP?usp=sharing. The resulting data directory should look like this:
    
    Pitch-Skeleton-User-Interface
    |-- Db
    |   |-- pretrain
    |   |   |-- epoch_210.pth
    |   |   |-- vitpose_Sk26.pth
    |   |   |-- yolo11m-seg.pt
    |   |   `-- yolov7_x_syncbn_fast_8x16b-300e_coco_20221124_215331-ef949a68.pth
    |   `-- Record
    |       |-- {video_name}.mp4        # 原始影片
    |       |-- {video_name}_Sk26.mp4   # 疊加骨架可視化影片
    |       `-- {video_name}.json       # Bounding box 與 26 個關節點座標
    `-- Src
        |-- UI_Control
        |-- mmpose_main
        |-- mmpretrain_main
        |-- mmyolo_main
        `-- tracker

### Demo
1. Demo command
    ```
    python Src\UI_Control\main.py
    ```
2. 2D 相機
    ```
    利用相機去進行骨架偵測
    ```
3. 2D 影片
    ```
    利用影片進行骨架偵測
    ```
    
