"""下載模型權重到 Db/pretrain/：python download_weights.py

- epoch_210.pth：骨架 ViTPose（Halpe 26 點），放在本專案 GitHub Release（weights-v1），
  只含推論用權重（328 MB），推論結果與原本 1.1 GB 的訓練存檔完全相同
- yolo26m-seg.pt：已在 git 裡，clone 下來就有；不見時才從 Ultralytics 下載

已經存在的檔案一律不動（包含原本 1.1 GB 的 epoch_210.pth），只補缺少的。只用 Python 標準函式庫。
"""
import hashlib
import os
import sys
import urllib.request

PRETRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Db", "pretrain")

WEIGHTS = [
    ("epoch_210.pth",
     "https://github.com/partner0487/Hitter-Skeleton/releases/download/weights-v1/epoch_210.pth",
     "e3452dccea6ae63698e8222f5f75c4303fa46a95e976267adf6249e90172e556"),
    ("yolo26m-seg.pt",
     "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt",
     "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name, url, expected):
    dst = os.path.join(PRETRAIN_DIR, name)
    tmp = dst + ".part"
    print(f"下載 {name} …")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MB", end="", flush=True)
    print()
    if sha256(tmp) != expected:
        os.remove(tmp)
        raise RuntimeError(f"{name} 檢查碼不符，下載可能不完整，請重新執行")
    os.replace(tmp, dst)
    print(f"  完成：{dst}")


def main():
    os.makedirs(PRETRAIN_DIR, exist_ok=True)
    failed = []
    for name, url, expected in WEIGHTS:
        if os.path.exists(os.path.join(PRETRAIN_DIR, name)):
            print(f"已存在，略過：{name}")
            continue
        try:
            download(name, url, expected)
        except Exception as e:
            print(f"  失敗：{e}")
            failed.append(name)
    if failed:
        print("下載失敗：" + "、".join(failed))
        sys.exit(1)
    print("權重都在 Db/pretrain/ 了")


if __name__ == "__main__":
    main()
