"""側拋／正投的自動錄影觸發：側面畫面 ROI 內出現明顯變化就觸發。

原本用 YOLO 找球、球「進入」ROI 才觸發，有三個問題：
- 偵測跟著即時骨架迴圈跑（約 30 次／秒），職業發球機的球速一次檢查間隔就飛過 1 m 以上，常整顆跳過 ROI
- 快速球在畫面上拖成殘影，YOLO 認不出是球
- 地上靜止的球若在 ROI 內，會一直被當成「球在裡面」，永遠等不到「進入」而不觸發

改成比較 ROI 內相鄰兩幀的差異，每個送出的相機畫面（60 fps）都檢查，運算量很小：
殘影越長變化越大，靜止的球不會產生變化。ROI 要畫在球路上、打者身體與球棒碰不到的位置
（打者前方、往投手方向約半個到一個身位），否則準備動作會提早觸發。

門檻以 tools/phase_truth.csv 的 27 支側拋／正投片段模擬（ROI 避開打者身體）：
全部觸發、在擊球前 2～19 幀；每 3 幀取 1 模擬三倍球速也全部觸發
"""
import cv2
import numpy as np


class MotionTrigger:
    DOWNSCALE = 4          # ROI 縮小倍率（降低雜訊與運算量）
    BLUR = 5               # 高斯模糊核大小
    DIFF_THRESH = 20       # 灰階差超過才算變化（壓掉相機雜訊）
    MIN_CHANGED_PX = 15    # 縮小後變化像素數門檻（模擬時 8～30 結果相近）

    def __init__(self):
        self.reset()

    def reset(self):
        """重新開始比較（換 ROI、錄完一段、模式切換時呼叫），下一幀只當基準不觸發"""
        self._prev = None
        self._roi = None

    def changed_pixels(self, frame: np.ndarray, roi) -> int:
        """與上一幀相比 ROI 內變化的像素數；第一幀或 ROI 不在畫面內回傳 0"""
        roi = tuple(int(v) for v in roi)
        if roi != self._roi:
            self._prev, self._roi = None, roi
        x, y, w, h = roi
        crop = frame[max(y, 0):max(y + h, 0), max(x, 0):max(x + w, 0)]
        if crop.size == 0 or min(crop.shape[:2]) < self.DOWNSCALE:
            return 0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        small = cv2.resize(gray, None, fx=1 / self.DOWNSCALE, fy=1 / self.DOWNSCALE, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (self.BLUR, self.BLUR), 0)
        prev, self._prev = self._prev, small
        if prev is None or prev.shape != small.shape:
            return 0
        return int(np.count_nonzero(cv2.absdiff(small, prev) > self.DIFF_THRESH))

    def update(self, frame: np.ndarray, roi) -> bool:
        """送入新的一幀，ROI 內有明顯變化時回傳 True"""
        return self.changed_pixels(frame, roi) >= self.MIN_CHANGED_PX
