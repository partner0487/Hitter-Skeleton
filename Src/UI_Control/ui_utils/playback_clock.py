"""播放時鐘：依實際經過的時間決定該播到第幾幀，回放速度與真實一致。

原本每幀固定等一段時間，沒算進畫圖本身的時間（3D 畫面、比較頁四個畫面每幀要幾十 ms），
實際都比設定的倍率慢。這裡改成：目標幀 = 起點 + 經過秒數 × fps × 倍率；
慢動作時每幀準時，畫面來不及時跳過中間幀。
"""
import time

TICK_MS = 5   # 檢查時鐘的間隔；實際換幀由經過時間決定


def rate_from_text(text: str, default: float = 1.0) -> float:
    """速度選單文字（0.25、0.5x 等）→ 倍率"""
    try:
        rate = float(text.lower().replace("x", "").strip())
    except ValueError:
        return default
    return rate if rate > 0 else default


class PlaybackClock:
    def __init__(self):
        self.t0 = None
        self.start_frame = 0
        self.frames_per_s = 60.0

    def start(self, frame: int, fps: float, rate: float):
        """從 frame 開始計時；播放中換倍率時以目前幀重新開始"""
        self.t0 = time.perf_counter()
        self.start_frame = int(frame)
        self.frames_per_s = fps * rate

    def target(self) -> int:
        return self.start_frame + int((time.perf_counter() - self.t0) * self.frames_per_s)
