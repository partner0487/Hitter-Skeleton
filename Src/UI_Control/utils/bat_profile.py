"""每位打者的球棒資料（棒長、棒重），存在 Db/<打者>/bat.json；沒填時用預設棒長。

用途：
- 2D 揮棒速度與側面比例尺：畫面上的球棒像素長度 = 棒長（utils/bat_speed2d.py），棒長不準速度就等比例偏差；
  甜蜜點（距棒頭 6 吋）在畫面上的位置也用棒長比例
- 3D：揮棒速度用三角化量到的棒頭，不需要棒長；有填棒長時在揮棒速度的提示裡比對實測棒長
  （偏短可能是握把端被手遮住，或遮罩少抓棒尖；後者會讓甜蜜點內縮、速度偏低）
- 棒重：目前只記錄（Blast 用來估揮棒功率），還沒有指標使用
單位與 Blast 相同：吋、盎司。
"""
import json
import os
from dataclasses import dataclass
from typing import Optional

from utils.bat_speed2d import BAT_LENGTH_M
from utils.paths import DB_DIR, clip_parts

BAT_FILE = "bat.json"
INCH_M = 0.0254
DEFAULT_LENGTH_IN = round(BAT_LENGTH_M / INCH_M, 1)   # 33.1 吋


@dataclass
class BatProfile:
    length_in: float
    weight_oz: Optional[float]   # None：沒填
    is_default: bool             # True：沒有 bat.json，用預設棒長

    @property
    def length_m(self) -> float:
        return BAT_LENGTH_M if self.is_default else self.length_in * INCH_M

    def describe(self) -> str:
        if self.is_default:
            return f"預設棒長 {BAT_LENGTH_M} m（未設定球棒）"
        weight = f"、{self.weight_oz:g} oz" if self.weight_oz else ""
        return f"棒長 {self.length_in:g} 吋（{self.length_m:.3f} m）{weight}"


def bat_file(player: str) -> str:
    return os.path.join(str(DB_DIR), player, BAT_FILE)


def load_bat(player: Optional[str]) -> BatProfile:
    default = BatProfile(DEFAULT_LENGTH_IN, None, True)
    if not player:
        return default
    try:
        with open(bat_file(player), encoding="utf-8") as f:
            data = json.load(f)
        length = float(data["length_in"])
    except (OSError, ValueError, KeyError, TypeError):
        return default
    weight = data.get("weight_oz")
    return BatProfile(length, float(weight) if weight else None, False)


def save_bat(player: str, length_in: float, weight_oz: Optional[float]) -> str:
    path = bat_file(player)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"length_in": round(length_in, 1), "weight_oz": round(weight_oz, 1) if weight_oz else None},
                  f, ensure_ascii=False, indent=1)
    return path


def reset_bat(player: str) -> None:
    try:
        os.remove(bat_file(player))
    except FileNotFoundError:
        pass


def bat_for_clip(path) -> BatProfile:
    """錄影路徑（Db/<打者>/...）對應打者的球棒資料"""
    parts = clip_parts(path)
    return load_bat(parts[0] if parts else None)
