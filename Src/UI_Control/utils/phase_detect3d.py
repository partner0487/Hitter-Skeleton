"""3D 頁的階段偵測。

離地、著地沿用 2D（utils/phase_detect2d.py）：腳的資料兩邊都是側面相機的 2D 骨架，
共用同一套判斷，兩頁的階段才會一致。
擊球點不同：2D 用 bat_side.json 的遮罩，3D 用 batht_df 已分好的棒頭座標取最高速幀；
3D 頁分析完後還會再用 3D 三角化的棒頭最高速覆寫，並以 detect_foot_phases 重算離地、著地。
"""
from typing import Optional

import numpy as np
import pandas as pd

from utils.phase_detect2d import PhaseDetector as PhaseDetector2D
from utils.phase_detect2d import PhaseResult  # noqa: F401  與 2D 共用，供 analysis3d_widget／compare_widget import


class PhaseDetector(PhaseDetector2D):
    # 3D 的擊球點最後由 3D 棒頭最高速覆寫，這裡不加 2D 的偏移
    IMPACT_OFFSET = 0

    def __init__(
        self,
        person_df: pd.DataFrame,
        batter_index: int,              # 0 右打、1 左打
        ball_df: pd.DataFrame = None,
        batht_df: pd.DataFrame = None,  # 球棒 head/tail 資料
        **kwargs,
    ):
        super().__init__(person_df=person_df, batter_index=batter_index, ball_df=ball_df, bat_df=None, **kwargs)
        self.batht_df = batht_df

    def _detect_impact(self, side: str = "right") -> Optional[int]:
        """棒頭極速法：batht_df 的棒頭座標相鄰幀位移最大的幀（間隔 > 3 幀不計）"""
        print("--- 開始偵測 Impact (棒頭極速法 3D) ---")
        if self.batht_df is None or self.batht_df.empty:
            return None

        bat_heads = {}
        for _, row in self.batht_df.iterrows():
            col = 'frame_num' if 'frame_num' in row else 'frame_number'
            if 'head' in row and row['head'] is not None and len(row['head']) >= 2:
                bat_heads[int(row[col])] = np.array(row['head'], dtype=np.float32)

        frames = sorted(bat_heads)
        max_speed, best_frame = -1.0, None
        for prev_f, curr_f in zip(frames, frames[1:]):
            if curr_f - prev_f > 3:  # 容許微小掉幀，斷層太大不計
                continue
            speed = float(np.linalg.norm(bat_heads[curr_f] - bat_heads[prev_f])) / (curr_f - prev_f)
            if speed > max_speed:
                max_speed, best_frame = speed, curr_f

        if best_frame is not None:
            print(f"✅ 成功找到擊球點(3D)！影格: {best_frame}, 棒頭最高速(px/frame): {max_speed:.2f}")
        return best_frame
