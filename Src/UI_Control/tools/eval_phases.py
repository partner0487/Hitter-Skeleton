"""用驗證集評估 2D 階段偵測：讀 phase_truth.csv 的人工標註（擊球點、著地），
對每個資料夾的 bat_side.json／person_side.json 跑與 2D 分析相同的 PhaseDetector，列出誤差與準確率。

phase_truth.csv 欄位：folder（相對 Db/）、impact、landing（未標註留空）、source（標註來源）、batter（left＝左打，留空＝右打）
幀號與 UI 進度條相同（從 0 起算）；擊球點以「球碰到球棒的那一幀」為準。

用法（在任何目錄）：
  python Src/UI_Control/tools/eval_phases.py
"""
import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

UI_CONTROL = Path(__file__).resolve().parents[1]
DB = UI_CONTROL.parents[1] / "Db"
TRUTH = Path(__file__).with_name("phase_truth.csv")
sys.path.insert(0, str(UI_CONTROL))

from utils.phase_detect2d import PhaseDetector  # noqa: E402

# 與 analysis2d_widget.detectPhases 相同的參數
DETECTOR_KWARGS = dict(th_lift=5.0, th_land=5.0, land_hold=3)


def read_records(folder, name):
    path = folder / name
    return pd.read_json(path, orient="records") if path.exists() else pd.DataFrame()


def detect(folder, batter_index=0):
    with contextlib.redirect_stdout(io.StringIO()):  # 偵測器會印很多除錯訊息
        detector = PhaseDetector(person_df=read_records(folder, "person_side.json"), batter_index=batter_index,
                                 ball_df=read_records(folder, "ball_side.json"),
                                 bat_df=read_records(folder, "bat_side.json"), **DETECTOR_KWARGS)
        return detector.detect_phases()


def summarize(name, errors):
    if not errors:
        return
    exact = sum(e == 0 for e in errors)
    near = sum(abs(e) <= 1 for e in errors)
    mean_abs = sum(abs(e) for e in errors) / len(errors)
    print(f"{name}：完全正確 {exact}/{len(errors)}，±1 幀內 {near}/{len(errors)}，平均誤差 {mean_abs:.2f} 幀")


def main():
    truth = pd.read_csv(TRUTH, encoding="utf-8")
    errors = {"impact": [], "landing": []}
    for row in truth.itertuples():
        folder = DB / row.folder
        if not (folder / "bat_side.json").exists():
            print(f"{row.folder}：找不到 bat_side.json，略過")
            continue
        phases = detect(folder, 1 if getattr(row, "batter", None) == "left" else 0)
        cells = []
        for key, predicted in (("impact", phases.F_impact), ("landing", phases.F_landing)):
            expected = getattr(row, key)
            if pd.isna(expected):
                continue
            expected = int(expected)
            if predicted is None:
                cells.append(f"{key} {expected} → 無")
                continue
            error = predicted - expected
            errors[key].append(error)
            cells.append(f"{key} {expected} → {predicted} ({error:+d})")
        print(f"{row.folder:40s} " + "｜".join(cells))
    print()
    summarize("擊球點", errors["impact"])
    summarize("著地", errors["landing"])


if __name__ == "__main__":
    main()
