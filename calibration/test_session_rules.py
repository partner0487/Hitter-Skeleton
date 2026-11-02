"""檢查校正端（calibration/stereo.py）與主程式（Src/UI_Control/utils/stereo_calibration.py）
對 session 名稱與打擊方向的判斷一致；兩邊各寫一份，改了一邊沒改另一邊時這裡會失敗。

執行：python calibration/test_session_rules.py（也可用 pytest）
"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Src" / "UI_Control"))

import stereo  # noqa: E402
from utils import stereo_calibration as main  # noqa: E402

NAMES = ["20261029_085223_righthit", "20261029_103826_lefthit", "20261019_000000_righthit",
         "20261101_120000", "20261101_120000_SN25083472", "20261101_120000_lefthitx",
         "2026110_120000", "20261101-120000", "Calibration", "notes"]
# (資料夾名稱, session.json 的 batter_side；None 表示沒有 session.json)
SESSIONS = [("20261029_085223_righthit", None), ("20261029_103826_lefthit", None),
            ("20261101_120000", None), ("20261101_120001", "left"), ("20261101_120002", "right"),
            ("20261101_120003_righthit", "left"), ("20261101_120004_lefthit", "bogus"),
            ("20261101_120005", "")]


def test_session_name_rule():
    for name in NAMES:
        assert bool(stereo.SESSION_NAME.fullmatch(name)) == bool(main._SESSION_NAME.fullmatch(name)), name


def test_side_names_and_suffixes():
    assert set(stereo.BATTER_SIDES) == set(main._SIDE_SUFFIX) == set(main._SIDE_NAME)
    assert stereo.BATTER_SIDES == main._SIDE_NAME
    for side, suffix in stereo.BATTER_SUFFIX.items():
        assert main._SIDE_SUFFIX[side] == "_" + suffix, side


def test_batter_side_of():
    with tempfile.TemporaryDirectory() as tmp:
        for name, side in SESSIONS:
            session = Path(tmp) / name
            session.mkdir()
            if side is not None:
                (session / "session.json").write_text(json.dumps({"batter_side": side}), encoding="utf-8")
            expected = stereo.batter_side_of(session)
            assert main.batter_side_of({}, session) == expected, name
            # stereo.json 的欄位只在 session 本身判斷不出來時才用
            for field in ("right", "left"):
                assert main.batter_side_of({"batter_side": field}, session) == (expected or field), (name, field)


if __name__ == "__main__":
    for test in (test_session_name_rule, test_side_names_and_suffixes, test_batter_side_of):
        test()
        print(f"✔ {test.__name__}")
