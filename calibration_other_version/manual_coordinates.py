"""Versioned image annotations for a common world origin, axes and local scale.

These annotations describe projected directions and a scale at the reference
segment's depth. They are deliberately not an intrinsic/extrinsic calibration.
All public functions return independent data and never modify their arguments.
"""
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile


SCHEMA = "manual_coordinates_v1"
UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
VIEW_NAMES = {"CF": "正面 CF", "CS": "側面 CS"}
LIMITATION = (
    "影像中的 XYZ 是三維座標軸的投影方向；mm/px 只適用於參考線段附近的同深度區域。"
    "這些標記不能單獨求出相機內外參、透視校正或真正的三維座標。"
)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} 必須是數值。".format(name))
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValueError("{} 必須是有限數值。".format(name))
    if not math.isfinite(result):
        raise ValueError("{} 必須是有限數值。".format(name))
    return result


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError("{} 格式不正確。".format(name))
    return value


def _image(value, name):
    value = _mapping(value, name)
    path = value.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("{} 尚未選擇影像。".format(name))
    dimensions = []
    for axis in ("width", "height"):
        number = value.get(axis)
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise ValueError("{} 的影像尺寸不正確。".format(name))
        dimensions.append(number)
    return {"path": path, "width": dimensions[0], "height": dimensions[1]}


def _point(value, image, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("{} 尚未標記。".format(name))
    x, y = [_number(number, name) for number in value]
    if not (0 <= x < image["width"] and 0 <= y < image["height"]):
        raise ValueError("{} 超出原始影像範圍。".format(name))
    return [x, y]


def _distance(first, second, name):
    distance = math.hypot(second[0] - first[0], second[1] - first[1])
    if not math.isfinite(distance) or distance <= 1e-6:
        raise ValueError("{} 的兩點不能重合。".format(name))
    return distance


def validate_document(document):
    """Validate required observations and recompute all derived quantities.

    Paths need not exist here: consumers can read data after images are moved.
    Image readers must check dimensions before displaying saved annotations.
    Stored derived numbers are never trusted on import.
    """
    document = _mapping(document, "標記檔")
    if (document.get("schema") != SCHEMA
            or type(document.get("schema_version")) is not int
            or document.get("schema_version") != 1):
        raise ValueError("不支援的標記檔版本；需要 manual_coordinates_v1。")
    if document.get("is_metric_3d_calibration", False) is not False:
        raise ValueError("手動影像標記不能宣告為三維外參。")
    views = _mapping(document.get("views"), "雙視角資料")
    result = {
        "schema": SCHEMA, "schema_version": 1,
        "kind": "image_axis_and_scale_annotations",
        "is_metric_3d_calibration": False,
        "created_at": str(document.get("created_at") or datetime.now(timezone.utc).isoformat()),
        "coordinate_convention": {
            "pixel_origin": "top_left", "pixel_x": "right", "pixel_y": "down",
            "world_axes": "right_handed_common_physical_origin",
            "world_origin": [0.0, 0.0, 0.0],
        },
        "limitation": LIMITATION,
        "notes": str(document.get("notes", "")),
        "views": {},
    }
    for role, title in VIEW_NAMES.items():
        view = _mapping(views.get(role), title)
        axes_image = _image(view.get("axes_image"), title + "座標影像")
        origin = _point(view.get("origin_px"), axes_image, title + "原點 O")
        axes = _mapping(view.get("axes_px"), title + " XYZ 軸")
        endpoints, directions = {}, {}
        for axis in ("X", "Y", "Z"):
            endpoint = _point(axes.get(axis), axes_image, title + " +" + axis)
            distance = _distance(origin, endpoint, title + " +" + axis)
            endpoints[axis] = endpoint
            directions[axis] = [(endpoint[i] - origin[i]) / distance for i in (0, 1)]
        scale = _mapping(view.get("scale"), title + "尺度")
        scale_image = _image(scale.get("image"), title + "尺度影像")
        points = scale.get("endpoints_px")
        if not isinstance(points, (list, tuple)) or len(points) != 2:
            raise ValueError(title + "請標記尺度線段的兩端 A、B。")
        points = [_point(point, scale_image, title + "尺度 " + label)
                  for label, point in zip(("A", "B"), points)]
        distance = _distance(points[0], points[1], title + "尺度")
        length = _number(scale.get("length"), title + "真實長度")
        if length <= 0:
            raise ValueError(title + "真實長度必須大於 0。")
        unit = scale.get("unit")
        if not isinstance(unit, str) or unit not in UNIT_TO_MM:
            raise ValueError(title + "長度單位必須是 mm、cm 或 m。")
        length_mm = length * UNIT_TO_MM[unit]
        ratio = length_mm / distance
        if not math.isfinite(length_mm) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(title + "真實長度超出可表示範圍。")
        result["views"][role] = {
            "axes_image": axes_image, "origin_px": origin,
            "axes_px": endpoints, "axis_unit_directions_px": directions,
            "scale": {
                "image": scale_image, "endpoints_px": points,
                "length": length, "unit": unit, "length_mm": length_mm,
                "distance_px": distance, "mm_per_pixel": ratio,
                "scope": "local_reference_segment_depth_only",
            },
        }
    return result


def make_document(views, notes=""):
    """Create a validated, complete document from the two annotated views."""
    return validate_document({
        "schema": SCHEMA, "schema_version": 1, "views": views, "notes": notes,
    })


def _images(document):
    for view in document["views"].values():
        yield view["axes_image"]
        yield view["scale"]["image"]


def save_document(path, document):
    """Atomically export JSON, storing image paths relative to the JSON folder."""
    destination = Path(path).resolve()
    result = validate_document(document)
    for image in _images(result):
        source = Path(image["path"]).resolve()
        try:
            image["path"] = os.path.relpath(str(source), str(destination.parent))
        except ValueError:  # Different Windows drives.
            image["path"] = str(source)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(destination.parent),
                prefix=destination.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(str(temporary), str(destination))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def load_document(path):
    """Read and validate JSON; resolve image paths relative to the saved file."""
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8-sig") as stream:
        document = json.load(stream)
    result = validate_document(document)
    for image in _images(result):
        image_path = Path(image["path"])
        if not image_path.is_absolute():
            image_path = source.parent / image_path
        image["path"] = str(image_path.resolve())
    return copy.deepcopy(result)
