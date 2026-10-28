"""Known-point wood-frame stereo PnP, with paired-pose consensus.

All points are original image pixels or millimetres. A pair must show the same
stationary physical frame in both cameras. The frame may move BETWEEN pairs;
only the camera-to-camera transform is compared between those different poses.
"""
import copy
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np


CONVENTION = "X_cs = R_cs_from_cf @ X_cf + t_cs_from_cf_mm"
WORLD_CONVENTION = "X_camera = R_camera_from_world @ X_world_mm + t_camera_from_world_mm"


def wood_frame_target():
    """The measured two-face frame; coordinates are fixed in millimetres.

    Front and side marks are different physical points on perpendicular faces.
    Both camera poses nevertheless refer to this SAME world coordinate system.
    The 0.3 cm cardboard offset is already included in Z=-3 and X=403.
    """
    front = [[80., 320., -3.], [320., 320., -3.],
             [320., 80., -3.], [80., 80., -3.]]
    side = [[403., 320., 30.], [403., 320., 270.],
            [403., 80., 270.], [403., 80., 30.]]
    front_ids, side_ids = [f"F{i}" for i in range(1, 5)], [f"S{i}" for i in range(1, 5)]
    return {
        "schema": "wood_frame_target_v1", "units": "mm",
        "points": [{"id": name, "xyz_mm": xyz} for name, xyz in
                   zip(front_ids + side_ids, front + side)],
        "view_point_ids": {"cf": front_ids, "cs": side_ids},
        "metadata": {
            "shape": "perpendicular_two_face_frame",
            "cardboard_thickness_mm": 3.,
            "cardboard_thickness_input": {"value": .3, "units": "cm"},
            "origin": "The supplied wood-frame world origin; coordinates preserved without reorientation.",
            "axes": "XYZ exactly as supplied for the measured frame.",
            "front_plane": "Z = -3 mm; F1,F2,F3,F4 in supplied physical order.",
            "side_plane": "X = 403 mm; S1,S2,S3,S4 in supplied physical order.",
            "thickness_note": "The 3 mm offset is already included in all point coordinates; do not apply it again.",
        },
    }


def rectangle_target(width_mm, height_mm, cardboard_thickness_mm=0.):
    """Four measured corners, X right, Z up, +Y from backing toward paper.

    The origin is the lower-left corner on the backing plane. Paper-face points
    are at Y=thickness; thickness is an explicit physical offset, not a scale.
    This is a rectangular target helper, not an assumed model of a user's frame.
    """
    width, height, thickness = map(float, (width_mm, height_mm, cardboard_thickness_mm))
    if not np.isfinite([width, height, thickness]).all() or min(width, height) <= 0 or thickness < 0:
        raise ValueError("木框寬、高須為正值；紙板厚度須為非負毫米數。")
    return {
        "schema": "wood_frame_target_v1", "units": "mm",
        "points": [{"id": name, "xyz_mm": xyz} for name, xyz in zip(
            ("P1", "P2", "P3", "P4"),
            ([0., thickness, 0.], [width, thickness, 0.],
             [width, thickness, height], [0., thickness, height]))],
        "metadata": {
            "shape": "rectangle", "width_mm": width, "height_mm": height,
            "cardboard_thickness_mm": thickness,
            "origin": "Lower-left corner on the backing plane, before the cardboard offset.",
            "axes": "X right, Z up, +Y from backing plane toward annotated paper face.",
            "point_order": "P1 lower-left, P2 lower-right, P3 upper-right, P4 upper-left; physical labels, not screen order.",
        },
    }


def validate_target(target):
    """Return ordered point IDs and an N x 3 array, rejecting ambiguous input."""
    if target.get("schema") != "wood_frame_target_v1" or target.get("units") != "mm":
        raise ValueError("需要 wood_frame_target_v1，座標單位固定為 mm。")
    points = target.get("points", [])
    ids = [p.get("id") for p in points]
    if len(ids) < 4 or any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("至少需要 4 個不重複且有名稱的木框控制點。")
    xyz = np.asarray([p.get("xyz_mm") for p in points], dtype=np.float64)
    if xyz.shape != (len(ids), 3) or not np.isfinite(xyz).all():
        raise ValueError("每個控制點須有有限的 XYZ 毫米座標。")
    singular = np.linalg.svd(xyz - xyz.mean(axis=0), compute_uv=False)
    if singular[0] <= 0 or singular[1] < singular[0] * 1e-6:
        raise ValueError("木框控制點不可全部重合或共線。")
    if len(np.unique(xyz, axis=0)) != len(xyz):
        raise ValueError("木框控制點的 XYZ 座標不可重複。")
    views = target.get("view_point_ids", {role: ids for role in ("cf", "cs")})
    if not isinstance(views, dict) or set(views) != {"cf", "cs"}:
        raise ValueError("木框 view_point_ids 必須定義 cf 與 cs 的控制點 ID。")
    for role in ("cf", "cs"):
        role_ids = views[role]
        if (not isinstance(role_ids, list) or len(role_ids) < 4
                or any(not isinstance(i, str) for i in role_ids)
                or len(set(role_ids)) != len(role_ids) or not set(role_ids).issubset(ids)):
            raise ValueError(f"{role.upper()} 至少須有 4 個不重複、存在於木框的控制點 ID。")
        subset = xyz[[ids.index(i) for i in role_ids]]
        singular = np.linalg.svd(subset - subset.mean(axis=0), compute_uv=False)
        if singular[0] <= 0 or singular[1] < singular[0] * 1e-6:
            raise ValueError(f"{role.upper()} 的木框控制點不可共線。")
    return ids, np.ascontiguousarray(xyz)


def _camera(camera, role):
    K = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
    dist = np.asarray(camera["distortion_coefficients"], dtype=np.float64)
    size = np.asarray(camera["image_size"], dtype=np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or min(K[0, 0], K[1, 1]) <= 0 or not np.allclose(K[2], [0, 0, 1])
            or abs(K[0, 1]) > 1e-8 or abs(K[1, 0]) > 1e-8
            or dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all()
            or size.shape != (2,) or not np.isfinite(size).all()
            or np.any(size <= 0) or not np.equal(size, size.astype(int)).all()):
        raise ValueError(f"{role.upper()} 內參、畸變係數或影像尺寸無效。")
    return K, dist.reshape(-1, 1), size.astype(int)


def _pixels(view, ids, size, pair_id, role):
    if list(view.get("image_size", [])) != size.tolist():
        raise ValueError(f"{pair_id} {role.upper()} 影像尺寸與內參不同，不能縮放或裁切後套用。")
    points = view.get("points", {})
    if not isinstance(points, dict) or set(points) != set(ids):
        raise ValueError(f"{pair_id} {role.upper()} 必須標記該視角全部指定 ID 的木框控制點。")
    pixels = np.asarray([points[i] for i in ids], dtype=np.float64)
    if (pixels.shape != (len(ids), 2) or not np.isfinite(pixels).all()
            or np.any(pixels < 0) or np.any(pixels >= size)):
        raise ValueError(f"{pair_id} {role.upper()} 點位須位於原始影像內且為有限像素座標。")
    if len(np.unique(pixels, axis=0)) != len(ids):
        raise ValueError(f"{pair_id} {role.upper()} 不可將不同控制點標在同一像素。")
    hull = cv2.convexHull(pixels.astype(np.float32))
    if cv2.contourArea(hull) < 25.:
        raise ValueError(f"{pair_id} {role.upper()} 標記幾乎共線或木框過小，請換角度拍攝。")
    return np.ascontiguousarray(pixels)


def _project(xyz, R, t, camera):
    K, dist, _ = camera
    return cv2.projectPoints(xyz, cv2.Rodrigues(R)[0], t, K, dist)[0].reshape(-1, 2)


def _rms(actual, predicted):
    return float(np.sqrt(np.mean(np.sum((actual - predicted) ** 2, axis=1))))


def _poses(xyz, pixels, camera):
    K, dist, _ = camera
    center = xyz.mean(axis=0)
    _, singular, basis_rows = np.linalg.svd(xyz - center, full_matrices=False)
    planar = bool(singular[-1] < singular[0] * 1e-7)
    seeds = []
    if planar:
        # IPPE is best conditioned on a centered Z=0 plane. Explicitly convert
        # arbitrary faces (including Z=-3 and X=403) then restore the SAME world
        # coordinates. Never let each camera's local plane become its world.
        basis = basis_rows.T
        if np.linalg.det(basis) < 0:
            basis[:, -1] *= -1
        local = np.ascontiguousarray((xyz - center) @ basis)
        local[:, 2] = 0.
        result = cv2.solvePnPGeneric(local, pixels, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if result[0]:
            for rvec, tvec in zip(result[1], result[2]):
                if np.isfinite(rvec).all() and np.isfinite(tvec).all():
                    R = cv2.Rodrigues(rvec)[0] @ basis.T
                    seeds.append((cv2.Rodrigues(R)[0], tvec.reshape(3, 1) - (R @ center).reshape(3, 1)))
    # SQPnP also handles exactly fronto-parallel IPPE degeneracies and provides
    # a useful independent seed without discarding either planar branch.
    result = cv2.solvePnPGeneric(xyz, pixels, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if result[0]:
        seeds.extend(zip(result[1], result[2]))
    poses = []
    for rvec, tvec in seeds:
        # Keep both IPPE branches: independent monocular "best" selections may
        # disagree, while several paired poses resolve the common stereo rig.
        rvec, tvec = np.array(rvec, float), np.array(tvec, float)
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            continue
        try:
            rvec, tvec = cv2.solvePnPRefineLM(xyz, pixels, K, dist, rvec, tvec)
        except cv2.error:
            continue
        R, t = cv2.Rodrigues(rvec)[0], tvec.reshape(3)
        depth = (xyz @ R.T + t)[:, 2]
        if not np.isfinite(depth).all() or np.min(depth) <= 1e-6:
            continue
        error = _rms(pixels, _project(xyz, R, t, camera))
        if not np.isfinite(error):
            continue
        if any(np.linalg.norm(R - p["R"]) < 1e-6 and np.linalg.norm(t - p["t"]) < 1e-4 for p in poses):
            continue
        poses.append({"R": R, "t": t, "rms_px": error, "min_depth_mm": float(depth.min())})
    return sorted(poses, key=lambda p: p["rms_px"]), planar


def _pair_error(R, t, pair, xyz_by_role, cameras):
    """Transfer each independent target pose across a proposed camera rig."""
    best = float("inf")
    for cf in pair["poses"]["cf"]:
        Rcs = R @ cf["R"]
        tcs = R @ cf["t"] + t
        if np.min((xyz_by_role["cs"] @ Rcs.T + tcs)[:, 2]) <= 0:
            continue
        forward = _rms(pair["pixels"]["cs"], _project(xyz_by_role["cs"], Rcs, tcs, cameras["cs"]))
        for cs in pair["poses"]["cs"]:
            Rcf = R.T @ cs["R"]
            tcf = R.T @ (cs["t"] - t)
            if np.min((xyz_by_role["cf"] @ Rcf.T + tcf)[:, 2]) <= 0:
                continue
            reverse = _rms(pair["pixels"]["cf"], _project(xyz_by_role["cf"], Rcf, tcf, cameras["cf"]))
            best = min(best, float(np.sqrt((forward ** 2 + reverse ** 2) / 2.)))
    return best


def _world_pose(pose):
    R, t = pose["R"], pose["t"]
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = R, t
    inverse = np.eye(4)
    inverse[:3, :3], inverse[:3, 3] = R.T, -R.T @ t
    return {"R_camera_from_world": R.tolist(), "t_camera_from_world_mm": t.tolist(),
            "T_camera_from_world_mm": transform.tolist(), "T_world_from_camera_mm": inverse.tolist(),
            "camera_center_world_mm": (-R.T @ t).tolist(), "rms_px": pose["rms_px"],
            "min_depth_mm": pose["min_depth_mm"]}


def fit_wood_frame_rig(cameras, target, pairs, *, min_pairs=3, max_rms_px=3.):
    """Choose ONE synchronized pair using all paired observations as checks.

    ``cameras`` has cf/cs intrinsic_matrix, distortion_coefficients, image_size.
    Each pair has a unique ``id`` and cf/cs views, each with image_size and
    ``points`` mapping that role's target point IDs to original [u,v] pixels.
    ``target.view_point_ids`` may specify different points on different faces
    for cf and cs; all their XYZ coordinates must share one physical frame.
    Optional
    image ``path`` and capture metadata are retained for reproducibility.

    Cross-validation means other frame poses check the candidate stereo rig.
    It is selection data, not an independent accuracy certification.
    """
    if not isinstance(min_pairs, int) or min_pairs < 3:
        raise ValueError("至少需要 3 組同步照片；建議拍攝約 15 組不同位置與角度。")
    if not np.isfinite(max_rms_px) or max_rms_px <= 0:
        raise ValueError("重投影誤差門檻須為正值。")
    ids, xyz = validate_target(target)
    view_ids = target.get("view_point_ids", {role: ids for role in ("cf", "cs")})
    xyz_by_role = {role: np.ascontiguousarray(xyz[[ids.index(i) for i in view_ids[role]]])
                   for role in ("cf", "cs")}
    camera_arrays = {role: _camera(cameras[role], role) for role in ("cf", "cs")}
    pairs = list(pairs)
    if len(pairs) < min_pairs:
        raise ValueError(f"至少需要 {min_pairs} 組同步照片；目前只有 {len(pairs)} 組。")
    names = [p.get("id") for p in pairs]
    if any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError("每組照片需要不重複且非空白的 id。")
    prepared, diagnostics = [], []
    for source in pairs:
        item = {"id": source["id"], "pixels": {}, "poses": {}}
        diagnostic = {"id": source["id"], "status": "rejected", "reason": "", "cameras": {}}
        for role in ("cf", "cs"):
            pixels = _pixels(source[role], view_ids[role], camera_arrays[role][2], source["id"], role)
            item["pixels"][role] = pixels
            try:
                solutions, planar = _poses(xyz_by_role[role], pixels, camera_arrays[role])
            except cv2.error as error:
                solutions, planar = [], False
                diagnostic["reason"] = f"{role.upper()} PnP 失敗：{error}"
            acceptable = [p for p in solutions if p["rms_px"] <= max_rms_px]
            item["poses"][role] = acceptable
            diagnostic["cameras"][role] = {
                "planar": planar, "solution_count": len(solutions),
                "accepted_solution_count": len(acceptable),
                "rms_px": solutions[0]["rms_px"] if solutions else None,
                "alternative_rms_gap_px": solutions[1]["rms_px"] - solutions[0]["rms_px"] if len(solutions) > 1 else None,
            }
            if not acceptable and not diagnostic["reason"]:
                diagnostic["reason"] = f"{role.upper()} 無正深度且重投影誤差 ≤ {max_rms_px:g} px 的 PnP 解。"
        diagnostics.append(diagnostic)
        if all(item["poses"].values()):
            item["diagnostic"] = diagnostic
            prepared.append(item)
    if len(prepared) < min_pairs:
        raise ValueError(f"只有 {len(prepared)} 組通過 PnP 檢查，至少需要 {min_pairs} 組。"
                         + " ".join(f"{d['id']}: {d['reason']}" for d in diagnostics if d["reason"]))
    proposals = []
    support_limit = max(3., 3. * max_rms_px)
    for pair in prepared:
        for cf in pair["poses"]["cf"]:
            for cs in pair["poses"]["cs"]:
                R = cs["R"] @ cf["R"].T
                t = cs["t"] - R @ cf["t"]
                if np.linalg.norm(t) <= 1e-3:
                    continue
                checks = [_pair_error(R, t, other, xyz_by_role, camera_arrays) for other in prepared]
                held = [e for e, other in zip(checks, prepared) if other is not pair]
                support = sum(e <= support_limit for e in checks)
                median = float(np.median(held))
                own = float(np.sqrt((cf["rms_px"] ** 2 + cs["rms_px"] ** 2) / 2))
                if not np.isfinite(median):
                    continue
                proposals.append({"pair": pair, "R": R, "t": t, "cf": cf, "cs": cs,
                                  "checks": checks, "support": support, "held_rms_px": median,
                                  "own_rms_px": own, "score_px": median + .25 * own})
    # A majority must agree; several views of an inconsistent/moved camera do
    # not become a valid rig merely because one photograph has a tiny residual.
    needed = max(min_pairs, len(prepared) // 2 + 1)
    eligible = [p for p in proposals if p["support"] >= needed and p["held_rms_px"] <= support_limit]
    if not eligible:
        raise ValueError("照片組之間無一致的雙相機外參。請確認兩台相機未移動、同組木框靜止且各視角點 ID 正確。")
    best = min(eligible, key=lambda p: (p["score_px"], -p["support"], p["pair"]["id"]))
    # A low monocular residual does not resolve planar mirror ambiguity. Refuse
    # two materially different stereo rigs with indistinguishable multi-pose
    # scores, instead of silently choosing whichever OpenCV returned first.
    ambiguity_margin = max(.25, .15 * best["score_px"])
    competing = []
    for proposal in eligible:
        if proposal["score_px"] > best["score_px"] + ambiguity_margin:
            continue
        angle = float(np.linalg.norm(cv2.Rodrigues(proposal["R"] @ best["R"].T)[0]))
        shift = float(np.linalg.norm(proposal["t"] - best["t"]))
        if angle > np.deg2rad(2.) or shift > max(10., .02 * np.linalg.norm(best["t"])):
            competing.append(proposal)
    if competing:
        raise ValueError("木框平面 PnP 仍有無法區分的外參解；請增加明顯傾斜、不同距離與位置的同步照片後重算。")
    for pair, cross_error in zip(prepared, best["checks"]):
        diagnostic = pair["diagnostic"]
        own_proposals = [p for p in proposals if p["pair"] is pair]
        chosen = min(own_proposals, key=lambda p: p["score_px"]) if own_proposals else None
        diagnostic.update({
            "status": "accepted" if cross_error <= support_limit else "rejected",
            "reason": "" if cross_error <= support_limit else "與多數照片的雙相機幾何不一致。",
            "cross_projection_rms_px": cross_error if np.isfinite(cross_error) else None,
            "selection_score_px": chosen["score_px"] if chosen else None,
            "other_pairs_median_rms_px": chosen["held_rms_px"] if chosen else None,
        })
    best["pair"]["diagnostic"]["status"] = "best"
    warnings = []
    if len(pairs) < 15:
        warnings.append("建議拍攝約 15 組，增加木框位置、傾斜角度及影像覆蓋範圍。")
    if any(d["status"] == "rejected" for d in diagnostics):
        warnings.append("部分照片未通過 PnP 或雙相機一致性檢查，詳見 candidate_diagnostics。")
    pose_rotations = [p["poses"]["cf"][0]["R"] for p in prepared]
    max_angle = max(float(np.linalg.norm(cv2.Rodrigues(R @ pose_rotations[0].T)[0]))
                    for R in pose_rotations)
    if max_angle < np.deg2rad(8):
        warnings.append("木框姿態變化小於 8°；請增加傾斜角度，降低平面 PnP 雙解的不確定性。")
    if best["held_rms_px"] > max_rms_px:
        warnings.append("其他照片的中位跨相機重投影誤差偏高，建議重新檢查點位與內參。")
    return {
        "schema": "metric_stereo_v1", "method": "pnp", "target_type": "wood_frame",
        "units": "mm", "coordinate_convention": CONVENTION,
        "R_cs_from_cf": best["R"].tolist(), "t_cs_from_cf_mm": best["t"].tolist(),
        "baseline_mm": float(np.linalg.norm(best["t"])), "intrinsics": copy.deepcopy(cameras),
        "wood_frame": copy.deepcopy(target), "best_pair_id": best["pair"]["id"],
        "world": {
            "source_pair_id": best["pair"]["id"], "units": "mm",
            "coordinate_convention": WORLD_CONVENTION,
            "description": "The selected pair's frame defines world coordinates. Other pairs only validate the fixed stereo rig.",
            "cameras": {role: _world_pose(best[role]) for role in ("cf", "cs")},
        },
        "fit": {
            "selection": "One synchronized pair, ranked by median bidirectional reprojection on other paired poses + 0.25 * own RMS.",
            "pair_rms_px": best["own_rms_px"], "other_pairs_median_rms_px": best["held_rms_px"],
            "score_px": best["score_px"], "support_pairs": best["support"],
            "support_threshold_px": support_limit, "max_pnp_rms_px": max_rms_px,
            "ambiguity_score_margin_px": ambiguity_margin,
        },
        "candidate_diagnostics": diagnostics, "source_pairs": copy.deepcopy(pairs),
        "training_pairs": [best["pair"]["id"]],
        "rejected_pairs": [d["id"] for d in diagnostics if d["status"] == "rejected"],
        "warnings": warnings, "independently_validated_for_measurement": False,
        "held_out_note": "Other poses are used for candidate selection; they are not an independent accuracy test.",
    }


def save_wood_frame_rig(path, rig):
    """Write a self-contained JSON atomically, without changing intrinsics."""
    if rig.get("target_type") != "wood_frame" or rig.get("schema") != "metric_stereo_v1":
        raise ValueError("只接受木框外參結果。")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    text = json.dumps(rig, ensure_ascii=False, indent=2, allow_nan=False, default=convert)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent),
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            stream.write(text)
        # Also validate the serialized representation before replacing a file.
        load_wood_frame_rig(temporary)
        os.replace(temporary, str(path))
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
    return path


def load_wood_frame_rig(path):
    """Load with existing stereo validation and verify both world transforms."""
    from Src.calibration.chessboard_cross.chessboard_extrinsic import load_rig
    rig = load_rig(path)
    if rig.get("target_type") != "wood_frame":
        raise ValueError("此檔案不是木框外參。")
    validate_target(rig["wood_frame"])
    world = rig["world"]
    if (world.get("units") != "mm" or world.get("coordinate_convention") != WORLD_CONVENTION
            or world.get("source_pair_id") != rig.get("best_pair_id")):
        raise ValueError("木框世界座標定義或來源照片組不一致。")
    transforms = {}
    for role in ("cf", "cs"):
        pose = world["cameras"][role]
        R = np.asarray(pose["R_camera_from_world"], float)
        t = np.asarray(pose["t_camera_from_world_mm"], float)
        T = np.asarray(pose["T_camera_from_world_mm"], float)
        inverse = np.asarray(pose["T_world_from_camera_mm"], float)
        if (R.shape != (3, 3) or t.shape != (3,) or T.shape != (4, 4) or inverse.shape != (4, 4)
                or not np.isfinite(R).all() or not np.isfinite(t).all()
                or not np.isfinite(T).all() or not np.isfinite(inverse).all()
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(R), 1., atol=1e-5)
                or not np.allclose(T[:3, :3], R) or not np.allclose(T[:3, 3], t)
                or not np.allclose(T[3], [0, 0, 0, 1]) or not np.allclose(T @ inverse, np.eye(4), atol=1e-5)):
            raise ValueError(f"{role.upper()} 世界座標轉換無效。")
        transforms[role] = T
    relative = transforms["cs"] @ np.linalg.inv(transforms["cf"])
    if (not np.allclose(relative[:3, :3], rig["R_cs_from_cf"], atol=1e-5)
            or not np.allclose(relative[:3, 3], rig["t_cs_from_cf_mm"], atol=1e-4)):
        raise ValueError("雙相機外參與所選照片組的世界座標不一致。")
    return rig
