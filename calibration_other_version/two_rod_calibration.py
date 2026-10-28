"""Eight-point stereo calibration from two independent four-ball rods."""
import copy
from itertools import combinations

import cv2
import numpy as np

from Src.calibration.calibration_workflow import save_json, validate_intrinsic
from Src.calibration.chessboard_cross.chessboard_metric import estimate_eight_point, triangulate, undistort
from Src.calibration.chessboard_cross.fluorescent_balls import DetectionConfig, detect_balls

IDS = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4']
GAPS = np.array([[200., 120., 180.], [100., 140., 210.]])


def rod_target(convention='centers'):
    if convention not in ('centers', 'surface_gaps'):
        raise ValueError('請選球心距離或球面淨空。')
    return {'schema': 'two_rod_target_v1', 'units': 'mm', 'ball_diameter_mm': 50.,
            'spacing_convention': convention, 'specified_spacings_mm': GAPS.tolist(),
            'center_spacings_mm': (GAPS + (50 if convention == 'surface_gaps' else 0)).tolist(),
            'point_ids': IDS, 'ordering': 'A1..A4 and B1..B4: physical top to bottom; same IDs in both views'}


def suggest_points(image, camera, target, config=None):
    """Reuse extractor detector; group by collinearity/projective cross ratio.

    A four-point line's cross ratio cannot distinguish reversal. All proposals
    require a human to confirm physical top/bottom in BOTH images.
    """
    candidates, _ = detect_balls(image, config or DetectionConfig())
    report = {'candidate_count': len(candidates), 'candidates': candidates, 'points': {},
              'note': '請核對 A/B 棒及實體上端；四點交比無法判斷整根棒倒置。'}
    if not 8 <= len(candidates) <= 16:
        report['note'] = '偵測到 {} 顆候選球；請調整 HSV／面積，或手動標記。'.format(len(candidates))
        return report
    pixels = np.array([p['center_px'] for p in candidates])
    corrected = undistort(pixels, camera, True)
    options = [[], []]
    for indices in combinations(range(len(pixels)), 4):
        p = corrected[list(indices)]
        _, _, vh = np.linalg.svd(p - p.mean(axis=0))
        direction = vh[0]
        if direction[1] < 0 or (abs(direction[1]) < 1e-6 and direction[0] < 0):
            direction = -direction
        positions = (p - p.mean(axis=0)) @ direction
        order = np.argsort(positions)
        s = positions[order]
        span = s[-1] - s[0]
        if span < 25 or np.min(np.diff(s)) < 3:
            continue
        linear = np.sqrt(np.mean(((p - p.mean(axis=0)) @ vh[1]) ** 2)) / span
        if linear > .035:
            continue
        ratio = (s[2]-s[0])*(s[3]-s[1])/((s[3]-s[0])*(s[2]-s[1]))
        ordered = [indices[i] for i in order]
        for rod, gaps in enumerate(target['center_spacings_mm']):
            a = np.r_[0., np.cumsum(gaps)]
            expected = (a[2]-a[0])*(a[3]-a[1])/((a[3]-a[0])*(a[2]-a[1]))
            error = abs(np.log(ratio / expected))
            if error < .12:
                options[rod].append((error + 3*linear, ordered))
    matches = [(a[0]+b[0], a[1]+b[1]) for a in options[0] for b in options[1]
               if not set(a[1]).intersection(b[1])]
    if matches:
        score, indices = min(matches, key=lambda x: x[0])
        report.update(points=dict(zip(IDS, pixels[indices].tolist())), grouping_score=float(score))
    else:
        report['note'] = '候選球無法可靠分成兩根棒；請手動標記或改善遮擋／曝光。'
    return report


def coverage_report(pairs, cameras):
    """Image-space coverage only; size variation is a depth proxy, not depth."""
    output = {}
    for role in ('cf', 'cs'):
        cells, angles = set(), set()
        spans_by_rod = {'A': [], 'B': []}
        for pair in pairs:
            if pair.get('excluded'):
                continue
            points = pair[role].get('points', {})
            if set(points) != set(IDS):
                continue
            width, height = cameras[role]['image_size']
            for rod, start in (('A', 0), ('B', 4)):
                p = np.array([points[i] for i in IDS[start:start+4]], float)
                center = p.mean(axis=0) / [width, height]
                cells.add((min(2, int(center[0]*3)), min(2, int(center[1]*3))))
                d = p[-1] - p[0]
                angle = np.degrees(np.arctan2(abs(d[0]), abs(d[1])))
                angles.add('直立' if angle < 20 else '斜向' if angle < 65 else '接近水平')
                spans_by_rod[rod].append(float(np.linalg.norm(d) / height))
        missing_cells = [y+x for j, y in enumerate(('上', '中', '下'))
                         for i, x in enumerate(('左', '中', '右')) if (i, j) not in cells]
        missing_angles = [a for a in ('直立', '斜向', '接近水平') if a not in angles]
        # Compare the SAME physical rod across captures. A and B have different
        # lengths and may be at different depths even in one stationary photo.
        ratios = {rod: max(spans)/min(spans) if spans and min(spans) > 0 else 1.
                  for rod, spans in spans_by_rod.items()}
        size_ratio = max(ratios.values())
        output[role] = {'covered_cells': len(cells), 'missing_positions': missing_cells,
                        'missing_projected_angles': missing_angles, 'apparent_size_ratio': size_ratio,
                        'apparent_size_ratio_by_rod': ratios,
                        'need_distance_variation': size_ratio < 1.35}
    return output


def _arrays(pairs, cameras):
    arrays = {r: [] for r in ('cf', 'cs')}
    names = set()
    for pair in pairs:
        if not isinstance(pair.get('id'), str) or pair['id'] in names:
            raise ValueError('組別 ID 無效或重複。')
        names.add(pair['id'])
        if pair.get('confirmed') is not True:
            raise ValueError('第 {} 組尚未確認兩視角 A/B 與球號。'.format(pair['id']))
        for role in arrays:
            view = pair[role]
            if list(view['image_size']) != list(cameras[role]['image_size']):
                raise ValueError('照片與內參尺寸不符。')
            if set(view['points']) != set(IDS):
                raise ValueError('每張照片需標記 A1–A4、B1–B4。')
            p = np.array([view['points'][key] for key in IDS], float)
            if (p.shape != (8, 2) or not np.isfinite(p).all() or np.any(p < 0)
                    or np.any(p >= np.array(view['image_size'])) or len(np.unique(p, axis=0)) != 8):
                raise ValueError('球心必須為影像內不重複的有限座標。')
            arrays[role].append(p)
    return {r: np.concatenate(p) for r, p in arrays.items()}


def _length_checks(xyz, valid, target):
    lengths, expected = [], []
    for rod, start in enumerate((0, 4)):
        positions = np.r_[0., np.cumsum(target['center_spacings_mm'][rod])]
        for a, b in combinations(range(4), 2):
            if valid[start+a] and valid[start+b]:
                lengths.append(float(np.linalg.norm(xyz[start+a]-xyz[start+b])))
                expected.append(float(positions[b]-positions[a]))
    return np.array(lengths), np.array(expected)


def _duplicate_observations(points, tolerance_px=1.):
    """Find repeated stereo poses before they can cross the train/check split.

    A new filename/ID does not make another independent capture. Require both
    views' corresponding ball centers to move by more than annotation jitter.
    This only groups poses within one pixel RMS in BOTH original-size images;
    changing one view is not a duplicate (pairing checks handle that instead).
    """
    views = {role: pixels.reshape(-1, len(IDS), 2) for role, pixels in points.items()}
    duplicates = []
    for index in range(1, len(views['cf'])):
        matches = np.ones(index, dtype=bool)
        for pixels in views.values():
            squared_rms = np.mean(np.sum((pixels[:index] - pixels[index]) ** 2, axis=2), axis=1)
            matches &= squared_rms <= tolerance_px ** 2
        if matches.any():
            duplicates.append((index, int(np.flatnonzero(matches)[0])))
    return duplicates


def fit_two_rod_rig(cameras, target, pairs, threshold_px=2.):
    if target != rod_target(target.get('spacing_convention')):
        raise ValueError('雙棒規格不符。')
    for camera in cameras.values():
        validate_intrinsic(camera)
    if not np.isfinite(threshold_px) or threshold_px <= 0:
        raise ValueError('誤差門檻須為正值。')
    pairs = [p for p in pairs if not p.get('excluded')]
    if len(pairs) < 15:
        raise ValueError('至少需要 15 組同步雙機照片（每台 15 張）。')
    all_points = _arrays(pairs, cameras)
    duplicates = _duplicate_observations(all_points)
    if duplicates:
        examples = '、'.join('{} ≈ {}'.format(pairs[index]['id'], pairs[earlier]['id'])
                            for index, earlier in duplicates[:6])
        raise ValueError('照片含重複或近乎相同的雙棒姿態（{}）；請略過重複組，移動棒子的位置／角度後補拍，'
                         '至少保留 15 組不同姿態。'.format(examples))
    held = pairs[::5]
    train = [p for i, p in enumerate(pairs) if i % 5]
    train_mask = np.arange(len(pairs)) % 5 != 0
    points = {role: pixels.reshape(-1, len(IDS), 2)[train_mask].reshape(-1, 2)
              for role, pixels in all_points.items()}
    R, t, inliers, details = estimate_eight_point(points['cf'], points['cs'], cameras, threshold_px)
    xyz, positive, errors, angles = triangulate(points['cf'], points['cs'], cameras, R, t)
    scales = []
    for i in range(len(train)):
        sl = slice(i*8, (i+1)*8)
        valid = inliers[sl] & positive[sl] & (errors[sl] <= threshold_px*2) & (angles[sl] >= 1)
        lengths, expected = _length_checks(xyz[sl], valid, target)
        if len(lengths) >= 6 and np.all(lengths > 1e-8):
            scales.append(float(np.median(expected/lengths)))
    if len(scales) < max(5, len(train)//2):
        raise ValueError('有效棒長不足；請核對球號、增加視差並補拍。')
    scale = float(np.median(scales))
    relative_mad = float(np.median(np.abs(np.array(scales)-scale))/scale)
    if relative_mad > .08:
        raise ValueError('各組尺度差異過大，請核對球心距離、棒子球號與同步。')
    t = t*scale
    all_xyz, all_depth, all_errors, all_parallax = triangulate(
        all_points['cf'], all_points['cs'], cameras, R, t)
    diagnostics = []
    for index, pair in enumerate(pairs):
        sl = slice(index*len(IDS), (index+1)*len(IDS))
        x, depth, err, parallax = (array[sl] for array in
                                  (all_xyz, all_depth, all_errors, all_parallax))
        valid = depth & (err <= 2*threshold_px) & (parallax >= 1)
        lengths, expected = _length_checks(x, valid, target)
        rel_error = float(np.sqrt(np.mean(((lengths-expected)/expected)**2))) if len(lengths) else None
        accepted = bool(valid.sum() >= 7 and len(lengths) >= 9 and rel_error is not None and rel_error <= .05)
        diagnostics.append({'id': pair['id'], 'held_out': not bool(train_mask[index]),
                            'accepted': accepted, 'valid_points': int(valid.sum()),
                            'reprojection_rms_px': float(np.sqrt(np.mean(err**2))),
                            'length_relative_rmse': rel_error,
                            'length_rmse_mm': float(np.sqrt(np.mean((lengths-expected)**2))) if len(lengths) else None})
    checks = [d for d in diagnostics if d['held_out']]
    if sum(d['accepted'] for d in checks) < max(2, int(np.ceil(.75*len(checks)))):
        raise ValueError('保留照片的棒長／重投影檢查未通過，請改善球號、同步或取樣角度。')
    if sum(d['accepted'] for d in diagnostics) < .75*len(pairs):
        raise ValueError('超過四分之一照片未通過幾何檢查，請修正標記或補拍。')
    details.update(mm_per_baseline_unit=scale, scale_relative_mad=relative_mad,
                   scale_source='known center-to-center distances within each independent rod')
    return {'schema': 'metric_stereo_v1', 'method': '8point', 'target_type': 'two_rods', 'units': 'mm',
            'coordinate_convention': 'X_cs = R_cs_from_cf @ X_cf + t_cs_from_cf_mm',
            'R_cs_from_cf': R.tolist(), 't_cs_from_cf_mm': t.tolist(),
            'baseline_mm': float(np.linalg.norm(t)), 'intrinsics': copy.deepcopy(cameras),
            'target': target, 'fit': details, 'candidate_diagnostics': diagnostics,
            'training_pairs': [p['id'] for p in train], 'held_out_pairs': [p['id'] for p in held],
            'source_pairs': copy.deepcopy(pairs), 'coverage': coverage_report(pairs, cameras),
            'independently_validated_for_measurement': False,
            'world_note': 'World is the CF camera frame; no fixed spatial relation between the rods is assumed.',
            'warnings': ['影像涵蓋與投影角度是補拍提示，無法直接確認真實三維角度及深度。',
                         '球面影像輪廓中心近似球心投影；正式量測請以額外已知距離驗證。']}


def save_two_rod_rig(path, rig):
    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        return value
    if rig.get('target_type') != 'two_rods' or rig.get('method') != '8point':
        raise ValueError('不是雙棒 8-point 外參。')
    save_json(path, plain(rig))
