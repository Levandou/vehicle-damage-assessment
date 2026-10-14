from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from damage_3d import find_damaged_parts
from diag import dlog


_RM_TRIED = False
_RM_AVAILABLE = False
_RM_IMPORT_ERR: Optional[str] = None
_RM_MODEL = None
_RM_DEVICE: Optional[str] = None
_RM_MAX_SIDE = 768


def _ensure_ssl_cert_for_torch_hub() -> None:
    import os
    if os.environ.get('SSL_CERT_FILE') and os.environ.get('REQUESTS_CA_BUNDLE'):
        return
    try:
        import certifi
        ca = certifi.where()
        os.environ.setdefault('SSL_CERT_FILE', ca)
        os.environ.setdefault('REQUESTS_CA_BUNDLE', ca)
    except Exception:
        pass


def _try_init_roma() -> bool:
    global _RM_TRIED, _RM_AVAILABLE, _RM_IMPORT_ERR, _RM_MODEL, _RM_DEVICE
    if _RM_TRIED:
        return _RM_AVAILABLE
    _RM_TRIED = True
    try:
        _ensure_ssl_cert_for_torch_hub()
        import torch
        from romatch import roma_outdoor
        _RM_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        _RM_MODEL = roma_outdoor(device=_RM_DEVICE)
        _RM_AVAILABLE = True
        dlog(f'[near.matcher] RoMa инициализирован, device={_RM_DEVICE}')
    except Exception as exc:
        _RM_IMPORT_ERR = str(exc)
        _RM_AVAILABLE = False
        dlog(f'[near.matcher] RoMa недоступен ({exc!s})')
    return _RM_AVAILABLE


def _resize_for_roma(
    img_bgr: np.ndarray, max_side: int,
) -> Tuple[np.ndarray, float, float]:
    h, w = img_bgr.shape[:2]
    if max(h, w) <= max_side:
        return img_bgr, 1.0, 1.0
    scale = float(max_side) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, w / float(new_w), h / float(new_h)


def _match_roma(
    near_bgr: np.ndarray,
    far_bgr: np.ndarray,
    *,
    max_side: int = _RM_MAX_SIDE,
    num_samples: int = 10000,
):
    import torch
    import tempfile

    t_total = time.perf_counter()
    near_resz, sx_n, sy_n = _resize_for_roma(near_bgr, max_side)
    far_resz, sx_f, sy_f = _resize_for_roma(far_bgr, max_side)

    with tempfile.TemporaryDirectory() as tmpd:
        near_path = str(Path(tmpd) / 'near.png')
        far_path = str(Path(tmpd) / 'far.png')
        cv2.imwrite(near_path, near_resz)
        cv2.imwrite(far_path, far_resz)

        t_match = time.perf_counter()
        with torch.inference_mode():
            warp, certainty = _RM_MODEL.match(near_path, far_path, device=_RM_DEVICE)
        dt_match = time.perf_counter() - t_match

        Hn, Wn = near_resz.shape[:2]
        Hf, Wf = far_resz.shape[:2]

        t_sample = time.perf_counter()
        torch.manual_seed(42)
        with torch.inference_mode():
            matches, c_sampled = _RM_MODEL.sample(warp, certainty, num=num_samples)
            kp_n, kp_f = _RM_MODEL.to_pixel_coordinates(
                matches, Hn, Wn, Hf, Wf,
            )
        dt_sample = time.perf_counter() - t_sample

    kp_n = kp_n.detach().cpu().numpy().astype(np.float32)
    kp_f = kp_f.detach().cpu().numpy().astype(np.float32)
    c_sampled = c_sampled.detach().cpu().numpy().astype(np.float32)

    kp_n[:, 0] *= sx_n
    kp_n[:, 1] *= sy_n
    kp_f[:, 0] *= sx_f
    kp_f[:, 1] *= sy_f

    dt_total = time.perf_counter() - t_total
    msg = (
        f'[RoMa timing] match={dt_match:.2f}s sample={dt_sample:.2f}s '
        f'total={dt_total:.2f}s '
        f'(near={Wn}x{Hn}, far={Wf}x{Hf}, device={_RM_DEVICE}, '
        f'samples={num_samples}, '
        f'cert_min={float(c_sampled.min()):.3f} '
        f'cert_max={float(c_sampled.max()):.3f})'
    )
    print(msg, flush=True)
    dlog(msg)

    return kp_n, kp_f, c_sampled


def _check_inlier_spread(
    pts_far_inliers: np.ndarray,
    far_hw: Tuple[int, int],
    min_frac: float = 0.04,
) -> Tuple[bool, Optional[str]]:
    if len(pts_far_inliers) == 0:
        return False, 'no_inliers'
    Hf, Wf = int(far_hw[0]), int(far_hw[1])
    far_area = float(Hf * Wf)
    xs = pts_far_inliers[:, 0]
    ys = pts_far_inliers[:, 1]
    bw = float(xs.max() - xs.min())
    bh = float(ys.max() - ys.min())
    spread_frac = (bw * bh) / max(1.0, far_area)
    if spread_frac < min_frac:
        return False, (
            f'inlier_spread_too_small: bbox_frac={spread_frac:.4f} '
            f'< min={min_frac}'
        )
    return True, None


def _check_warp_sanity(
    H: np.ndarray,
    near_hw: Tuple[int, int],
    far_hw: Tuple[int, int],
    *,
    max_corner_excess: float = 2.0,
    min_area_frac: float = 0.001,
    max_aspect: float = 15.0,
) -> Tuple[bool, Optional[str]]:
    Hn, Wn = int(near_hw[0]), int(near_hw[1])
    Hf, Wf = int(far_hw[0]), int(far_hw[1])
    far_area = float(Hf * Wf)

    corners_near = np.array(
        [[0, 0], [Wn, 0], [Wn, Hn], [0, Hn]], dtype=np.float32
    ).reshape(-1, 1, 2)
    try:
        corners_far = cv2.perspectiveTransform(corners_near, H).reshape(-1, 2)
    except cv2.error as e:
        return False, f'perspectiveTransform_failed: {e}'

    if not np.all(np.isfinite(corners_far)):
        return False, 'warped_corners_not_finite'

    bound = max(Wf, Hf) * max_corner_excess
    if np.any(np.abs(corners_far) > bound):
        return False, (
            f'warped_corners_out_of_bounds: '
            f'|coord|>{bound:.0f} (far_size={Wf}x{Hf})'
        )

    pts = corners_far
    area = 0.5 * abs(
        (pts[0, 0] - pts[2, 0]) * (pts[1, 1] - pts[3, 1])
        - (pts[1, 0] - pts[3, 0]) * (pts[0, 1] - pts[2, 1])
    )
    area_frac = area / max(1.0, far_area)
    if area_frac < min_area_frac:
        return False, (
            f'warped_area_too_small: area_frac={area_frac:.5f} '
            f'< min={min_area_frac}'
        )

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    signs = np.sign([_cross(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4])
                     for i in range(4)])
    if len(set(s for s in signs if s != 0)) > 1:
        return False, 'warped_polygon_not_convex'

    bw = float(pts[:, 0].max() - pts[:, 0].min())
    bh = float(pts[:, 1].max() - pts[:, 1].min())
    if bw < 1 or bh < 1:
        return False, f'warped_bbox_degenerate: bw={bw:.1f} bh={bh:.1f}'
    aspect = max(bw, bh) / max(1.0, min(bw, bh))
    if aspect > max_aspect:
        return False, (
            f'warped_aspect_extreme: aspect={aspect:.1f} > max={max_aspect}'
        )

    return True, None


def _compute_shared_parts_masks(
    near_damage_masks: List[Dict],
    near_part_masks: List[Dict],
    far_part_masks: List[Dict],
    near_hw: Tuple[int, int],
    far_hw: Tuple[int, int],
    *,
    min_overlap_px: int = 200,
) -> Optional[Dict[str, Any]]:
    if not near_part_masks or not far_part_masks:
        return None
    near_labels = {p.get('label') for p in near_part_masks if p.get('label')}
    far_labels = {p.get('label') for p in far_part_masks if p.get('label')}
    common_labels = sorted(near_labels & far_labels)
    if not common_labels:
        return None

    Hn, Wn = int(near_hw[0]), int(near_hw[1])
    Hf, Wf = int(far_hw[0]), int(far_hw[1])
    near_mask = np.zeros((Hn, Wn), dtype=bool)
    far_mask = np.zeros((Hf, Wf), dtype=bool)

    for p in near_part_masks:
        if p.get('label') in common_labels:
            near_mask |= np.asarray(p['mask']) > 0
    for d in near_damage_masks or []:
        near_mask |= np.asarray(d['mask']) > 0

    for p in far_part_masks:
        if p.get('label') in common_labels:
            far_mask |= np.asarray(p['mask']) > 0

    damaged = find_damaged_parts(
        near_damage_masks, near_part_masks, min_overlap_px=min_overlap_px,
    )
    primary_label = None
    overlap_px = 0
    if damaged:
        primary = max(damaged, key=lambda d: d.get('overlap_px', 0))
        primary_label = primary['part']
        overlap_px = int(primary.get('overlap_px', 0))

    return {
        'near_mask': near_mask.astype(np.uint8),
        'far_mask': far_mask.astype(np.uint8),
        'common_labels': common_labels,
        'primary_part': primary_label,
        'overlap_px': overlap_px,
    }


def match_near_to_far_roma_masked(
    near_bgr: np.ndarray,
    far_bgr: np.ndarray,
    near_damage_masks: List[Dict],
    near_part_masks: List[Dict],
    far_part_masks: List[Dict],
    *,
    ransac_thresh: float = 5.0,
    min_inliers: int = 8,
    min_inlier_ratio: float = 0.08,
    min_inlier_spread_frac: float = 0.02,
    min_overlap_px: int = 200,
    mask_dilate_px: int = 20,
    max_side: int = _RM_MAX_SIDE,
    num_samples: int = 10000,
    top_n: int = 2000,
    certainty_threshold: float = 0.5,
) -> Optional[Dict[str, Any]]:
    t_total = time.perf_counter()

    t_init = time.perf_counter()
    if not _try_init_roma():
        dlog('[near.matcher.roma] RoMa not available, fallback')
        return None
    dt_init = time.perf_counter() - t_init

    Hn, Wn = near_bgr.shape[:2]
    Hf, Wf = far_bgr.shape[:2]

    shared = _compute_shared_parts_masks(
        near_damage_masks, near_part_masks, far_part_masks,
        near_hw=(Hn, Wn), far_hw=(Hf, Wf),
        min_overlap_px=min_overlap_px,
    )
    if shared is None:
        dlog('[near.matcher.roma] no common parts between NEAR and FAR, fallback')
        return None

    near_keep = shared['near_mask'].copy()
    far_keep = shared['far_mask'].copy()
    if mask_dilate_px > 0:
        k = max(1, int(mask_dilate_px) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        near_keep = cv2.dilate(near_keep, kernel)
        far_keep = cv2.dilate(far_keep, kernel)

    near_filt = near_bgr.copy()
    far_filt = far_bgr.copy()
    near_filt[near_keep == 0] = 0
    far_filt[far_keep == 0] = 0

    print(
        f'[RoMa-masked] common parts: {shared["common_labels"]} | '
        f'near_keep_px={int(near_keep.sum())}, far_keep_px={int(far_keep.sum())} | '
        f'max_side={max_side}, samples={num_samples}, top_n={top_n}',
        flush=True,
    )
    dlog(
        f'[near.matcher.roma] common_labels={shared["common_labels"]} '
        f'primary={shared["primary_part"]!r} '
        f'near_keep_px={int(near_keep.sum())} far_keep_px={int(far_keep.sum())} '
        f'max_side={max_side} samples={num_samples} top_n={top_n}'
    )

    try:
        pts_near, pts_far, scores = _match_roma(
            near_filt, far_filt, max_side=max_side, num_samples=num_samples,
        )
    except Exception as exc:
        dlog(f'[near.matcher.roma] _match_roma failed: {exc}')
        return None

    if len(pts_near) == 0:
        dlog('[near.matcher.roma] RoMa returned 0 matches')
        return None

    order = np.argsort(-scores)
    top_idx = order[: int(min(top_n, len(order)))]
    pts_near = pts_near[top_idx]
    pts_far = pts_far[top_idx]
    scores = scores[top_idx]

    mask_cert = scores >= certainty_threshold
    if int(mask_cert.sum()) >= 4:
        pts_near = pts_near[mask_cert]
        pts_far = pts_far[mask_cert]
        scores = scores[mask_cert]

    n_before = int(len(pts_near))
    if n_before < 4:
        dlog(f'[near.matcher.roma] <4 matches after certainty filter ({n_before})')
        return None

    yn = np.clip(np.round(pts_near[:, 1]).astype(int), 0, Hn - 1)
    xn = np.clip(np.round(pts_near[:, 0]).astype(int), 0, Wn - 1)
    yf = np.clip(np.round(pts_far[:, 1]).astype(int), 0, Hf - 1)
    xf = np.clip(np.round(pts_far[:, 0]).astype(int), 0, Wf - 1)
    valid = (near_keep[yn, xn] > 0) & (far_keep[yf, xf] > 0)
    n_kept = int(valid.sum())
    dlog(
        f'[near.matcher.roma] mask filter: kept {n_kept}/{n_before} '
        f'(certainty cutoff applied: {int(mask_cert.sum())})'
    )
    if n_kept < 4:
        return None
    pts_near = pts_near[valid]
    pts_far = pts_far[valid]
    scores = scores[valid]

    H_mat, inl = cv2.findHomography(
        pts_near, pts_far,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=10000, confidence=0.999,
    )
    if H_mat is None or inl is None:
        dlog('[near.matcher.roma] findHomography returned None')
        return None
    inliers = inl.ravel().astype(bool)
    n_inl = int(inliers.sum())
    n_tot = int(len(pts_near))
    inlier_ratio = float(n_inl) / max(1, n_tot)

    degenerate_reasons: List[str] = []
    if n_inl < min_inliers:
        degenerate_reasons.append(f'too_few_inliers: {n_inl} < {min_inliers}')
    if inlier_ratio < min_inlier_ratio:
        degenerate_reasons.append(
            f'inlier_ratio_too_low: {inlier_ratio:.3f} < {min_inlier_ratio}'
        )
    ok_spread, reason_spread = _check_inlier_spread(
        pts_far[inliers], far_bgr.shape[:2], min_frac=min_inlier_spread_frac,
    )
    if not ok_spread:
        degenerate_reasons.append(reason_spread or 'spread_failed')
    ok_warp, reason_warp = _check_warp_sanity(
        H_mat, near_bgr.shape[:2], far_bgr.shape[:2],
    )
    if not ok_warp:
        degenerate_reasons.append(reason_warp or 'warp_failed')
    if degenerate_reasons:
        dlog(
            f'[near.matcher.roma] would-be degenerate (gates disabled): '
            f'{"; ".join(degenerate_reasons)}'
        )
    degenerate = False
    degenerate_reason = None

    dlog(
        f'[near.matcher.roma] H готова: matches={n_tot} inliers={n_inl} '
        f'ratio={inlier_ratio:.3f} degenerate={degenerate} reason={degenerate_reason}'
    )

    dt_total = time.perf_counter() - t_total
    summary = (
        f'[RoMa-masked total] {dt_total:.2f}s '
        f'(init={dt_init:.2f}s, '
        f'primary={shared["primary_part"]!r}, parts={shared["common_labels"]}, '
        f'matches={n_tot}, inliers={n_inl}, degenerate={degenerate})'
    )
    print(summary, flush=True)
    dlog(summary)

    return {
        'H': H_mat,
        'method': 'roma_masked',
        'fallback_reason': None,
        'pts_near': pts_near,
        'pts_far': pts_far,
        'scores': scores,
        'inliers': inliers,
        'n_matches': n_tot,
        'n_inliers': n_inl,
        'degenerate': degenerate,
        'degenerate_reason': degenerate_reason,
        'crop_info': {
            'primary_part': shared['primary_part'],
            'overlap_px': shared['overlap_px'],
            'common_labels': shared['common_labels'],
            'sub_method': 'roma_outdoor',
            'mask_dilate_px': int(mask_dilate_px),
            'max_side': int(max_side),
            'num_samples': int(num_samples),
            'top_n': int(top_n),
            'certainty_threshold': float(certainty_threshold),
            'n_matches_before_mask_filter': int(n_before),
            'n_matches_after_mask_filter': int(n_kept),
            'total_seconds': float(dt_total),
        },
    }


def make_masked_inputs_overlay(
    near_bgr: np.ndarray,
    far_bgr: np.ndarray,
    near_keep: np.ndarray,
    far_keep: np.ndarray,
) -> np.ndarray:
    near_filt = near_bgr.copy()
    far_filt = far_bgr.copy()
    near_filt[near_keep == 0] = 0
    far_filt[far_keep == 0] = 0
    Hn, Wn = near_filt.shape[:2]
    Hf, Wf = far_filt.shape[:2]
    H = max(Hn, Hf)
    canvas = np.zeros((H, Wn + Wf, 3), dtype=np.uint8)
    canvas[:Hn, :Wn] = near_filt
    canvas[:Hf, Wn:Wn + Wf] = far_filt
    label = 'masked inputs to RoMa: NEAR | FAR'
    cv2.putText(canvas, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def warp_damage_masks_to_far(
    damage_masks_near: List[Dict],
    H: np.ndarray,
    far_hw: Tuple[int, ...],
) -> List[Dict]:
    if H is None:
        return []
    H_far, W_far = int(far_hw[0]), int(far_hw[1])
    out: List[Dict] = []
    for item in damage_masks_near:
        m = np.asarray(item['mask'], dtype=np.uint8)
        if m.size == 0 or int(m.sum()) == 0:
            continue
        m255 = (m > 0).astype(np.uint8) * 255
        m_far = cv2.warpPerspective(
            m255, H, (W_far, H_far), flags=cv2.INTER_NEAREST,
        )
        m_far_bin = (m_far > 0).astype(np.uint8)
        if int(m_far_bin.sum()) == 0:
            continue
        out.append({**item, 'mask': m_far_bin})
    return out


def make_matches_overlay(
    near_bgr: np.ndarray,
    far_bgr: np.ndarray,
    pts_near: np.ndarray,
    pts_far: np.ndarray,
    scores: Optional[np.ndarray] = None,
    max_lines: int = 300,
) -> np.ndarray:
    Hn, Wn = near_bgr.shape[:2]
    Hf, Wf = far_bgr.shape[:2]
    H_canvas = max(Hn, Hf)
    canvas = np.zeros((H_canvas, Wn + Wf, 3), dtype=np.uint8)
    canvas[:Hn, :Wn] = near_bgr
    canvas[:Hf, Wn:Wn + Wf] = far_bgr

    n = len(pts_near)
    if n == 0:
        return canvas
    if n > max_lines:
        if scores is not None:
            order = np.argsort(-scores)[:max_lines]
        else:
            order = np.random.choice(n, max_lines, replace=False)
        pts_near = pts_near[order]
        pts_far = pts_far[order]
        if scores is not None:
            scores = scores[order]
        n = len(pts_near)

    for i in range(n):
        x1 = int(round(float(pts_near[i, 0])))
        y1 = int(round(float(pts_near[i, 1])))
        x2 = int(round(float(pts_far[i, 0]))) + Wn
        y2 = int(round(float(pts_far[i, 1])))
        s = float(scores[i]) if scores is not None else 0.7
        s = max(0.0, min(1.0, s))
        color = (
            int(np.clip(255 * (1.0 - s), 0, 255)),
            int(np.clip(255 * s, 0, 255)),
            80,
        )
        cv2.line(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, (0, 0, 255), -1)
        cv2.circle(canvas, (x2, y2), 2, (0, 0, 255), -1)

    label = f'matches: {n}'
    cv2.putText(canvas, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def make_warped_damages_on_far_overlay(
    far_bgr: np.ndarray,
    damage_masks_far_warped: List[Dict],
) -> np.ndarray:
    overlay = far_bgr.copy()
    red = np.array([0, 0, 220], dtype=np.float32)
    for item in damage_masks_far_warped:
        m = np.asarray(item['mask']) > 0
        if not m.any():
            continue
        overlay[m] = (overlay[m].astype(np.float32) * 0.4 + red * 0.6).astype(np.uint8)
    cv2.putText(
        overlay,
        f'warped damages (from near): {len(damage_masks_far_warped)}',
        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 220), 2, cv2.LINE_AA,
    )
    return overlay
