from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from damage_3d import (
    ensure_dir,
    find_damaged_parts,
    make_camera_matrix,
    make_damages_overlay,
    make_parts_overlay,
    overlay_mesh_on_photo,
    ray_cast,
    render_wireframe,
)
from diag import Stopwatch, dlog, init_pipeline_diag_log
from near_matchers import (
    _compute_shared_parts_masks,
    make_masked_inputs_overlay,
    make_matches_overlay,
    make_warped_damages_on_far_overlay,
    match_near_to_far_roma_masked,
    warp_damage_masks_to_far,
)
from near_pose import (
    _estimate_focal_near_px,
    _render_damaged_faces_on_photo,
    solve_near_pose_via_correspondences,
)


def _save_two_view_report(
    out_dir: Path,
    report: Dict[str, Any],
    stopwatch: Optional['Stopwatch'] = None,
) -> Dict[str, Any]:
    del out_dir
    if stopwatch is not None and 'timings' not in report:
        report['timings'] = stopwatch.finalize()
    return report


def _union_for_labels(part_masks, labels) -> Optional[np.ndarray]:
    if not part_masks:
        return None
    h, w = part_masks[0]['mask'].shape[:2]
    out = np.zeros((h, w), dtype=bool)
    used = False
    for p in part_masks:
        if p.get('label') in labels:
            out |= (np.asarray(p['mask']) > 0)
            used = True
    return out if used else None


def _mask_shape_descriptor(mask: np.ndarray) -> Optional[Dict[str, float]]:
    m = np.asarray(mask) > 0
    if not m.any():
        return None
    ys, xs = np.where(m)
    n = int(len(xs))
    if n < 16:
        return None

    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    lam_major = float(eigvals[1])
    lam_minor = float(max(eigvals[0], 1e-9))
    vec_major = eigvecs[:, 1]
    angle_rad = math.atan2(float(vec_major[1]), float(vec_major[0]))
    angle_deg = math.degrees(angle_rad) % 180.0
    aspect = float(math.sqrt(lam_major / lam_minor))

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bbox_area = max((x2 - x1) * (y2 - y1), 1)
    fill = float(n) / float(bbox_area)

    return {
        'angle_deg': float(angle_deg),
        'aspect': float(aspect),
        'fill': float(fill),
        'area_px': int(n),
    }


def _shape_descriptor_distance(d1: Dict[str, float], d2: Dict[str, float]) -> float:
    a1, a2 = float(d1['angle_deg']), float(d2['angle_deg'])
    da = min(abs(a1 - a2), 180.0 - abs(a1 - a2))
    da_norm = da / 90.0

    asp1, asp2 = float(d1['aspect']), float(d2['aspect'])
    asp_log = abs(math.log(max(asp1, 1e-6) / max(asp2, 1e-6))) / math.log(3.0)
    asp_norm = min(asp_log, 1.0)

    df = abs(float(d1['fill']) - float(d2['fill']))

    return float(0.55 * da_norm + 0.25 * asp_norm + 0.20 * df)


def select_best_far_for_near(
    near_bgr: np.ndarray,
    near_damage_masks: List[Dict],
    near_part_masks: List[Dict],
    far_contexts: List[Dict[str, Any]],
    *,
    min_overlap_px: int = 200,
) -> Dict[str, Any]:
    damaged_pairs = find_damaged_parts(
        near_damage_masks, near_part_masks, min_overlap_px=min_overlap_px,
    )
    damaged_labels = sorted({dp['part'] for dp in damaged_pairs})

    scored: List[Dict[str, Any]] = []

    if not damaged_labels:
        for fc in far_contexts:
            total = sum(
                int((np.asarray(p['mask']) > 0).sum())
                for p in fc['part_masks']
            )
            scored.append({
                'far_idx': fc['idx'], 'far_path': fc['path'],
                'damaged_labels': [], 'part_area_score': total,
                'shape_distance': None, 'shape_similarity': None,
            })
        scored.sort(key=lambda x: -x['part_area_score'])
        if not scored or scored[0]['part_area_score'] == 0:
            return {
                'winner_idx': None, 'scores': scored,
                'damaged_labels': [],
                'reason': 'no_damaged_parts_on_near_and_no_parts_on_any_far',
            }
        return {
            'winner_idx': scored[0]['far_idx'], 'scores': scored,
            'damaged_labels': [],
            'reason': 'fallback: no damaged part on NEAR — picked FAR with largest total parts area',
        }

    near_dmg_region = _union_for_labels(near_part_masks, set(damaged_labels))
    near_desc = _mask_shape_descriptor(
        near_dmg_region.astype(np.uint8) if near_dmg_region is not None else None,
    ) if near_dmg_region is not None else None

    for fc in far_contexts:
        label_areas = {
            p.get('label'): int((np.asarray(p['mask']) > 0).sum())
            for p in fc['part_masks']
        }
        area_score = sum(label_areas.get(lbl, 0) for lbl in damaged_labels)

        if area_score == 0 or near_desc is None:
            scored.append({
                'far_idx': fc['idx'], 'far_path': fc['path'],
                'damaged_labels': damaged_labels,
                'part_area_score': int(area_score),
                'shape_distance': None, 'shape_similarity': 0.0,
                'near_descriptor': near_desc, 'far_descriptor': None,
            })
            continue

        far_region = _union_for_labels(fc['part_masks'], set(damaged_labels))
        far_desc = _mask_shape_descriptor(
            far_region.astype(np.uint8) if far_region is not None else None,
        ) if far_region is not None else None

        if far_desc is None:
            scored.append({
                'far_idx': fc['idx'], 'far_path': fc['path'],
                'damaged_labels': damaged_labels,
                'part_area_score': int(area_score),
                'shape_distance': None, 'shape_similarity': 0.0,
                'near_descriptor': near_desc, 'far_descriptor': None,
            })
            continue

        sdist = _shape_descriptor_distance(near_desc, far_desc)
        ssim = math.exp(-2.0 * sdist)
        scored.append({
            'far_idx': fc['idx'], 'far_path': fc['path'],
            'damaged_labels': damaged_labels,
            'part_area_score': int(area_score),
            'shape_distance': float(sdist),
            'shape_similarity': float(ssim),
            'near_descriptor': near_desc,
            'far_descriptor': far_desc,
        })

    survivors = [s for s in scored if s['part_area_score'] > 0]
    if not survivors:
        return {
            'winner_idx': None, 'scores': scored,
            'damaged_labels': damaged_labels,
            'reason': f'no FAR contains damaged labels {damaged_labels}',
        }
    if len(survivors) == 1:
        return {
            'winner_idx': survivors[0]['far_idx'], 'scores': scored,
            'damaged_labels': damaged_labels,
            'reason': f'single FAR has damaged labels {damaged_labels}',
        }
    if near_desc is None:
        survivors.sort(key=lambda x: -x['part_area_score'])
        return {
            'winner_idx': survivors[0]['far_idx'], 'scores': scored,
            'damaged_labels': damaged_labels,
            'reason': 'NEAR damaged region empty — fallback to part-area',
        }

    survivors.sort(key=lambda x: -(x['shape_similarity'] or 0.0))
    winner = survivors[0]
    scored.sort(key=lambda x: -(x.get('shape_similarity') or 0.0))

    return {
        'winner_idx': winner['far_idx'], 'scores': scored,
        'damaged_labels': damaged_labels,
        'reason': (
            f"shape descriptor ranking (angle/aspect/fill); "
            f"winner shape_dist={winner['shape_distance']:.3f}, "
            f"shape_sim={winner['shape_similarity']:.3f}, "
            f"part_area_px={winner['part_area_score']}"
        ),
    }


def _run_hidden_damage_classifier(
    pipeline: Any,
    severity_per_part: List[Dict[str, Any]],
    report: Dict[str, Any],
    sw: Optional['Stopwatch'] = None,
) -> None:
    classifier = getattr(pipeline, 'hidden_damage_classifier', None)
    if classifier is None:
        report['hidden_damage'] = {
            'status': 'SKIPPED',
            'reason': 'classifier_not_loaded',
        }
        return

    section_name = 'hidden_damage classifier (XGBoost)'
    section_ctx = sw.section(section_name) if sw is not None else None
    try:
        if section_ctx is not None:
            section_ctx.__enter__()
        try:
            result = classifier.predict_from_severity_list(severity_per_part or [])
        except Exception as exc:
            result = {
                'status': 'ERROR',
                'error': f'predict_failed: {exc}',
                'threshold': float(getattr(classifier, 'threshold', 0.6)),
            }
            dlog(f'[hidden_damage] predict raised: {exc}')
            print(f'[hidden_damage] predict raised: {exc}', flush=True)
        report['hidden_damage'] = result

        if result.get('status') in ('OK', 'EXPERT_REQUIRED'):
            print(
                f'[hidden_damage] confidence={result.get("hidden_damage_confidence")}  '
                f'threshold={result.get("threshold")}  status={result.get("status")}',
                flush=True,
            )
    finally:
        if section_ctx is not None:
            section_ctx.__exit__(None, None, None)


def run_two_view(
    near_path: str,
    far_path: str,
    output_dir: str,
    pipeline,
) -> Dict[str, Any]:
    out_dir = ensure_dir(Path(output_dir))
    init_pipeline_diag_log(out_dir, console_echo=False)
    dlog(f'[near.two_view] near={near_path} far={far_path}')

    sw = Stopwatch('two_view', echo_steps=True)

    cv2.setRNGSeed(42)
    np.random.seed(42)

    with sw.section('imread NEAR + FAR'):
        far_bgr = cv2.imread(str(far_path))
        near_bgr = cv2.imread(str(near_path))
        if far_bgr is None:
            raise FileNotFoundError(f'Не удалось открыть дальнее фото: {far_path}')
        if near_bgr is None:
            raise FileNotFoundError(f'Не удалось открыть ближнее фото: {near_path}')

    with sw.section('FAR: process_image_core'):
        far_core = pipeline._process_image_core(
            far_bgr, out_dir,
            skip_damages=True,
            skip_quality_gate=True,
            file_prefix='far_',
            source_image_path=str(far_path),
            stopwatch=sw,
        )
        far_report = far_core['report']

    report: Dict[str, Any] = {
        'mode': 'two-view',
        'status': 'OK',
        'near_image_path': str(Path(near_path).resolve()),
        'far_image_path': str(Path(far_path).resolve()),
        'far': far_report,
        'near': {},
        'matching': None,
        'files': dict(far_report.get('files', {})),
        'damaged_parts_2d': [],
        'severity_per_part': [],
        'photo_quality_assessment': {'far': None, 'near': None},
    }

    car_finder = getattr(pipeline, 'car_finder', None)
    far_car_mask = None
    if car_finder is not None:
        try:
            with sw.section('FAR: car segmentation (yolo11n-seg)'):
                car_res = car_finder.find_largest_car(far_bgr)
            if car_res.found and car_res.mask is not None:
                far_car_mask = (car_res.mask > 0).astype(np.uint8)
                report['far_car_segment'] = {
                    'found': True,
                    'confidence': float(car_res.confidence or 0.0),
                    'area_pixels': int(car_res.area_pixels or 0),
                    'bbox_xyxy': list(car_res.bbox_xyxy) if car_res.bbox_xyxy else None,
                }
                kept_parts = []
                dropped = 0
                for p in far_core['part_masks']:
                    m = np.asarray(p['mask']) > 0
                    inside = np.logical_and(m, far_car_mask > 0).astype(np.uint8)
                    if int(inside.sum()) >= pipeline.min_overlap_px:
                        new_p = dict(p)
                        new_p['mask'] = inside
                        kept_parts.append(new_p)
                    else:
                        dropped += 1
                far_core['part_masks'] = kept_parts
                report['far_part_masks_filtered'] = {
                    'kept': len(kept_parts), 'dropped_outside_car': int(dropped),
                }
                with sw.section('FAR: IQA evaluate_in_region(car_mask)'):
                    far_q = pipeline.quality_gate.evaluate_in_region(far_bgr, far_car_mask)
                report['photo_quality_assessment']['far'] = far_q
                print(
                    f'[two_view] FAR car_seg: conf={car_res.confidence:.2f} '
                    f'area={car_res.area_pixels}, parts kept {len(kept_parts)}/{len(kept_parts)+dropped}, '
                    f'quality={far_q.get("status")}',
                    flush=True,
                )
            else:
                report['far_car_segment'] = {'found': False}
                print('[two_view] FAR car_seg: no car found, quality fallback to full-frame', flush=True)
        except Exception as exc:
            dlog(f'[two_view] car_finder failed: {exc}')
            print(f'[two_view] car_finder failed: {exc}', flush=True)
    if report['photo_quality_assessment']['far'] is None:
        with sw.section('FAR: IQA full-frame (fallback)'):
            report['photo_quality_assessment']['far'] = pipeline.quality_gate.evaluate(far_bgr)

    far_q_status = (report['photo_quality_assessment']['far'] or {}).get('status')
    if far_q_status == 'REJECT':
        far_reasons = ','.join(
            (report['photo_quality_assessment']['far'] or {}).get('reject_reasons') or []
        )
        report['status'] = 'UNABLE_TO_DETERMINE'
        report['error'] = f'far_quality_rejected: {far_reasons}' if far_reasons else 'far_quality_rejected'
        print(f'[two_view] STOP: FAR quality REJECT ({far_reasons})', flush=True)
        return _save_two_view_report(out_dir, report, sw)

    if far_core['pose'] is None:
        err = far_core.get('pose_error') or far_report.get('error') or 'no_pose'
        report['status'] = 'UNABLE_TO_DETERMINE'
        report['error'] = f'far_pose_not_recovered: {err}'
        return _save_two_view_report(out_dir, report, sw)

    with sw.section('NEAR: damages segmentation (Mask R-CNN)'):
        near_damage_masks = pipeline.damages.predict(near_bgr)
    with sw.section('NEAR: parts segmentation (DeepLabV3+)'):
        near_part_masks = pipeline.details.predict(near_bgr)

    if near_part_masks:
        near_parts_union = np.zeros(near_bgr.shape[:2], dtype=np.uint8)
        for p in near_part_masks:
            near_parts_union = np.maximum(
                near_parts_union, (np.asarray(p['mask']) > 0).astype(np.uint8),
            )
        if int(near_parts_union.sum()) > 0:
            try:
                with sw.section('NEAR: IQA evaluate_in_region(parts_union)'):
                    near_q = pipeline.quality_gate.evaluate_in_region(near_bgr, near_parts_union)
                report['photo_quality_assessment']['near'] = near_q
                print(
                    f'[two_view] NEAR quality on parts-union: '
                    f'{int(near_parts_union.sum())} px, status={near_q.get("status")}',
                    flush=True,
                )
            except Exception as exc:
                dlog(f'[two_view] NEAR quality_in_region failed: {exc}')
    if report['photo_quality_assessment']['near'] is None:
        with sw.section('NEAR: IQA full-frame (fallback)'):
            report['photo_quality_assessment']['near'] = pipeline.quality_gate.evaluate(near_bgr)

    near_q_status = (report['photo_quality_assessment']['near'] or {}).get('status')
    if near_q_status == 'REJECT':
        near_reasons = ','.join(
            (report['photo_quality_assessment']['near'] or {}).get('reject_reasons') or []
        )
        report['status'] = 'UNABLE_TO_DETERMINE'
        report['error'] = (
            f'near_quality_rejected: {near_reasons}' if near_reasons else 'near_quality_rejected'
        )
        print(f'[two_view] STOP: NEAR quality REJECT ({near_reasons})', flush=True)
        return _save_two_view_report(out_dir, report, sw)

    near_q_status = (report['photo_quality_assessment']['near'] or {}).get('status')
    if near_q_status == 'REJECT':
        near_reasons = ','.join(
            (report['photo_quality_assessment']['near'] or {}).get('reject_reasons') or []
        )
        report['status'] = 'UNABLE_TO_DETERMINE'
        report['error'] = (
            f'near_quality_rejected: {near_reasons}' if near_reasons else 'near_quality_rejected'
        )
        print(f'[two_view] STOP: NEAR quality REJECT ({near_reasons})', flush=True)
        return _save_two_view_report(out_dir, report, sw)

    report['near'] = {
        'image_size': {'width': int(near_bgr.shape[1]), 'height': int(near_bgr.shape[0])},
        'damages_detected': [
            {'label': d['label'], 'conf': float(d.get('conf', 0.0)),
             'pixel_count': int((d['mask'] > 0).sum())}
            for d in near_damage_masks
        ],
        'parts_detected': [
            {'label': p['label'], 'conf': float(p.get('conf', 0.0)),
             'pixel_count': int((p['mask'] > 0).sum())}
            for p in near_part_masks
        ],
    }

    if near_part_masks:
        ov = make_parts_overlay(near_bgr, near_part_masks)
        p = out_dir / 'near_parts_overlay.jpg'
        cv2.imwrite(str(p), ov)
        report['files']['near_parts_overlay'] = str(p)
    if near_damage_masks:
        ov = make_damages_overlay(near_bgr, near_damage_masks)
        p = out_dir / 'near_damages_overlay.jpg'
        cv2.imwrite(str(p), ov)
        report['files']['near_damages_overlay'] = str(p)

    if not near_damage_masks:
        report['error'] = 'no_damages_detected_on_near'
        return _save_two_view_report(out_dir, report, sw)

    match_info: Optional[Dict[str, Any]] = None
    matcher_used: Optional[str] = None

    print('[two_view] Trying RoMa-masked matcher (may take 15-60s on CPU)...', flush=True)
    roma_info: Optional[Dict[str, Any]] = None
    try:
        with sw.section('NEAR↔FAR matching (RoMa-masked)'):
            roma_info = match_near_to_far_roma_masked(
                near_bgr, far_bgr,
                near_damage_masks, near_part_masks, far_core['part_masks'],
                min_overlap_px=pipeline.min_overlap_px,
            )
    except Exception as exc:
        dlog(f'[two_view] roma_masked raised: {exc}')
        print(f'[two_view] RoMa-masked raised exception: {exc}', flush=True)
        roma_info = None

    if roma_info is not None:
        try:
            shared_vis = _compute_shared_parts_masks(
                near_damage_masks, near_part_masks, far_core['part_masks'],
                near_hw=near_bgr.shape[:2], far_hw=far_bgr.shape[:2],
                min_overlap_px=pipeline.min_overlap_px,
            )
            if shared_vis is not None:
                near_keep_vis = shared_vis['near_mask'].copy()
                far_keep_vis = shared_vis['far_mask'].copy()
                kp = roma_info['crop_info'].get('mask_dilate_px', 0)
                if kp > 0:
                    k = max(1, int(kp) * 2 + 1)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                    near_keep_vis = cv2.dilate(near_keep_vis, kernel)
                    far_keep_vis = cv2.dilate(far_keep_vis, kernel)
                masked_vis = make_masked_inputs_overlay(
                    near_bgr, far_bgr, near_keep_vis, far_keep_vis,
                )
                p_mvis = out_dir / 'roma_masked_inputs.jpg'
                cv2.imwrite(str(p_mvis), masked_vis)
                report['files']['roma_masked_inputs'] = str(p_mvis)
        except Exception as vis_exc:
            dlog(f'[two_view] masked_inputs vis failed: {vis_exc}')

    if roma_info is not None and not roma_info.get('degenerate', False):
        match_info = roma_info
        matcher_used = 'roma_masked'
        print(
            f'[two_view] === MATCHER USED: roma_masked === '
            f'inliers={match_info["n_inliers"]}/{match_info["n_matches"]} '
            f'ratio={match_info["n_inliers"]/max(1,match_info["n_matches"]):.3f}',
            flush=True,
        )
    elif roma_info is not None:
        print(
            f'[two_view] RoMa-masked DEGENERATE: '
            f'{roma_info.get("degenerate_reason")} — falling back to cropped LoFTR',
            flush=True,
        )
        dlog(
            f'[two_view] roma_masked degenerate: '
            f'{roma_info.get("degenerate_reason")} — trying cropped LoFTR next'
        )
    else:
        print('[two_view] RoMa-masked returned None — trying cropped LoFTR', flush=True)

    if match_info is None:
        report['error'] = 'roma_only_mode: RoMa did not return a match (no LoFTR fallback enabled)'
        print('[two_view] RoMa returned no result, and LoFTR fallbacks are disabled', flush=True)
        return _save_two_view_report(out_dir, report, sw)

    report['matching'] = {
        'method': match_info['method'],
        'matcher_used': matcher_used,
        'n_matches': int(match_info['n_matches']),
        'n_inliers': int(match_info['n_inliers']),
        'inlier_ratio': (
            float(match_info['n_inliers']) / max(1, match_info['n_matches'])
        ),
        'fallback_reason': match_info.get('fallback_reason'),
        'H': match_info['H'].tolist(),
        'degenerate': bool(match_info.get('degenerate', False)),
        'degenerate_reason': match_info.get('degenerate_reason'),
        'crop_info': match_info.get('crop_info'),
    }

    inl = match_info['inliers']
    matches_img = make_matches_overlay(
        near_bgr, far_bgr,
        match_info['pts_near'][inl],
        match_info['pts_far'][inl],
        scores=match_info['scores'][inl] if match_info['scores'] is not None else None,
        max_lines=300,
    )
    p_matches = out_dir / 'near_far_matches.jpg'
    cv2.imwrite(str(p_matches), matches_img)
    report['files']['matches_overlay'] = str(p_matches)

    R_far = np.asarray(far_core['pose']['R'], dtype=np.float64)
    t_far = np.asarray(far_core['pose']['t'], dtype=np.float64).reshape(3)
    K_far = np.asarray(far_core['pose']['K'], dtype=np.float64)

    inl = match_info['inliers']
    pts_near_inl = match_info['pts_near'][inl]
    pts_far_inl = match_info['pts_far'][inl]

    Hn, Wn = near_bgr.shape[:2]
    focal_near_px, focal_src = _estimate_focal_near_px(
        near_path, Wn, K_far, far_bgr.shape[1],
    )
    K_near, _ = make_camera_matrix(Wn, Hn, focal=focal_near_px)

    pnp_info: Optional[Dict[str, Any]] = None
    if len(pts_near_inl) >= 6:
        print(
            f'[two_view] Trying PnP NEAR via 3D-mesh '
            f'({len(pts_near_inl)} correspondences, focal_src={focal_src})...',
            flush=True,
        )
        try:
            with sw.section('PnP NEAR via 3D-mesh (RoMa correspondences)'):
                pnp_info = solve_near_pose_via_correspondences(
                    pts_near_inl, pts_far_inl,
                    R_far=R_far, t_far=t_far, K_far=K_far, K_near=K_near,
                    mesh_verts=pipeline.full_ray_verts,
                    mesh_faces=pipeline.full_ray_faces,
                    ransac_reproj_err_px=pipeline.pnp_ransac_reproj_err_px,
                )
        except Exception as exc:
            dlog(f'[two_view] solve_near_pose_via_correspondences raised: {exc}')
            print(f'[two_view] PnP NEAR raised: {exc}', flush=True)
            pnp_info = None

    if pnp_info is not None:
        print(
            f'[two_view] === PnP NEAR OK === '
            f'hits={pnp_info["n_ray_hits"]}/{pnp_info["n_correspondences"]} '
            f'inliers={pnp_info["n_pnp_inliers"]} '
            f'mean_err={pnp_info["mean_reproj_err_px"]:.2f}px',
            flush=True,
        )
        report['pnp_near'] = {
            'success': True,
            'n_correspondences': pnp_info['n_correspondences'],
            'n_ray_hits': pnp_info['n_ray_hits'],
            'n_pnp_inliers': pnp_info['n_pnp_inliers'],
            'mean_reproj_err_px': pnp_info['mean_reproj_err_px'],
            'max_reproj_err_px': pnp_info['max_reproj_err_px'],
            'focal_near_px': float(focal_near_px),
            'focal_source': focal_src,
            'R': pnp_info['R'].tolist(),
            't': pnp_info['t'].tolist(),
            'K_near': pnp_info['K'].tolist(),
        }

        try:
            with sw.section('NEAR: pose overlays (full-mesh wireframe + render)'):
                R_n = pnp_info['R']
                t_n = pnp_info['t']
                K_n = pnp_info['K']
                pose_ov = overlay_mesh_on_photo(
                    near_bgr, pipeline.full_render_verts, pipeline.full_render_faces,
                    R_n, t_n, K_n,
                    line_color=(0, 255, 140), fill_color=(0, 255, 140),
                    alpha_fill=0.18, line_thickness=1,
                )
                cv2.putText(
                    pose_ov,
                    f'NEAR pose=PnP via RoMa  '
                    f'inliers={pnp_info["n_pnp_inliers"]}/{pnp_info["n_ray_hits"]}  '
                    f'mean_err={pnp_info["mean_reproj_err_px"]:.2f}px',
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 255, 128), 2, cv2.LINE_AA,
                )
                p_near_ov = out_dir / 'near_pose_overlay.jpg'
                cv2.imwrite(str(p_near_ov), pose_ov)
                report['files']['near_pose_overlay'] = str(p_near_ov)

                standalone = render_wireframe(
                    pipeline.full_render_verts, pipeline.full_render_faces,
                    R_n, t_n, K_n,
                    near_bgr.shape[0], near_bgr.shape[1],
                )
                p_near_rd = out_dir / 'near_pose_render.jpg'
                cv2.imwrite(str(p_near_rd), standalone)
                report['files']['near_pose_render'] = str(p_near_rd)
        except Exception as vis_exc:
            dlog(f'[two_view] NEAR pose vis failed: {vis_exc}')
            print(f'[two_view] NEAR pose vis failed: {vis_exc}', flush=True)
    else:
        print('[two_view] PnP NEAR did not converge — falling back to homography warp', flush=True)
        report['pnp_near'] = {'success': False, 'focal_near_px': float(focal_near_px),
                              'focal_source': focal_src}

    if pnp_info is not None:
        R_n = pnp_info['R']
        t_n = pnp_info['t']
        K_n = pnp_info['K']

        try:
          with sw.section('NEAR: damage_on_mesh viz (full-mesh ray-cast + render)'):
            near_damage_union = np.zeros(near_bgr.shape[:2], dtype=np.uint8)
            for d in near_damage_masks:
                near_damage_union = np.maximum(
                    near_damage_union,
                    (np.asarray(d['mask']) > 0).astype(np.uint8),
                )
            hits_for_vis = ray_cast(
                near_damage_union,
                pipeline.full_ray_verts, pipeline.full_ray_faces,
                R_n.astype(np.float64), t_n.astype(np.float64), K_n.astype(np.float64),
                subsample=2,
            )
            damaged_face_idx = {fi for fi, _ in hits_for_vis}

            vis = overlay_mesh_on_photo(
                near_bgr, pipeline.full_render_verts, pipeline.full_render_faces,
                R_n, t_n, K_n,
                line_color=(140, 220, 80), fill_color=(180, 230, 100),
                alpha_fill=0.06, line_thickness=1,
            )
            vis = _render_damaged_faces_on_photo(
                vis, pipeline.full_ray_verts, pipeline.full_ray_faces,
                R_n, t_n, K_n, damaged_face_idx,
                fill_color=(35, 35, 200), alpha=0.55,
            )
            pure_red = np.array([0, 0, 230], dtype=np.float32)
            for d in near_damage_masks:
                m = (np.asarray(d['mask']) > 0)
                if m.any():
                    vis[m] = (
                        vis[m].astype(np.float32) * 0.45 + pure_red * 0.55
                    ).astype(np.uint8)
            label = (
                f'NEAR: damage mask (bright red) + mesh wireframe + '
                f'ray-cast hit faces (dark red)  '
                f'[{len(damaged_face_idx)} hit faces, {len(hits_for_vis)} hits]'
            )
            cv2.putText(vis, label, (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(vis, label, (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)
            p_dmesh = out_dir / 'near_damage_on_mesh.jpg'
            cv2.imwrite(str(p_dmesh), vis)
            report['files']['near_damage_on_mesh'] = str(p_dmesh)
            print(
                f'[two_view] near_damage_on_mesh saved: '
                f'{len(damaged_face_idx)} damaged faces, '
                f'{len(hits_for_vis)} ray hits',
                flush=True,
            )
        except Exception as vis_exc:
            dlog(f'[two_view] near_damage_on_mesh viz failed: {vis_exc}')
            print(f'[two_view] near_damage_on_mesh viz failed: {vis_exc}', flush=True)

        with sw.section('NEAR: find_damaged_parts (damage ∩ parts)'):
            damaged_near = find_damaged_parts(
                near_damage_masks, near_part_masks,
                min_overlap_px=pipeline.min_overlap_px,
            )
        report['damaged_parts_2d'] = [
            {'part': dp['part'], 'damage_idx': dp['damage_idx'],
             'overlap_px': dp['overlap_px']}
            for dp in damaged_near
        ]
        if not damaged_near:
            report['error'] = 'no_part_damage_intersection_on_near'
            print(
                '[two_view] No intersection between NEAR damages and NEAR parts — '
                'damaged part undetermined',
                flush=True,
            )
            return _save_two_view_report(out_dir, report, sw)

        with sw.section('severity_per_part (ray-cast on part-mesh)'):
            severity = pipeline.severity_per_part(
                damaged_near, R_n, t_n, K_n, near_bgr, out_dir,
            )
        report['severity_per_part'] = severity
        print(
            f'[two_view] severity via NEAR-3D: '
            f'{len(damaged_near)} damaged parts, '
            f'severities={[round(s.get("severity_percent") or 0, 3) for s in severity]}',
            flush=True,
        )

        _run_hidden_damage_classifier(pipeline, severity, report, sw)

        return _save_two_view_report(out_dir, report, sw)

    print(
        '[two_view] PnP NEAR did not converge — fallback to homography warp + FAR severity',
        flush=True,
    )
    with sw.section('fallback: warp NEAR damages to FAR (H matrix)'):
        damage_masks_far_warped = warp_damage_masks_to_far(
            near_damage_masks, match_info['H'], far_bgr.shape[:2],
        )
        warped_overlay = make_warped_damages_on_far_overlay(far_bgr, damage_masks_far_warped)
        p_warp = out_dir / 'damages_warped_on_far.jpg'
        cv2.imwrite(str(p_warp), warped_overlay)
        report['files']['damages_warped_on_far'] = str(p_warp)

    if not damage_masks_far_warped:
        report['error'] = 'warped_damage_masks_empty'
        return _save_two_view_report(out_dir, report, sw)

    far_part_masks = far_core['part_masks']
    if not far_part_masks:
        report['error'] = 'no_parts_detected_on_far'
        return _save_two_view_report(out_dir, report, sw)

    with sw.section('FAR: find_damaged_parts (warped_damage ∩ FAR_parts)'):
        damaged = find_damaged_parts(
            damage_masks_far_warped, far_part_masks,
            min_overlap_px=pipeline.min_overlap_px,
        )
    report['damaged_parts_2d'] = [
        {'part': dp['part'], 'damage_idx': dp['damage_idx'], 'overlap_px': dp['overlap_px']}
        for dp in damaged
    ]
    if not damaged:
        report['error'] = 'no_part_damage_intersection_after_warp'
        return _save_two_view_report(out_dir, report, sw)

    pose = far_core['pose']
    with sw.section('severity_per_part (FAR ray-cast, fallback path)'):
        severity = pipeline.severity_per_part(
            damaged, pose['R'], pose['t'], pose['K'],
            far_core['img_bgr'], out_dir,
        )
    report['severity_per_part'] = severity

    _run_hidden_damage_classifier(pipeline, severity, report, sw)

    return _save_two_view_report(out_dir, report, sw)
