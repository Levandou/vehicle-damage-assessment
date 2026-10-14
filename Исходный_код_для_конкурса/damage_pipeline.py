
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from cv_models import (
    AdaptivePhotoQualityGate,
    DamagesYoloSegmenter,
    KeypointsYoloDetector,
    LargestCarFinder,
    PARENT_OF,
    PartsYoloSegmenter,
)
from damage_3d import (
    _read_focal_exif,
    annotate_keypoints,
    build_correspondences_from_yolo_keypoints,
    build_part_lookup,
    compute_severity,
    draw_severity_on_photo,
    ensure_dir,
    find_damaged_parts,
    get_keypoint_world_positions,
    load_glb,
    load_glb_chunks,
    make_camera_matrix,
    make_damages_overlay,
    make_keypoints_visualization,
    make_parts_overlay,
    overlay_mesh_on_photo,
    overlay_projected_hit_points,
    ray_cast,
    render_mesh_debug_view,
    render_wireframe,
    resolve_label_to_glb_key,
    safe_name,
    simplify_mesh,
    try_solve_pnp,
    visualize_mesh_damage_debug,
)
from contextlib import nullcontext

from diag import Stopwatch, dlog


def _all_kp_node_names() -> List[str]:
    names = []
    for prefix, parent in PARENT_OF.items():
        for k in range(1, 41):
            names.append(f'{parent}_{k}')
    return names


class DamageEstimationPipeline:

    def __init__(
        self,
        details_model_path: str,
        damages_model_path: str,
        points_model_path: str,
        glb_path: str,
        car_model_path: Optional[str] = None,
        hidden_damage_model_path: Optional[str] = None,
        hidden_damage_threshold: Optional[float] = None,
        glb_up_axis: str = 'auto',
        glb_flip_up: bool = False,
        max_faces: int = 40000,
        render_max_faces: int = 320000,
        ray_subsample: int = 4,
        min_overlap_px: int = 200,
        ok_iou_threshold: float = 0.30,
        focal_px: Optional[float] = None,
        apply_quality_preprocess: bool = False,
        kp_conf_min: float = 0.50,
        kp_dup_px: float = 10.0,
        pnp_ransac_reproj_err_px: float = 8.0,
        pnp_ransac_threshold_ladder_px: Tuple[float, ...] = (8.0, 15.0, 25.0),
    ):
        self.details = PartsYoloSegmenter(details_model_path)
        self.damages = DamagesYoloSegmenter(damages_model_path)
        self.points = KeypointsYoloDetector(points_model_path)
        self.quality_gate = AdaptivePhotoQualityGate()
        self.apply_quality_preprocess = bool(apply_quality_preprocess)

        self.full_verts, self.full_faces, self.part_meshes_raw = load_glb(
            glb_path, up_axis=glb_up_axis, flip_up=glb_flip_up,
            debug_axes=False, apply_reorient=False,
        )
        self.j_data, self.bin_data = load_glb_chunks(Path(glb_path))
        self.kp_xyz = get_keypoint_world_positions(self.j_data, _all_kp_node_names())
        self.part_lookup = build_part_lookup(self.part_meshes_raw)

        self.max_faces = max_faces
        self.render_max_faces = render_max_faces

        self.full_ray_verts, self.full_ray_faces = simplify_mesh(
            self.full_verts, self.full_faces, max_faces=max_faces, tag='ray mesh (full car)',
        )
        self.full_render_verts, self.full_render_faces = simplify_mesh(
            self.full_verts, self.full_faces, max_faces=render_max_faces, tag='render mesh (full car)',
        )

        self.part_ray_meshes: Dict[str, tuple] = {}
        self.part_render_meshes: Dict[str, tuple] = {}
        for pname, (pv, pf) in self.part_meshes_raw.items():
            if pname == '__unnamed__':
                continue
            rv, rf = simplify_mesh(pv, pf, max_faces=max_faces, quiet=True)
            self.part_ray_meshes[pname] = (rv, rf)
            rv2, rf2 = simplify_mesh(pv, pf, max_faces=render_max_faces, quiet=True)
            self.part_render_meshes[pname] = (rv2, rf2)

        self.ray_subsample = ray_subsample
        self.min_overlap_px = min_overlap_px
        self.ok_iou_threshold = ok_iou_threshold
        self.focal_px = focal_px
        self.kp_conf_min = float(kp_conf_min)
        self.kp_dup_px = float(kp_dup_px)
        self.pnp_ransac_reproj_err_px = float(pnp_ransac_reproj_err_px)
        self.pnp_ransac_threshold_ladder_px = tuple(float(x) for x in pnp_ransac_threshold_ladder_px)

        self.car_finder: Optional[Any] = None
        if car_model_path:
            try:
                self.car_finder = LargestCarFinder(car_model_path)
                dlog(f'[pipeline.init] LargestCarFinder loaded from {car_model_path}')
            except Exception as exc:
                dlog(f'[pipeline.init] LargestCarFinder load failed ({car_model_path}): {exc}')
                self.car_finder = None

        self.hidden_damage_classifier: Optional[Any] = None
        if hidden_damage_model_path:
            try:
                from hidden_damage import HiddenDamageClassifier
                self.hidden_damage_classifier = HiddenDamageClassifier(
                    hidden_damage_model_path,
                    threshold_override=hidden_damage_threshold,
                )
                dlog(
                    f'[pipeline.init] HiddenDamageClassifier loaded from '
                    f'{hidden_damage_model_path} (threshold='
                    f'{self.hidden_damage_classifier.threshold})'
                )
            except Exception as exc:
                dlog(
                    f'[pipeline.init] HiddenDamageClassifier load failed '
                    f'({hidden_damage_model_path}): {exc}'
                )
                self.hidden_damage_classifier = None


    def _diagnose_keypoints(self, yolo_keypoints: List[Dict]) -> Dict[str, Any]:
        glb_nodes_present = sorted(self.kp_xyz.keys())
        dlog(
            f'[pose/keypoints] GLB содержит {len(glb_nodes_present)} keypoint-нод: '
            f'{glb_nodes_present}'
        )
        dlog(f'[pose/keypoints] YOLO нашёл {len(yolo_keypoints)} точек:')

        per_kp: List[Dict[str, Any]] = []
        matched: List[Dict[str, Any]] = []
        dropped_no_node: List[Dict[str, Any]] = []
        dropped_no_mapping: List[Dict[str, Any]] = []

        for idx, kp in enumerate(yolo_keypoints):
            cls = kp.get('cls_name')
            glb = kp.get('glb_name')
            cx = float(kp.get('cx') or 0.0)
            cy = float(kp.get('cy') or 0.0)
            conf = float(kp.get('conf') or 0.0)
            entry = {
                'idx': idx,
                'cls_name': cls,
                'glb_name': glb,
                'cx': cx, 'cy': cy, 'conf': conf,
                'status': None,
            }
            if not glb:
                entry['status'] = 'NO_PARENT_MAPPING'
                dropped_no_mapping.append(entry)
                dlog(
                    f'[pose/keypoints]   #{idx:02d} cls={cls!r:>10} -> DROP: '
                    f'в PARENT_OF нет правила маппинга в GLB-ноду'
                )
            elif glb in self.kp_xyz:
                entry['status'] = 'MATCHED'
                matched.append(entry)
                dlog(
                    f'[pose/keypoints]   #{idx:02d} cls={cls!r:>10} -> MATCH glb={glb!r:<24} '
                    f'cx={cx:7.1f} cy={cy:7.1f} conf={conf:.3f}'
                )
            else:
                entry['status'] = 'NO_NODE_IN_GLB'
                dropped_no_node.append(entry)
                dlog(
                    f'[pose/keypoints]   #{idx:02d} cls={cls!r:>10} -> DROP: '
                    f'ожидался узел {glb!r} в GLB, но его там нет'
                )
            per_kp.append(entry)

        dlog(
            f'[pose/keypoints] SUMMARY total={len(yolo_keypoints)} '
            f'matched={len(matched)} '
            f'dropped_no_node_in_glb={len(dropped_no_node)} '
            f'dropped_no_parent_mapping={len(dropped_no_mapping)} '
            f'pnp_min_required=4'
        )

        return {
            'glb_keypoint_nodes_present': glb_nodes_present,
            'yolo_keypoints_total': len(yolo_keypoints),
            'matched_count': len(matched),
            'dropped_no_node_in_glb_count': len(dropped_no_node),
            'dropped_no_parent_mapping_count': len(dropped_no_mapping),
            'per_keypoint': per_kp,
            'matched': matched,
            'dropped_no_node_in_glb': dropped_no_node,
            'dropped_no_parent_mapping': dropped_no_mapping,
        }

    def _prefilter_keypoints(
        self,
        yolo_keypoints: List[Dict],
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        diag: Dict[str, Any] = {
            'conf_min': self.kp_conf_min,
            'dup_px': self.kp_dup_px,
            'n_input': len(yolo_keypoints),
            'dropped_low_conf': [],
            'dropped_duplicate': [],
        }

        kept: List[Dict] = []
        for kp in yolo_keypoints:
            conf = float(kp.get('conf') or 0.0)
            if conf < self.kp_conf_min:
                diag['dropped_low_conf'].append({
                    'cls_name': kp.get('cls_name'),
                    'glb_name': kp.get('glb_name'),
                    'cx': float(kp.get('cx') or 0.0),
                    'cy': float(kp.get('cy') or 0.0),
                    'conf': conf,
                })
                dlog(
                    f'[pose/prefilter] DROP low_conf cls={kp.get("cls_name")!r:>10} '
                    f'glb={kp.get("glb_name")!r:<24} conf={conf:.3f} < {self.kp_conf_min:.2f}'
                )
                continue
            kept.append(kp)

        kept_sorted = sorted(kept, key=lambda k: float(k.get('conf') or 0.0), reverse=True)
        accepted: List[Dict] = []
        for kp in kept_sorted:
            cx = float(kp.get('cx') or 0.0)
            cy = float(kp.get('cy') or 0.0)
            dup_of = None
            for a in accepted:
                ax = float(a.get('cx') or 0.0)
                ay = float(a.get('cy') or 0.0)
                if (cx - ax) ** 2 + (cy - ay) ** 2 < self.kp_dup_px ** 2:
                    dup_of = a
                    break
            if dup_of is not None:
                diag['dropped_duplicate'].append({
                    'cls_name': kp.get('cls_name'),
                    'glb_name': kp.get('glb_name'),
                    'cx': cx, 'cy': cy,
                    'conf': float(kp.get('conf') or 0.0),
                    'duplicate_of_cls': dup_of.get('cls_name'),
                    'duplicate_of_glb': dup_of.get('glb_name'),
                })
                dlog(
                    f'[pose/prefilter] DROP duplicate cls={kp.get("cls_name")!r:>10} '
                    f'glb={kp.get("glb_name")!r:<24} (~ {dup_of.get("cls_name")!r}/'
                    f'{dup_of.get("glb_name")!r}, < {self.kp_dup_px:.1f}px)'
                )
                continue
            accepted.append(kp)

        diag['n_output'] = len(accepted)
        dlog(
            f'[pose/prefilter] SUMMARY in={len(yolo_keypoints)} '
            f'low_conf_dropped={len(diag["dropped_low_conf"])} '
            f'duplicate_dropped={len(diag["dropped_duplicate"])} '
            f'kept={len(accepted)}  (conf_min={self.kp_conf_min:.2f}, dup_px={self.kp_dup_px:.1f})'
        )
        return accepted, diag

    def _build_focal_candidates(
        self,
        width: int,
        source_image_path: Optional[str],
    ) -> List[Tuple[float, str]]:
        candidates: List[Tuple[float, str]] = []
        if self.focal_px is not None and float(self.focal_px) > 0:
            candidates.append((float(self.focal_px), 'override'))
            dlog(
                f'[K/source] жёсткий focal_px={self.focal_px:.2f} из __init__ — '
                'grid-search/EXIF отключены.'
            )
            return candidates

        if source_image_path:
            f35mm, fmm, make, model = _read_focal_exif(source_image_path)
            if f35mm and f35mm > 0:
                f_exif_px = float(f35mm) * float(width) / 36.0
                candidates.append((f_exif_px, 'exif'))
                dlog(
                    f'[K/exif] image={source_image_path} '
                    f'FocalLengthIn35mmFilm={f35mm}mm FocalLength={fmm}mm '
                    f'Make={make} Model={model} -> focal_px={f_exif_px:.2f}'
                )
            else:
                dlog(
                    f'[K/exif] image={source_image_path} '
                    f'FocalLengthIn35mmFilm отсутствует; FocalLength={fmm}mm '
                    f'Make={make} Model={model} -> только grid.'
                )
        else:
            dlog('[K/exif] source_image_path не передан, EXIF не читаем; только grid.')

        for factor in (0.5, 0.65, 0.72, 0.85, 1.0, 1.2):
            candidates.append((float(factor) * float(width), f'grid_{factor:.2f}W'))
        return candidates

    def _run_pnp_for_focal(
        self,
        obj_pts: np.ndarray,
        img_pts: np.ndarray,
        width: int,
        height: int,
        focal_px: float,
    ) -> Optional[Dict[str, Any]]:
        K_f, dist_f = make_camera_matrix(width, height, focal=focal_px)
        n_total = int(obj_pts.shape[0])
        min_inliers_for_win = min(6, n_total)

        best_with_quorum: Optional[Tuple[str, Dict[str, Any], float]] = None
        best_any: Optional[Tuple[str, Dict[str, Any], float]] = None

        for thresh in self.pnp_ransac_threshold_ladder_px:
            results = try_solve_pnp(
                obj_pts, img_pts, K_f, dist_f,
                ransac_reproj_err_px=thresh,
                use_ransac=True,
            )
            ok = {n: r for n, r in results.items() if r.get('ok')}
            if not ok:
                continue
            for nm, r in ok.items():
                err = r['mean_err_px']
                if best_any is None or err < best_any[1]['mean_err_px']:
                    best_any = (nm, r, thresh)
            with_quorum = {n: r for n, r in ok.items() if r.get('n_inliers', 0) >= min_inliers_for_win}
            if with_quorum:
                win_name = min(with_quorum, key=lambda n: with_quorum[n]['mean_err_px'])
                win = with_quorum[win_name]
                best_with_quorum = (win_name, win, thresh)
                break

        if best_with_quorum is not None:
            win_name, win, thresh = best_with_quorum
            return {**win, 'method': win_name, 'K': K_f, 'used_ransac_threshold_px': thresh}

        if best_any is not None:
            win_name, win, thresh = best_any
            return {
                **win, 'method': win_name, 'K': K_f,
                'used_ransac_threshold_px': thresh,
                'low_quality_no_quorum': True,
            }

        results = try_solve_pnp(
            obj_pts, img_pts, K_f, dist_f,
            use_ransac=False,
        )
        ok = {n: r for n, r in results.items() if r.get('ok')}
        if not ok:
            return None
        win_name = min(ok, key=lambda n: ok[n]['mean_err_px'])
        win = ok[win_name]
        return {**win, 'method': win_name, 'K': K_f, 'ransac_failed': True}

    def _solve_pose(
        self,
        img_bgr: np.ndarray,
        yolo_keypoints: List[Dict],
        source_image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        h, w = img_bgr.shape[:2]

        kp_diag = self._diagnose_keypoints(yolo_keypoints)
        prefiltered_kps, prefilter_diag = self._prefilter_keypoints(yolo_keypoints)
        usable_kps = [k for k in prefiltered_kps if k.get('glb_name') in self.kp_xyz]

        prefilter_fallback = False
        if len(usable_kps) < 4:
            dlog(
                f'[pose/prefilter] WARN после prefilter осталось {len(usable_kps)} '
                f'matched keypoints (<4) — откатываемся на нефильтрованный набор '
                f'(RANSAC выполнит отсев аутлаеров).'
            )
            usable_kps = [k for k in yolo_keypoints if k.get('glb_name') in self.kp_xyz]
            prefilter_fallback = True

        pnp_meta: Dict[str, Any] = {
            'method': None,
            'n_keypoints_total': len(yolo_keypoints),
            'n_keypoints_after_prefilter': len(prefiltered_kps),
            'n_keypoints_matched_glb': len(usable_kps),
            'keypoints_diagnostics': kp_diag,
            'prefilter': prefilter_diag,
            'prefilter_fallback_used': prefilter_fallback,
            'ransac_reproj_err_px': self.pnp_ransac_reproj_err_px,
        }

        if len(usable_kps) < 4:
            raise RuntimeError(
                f'PnP не запущен: matched keypoints={len(usable_kps)} < 4. '
                f'Силуэтный fallback отключён. '
                f'Смотри keypoints_diagnostics в отчёте и [pose/keypoints] в логе для разбора.'
            )

        obj_pts, img_pts, names = build_correspondences_from_yolo_keypoints(
            usable_kps, self.kp_xyz,
        )

        candidates = self._build_focal_candidates(w, source_image_path)

        def _score(w: Dict[str, Any]) -> Tuple[int, float]:
            return (int(w.get('n_inliers', 0)), -float(w['mean_err_px']))

        best: Optional[Tuple[float, str, Dict[str, Any]]] = None
        for f_px, label in candidates:
            win = self._run_pnp_for_focal(obj_pts, img_pts, w, h, f_px)
            if win is None:
                dlog(f'[K/grid] f={f_px:8.2f} ({label:>16}): ВСЕ PnP-методы не сошлись')
                continue
            improved = best is None or _score(win) > _score(best[2])
            flags = []
            if win.get('low_quality_no_quorum'):
                flags.append('NO_QUORUM')
            if win.get('ransac_failed'):
                flags.append('NO_RANSAC')
            flag_s = (' [' + ','.join(flags) + ']') if flags else ''
            dlog(
                f'[K/grid] f={f_px:8.2f} ({label:>16}): best={win["method"]:<9} '
                f'mean_err={win["mean_err_px"]:7.2f}px max_err={win["max_err_px"]:7.2f}px '
                f'inliers={win.get("n_inliers", "?")}/{win.get("n_total", "?")} '
                f'thr={win.get("used_ransac_threshold_px", "?")}px{flag_s}'
                f'{"   <- BEST" if improved else ""}'
            )
            if improved:
                best = (f_px, label, win)

        if best is None:
            raise RuntimeError(
                f'PnP не сошёлся ни одним методом ни с одним focal_px из '
                f'{len(candidates)} кандидатов: {[c[1] for c in candidates]}'
            )

        if best[1] != 'override':
            anchor = best[0]
            for factor in (0.95, 1.05, 0.9, 1.1):
                f_try = anchor * factor
                win = self._run_pnp_for_focal(obj_pts, img_pts, w, h, f_try)
                if win is None:
                    continue
                improved = _score(win) > _score(best[2])
                dlog(
                    f'[K/refine] f={f_try:8.2f} (x{factor:.2f} anchor): '
                    f'best={win["method"]:<9} mean_err={win["mean_err_px"]:7.2f}px '
                    f'inliers={win.get("n_inliers", "?")}/{win.get("n_total", "?")}'
                    f'{"   <- BEST" if improved else ""}'
                )
                if improved:
                    best = (f_try, f'refine_x{factor:.2f}', win)

        f_final, label_final, win = best
        K_final = win['K']
        R = cv2.Rodrigues(win['rvec_np'])[0]
        t = win['tvec_np'].flatten()

        hfov_deg = float(2.0 * np.degrees(np.arctan(0.5 * float(w) / float(f_final))))
        dlog(
            f'[K/winner] focal_px={f_final:.2f} source={label_final} '
            f'(focal/W={f_final / w:.3f}, HFOV~{hfov_deg:.1f}°)'
        )
        inlier_idx = win.get('inlier_idx', list(range(len(names))))
        outlier_idx = win.get('outlier_idx', [])
        inlier_set = set(inlier_idx)
        inlier_names = [names[i] for i in inlier_idx]
        outlier_names = [names[i] for i in outlier_idx]

        quality_tag = ''
        if win.get('ransac_failed'):
            quality_tag = '  !! ВСЕ RANSAC-пороги провалились, использован non-RANSAC PnP (поза ненадёжна)'
        elif win.get('low_quality_no_quorum'):
            quality_tag = '  ! NO_QUORUM: ни один порог не дал >=6 inliers, поза ниже стандартного качества'

        dlog(
            f'[pose/pnp] WINNER method={win["method"]} '
            f'mean_err_px(inliers)={win["mean_err_px"]:.3f} '
            f'max_err_px(inliers)={win["max_err_px"]:.3f} '
            f'inliers={win.get("n_inliers", len(names))}/{win.get("n_total", len(names))} '
            f'thr={win.get("used_ransac_threshold_px", "?")}px '
            f'||t||={float(np.linalg.norm(t)):.3f}{quality_tag}'
        )
        if outlier_names:
            dlog(f'[pose/pnp] RANSAC OUTLIERS (отброшены): {outlier_names}')
        for i, (nm, e) in enumerate(zip(names, win['per_point_err'])):
            tag = 'IN ' if i in inlier_set else 'OUT'
            dlog(f'[pose/pnp]   [{tag}] {nm:<28} reproj_err={e:6.2f}px')

        pnp_meta.update({
            'method': 'pnp',
            'best_method': win['method'],
            'mean_err_px': float(win['mean_err_px']),
            'max_err_px': float(win['max_err_px']),
            'names': names,
            'image_pts': img_pts.tolist(),
            'object_pts': obj_pts.tolist(),
            'projected_pts': win['projected'],
            'per_point_err': win['per_point_err'],
            'focal_px': float(f_final),
            'focal_source': label_final,
            'focal_to_width_ratio': float(f_final / w),
            'hfov_deg_est': hfov_deg,
            'ransac_inlier_idx': inlier_idx,
            'ransac_outlier_idx': outlier_idx,
            'ransac_inlier_names': inlier_names,
            'ransac_outlier_names': outlier_names,
            'n_inliers': int(win.get('n_inliers', len(names))),
            'n_outliers': int(win.get('n_outliers', 0)),
            'used_ransac_threshold_px': win.get('used_ransac_threshold_px'),
            'low_quality_no_quorum': bool(win.get('low_quality_no_quorum', False)),
            'ransac_failed_used_plain_pnp': bool(win.get('ransac_failed', False)),
        })
        return {'R': R, 't': t, 'K': K_final, 'meta': pnp_meta}


    def severity_per_part(
        self,
        damaged_parts: List[Dict],
        R: np.ndarray, t: np.ndarray, K: np.ndarray,
        img_bgr: np.ndarray,
        out_dir: Path,
    ) -> List[Dict]:
        h, w = img_bgr.shape[:2]
        R64 = np.asarray(R, dtype=np.float64)
        t64 = np.asarray(t, dtype=np.float64).reshape(3)
        K64 = np.asarray(K, dtype=np.float64)

        results = []
        for idx, dp in enumerate(damaged_parts):
            label = dp['part']
            local_dmg = np.logical_and(dp['damage_mask'] > 0, dp['part_mask'] > 0).astype(np.uint8)
            local_px = int(local_dmg.sum())
            glb_key = resolve_label_to_glb_key(label, self.part_lookup, self.part_meshes_raw)

            base = {
                'part': label,
                'glb_part_key': glb_key,
                'damage_idx': dp['damage_idx'],
                'overlap_px': dp['overlap_px'],
                'local_damage_px': local_px,
            }

            if local_px == 0:
                results.append({**base, 'status': 'NO_LOCAL_DAMAGE', 'severity': None})
                continue

            if glb_key and glb_key in self.part_ray_meshes:
                ray_verts, ray_faces = self.part_ray_meshes[glb_key]
                rend_v, rend_f = self.part_render_meshes.get(
                    glb_key, (self.full_render_verts, self.full_render_faces),
                )
                ray_src = 'part'
            else:
                ray_verts, ray_faces = self.full_ray_verts, self.full_ray_faces
                rend_v, rend_f = self.full_render_verts, self.full_render_faces
                ray_src = 'full_fallback'

            hits = ray_cast(local_dmg, ray_verts, ray_faces, R64, t64, K64,
                            subsample=self.ray_subsample)
            if not hits:
                results.append({**base, 'ray_mesh_source': ray_src,
                                'status': 'NO_RAY_HITS', 'severity': None})
                continue

            sev, dmg_a, tot_a = compute_severity(hits, ray_verts, ray_faces)

            sev_photo = draw_severity_on_photo(img_bgr, local_dmg, hits, R64, t64, K64, sev)
            sev_photo_path = out_dir / f"severity_photo_{idx}_{safe_name(label)}.jpg"
            cv2.imwrite(str(sev_photo_path), sev_photo)

            mesh_view = render_mesh_debug_view(
                rend_v, rend_f, R64, t64, K64, h, w,
                highlight_faces={fi for fi, _ in hits},
            )
            mesh_view = overlay_projected_hit_points(
                mesh_view, hits, R64, t64, K64, color=(35, 55, 235), radius=4,
            )
            sev_mesh_path = out_dir / f"severity_mesh_{idx}_{safe_name(label)}.jpg"
            visualize_mesh_damage_debug(mesh_view, sev, len(hits), sev_mesh_path)

            results.append({
                **base,
                'ray_mesh_source': ray_src,
                'severity': round(float(sev), 6),
                'severity_percent': round(float(sev) * 100.0, 4),
                'damage_area_3d': float(dmg_a),
                'part_total_area_3d': float(tot_a),
                'hit_points': len(hits),
                'status': 'OK',
                'files': {
                    'severity_photo': str(sev_photo_path),
                    'severity_mesh': str(sev_mesh_path),
                },
            })

        return results


    def _process_image_core(
        self,
        img_bgr: np.ndarray,
        out_dir: Path,
        *,
        skip_damages: bool = False,
        skip_keypoints: bool = False,
        skip_quality_gate: bool = False,
        file_prefix: str = '',
        source_image_path: Optional[str] = None,
        stopwatch: Optional[Stopwatch] = None,
    ) -> Dict[str, Any]:
        out_dir = Path(out_dir)
        h, w = img_bgr.shape[:2]

        prefix_tag = f'[{file_prefix.rstrip("_") or "core"}]'

        def _sw(name: str):
            if stopwatch is None:
                return nullcontext()
            return stopwatch.section(f'{prefix_tag} {name}')

        report: Dict[str, Any] = {
            'image_size': {'width': w, 'height': h},
            'parts_detected': [],
            'damages_detected': [],
            'keypoints_detected': [],
            'files': {},
            'severity_per_part': [],
            'pose': None,
            'damaged_parts_2d': [],
            'photo_quality': None,
        }

        if not skip_quality_gate:
            with _sw('quality gate (full-frame)'):
                quality_before = self.quality_gate.evaluate(img_bgr)
                applied_ops: List[Dict[str, Any]] = []
                if quality_before['status'] == 'REJECT':
                    report['photo_quality'] = {
                        'before': quality_before,
                        'after': quality_before,
                        'applied_ops': [],
                        'status': 'REJECT',
                    }
                    report['error'] = 'photo_quality_rejected: ' + ','.join(
                        quality_before.get('reject_reasons') or []
                    )
                    return {
                        'img_bgr': img_bgr,
                        'part_masks': [], 'damage_masks': [], 'keypoints': [],
                        'pose': None, 'pose_error': None, 'report': report,
                    }

                if quality_before['status'] == 'FIX_AND_RETRY' and self.apply_quality_preprocess:
                    img_bgr, applied_ops = self.quality_gate.preprocess(img_bgr, quality_before)
                quality_after = self.quality_gate.evaluate(img_bgr)
                report['photo_quality'] = {
                    'before': quality_before,
                    'after': quality_after,
                    'applied_ops': applied_ops,
                    'status': quality_after['status'],
                }
                if quality_after['status'] == 'REJECT':
                    report['error'] = 'photo_quality_rejected_after_fix: ' + ','.join(
                        quality_after.get('reject_reasons') or []
                    )
                    return {
                        'img_bgr': img_bgr,
                        'part_masks': [], 'damage_masks': [], 'keypoints': [],
                        'pose': None, 'pose_error': None, 'report': report,
                    }

        with _sw('parts segmentation (DeepLabV3+)'):
            part_masks = self.details.predict(img_bgr)
        if skip_damages:
            damage_masks: List[Dict] = []
        else:
            with _sw('damages segmentation (Mask R-CNN)'):
                damage_masks = self.damages.predict(img_bgr)
        if skip_keypoints:
            yolo_keypoints: List[Dict] = []
        else:
            with _sw('keypoints (HRNet-W32)'):
                yolo_keypoints = self.points.predict(img_bgr)

        dlog(
            f'[pipeline.core] prefix={file_prefix!r} '
            f'parts={len(part_masks)} '
            f'damages={"SKIPPED" if skip_damages else len(damage_masks)} '
            f'keypoints={"SKIPPED" if skip_keypoints else len(yolo_keypoints)}'
        )

        report['parts_detected'] = [
            {'label': p['label'], 'conf': float(p.get('conf', 0.0)),
             'pixel_count': int((p['mask'] > 0).sum())}
            for p in part_masks
        ]
        report['damages_detected'] = [
            {'label': d['label'], 'conf': float(d.get('conf', 0.0)),
             'pixel_count': int((d['mask'] > 0).sum())}
            for d in damage_masks
        ]
        report['keypoints_detected'] = [
            {'cls_name': k['cls_name'], 'glb_name': k.get('glb_name'),
             'cx': k['cx'], 'cy': k['cy'], 'conf': k['conf']}
            for k in yolo_keypoints
        ]

        with _sw('detection overlays (parts/damages/keypoints)'):
            if part_masks:
                parts_overlay = make_parts_overlay(img_bgr, part_masks)
                p1 = out_dir / f'{file_prefix}parts_overlay.jpg'
                cv2.imwrite(str(p1), parts_overlay)
                report['files']['parts_overlay'] = str(p1)

            if damage_masks:
                dmg_overlay = make_damages_overlay(img_bgr, damage_masks)
                p2 = out_dir / f'{file_prefix}damages_overlay.jpg'
                cv2.imwrite(str(p2), dmg_overlay)
                report['files']['damages_overlay'] = str(p2)

            if yolo_keypoints:
                kp_vis = make_keypoints_visualization(img_bgr, yolo_keypoints)
                p3 = out_dir / f'{file_prefix}keypoints.jpg'
                cv2.imwrite(str(p3), kp_vis)
                report['files']['keypoints'] = str(p3)

        if part_masks and damage_masks:
            damaged = find_damaged_parts(
                damage_masks, part_masks, min_overlap_px=self.min_overlap_px,
            )
            report['damaged_parts_2d'] = [
                {'part': dp['part'], 'damage_idx': dp['damage_idx'], 'overlap_px': dp['overlap_px']}
                for dp in damaged
            ]

        pose = None
        pose_error: Optional[str] = None
        if yolo_keypoints:
            with _sw('pose estimation (PnP + focal grid-search)'):
                try:
                    pose = self._solve_pose(
                        img_bgr, yolo_keypoints, source_image_path=source_image_path,
                    )
                    report['pose'] = pose['meta']
                except Exception as exc:
                    pose_error = str(exc)
                    report['error'] = f'pose_estimation_failed: {exc}'
                    try:
                        report['pose'] = {
                            'method': None,
                            'error': str(exc),
                            'keypoints_diagnostics': self._diagnose_keypoints(yolo_keypoints),
                        }
                    except Exception:
                        pass

        if pose is not None:
            with _sw('pose overlays + reprojection viz'):
                R, t, K = pose['R'], pose['t'], pose['K']
                pose_overlay_img = overlay_mesh_on_photo(
                    img_bgr, self.full_render_verts, self.full_render_faces, R, t, K,
                    line_color=(0, 255, 140), fill_color=(0, 255, 140),
                    alpha_fill=0.18, line_thickness=1,
                )
                cv2.putText(
                    pose_overlay_img,
                    f'pose=PnP({pose["meta"]["best_method"]})  '
                    f'mean_err={pose["meta"]["mean_err_px"]:.2f}px',
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 128), 2, cv2.LINE_AA,
                )
                pose_overlay_path = out_dir / f'{file_prefix}pose_overlay.jpg'
                cv2.imwrite(str(pose_overlay_path), pose_overlay_img)
                report['files']['pose_overlay'] = str(pose_overlay_path)

                standalone = render_wireframe(
                    self.full_render_verts, self.full_render_faces, R, t, K, h, w,
                )
                standalone_path = out_dir / f'{file_prefix}pose_render.jpg'
                cv2.imwrite(str(standalone_path), standalone)
                report['files']['pose_render'] = str(standalone_path)

                img_pts = np.array(pose['meta']['image_pts'], dtype=np.float32)
                proj_pts = np.array(pose['meta']['projected_pts'], dtype=np.float32)
                inlier_set = set(pose['meta'].get('ransac_inlier_idx', range(len(pose['meta']['names']))))
                inlier_mask = [i in inlier_set for i in range(len(pose['meta']['names']))]
                kp_reproj = annotate_keypoints(
                    img_bgr, img_pts, proj_pts, pose['meta']['names'],
                    inlier_mask=inlier_mask,
                )
                kp_reproj_path = out_dir / f'{file_prefix}keypoints_reprojection.jpg'
                cv2.imwrite(str(kp_reproj_path), kp_reproj)
                report['files']['keypoints_reprojection'] = str(kp_reproj_path)

        return {
            'img_bgr': img_bgr,
            'part_masks': part_masks,
            'damage_masks': damage_masks,
            'keypoints': yolo_keypoints,
            'pose': pose,
            'pose_error': pose_error,
            'report': report,
        }

    @staticmethod
    def _save_report(out_dir: Path, report: Dict[str, Any]) -> Dict[str, Any]:
        path = out_dir / 'report.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        report['files']['report_json'] = str(path)
        return report
